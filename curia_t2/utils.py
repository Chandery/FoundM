from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler


@dataclass
class Sample:
    case_id: str
    image_path: Path
    label: int | None


def _pick_axis(plane: str) -> int:
    plane = plane.lower()
    if plane == "axial":
        return 2
    if plane == "coronal":
        return 1
    if plane == "sagittal":
        return 0
    raise ValueError(f"Unsupported plane={plane}. Use axial/coronal/sagittal")


def _sample_indices(length: int, num_slices: int) -> np.ndarray:
    if length <= 0:
        return np.array([0], dtype=np.int64)
    if num_slices <= 1:
        return np.array([length // 2], dtype=np.int64)
    return np.linspace(0, length - 1, num=num_slices, dtype=np.int64)


def _extract_slices(volume: np.ndarray, axis: int, num_slices: int) -> list[np.ndarray]:
    volume = np.moveaxis(volume, axis, 0)
    indices = _sample_indices(volume.shape[0], num_slices)
    out: list[np.ndarray] = []
    for i in indices:
        s = volume[i].astype(np.float32, copy=False)
        out.append(s)
    return out


def read_volume(path: Path) -> np.ndarray:
    arr = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={arr.shape} from {path}")
    return arr


class CTVolumeDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        labels_csv: str | Path,
        label_col: str,
        split: str | None,
        num_slices: int,
        plane: str,
        require_label: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.df = pd.read_csv(labels_csv)
        self.label_col = label_col
        self.num_slices = int(num_slices)
        self.axis = _pick_axis(plane)
        self.require_label = require_label

        if split is not None:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.samples: list[Sample] = []
        for _, row in self.df.iterrows():
            case_id = str(row["case_id"])
            image_path = self.image_dir / case_id
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image: {image_path}")

            label: int | None
            if label_col in row and not pd.isna(row[label_col]):
                label = int(row[label_col])
            else:
                label = None

            if self.require_label and label is None:
                raise ValueError(f"Label missing for case_id={case_id}, label_col={label_col}")

            self.samples.append(Sample(case_id=case_id, image_path=image_path, label=label))

    def __len__(self) -> int:
        return len(self.samples)

    def labels(self) -> list[int]:
        out: list[int] = []
        for s in self.samples:
            if s.label is None:
                raise ValueError("Dataset contains missing labels; cannot build label list.")
            out.append(int(s.label))
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        vol = read_volume(sample.image_path)
        slices = _extract_slices(vol, self.axis, self.num_slices)
        item: dict[str, Any] = {
            "case_id": sample.case_id,
            "slices": slices,
        }
        if sample.label is not None:
            item["label"] = sample.label
        return item


class CuriaSliceAggregator(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    @staticmethod
    def _extract_embedding(outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state"):
            hs = outputs.last_hidden_state
            return hs[:, 0] if hs.ndim == 3 else hs
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            x = outputs[0]
            return x[:, 0] if x.ndim == 3 else x
        raise RuntimeError("Could not extract embeddings from backbone outputs")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, S, C, H, W]
        bsz, n_slices, c, h, w = pixel_values.shape
        flat = pixel_values.reshape(bsz * n_slices, c, h, w)
        outputs = self.backbone(pixel_values=flat)
        emb = self._extract_embedding(outputs)  # [B*S, D]
        emb = emb.reshape(bsz, n_slices, -1)

        # Attention pooling across slices.
        q = self.query / (self.query.norm() + 1e-6)
        scores = torch.einsum("bsd,d->bs", emb, q)
        attn = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bs,bsd->bd", attn, emb)

        logits = self.classifier(self.dropout(pooled))
        return logits


def collate_with_processor(batch: list[dict[str, Any]], processor: Any, device: torch.device) -> dict[str, Any]:
    case_ids = [x["case_id"] for x in batch]
    labels = [x.get("label", None) for x in batch]
    slices_per_case = [x["slices"] for x in batch]
    n_slices = len(slices_per_case[0])
    if any(len(s) != n_slices for s in slices_per_case):
        raise ValueError("All samples in batch must have same num_slices")

    flat_slices: list[np.ndarray] = []
    for slices in slices_per_case:
        flat_slices.extend(slices)

    proc = processor(images=flat_slices, return_tensors="pt")
    pixel_values = proc["pixel_values"]  # [B*S, C, H, W]

    bsz = len(batch)
    pixel_values = pixel_values.reshape(bsz, n_slices, *pixel_values.shape[1:]).to(device)

    out: dict[str, Any] = {
        "case_id": case_ids,
        "pixel_values": pixel_values,
    }
    if all(l is not None for l in labels):
        out["labels"] = torch.tensor(labels, dtype=torch.long, device=device)
    return out


class BalancedBatchSampler(Sampler[list[int]]):
    """
    Class-balanced batch sampler.
    Each batch contains equal number of samples per class.
    """

    def __init__(self, labels: list[int], batch_size: int, drop_last: bool = True, seed: int = 42) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

        self.classes = np.unique(self.labels).tolist()
        self.n_classes = len(self.classes)
        if self.n_classes < 2:
            raise ValueError("BalancedBatchSampler requires at least 2 classes.")
        if self.batch_size % self.n_classes != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by number of classes ({self.n_classes})."
            )
        self.samples_per_class = self.batch_size // self.n_classes
        self.class_to_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        for c, ids in self.class_to_indices.items():
            if len(ids) == 0:
                raise ValueError(f"Class {c} has no samples.")

        n = len(self.labels)
        self.num_batches = (n // self.batch_size) if self.drop_last else int(np.ceil(n / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        for _ in range(self.num_batches):
            batch: list[int] = []
            for c in self.classes:
                cls_ids = self.class_to_indices[c]
                chosen = rng.choice(cls_ids, size=self.samples_per_class, replace=True)
                batch.extend(chosen.tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
