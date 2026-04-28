from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Any

if not os.environ.get("HF_MODULES_CACHE"):
    candidates = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "modules")
    candidates.append(Path.cwd() / ".hf_cache" / "modules")
    candidates.append(Path("/tmp/hf_modules_cache"))
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            os.environ["HF_MODULES_CACHE"] = str(c)
            break
        except Exception:
            continue

import lightning as L
import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from transformers import AutoImageProcessor, AutoModel

from utils import BalancedBatchSampler, CTVolumeDataset, CuriaSliceAggregator, collate_with_processor


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Train Curia Task2 classifier (Lightning + YAML config)")
    ap.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    return ap.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    return cfg


def stratified_take_indices(labels: list[int], max_samples: int, seed: int) -> list[int]:
    if max_samples <= 0 or max_samples >= len(labels):
        return list(range(len(labels)))
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels_arr).tolist()
    chunks: list[np.ndarray] = []
    per_class = max(1, max_samples // len(classes))
    for c in classes:
        idx = np.where(labels_arr == c)[0]
        if len(idx) == 0:
            continue
        take = min(len(idx), per_class)
        chunks.append(rng.choice(idx, size=take, replace=False))
    base = np.concatenate(chunks) if chunks else np.array([], dtype=np.int64)
    if len(base) < max_samples:
        remain_pool = np.setdiff1d(np.arange(len(labels_arr)), base, assume_unique=False)
        if len(remain_pool) > 0:
            extra = rng.choice(remain_pool, size=min(max_samples - len(base), len(remain_pool)), replace=False)
            base = np.concatenate([base, extra])
    rng.shuffle(base)
    return base[:max_samples].tolist()


def infer_hidden_size(backbone: nn.Module) -> int:
    cfg = getattr(backbone, "config", None)
    for key in ["hidden_size", "projection_dim", "embed_dim", "dim"]:
        if cfg is not None and hasattr(cfg, key):
            return int(getattr(cfg, key))
    raise RuntimeError("Could not infer hidden size from backbone.config")


def _normalize_devices(devices_cfg: Any) -> Any:
    if devices_cfg is None:
        return "auto"
    if isinstance(devices_cfg, int):
        return devices_cfg
    if isinstance(devices_cfg, (list, tuple)):
        out = []
        for d in devices_cfg:
            if isinstance(d, str) and d.startswith("cuda:"):
                out.append(int(d.split(":", 1)[1]))
            else:
                out.append(int(d))
        return out
    if isinstance(devices_cfg, str):
        s = devices_cfg.strip()
        if s.lower() == "auto":
            return "auto"
        if s.startswith("cuda:"):
            return [int(s.split(":", 1)[1])]
        if "," in s:
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        if s.isdigit():
            return [int(s)]
    return devices_cfg


class CuriaDataModule(L.LightningDataModule):
    def __init__(self, cfg: dict[str, Any], processor: Any, seed: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.processor = processor
        self.seed = int(seed)
        self.train_dataset = None
        self.val_dataset = None
        self.train_labels: list[int] = []
        self._train_batch_sampler = None

    def setup(self, stage: str | None = None) -> None:
        data_cfg = self.cfg["data"]
        train_ds = CTVolumeDataset(
            image_dir=data_cfg["image_dir"],
            labels_csv=data_cfg["labels_csv"],
            label_col=data_cfg["label_col"],
            split=data_cfg.get("train_split", "train"),
            num_slices=int(data_cfg.get("num_slices", 32)),
            plane=data_cfg.get("plane", "axial"),
            require_label=True,
        )
        val_ds = CTVolumeDataset(
            image_dir=data_cfg["image_dir"],
            labels_csv=data_cfg["labels_csv"],
            label_col=data_cfg["label_col"],
            split=data_cfg.get("val_split", "val"),
            num_slices=int(data_cfg.get("num_slices", 32)),
            plane=data_cfg.get("plane", "axial"),
            require_label=True,
        )

        max_train = int(data_cfg.get("max_train_samples", 0))
        max_val = int(data_cfg.get("max_val_samples", 0))
        if max_train > 0:
            t_idx = stratified_take_indices(train_ds.labels(), max_train, seed=self.seed)
            train_ds = Subset(train_ds, t_idx)
        if max_val > 0:
            v_idx = stratified_take_indices(val_ds.labels(), max_val, seed=self.seed + 1)
            val_ds = Subset(val_ds, v_idx)

        if isinstance(train_ds, Subset):
            base = train_ds.dataset
            self.train_labels = [base.labels()[i] for i in train_ds.indices]
        else:
            self.train_labels = train_ds.labels()

        unique, counts = np.unique(np.asarray(self.train_labels), return_counts=True)
        print("train label distribution:", {int(k): int(v) for k, v in zip(unique, counts)})

        self.train_dataset = train_ds
        self.val_dataset = val_ds

    def train_dataloader(self):
        data_cfg = self.cfg["data"]
        batch_size = int(data_cfg.get("batch_size", 2))
        num_workers = int(data_cfg.get("num_workers", 0))
        use_balanced = bool(data_cfg.get("balanced_sampler", True))
        collate_fn = partial(collate_with_processor, processor=self.processor, device=torch.device("cpu"))

        if use_balanced:
            self._train_batch_sampler = BalancedBatchSampler(
                labels=self.train_labels,
                batch_size=batch_size,
                drop_last=True,
                seed=self.seed,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=self._train_batch_sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        data_cfg = self.cfg["data"]
        collate_fn = partial(collate_with_processor, processor=self.processor, device=torch.device("cpu"))
        return DataLoader(
            self.val_dataset,
            batch_size=int(data_cfg.get("batch_size", 2)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def set_train_epoch(self, epoch: int) -> None:
        if self._train_batch_sampler is not None:
            self._train_batch_sampler.set_epoch(epoch)


class CuriaTask2LightningModule(L.LightningModule):
    def __init__(self, cfg: dict[str, Any], backbone: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]
        hidden_size = infer_hidden_size(backbone)
        self.model = CuriaSliceAggregator(
            backbone=backbone,
            hidden_size=hidden_size,
            num_classes=int(model_cfg.get("num_classes", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.val_labels: list[int] = []
        self.val_preds: list[int] = []
        self.val_probs: list[list[float]] = []

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)

    def on_train_epoch_start(self) -> None:
        dm = self.trainer.datamodule
        if dm is not None and hasattr(dm, "set_train_epoch"):
            dm.set_train_epoch(self.current_epoch)

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        logits = self(batch["pixel_values"])
        loss = self.criterion(logits, batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["labels"].shape[0])
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_labels.clear()
        self.val_preds.clear()
        self.val_probs.clear()

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        logits = self(batch["pixel_values"])
        loss = self.criterion(logits, batch["labels"])
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        self.val_labels.extend(batch["labels"].detach().cpu().tolist())
        self.val_preds.extend(preds.detach().cpu().tolist())
        self.val_probs.extend(probs.detach().cpu().tolist())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["labels"].shape[0])

    def on_validation_epoch_end(self) -> None:
        if not self.val_labels:
            return
        y_true = np.asarray(self.val_labels, dtype=np.int64)
        y_pred = np.asarray(self.val_preds, dtype=np.int64)
        y_prob = np.asarray(self.val_probs, dtype=np.float32)

        balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) > 1:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                auroc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                auroc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        else:
            auroc = 0.0

        self.log("val_balanced_acc", balanced_acc, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val_auroc", auroc, prog_bar=True, on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        optim_cfg = self.cfg["optim"]
        trainable = [p for p in self.parameters() if p.requires_grad]
        return AdamW(trainable, lr=float(optim_cfg.get("lr", 2e-4)), weight_decay=float(optim_cfg.get("wd", 1e-4)))


def build_backbone_and_processor(cfg: dict[str, Any]) -> tuple[Any, nn.Module]:
    model_cfg = cfg["model"]
    cache_dir = model_cfg.get("cache_dir")
    if not os.environ.get("HF_MODULES_CACHE"):
        base_cache = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "huggingface")
        modules_cache = base_cache / "modules"
        modules_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_MODULES_CACHE"] = str(modules_cache)

    token = model_cfg.get("hf_token") if model_cfg.get("hf_token") else True
    model_src = model_cfg.get("local_model_path") or model_cfg.get("model_id", "raidium/curia")
    local_only = bool(model_cfg.get("local_model_path"))

    processor = AutoImageProcessor.from_pretrained(
        model_src,
        trust_remote_code=True,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_only,
    )
    backbone = AutoModel.from_pretrained(
        model_src,
        trust_remote_code=True,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_only,
    )

    if not bool(model_cfg.get("unfreeze_backbone", False)):
        for p in backbone.parameters():
            p.requires_grad = False

    return processor, backbone


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    L.seed_everything(seed, workers=True)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    processor, backbone = build_backbone_and_processor(cfg)
    data_module = CuriaDataModule(cfg=cfg, processor=processor, seed=seed)
    module = CuriaTask2LightningModule(cfg=cfg, backbone=backbone)

    ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=str(out_dir),
        filename="best",
        monitor="val_balanced_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")

    logger = L.pytorch.loggers.CSVLogger(save_dir=str(out_dir), name="logs")

    trainer_cfg = cfg["trainer"]
    accelerator = trainer_cfg.get("accelerator", "auto")
    devices = _normalize_devices(trainer_cfg.get("devices", "auto"))
    strategy = trainer_cfg.get("strategy", "auto")

    if isinstance(devices, list) and len(devices) == 1:
        if accelerator in ("auto", "gpu"):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(devices[0]))
            accelerator = "gpu"
            devices = 1
            strategy = "auto"

    # If user requests single-device training, ignore external distributed launcher env.
    if strategy == "single_device":
        for k in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"]:
            os.environ.pop(k, None)

    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requests GPU training, but CUDA is not available in current runtime. "
            "For local smoke test, set trainer.accelerator=cpu and trainer.devices=1."
        )

    trainer = L.Trainer(
        max_epochs=int(trainer_cfg.get("epochs", 10)),
        fast_dev_run=bool(trainer_cfg.get("fast_dev_run", False)),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=trainer_cfg.get("precision", "32-true"),
        deterministic=bool(trainer_cfg.get("deterministic", True)),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 10)),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 0.0)),
        num_sanity_val_steps=int(trainer_cfg.get("num_sanity_val_steps", 2)),
        callbacks=[ckpt_cb, lr_monitor],
        logger=logger,
    )

    trainer.fit(module, datamodule=data_module)

    best_metrics = {
        "best_model_path": ckpt_cb.best_model_path,
        "best_val_balanced_acc": float(ckpt_cb.best_model_score.item()) if ckpt_cb.best_model_score is not None else None,
    }
    with open(out_dir / "best.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(best_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
