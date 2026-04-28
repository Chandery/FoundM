from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


def case_to_feature_name(case_id: str) -> str:
    if case_id.endswith('.nii.gz'):
        return case_id[:-7]
    if case_id.endswith('.h5'):
        return case_id[:-3]
    return case_id


class H5FeatureDataset(Dataset):
    def __init__(
        self,
        features_dir: str | Path,
        labels_csv: str | Path,
        label_col: str,
        split: str,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.df = pd.read_csv(labels_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.label_col = label_col
        if label_col not in self.df.columns:
            raise ValueError(f'missing label column: {label_col}')

        self.records: list[dict[str, Any]] = []
        for _, row in self.df.iterrows():
            case_id = str(row['case_id'])
            feat_name = case_to_feature_name(case_id)
            feat_path = self.features_dir / f'{feat_name}.h5'
            if not feat_path.exists():
                raise FileNotFoundError(f'missing feature file: {feat_path}')
            self.records.append({
                'case_id': case_id,
                'feat_path': feat_path,
                'label': int(row[label_col]),
            })

    def __len__(self) -> int:
        return len(self.records)

    def labels(self) -> list[int]:
        return [int(r['label']) for r in self.records]

    def feature_dim(self) -> int:
        with h5py.File(self.records[0]['feat_path'], 'r') as f:
            y = np.asarray(f['y_hat'][:], dtype=np.float32)
        return int(y.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        with h5py.File(r['feat_path'], 'r') as f:
            x = np.asarray(f['y_hat'][:], dtype=np.float32)
        return {
            'case_id': r['case_id'],
            'x': torch.from_numpy(x),
            'y': torch.tensor(r['label'], dtype=torch.long),
        }


def collate_features(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'case_id': [b['case_id'] for b in batch],
        'x': torch.stack([b['x'] for b in batch], dim=0),
        'y': torch.stack([b['y'] for b in batch], dim=0),
    }


class BalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, labels: list[int], batch_size: int, drop_last: bool = True, seed: int = 42):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        self.classes = np.unique(self.labels).tolist()
        self.n_classes = len(self.classes)
        if self.n_classes < 2:
            raise ValueError('BalancedBatchSampler requires >=2 classes')
        if self.batch_size % self.n_classes != 0:
            raise ValueError(f'batch_size ({self.batch_size}) must be divisible by num_classes ({self.n_classes})')
        self.k = self.batch_size // self.n_classes

        self.cls_idx = {c: np.where(self.labels == c)[0] for c in self.classes}
        n = len(self.labels)
        self.n_batches = (n // self.batch_size) if self.drop_last else int(np.ceil(n / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.n_batches):
            b = []
            for c in self.classes:
                pick = rng.choice(self.cls_idx[c], size=self.k, replace=True)
                b.extend(pick.tolist())
            rng.shuffle(b)
            yield b
