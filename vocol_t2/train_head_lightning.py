from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from utils import BalancedBatchSampler, H5FeatureDataset, collate_features


class HeadModule(L.LightningModule):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg['model']
        hidden = int(mcfg.get('hidden_dim', 0))
        dropout = float(mcfg.get('dropout', 0.1))
        num_classes = int(mcfg.get('num_classes', 2))

        print(f"hidden_dim:{hidden}")
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.val_y = []
        self.val_p = []
        self.val_prob = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        logits = self(batch['x'])
        loss = self.criterion(logits, batch['y'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch['y'].shape[0])
        return loss

    def on_validation_epoch_start(self):
        self.val_y.clear(); self.val_p.clear(); self.val_prob.clear()

    def validation_step(self, batch, batch_idx):
        logits = self(batch['x'])
        loss = self.criterion(logits, batch['y'])
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        self.val_y.extend(batch['y'].detach().cpu().tolist())
        self.val_p.extend(preds.detach().cpu().tolist())
        self.val_prob.extend(probs.detach().cpu().tolist())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=batch['y'].shape[0])

    def on_validation_epoch_end(self):
        if len(self.val_y) == 0:
            return
        y = np.asarray(self.val_y, dtype=np.int64)
        p = np.asarray(self.val_p, dtype=np.int64)
        prob = np.asarray(self.val_prob, dtype=np.float32)

        bacc = float(balanced_accuracy_score(y, p))
        f1 = float(f1_score(y, p))
        if len(np.unique(y)) > 1:
            auroc = float(roc_auc_score(y, prob[:, 1]))
        else:
            auroc = 0.0

        self.log('val_balanced_acc', bacc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_auroc', auroc, prog_bar=True)

    def configure_optimizers(self):
        ocfg = self.cfg['optim']
        return AdamW(self.parameters(), lr=float(ocfg.get('lr', 2e-4)), weight_decay=float(ocfg.get('wd', 1e-4)))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser('Train classification head on CT-NEXUS h5 features')
    ap.add_argument('--config', type=str, required=True)
    return ap.parse_args()


def apply_runtime_env(cfg: dict) -> None:
    rcfg = cfg.get('runtime', {})
    if rcfg.get('hf_modules_cache'):
        os.environ.setdefault('HF_MODULES_CACHE', str(rcfg['hf_modules_cache']))
    if rcfg.get('mplconfigdir'):
        os.environ.setdefault('MPLCONFIGDIR', str(rcfg['mplconfigdir']))
    if rcfg.get('cuda_visible_devices') is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rcfg['cuda_visible_devices'])


def resolve_devices_and_accelerator(tcfg: dict) -> tuple[str, object]:
    accelerator = tcfg.get('accelerator', 'auto')
    devices = tcfg.get('devices', 'auto')
    if isinstance(devices, str) and devices.startswith('cuda:'):
        accelerator = 'gpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = devices.split(':', 1)[1]
        devices = 1
    return accelerator, devices


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    apply_runtime_env(cfg)
    L.seed_everything(int(cfg.get('seed', 42)), workers=True)

    dcfg = cfg['data']
    tcfg = cfg['trainer']
    out_dir = Path(cfg['paths']['run_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = H5FeatureDataset(cfg['paths']['features_dir'], cfg['paths']['labels_csv'], dcfg.get('label_col','covid'), dcfg.get('train_split','train'))
    val_ds = H5FeatureDataset(cfg['paths']['features_dir'], cfg['paths']['labels_csv'], dcfg.get('label_col','covid'), dcfg.get('val_split','val'))

    labels = train_ds.labels()
    uniq, cnt = np.unique(np.asarray(labels), return_counts=True)
    print('train label distribution:', {int(k): int(v) for k, v in zip(uniq, cnt)})

    if bool(dcfg.get('balanced_sampler', True)):
        bsamp = BalancedBatchSampler(labels, int(dcfg.get('batch_size', 8)), seed=int(cfg.get('seed', 42)))
        train_loader = DataLoader(train_ds, batch_sampler=bsamp, num_workers=int(dcfg.get('num_workers', 0)), collate_fn=collate_features)
    else:
        train_loader = DataLoader(train_ds, batch_size=int(dcfg.get('batch_size', 8)), shuffle=True, num_workers=int(dcfg.get('num_workers', 0)), collate_fn=collate_features)

    val_loader = DataLoader(val_ds, batch_size=int(dcfg.get('batch_size', 8)), shuffle=False, num_workers=int(dcfg.get('num_workers', 0)), collate_fn=collate_features)

    in_dim = train_ds.feature_dim()
    print(cfg)
    module = HeadModule(in_dim=in_dim, cfg=cfg)

    ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=str(out_dir),
        filename='best',
        monitor='val_balanced_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
    )
    logger = L.pytorch.loggers.CSVLogger(save_dir=str(out_dir), name='logs')

    accelerator, devices = resolve_devices_and_accelerator(tcfg)
    trainer = L.Trainer(
        max_epochs=int(tcfg.get('epochs', 20)),
        fast_dev_run=bool(tcfg.get('fast_dev_run', False)),
        accelerator=accelerator,
        devices=devices,
        strategy=tcfg.get('strategy', 'auto'),
        precision=tcfg.get('precision', '32-true'),
        deterministic=bool(tcfg.get('deterministic', True)),
        log_every_n_steps=int(tcfg.get('log_every_n_steps', 10)),
        callbacks=[ckpt_cb],
        logger=logger,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best = {
        'best_model_path': ckpt_cb.best_model_path,
        'best_val_balanced_acc': float(ckpt_cb.best_model_score.item()) if ckpt_cb.best_model_score is not None else None,
    }
    (out_dir / 'best.json').write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(best, ensure_ascii=False))


if __name__ == '__main__':
    main()
