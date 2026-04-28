from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn
from tqdm import tqdm

from utils import H5FeatureDataset


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser('Predict with trained head on CT-NEXUS features')
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--checkpoint', type=str, default=None)
    ap.add_argument('--output_csv', type=str, default=None)
    return ap.parse_args()


def build_head(in_dim: int, cfg: dict) -> nn.Module:
    mcfg = cfg['model']
    hidden = int(mcfg.get('hidden_dim', 0))
    dropout = float(mcfg.get('dropout', 0.1))
    num_classes = int(mcfg.get('num_classes', 2))
    if hidden > 0:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    return nn.Linear(in_dim, num_classes)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    dcfg = cfg['data']
    split = cfg.get('predict', {}).get('split', dcfg.get('val_split', 'val'))
    ds = H5FeatureDataset(cfg['paths']['features_dir'], cfg['paths']['labels_csv'], dcfg.get('label_col','covid'), split)

    in_dim = ds.feature_dim()
    head = build_head(in_dim, cfg)

    ckpt_path = args.checkpoint or cfg.get('predict', {}).get('checkpoint') or str(Path(cfg['paths']['run_dir']) / 'best.ckpt')
    out_csv = args.output_csv or cfg.get('predict', {}).get('output_csv') or str(Path(cfg['paths']['run_dir']) / f'{split}_predictions.csv')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state = {k[len('net.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('net.')}
    elif 'model_state' in ckpt:
        state = ckpt['model_state']
    else:
        raise ValueError('Unsupported checkpoint format')
    head.load_state_dict(state, strict=True)
    head.eval()

    rows = []
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc='predict'):
            sample = ds[i]
            x = sample['x'].unsqueeze(0)
            logits = head(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(logits, dim=1).item())
            rows.append({
                'case_id': sample['case_id'],
                'pred': pred,
                'prob_0': float(probs[0].item()),
                'prob_1': float(probs[1].item()),
            })

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Saved predictions to {out}')


if __name__ == '__main__':
    main()
