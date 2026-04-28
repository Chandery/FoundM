from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser('Extract VoCo-L features from NIfTI images')
    ap.add_argument('--config', type=str, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    p = cfg['paths']
    e = cfg['extract']

    script = Path(p['voco_root']) / 'extract_feat_LP.py'
    if not script.exists():
        raise FileNotFoundError(script)

    out_dir = Path(p['features_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    rcfg = cfg.get('runtime', {})
    env.setdefault('MPLCONFIGDIR', str(rcfg.get('mplconfigdir', '/tmp/mpl_cfg')))
    if rcfg.get('hf_modules_cache'):
        env.setdefault('HF_MODULES_CACHE', str(rcfg['hf_modules_cache']))
    if rcfg.get('cuda_visible_devices') is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(rcfg['cuda_visible_devices'])

    cmd = [
        sys.executable,
        str(script),
        '-i', str(p['images_dir']),
        '-o', str(out_dir),
        '--checkpoint', str(p['checkpoint_path']),
        '--feature_size', str(e.get('feature_size', 96)),
        '--batch_size', str(e.get('batch_size', 1)),
        '--num_workers', str(e.get('num_workers', 4)),
        '--roi_size', *(str(x) for x in e.get('roi_size', [336, 336, 320])),
        '--spacing', *(str(x) for x in e.get('spacing', [1.5, 1.5, 1.5])),
    ]

    masks_path = e.get('masks_path')
    if masks_path:
        cmd.extend(['--masks_path', str(masks_path)])
    fg_labels = e.get('fg_labels')
    if fg_labels:
        cmd.extend(['--fg_labels', *(str(x) for x in fg_labels)])

    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == '__main__':
    main()
