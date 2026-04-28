from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser('Extract CT-NEXUS features from NIfTI images')
    ap.add_argument('--config', type=str, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    p = cfg['paths']
    e = cfg['extract']

    script = Path(p['ctnexus_root']) / 'src' / 'feature_extraction' / 'extract_feat_LP.py'
    if not script.exists():
        raise FileNotFoundError(script)

    out_dir = Path(p['features_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['PYTHONPATH'] = f"{Path(p['ctnexus_root']) / 'src'}:{env.get('PYTHONPATH','')}"
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
        '--batch_size', str(e.get('batch_size', 1)),
        '--num_workers', str(e.get('num_workers', 0)),
        '--num_classes', str(e.get('num_classes', 2)),
    ]

    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == '__main__':
    main()
