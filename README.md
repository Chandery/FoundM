# FoundM Workspace

Multi-track medical imaging workspace with a standardized harness for `curia_t2` and related baselines.

## Repository Layout

- `curia_t2/`: active Curia Task2 pipeline (main working scope)
- `ctnexus_t2/`: CT-NEXUS style Task2 baseline code
- `vocol_t2/`: VoCo style Task2 baseline code
- `CT-NEXUS/`: upstream legacy repo (tracked as git submodule)
- `Large-Scale-Medical_CVPR26CTFM/`: upstream legacy repo (tracked as git submodule)
- `AGENTS.md`, `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`: harness/state files

## Environment

Canonical conda env:

```bash
conda activate curia-t2
```

## Quick Start

From repo root:

```bash
./init.sh
```

Core verification commands:

```bash
conda run -n curia-t2 python -m py_compile curia_t2/*.py
conda run -n curia-t2 python curia_t2/train_curia_task2.py --help
conda run -n curia-t2 python curia_t2/predict_curia_task2.py --help
conda run -n curia-t2 python curia_t2/download_curia.py --help
```

## Multi-Server Sync (Git + Submodules)

Initial clone on a new server:

```bash
git clone git@github.com:Chandery/FoundM.git
cd FoundM
git submodule update --init --recursive
./init.sh
```

Daily sync:

```bash
git pull --rebase
git submodule update --init --recursive
```

When submodule pointers change:

```bash
git add .gitmodules CT-NEXUS Large-Scale-Medical_CVPR26CTFM
git commit -m "chore: update submodule pointers"
git push
```

## Tracking Policy

- Tracked: source code, configs, harness files.
- Ignored: caches, datasets, checkpoints, run outputs (`*/runs/`, `*.ckpt`, `*.pt`, `*.h5`, etc.).

See `.gitignore` for exact rules.
