# Progress Log

## 2026-04-28

### Completed
- Standardized runtime environment to `curia-t2`.
- Removed old conda env `FoundM` after successful migration.
- Converted harness paths from hardcoded absolute paths to repository-relative paths:
  - `init.sh` now derives repo root dynamically.
  - `curia_t2/README.md` command examples now use relative paths.
- Refreshed harness metadata and continuity files for multi-server portability.

### Verification Evidence
- `./init.sh` passes (`[init] done`) with `curia-t2`.
- `conda run -n curia-t2 python -m py_compile curia_t2/*.py` passes.
- `conda run -n curia-t2 python curia_t2/train_curia_task2.py --help` passes.
- `conda run -n curia-t2 python curia_t2/predict_curia_task2.py --help` passes.
- `conda run -n curia-t2 python curia_t2/download_curia.py --help` passes.

### Next Actions
1. Start `feat-005` full baseline training with `curia_t2/configs/train_covid.yaml`.
2. Run val prediction and export metrics CSV/JSON.
3. Record `feat-005` evidence in `feature_list.json`.

## 2026-04-24

### Completed
- Migrated training pipeline to Lightning + YAML config:
  - `curia_t2/train_curia_task2.py`
  - `curia_t2/configs/train_covid.yaml`
  - `curia_t2/configs/train_covid_smoke.yaml`
- Maintained class-balanced training batches via `BalancedBatchSampler`.
- Kept explicit split boundaries:
  - training uses `train_split`
  - validation/testing uses `val_split`.
- Updated prediction loader to support both legacy `best.pt` and Lightning `best.ckpt`.
- Updated docs in `curia_t2/README.md`.

### Verification Evidence
- `./init.sh` passes after refactor.
- Lightning smoke training succeeded:
  - config: `curia_t2/configs/train_covid_smoke.yaml`
  - output ckpt: `curia_t2/runs/covid_lightning_smoke/best.ckpt`
- Prediction from Lightning checkpoint succeeded:
  - output CSV: `curia_t2/runs/covid_lightning_smoke/val_predictions.csv`
  - 65 lines (header + 64 val samples)

### Next Actions
1. Edit `curia_t2/configs/train_covid.yaml` as desired.
2. Launch full training run.
3. Predict on `val` with produced `best.ckpt`.
4. Compare metrics and decide whether to set `model.unfreeze_backbone=true`.
