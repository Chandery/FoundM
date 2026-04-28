# Session Handoff

## Current State
- Harness paths are repository-relative and portable across servers.
- Canonical conda env is `curia-t2` (old `FoundM` env removed).
- Curia backbone-only cache and Lightning training flow are in place.
- Next pending feature is `feat-005` (full baseline training).

## Restart Commands

```bash
cd /data/chenduoyou/FoundM
./init.sh
```

## Main Config
- `curia_t2/configs/train_covid.yaml`

## Proven Smoke Run
- Config: `curia_t2/configs/train_covid_smoke.yaml`
- Checkpoint: `curia_t2/runs/covid_lightning_smoke/best.ckpt`
- Predictions: `curia_t2/runs/covid_lightning_smoke/val_predictions.csv`

## Full Training Command

```bash
conda run -n curia-t2 python curia_t2/train_curia_task2.py \
  --config curia_t2/configs/train_covid.yaml
```

## Validation Prediction Command

```bash
conda run -n curia-t2 python curia_t2/predict_curia_task2.py \
  --image_dir CVPR26-3DCTFMCompetition/COVID-CT/images \
  --labels_csv CVPR26-3DCTFMCompetition/COVID-CT/labels/covid.csv \
  --label_col covid \
  --split val \
  --local_model_path ./.hf_cache/models--raidium--curia/snapshots/9657dc56276bc6c9503ef6f8d060879c8bee482f \
  --checkpoint curia_t2/runs/covid_full/best.ckpt \
  --output_csv curia_t2/runs/covid_full/val_predictions.csv
```

## Metric Evaluation Command

```bash
conda run -n curia-t2 python curia_t2/eval_predictions.py \
  --pred_csv curia_t2/runs/covid_full/val_predictions.csv \
  --labels_csv CVPR26-3DCTFMCompetition/COVID-CT/labels/covid.csv \
  --label_col covid \
  --split val \
  --prob_col prob_1 \
  --out_json curia_t2/runs/covid_full/val_metrics.json \
  --out_csv curia_t2/runs/covid_full/val_metrics.csv
```

## Notes
- Binary task with balanced sampler requires even `data.batch_size`.
- For quick startup checks, use `curia_t2/configs/train_covid_smoke.yaml`.
