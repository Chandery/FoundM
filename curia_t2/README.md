# Curia Task2 Pipeline (Workshop)

Minimal pipeline for:

1. Download/cache Curia foundation backbone
2. Train a slice-aggregation classifier on NIfTI volumes
3. Run prediction and export CSV

## Environment

Use your prepared conda env (Python 3.11 + torch + transformers + lightning + datasets + nibabel + SimpleITK).

## 1) Cache Curia model

```bash
python curia_t2/download_curia.py \
  --model_id raidium/curia \
  --cache_dir ./.hf_cache \
  --hf_token "$HF_TOKEN"
```

By default this downloads **backbone-only** files (not all downstream heads).
If you need the full repository, add `--full_repo`.

## 2) Train on COVID-CT (Lightning + Config)

Edit config:
- `curia_t2/configs/train_covid.yaml`

Run:

```bash
python curia_t2/train_curia_task2.py --config curia_t2/configs/train_covid.yaml
```

Key notes:
- Training uses `data.train_split` (default `train`)
- Validation uses `data.val_split` (default `val`)
- Balanced class sampling is controlled by `data.balanced_sampler` (default `true`)
- Keep `data.batch_size` divisible by class count (binary task => even number)

## 3) Predict

```bash
python curia_t2/predict_curia_task2.py \
  --image_dir CVPR26-3DCTFMCompetition/COVID-CT/images \
  --labels_csv CVPR26-3DCTFMCompetition/COVID-CT/labels/covid.csv \
  --label_col covid \
  --split val \
  --local_model_path ./.hf_cache/models--raidium--curia/snapshots/9657dc56276bc6c9503ef6f8d060879c8bee482f \
  --checkpoint curia_t2/runs/covid_full/best.ckpt \
  --output_csv curia_t2/runs/covid_full/val_predictions.csv
```

Output CSV columns:
- `case_id`
- `pred`
- `prob_0 ... prob_{num_classes-1}`


## 4) Evaluate Metrics

```bash
python curia_t2/eval_predictions.py \
  --pred_csv curia_t2/runs/covid_full/val_predictions.csv \
  --labels_csv CVPR26-3DCTFMCompetition/COVID-CT/labels/covid.csv \
  --label_col covid \
  --split val \
  --prob_col prob_1 \
  --out_json curia_t2/runs/covid_full/val_metrics.json \
  --out_csv curia_t2/runs/covid_full/val_metrics.csv
```
