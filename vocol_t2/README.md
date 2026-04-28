# VoCo-L Task2 Pipeline

## 1) Feature Extraction

```bash
conda run -n curia-t2 python vocol_t2/extract_features.py --config vocol_t2/configs/covid.yaml
```

## 2) Train Head (Lightning)

```bash
conda run -n curia-t2 python vocol_t2/train_head_lightning.py --config vocol_t2/configs/covid.yaml
```

## 3) Predict on Validation

```bash
conda run -n curia-t2 python vocol_t2/predict_head.py --config vocol_t2/configs/covid.yaml
```

## 4) Evaluate Metrics (+ confusion matrix PNG)

```bash
conda run -n curia-t2 python vocol_t2/eval_predictions.py --config vocol_t2/configs/covid.yaml
```

## One-command run

```bash
bash vocol_t2/run_all.sh /data/home/3dgs/FoundM/vocol_t2/configs/covid.yaml curia-t2
```
