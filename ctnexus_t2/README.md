# CT-NEXUS Task2 Pipeline

## 1) Feature Extraction

```bash
conda run -n curia-t2 python ctnexus_t2/extract_features.py --config ctnexus_t2/configs/covid.yaml
```

## 2) Train Head (Lightning)

```bash
conda run -n curia-t2 python ctnexus_t2/train_head_lightning.py --config ctnexus_t2/configs/covid.yaml
```

## 3) Predict on Validation

```bash
conda run -n curia-t2 python ctnexus_t2/predict_head.py --config ctnexus_t2/configs/covid.yaml
```

## 4) Evaluate Metrics

```bash
conda run -n curia-t2 python ctnexus_t2/eval_predictions.py --config ctnexus_t2/configs/covid.yaml
```

`ctnexus_t2/configs/covid.yaml` uses `runtime.cuda_visible_devices: "1"` and `trainer.devices: 1`, so extraction/training are pinned to one GPU.
