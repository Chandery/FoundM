from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser('Evaluate CT-NEXUS predictions with config')
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--pred_csv', type=str, default=None)
    ap.add_argument('--out_json', type=str, default=None)
    ap.add_argument('--out_csv', type=str, default=None)
    ap.add_argument('--cm_png', type=str, default=None)
    return ap.parse_args()


def plot_confusion_matrix(cm: np.ndarray, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=['Pred 0', 'Pred 1'],
        yticklabels=['True 0', 'True 1'],
        ylabel='Ground Truth',
        xlabel='Prediction',
        title='Confusion Matrix',
    )

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f'{int(cm[i, j])}',
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=12,
            )
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    dcfg = cfg['data']
    pcfg = cfg.get('predict', {})
    paths = cfg['paths']
    split = pcfg.get('split', dcfg.get('val_split', 'val'))

    pred_csv = args.pred_csv or pcfg.get('output_csv') or str(Path(paths['run_dir']) / f'{split}_predictions.csv')
    out_json = args.out_json or str(Path(paths['run_dir']) / f'{split}_metrics.json')
    out_csv = args.out_csv or str(Path(paths['run_dir']) / f'{split}_metrics.csv')
    cm_png = args.cm_png or str(Path(paths['run_dir']) / f'{split}_confusion_matrix.png')
    labels_csv = paths['labels_csv']
    label_col = dcfg.get('label_col', 'covid')
    prob_col = 'prob_1'

    df_pred = pd.read_csv(pred_csv)
    df_label = pd.read_csv(labels_csv)

    df_eval = df_label[df_label['split'] == split][['case_id', label_col]].copy()
    merged = df_eval.merge(df_pred, on='case_id', how='inner')

    missing = sorted(set(df_eval['case_id']) - set(merged['case_id']))
    extra = sorted(set(df_pred['case_id']) - set(merged['case_id']))

    y_true = merged[label_col].astype(int)
    y_pred = merged['pred'].astype(int)
    y_prob = merged[prob_col].astype(float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        'pred_csv': str(pred_csv),
        'labels_csv': str(labels_csv),
        'label_col': str(label_col),
        'split': str(split),
        'num_samples': int(len(merged)),
        'num_missing_from_pred': int(len(missing)),
        'num_extra_in_pred': int(len(extra)),
        'acc': float(accuracy_score(y_true, y_pred)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'auroc': float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else None,
        'sensitivity': float(recall_score(y_true, y_pred, pos_label=1)),
        'specificity': float(recall_score(y_true, y_pred, pos_label=0)),
        'confusion_matrix_0_1': cm.tolist(),
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    plot_confusion_matrix(cm, cm_png)
    print(f'Saved JSON metrics: {out_json}')
    print(f'Saved CSV metrics: {out_csv}')
    print(f'Saved confusion matrix plot: {cm_png}')


if __name__ == '__main__':
    main()
