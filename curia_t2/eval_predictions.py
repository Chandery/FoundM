from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Evaluate prediction CSV against label CSV")
    ap.add_argument("--pred_csv", type=str, required=True, help="Prediction CSV with case_id,pred,prob_* columns")
    ap.add_argument("--labels_csv", type=str, required=True, help="Ground-truth CSV (contains case_id, label column, split)")
    ap.add_argument("--label_col", type=str, default="covid")
    ap.add_argument("--split", type=str, default="val", help="Evaluate on this split in labels CSV")
    ap.add_argument("--prob_col", type=str, default="prob_1", help="Probability column for positive class")
    ap.add_argument("--out_json", type=str, default=None, help="Optional path to save metrics JSON")
    ap.add_argument("--out_csv", type=str, default=None, help="Optional path to save metrics CSV")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    pred_path = Path(args.pred_csv)
    label_path = Path(args.labels_csv)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing pred_csv: {pred_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing labels_csv: {label_path}")

    df_pred = pd.read_csv(pred_path)
    df_label = pd.read_csv(label_path)

    if "case_id" not in df_pred.columns:
        raise ValueError("pred_csv must contain column: case_id")
    if "pred" not in df_pred.columns:
        raise ValueError("pred_csv must contain column: pred")
    if args.prob_col not in df_pred.columns:
        raise ValueError(f"pred_csv missing probability column: {args.prob_col}")
    if "case_id" not in df_label.columns:
        raise ValueError("labels_csv must contain column: case_id")
    if args.label_col not in df_label.columns:
        raise ValueError(f"labels_csv missing label column: {args.label_col}")
    if "split" not in df_label.columns:
        raise ValueError("labels_csv must contain column: split")

    df_eval = df_label[df_label["split"] == args.split][["case_id", args.label_col]].copy()
    merged = df_eval.merge(df_pred, on="case_id", how="inner")

    missing = sorted(set(df_eval["case_id"]) - set(merged["case_id"]))
    extra = sorted(set(df_pred["case_id"]) - set(merged["case_id"]))

    y_true = merged[args.label_col].astype(int)
    y_pred = merged["pred"].astype(int)
    y_prob = merged[args.prob_col].astype(float)

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, pos_label=1))
    spec = float(recall_score(y_true, y_pred, pos_label=0))
    auroc = float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    metrics = {
        "pred_csv": str(pred_path),
        "labels_csv": str(label_path),
        "label_col": args.label_col,
        "split": args.split,
        "num_samples": int(len(merged)),
        "num_missing_from_pred": int(len(missing)),
        "num_extra_in_pred": int(len(extra)),
        "acc": acc,
        "balanced_acc": bacc,
        "f1": f1,
        "auroc": auroc,
        "sensitivity": sens,
        "specificity": spec,
        "confusion_matrix_0_1": cm,
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON metrics: {out_json}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(out_csv, index=False)
        print(f"Saved CSV metrics: {out_csv}")


if __name__ == "__main__":
    main()
