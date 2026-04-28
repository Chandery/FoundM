from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
if not os.environ.get("HF_MODULES_CACHE"):
    candidates = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "modules")
    candidates.append(Path.cwd() / ".hf_cache" / "modules")
    candidates.append(Path("/tmp/hf_modules_cache"))
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            os.environ["HF_MODULES_CACHE"] = str(c)
            break
        except Exception:
            continue

from transformers import AutoImageProcessor, AutoModel

from utils import CTVolumeDataset, CuriaSliceAggregator, collate_with_processor


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Predict with trained Curia Task2 model")
    ap.add_argument("--config", type=str, default=None, help="YAML config path (recommended)")

    ap.add_argument("--image_dir", type=str, default=None)
    ap.add_argument("--labels_csv", type=str, default=None)
    ap.add_argument("--label_col", type=str, default=None)
    ap.add_argument("--split", type=str, default=None)

    ap.add_argument("--model_id", type=str, default=None)
    ap.add_argument("--local_model_path", type=str, default=None, help="Local snapshot path (preferred for offline use)")
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--cache_dir", type=str, default=None)

    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--num_slices", type=int, default=None)
    ap.add_argument("--plane", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)

    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--output_csv", type=str, default=None)
    return ap.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    return cfg


def merge_with_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        required = ["image_dir", "labels_csv", "label_col", "split", "num_classes", "num_slices", "plane", "batch_size", "num_workers", "checkpoint", "output_csv"]
        missing = [k for k in required if getattr(args, k) is None]
        if missing:
            raise ValueError(f"Missing required args without --config: {missing}")
        return args

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    pred_cfg = cfg.get("predict", {})
    output_cfg = cfg.get("output", {})

    args.image_dir = args.image_dir or data_cfg.get("image_dir")
    args.labels_csv = args.labels_csv or data_cfg.get("labels_csv")
    args.label_col = args.label_col or data_cfg.get("label_col", "covid")
    args.split = args.split or pred_cfg.get("split", data_cfg.get("val_split", "val"))

    args.model_id = args.model_id or model_cfg.get("model_id", "raidium/curia")
    args.local_model_path = args.local_model_path or model_cfg.get("local_model_path")
    args.hf_token = args.hf_token or model_cfg.get("hf_token")
    args.cache_dir = args.cache_dir or model_cfg.get("cache_dir")

    args.num_classes = args.num_classes or int(model_cfg.get("num_classes", 2))
    args.num_slices = args.num_slices or int(data_cfg.get("num_slices", 32))
    args.plane = args.plane or data_cfg.get("plane", "axial")
    args.batch_size = args.batch_size or int(data_cfg.get("batch_size", 2))
    args.num_workers = args.num_workers if args.num_workers is not None else int(data_cfg.get("num_workers", 0))

    default_ckpt = Path(output_cfg.get("dir", ".")) / "best.ckpt"
    default_csv = Path(output_cfg.get("dir", ".")) / "val_predictions.csv"
    args.checkpoint = args.checkpoint or pred_cfg.get("checkpoint") or str(default_ckpt)
    args.output_csv = args.output_csv or pred_cfg.get("output_csv") or str(default_csv)

    required = ["image_dir", "labels_csv", "label_col", "split", "num_classes", "num_slices", "plane", "batch_size", "num_workers", "checkpoint", "output_csv"]
    missing = [k for k in required if getattr(args, k) is None]
    if missing:
        raise ValueError(f"Missing required fields after config merge: {missing}")
    return args


def infer_hidden_size(backbone: torch.nn.Module) -> int:
    cfg = getattr(backbone, "config", None)
    for key in ["hidden_size", "projection_dim", "embed_dim", "dim"]:
        if cfg is not None and hasattr(cfg, key):
            return int(getattr(cfg, key))
    raise RuntimeError("Could not infer hidden size from backbone.config")


def main() -> None:
    args = merge_with_config(parse_args())

    if not os.environ.get("HF_MODULES_CACHE"):
        base_cache = Path(args.cache_dir) if args.cache_dir else (Path.home() / ".cache" / "huggingface")
        modules_cache = base_cache / "modules"
        modules_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_MODULES_CACHE"] = str(modules_cache)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token = args.hf_token if args.hf_token else True
    model_src = args.local_model_path if args.local_model_path else args.model_id
    local_only = bool(args.local_model_path)

    processor = AutoImageProcessor.from_pretrained(
        model_src,
        trust_remote_code=True,
        token=token,
        cache_dir=args.cache_dir,
        local_files_only=local_only,
    )
    backbone = AutoModel.from_pretrained(
        model_src,
        trust_remote_code=True,
        token=token,
        cache_dir=args.cache_dir,
        local_files_only=local_only,
    )

    hidden_size = infer_hidden_size(backbone)
    model = CuriaSliceAggregator(
        backbone=backbone,
        hidden_size=hidden_size,
        num_classes=args.num_classes,
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw = ckpt["state_dict"]
        state_dict = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    else:
        raise ValueError(f"Unsupported checkpoint format: {args.checkpoint}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    ds = CTVolumeDataset(
        image_dir=args.image_dir,
        labels_csv=args.labels_csv,
        label_col=args.label_col,
        split=args.split,
        num_slices=args.num_slices,
        plane=args.plane,
        require_label=False,
    )

    collate_fn = partial(collate_with_processor, processor=processor, device=device)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    rows = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="predict"):
            logits = model(batch["pixel_values"])
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            for i, case_id in enumerate(batch["case_id"]):
                row = {
                    "case_id": case_id,
                    "pred": int(preds[i].item()),
                }
                for c in range(probs.shape[1]):
                    row[f"prob_{c}"] = float(probs[i, c].item())
                rows.append(row)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved predictions to {out}")


if __name__ == "__main__":
    main()
