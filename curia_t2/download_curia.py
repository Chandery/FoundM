from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

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


def main() -> None:
    ap = argparse.ArgumentParser("Download and cache Curia backbone")
    ap.add_argument("--model_id", type=str, default="raidium/curia")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument(
        "--full_repo",
        action="store_true",
        help="Download full repo including all downstream task heads (large). Default: backbone-only.",
    )
    args = ap.parse_args()

    if not os.environ.get("HF_MODULES_CACHE"):
        base_cache = Path(args.cache_dir) if args.cache_dir else (Path.home() / ".cache" / "huggingface")
        modules_cache = base_cache / "modules"
        modules_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_MODULES_CACHE"] = str(modules_cache)

    token = args.hf_token if args.hf_token else True
    allow_patterns = None
    if not args.full_repo:
        # Root backbone artifacts only. Avoid downloading task-specific heads in subfolders.
        allow_patterns = [
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "curia_image_processor.py",
            "modeling_dinov2.py",
            "README.md",
            "LICENSE",
            ".gitattributes",
        ]

    local_path = snapshot_download(
        repo_id=args.model_id,
        cache_dir=args.cache_dir,
        token=token,
        allow_patterns=allow_patterns,
    )

    AutoImageProcessor.from_pretrained(
        local_path,
        trust_remote_code=True,
    )
    AutoModel.from_pretrained(
        local_path,
        trust_remote_code=True,
    )
    mode = "full repo" if args.full_repo else "backbone-only"
    print(f"Curia cached ({mode}) at: {local_path}")


if __name__ == "__main__":
    main()
