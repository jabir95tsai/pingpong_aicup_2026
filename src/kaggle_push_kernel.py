"""Push a notebook to Kaggle as a private kernel.

Usage:
    python src/kaggle_push_kernel.py --notebook notebooks/kaggle_autogluon_all_models_starter.ipynb \
                                     --slug autogluon-full \
                                     --title "AICUP AutoGluon all models" \
                                     --gpu

Produces:
    kaggle_kernel_<slug>/kernel-metadata.json
    kaggle_kernel_<slug>/<basename>.ipynb
Then runs `kaggle kernels push -p ...`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OWNER = "jabir95tsai"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--notebook", required=True, help="Path to local .ipynb")
    p.add_argument("--slug", required=True, help="Kernel slug (URL component)")
    p.add_argument("--title", required=True, help="Human title")
    p.add_argument("--gpu", action="store_true", help="Enable GPU T4 x2 (uses weekly quota)")
    p.add_argument("--no-internet", action="store_true", help="Disable internet (no pip-install)")
    p.add_argument("--dataset", default="jabir95tsai/aicup2026-pingpong-private",
                   help="Dataset to attach")
    p.add_argument("--public", action="store_true", help="Make kernel PUBLIC (default: private)")
    args = p.parse_args()

    nb_src = PROJ / args.notebook
    assert nb_src.exists(), f"Notebook not found: {nb_src}"

    stage = PROJ / f"kaggle_kernel_{args.slug}"
    stage.mkdir(exist_ok=True)
    shutil.copy2(nb_src, stage / nb_src.name)

    meta = {
        "id": f"{OWNER}/{args.slug}",
        "title": args.title,
        "code_file": nb_src.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true" if not args.public else "false",
        "enable_gpu": "true" if args.gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "false" if args.no_internet else "true",
        "dataset_sources": [args.dataset],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (stage / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {stage / 'kernel-metadata.json'}")

    rc = subprocess.call(["kaggle", "kernels", "push", "-p", str(stage)])
    if rc != 0:
        sys.exit(rc)
    print()
    print(f"Pushed.  https://www.kaggle.com/code/{OWNER}/{args.slug}")
    print(f"  visibility: {'PUBLIC' if args.public else 'PRIVATE'}")
    print(f"  accelerator: {'GPU T4 x2' if args.gpu else 'CPU'}")
    print(f"  internet: {'OFF' if args.no_internet else 'ON'}")


if __name__ == "__main__":
    main()
