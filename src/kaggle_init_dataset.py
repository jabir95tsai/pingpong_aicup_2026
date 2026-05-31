"""Create / update a PRIVATE Kaggle dataset for AICUP 2026.

The dataset bundles legal training data + a handful of OOF arrays needed
for blend experiments on Kaggle. Everything stays under the user's account
and is NOT shared with anyone.

USAGE:
    # First time (creates the dataset; uses metadata in kaggle_dataset/dataset-metadata.json)
    python -u src/kaggle_init_dataset.py --create

    # Subsequent updates (after adding files / new OOF arrays)
    python -u src/kaggle_init_dataset.py --update --note "added v15feat_b OOF"

Outputs:
    kaggle_dataset/  (the staging dir Kaggle uploads from)
        train.csv
        test.csv
        test_new.csv
        oof/<tag>_oof_act.npy ...
        dataset-metadata.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"
OOF = PROJ / "oof_predictions"
STAGE = PROJ / "kaggle_dataset"

# These OOF tags ship to Kaggle so we can run blend-swap audits there
# AND so AutoGluon can use them as meta-features in stacking notebooks.
#
# Set OOF_TAGS_FULL=True to ship every OOF tag (~2 GB) — useful for AutoGluon
# meta-stacking. Set False to keep only R-034 baseline (~150 MB).
OOF_TAGS_R034_BASELINE = [
    "v11_aug_oldtest",
    "v11plus",
    "v13_oldtest",
    "v14_seed2_v15feat_a",  # R-034 LB best component
    "v16_avg3",
]

# Extra tags useful for AutoGluon meta-stacking input (NEW SIGNAL CLASS
# components + diversity). Includes everything that has dOV >= -0.005 in
# the parked audit so we don't pre-filter ourselves out.
OOF_TAGS_STACKING_EXTRA = [
    # Other LB-tested baselines (for diversity)
    "v11_aug", "v11", "v13", "v14_seed2", "v14_seed0", "v14_seed1",
    "v16_testhist_aug", "v17_momentum",
    # NEW SIGNAL CLASS — never LB-tested but parked-audit STAGE 1
    "meta_stack", "meta_stack_v2_logistic", "sn2_expert",
    # v14 variants with different features (B-feature class)
    "v14_recvhand", "v14_recvprofile",
    # Mulminet family (LB-failed but still has signal in blend)
    "v11_mulminet_aug_avg3", "v11_mulminet_aug_avg2",
    "v11_mulminet_aug_oldtest", "v11_mulminet_pretrained_aug",
    "v11_mulminet",
    # Oldtest seeds (for averaging on Kaggle if desired)
    "v11_aug_oldtest_avg2", "v11_aug_oldtest_avg3",
    "v13_oldtest_avg2", "v13_oldtest_avg3",
    "v14_seed0_oldtest", "v14_seed1_oldtest", "v14_seed2_oldtest",
    "v14_oldtest_avg2", "v14_avg3",
    "v16_testhist_aug_oldtest_avg3", "v16_testhist_aug_oldtest_avg5",
    # v11plus variants
    "v11plus_oldtest", "v11plus_oldtest_avg2", "v11plus_aug",
    # Other GBM variants
    "v12_5f", "v15_hist_only", "v18",
]


def get_oof_tags_to_ship() -> list[str]:
    """Return the list of OOF tags to copy. Toggle via env var."""
    if os.environ.get("OOF_TAGS_FULL", "1") == "1":
        return OOF_TAGS_R034_BASELINE + OOF_TAGS_STACKING_EXTRA
    return OOF_TAGS_R034_BASELINE


# Source files to ship for feature engineering. The notebook does:
#   sys.path.insert(0, '/kaggle/input/aicup2026-pingpong-private/code/')
# then imports features_v15feat_b as usual.
SRC_FILES_TO_SHIP = [
    "config.py",
    "data_cleaning.py",  # for clean_data() used by transformer trainers
    # Feature dependency chain for v15feat_b
    "features_v3.py", "features_v4.py", "features_v5.py", "features_v6.py",
    "features_v7.py", "features_v9.py", "features_v15feat.py",
    "features_v15feat_b.py",
    # Player profile features
    "features_v9_recvhand.py", "features_v9_recvprofile.py",
    # R-029b features used by sgp_prefix_v3
    "features_sgp_prefix_v3.py",
    # R-032 v2 (LORO cross-rally match-pair features) + R-047 v15feat_c
    "features_v16match_v2.py",
    "features_v15feat_c.py",
    # All trainer scripts for Kaggle parallel retrain
    "train_v11_transformer.py",        # v11, v11_aug, v11plus
    "train_v11_mulminet.py",            # v11_mulminet, *_aug
    "train_v11_mulminet_pretrained.py", # v11_mulminet_pretrained_aug
    "train_v11_mulminet_softf1.py",     # R-031
    "train_v13.py",                     # v13 GBM
    "train_v14.py",                     # v14 GBM + feature variants
    "train_v16_testhist_aug.py",        # v16 family
    "sgp_prefix_v3.py",                 # R-030 SGP specialist
    "train_causal_lm_v1.py",            # R-066 Path B causal LM
]

# Parquet/aux data files to ship as top-level data dataset items.
AUX_DATA_FILES = [
    "test_history_pairs_new.parquet",   # --aug-parquet for transformers
]

# Pre-trained model checkpoints to ship (e.g., R-031 Phase A CE warmup ckpt
# so that the Phase B-only kernel can skip the 12-hr Phase A entirely).
# Source paths are relative to project root.
MODEL_CHECKPOINTS = [
    # Already staged manually at kaggle_dataset/ce_checkpoint/best_ce_fold1.pt
]

DATASET_OWNER = "jabir95tsai"  # match the Kaggle username; edit if different
DATASET_SLUG = "aicup2026-pingpong-private"
DATASET_TITLE = "AICUP 2026 Pingpong (private)"


def write_metadata() -> None:
    """Write the dataset-metadata.json that Kaggle CLI requires."""
    STAGE.mkdir(exist_ok=True)
    meta = {
        "title": DATASET_TITLE,
        "id": f"{DATASET_OWNER}/{DATASET_SLUG}",
        "licenses": [{"name": "unknown"}],
        # Visibility is controlled by --public flag at create time; we always
        # omit it here to default to PRIVATE.
    }
    (STAGE / "dataset-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote {STAGE / 'dataset-metadata.json'}")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  [skip] {src} not found")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied {src.name}  ({dst.stat().st_size // 1024} KB)")


def stage_files() -> None:
    """Copy data + OOF arrays + src code into the kaggle_dataset/ staging dir."""
    print("Staging data files ...")
    # sample_submission.csv REQUIRED by config.py's _has_required_data_files check
    # (config.py PINGPONG_DATA_DIR validation needs train.csv + sample_submission.csv).
    for fn in ["train.csv", "test.csv", "test_new.csv", "sample_submission.csv"]:
        copy_file(DATA / fn, STAGE / fn)

    print("Staging auxiliary data ...")
    for fn in AUX_DATA_FILES:
        copy_file(DATA / fn, STAGE / fn)

    tags = get_oof_tags_to_ship()
    print(f"Staging OOF arrays for {len(tags)} components ...")
    oof_dir = STAGE / "oof"
    oof_dir.mkdir(exist_ok=True)
    skipped = []
    for tag in tags:
        ok = False
        for suffix in [
            "oof_act", "oof_pt", "oof_srv",
            "oof_y_act", "oof_y_pt", "oof_y_srv",
            "oof_mask", "oof_nsn",
            "test_act", "test_pt", "test_srv", "test_rally_uid",
        ]:
            fp = OOF / f"{tag}_{suffix}.npy"
            if fp.exists():
                copy_file(fp, oof_dir / fp.name)
                ok = True
        if not ok:
            skipped.append(tag)
    if skipped:
        print(f"  [warn] {len(skipped)} tags had NO files (will be ignored on Kaggle):")
        for t in skipped:
            print(f"     - {t}")

    print(f"Staging source code ...")
    code_dir = STAGE / "code"
    code_dir.mkdir(exist_ok=True)
    src_root = PROJ / "src"
    for fn in SRC_FILES_TO_SHIP:
        copy_file(src_root / fn, code_dir / fn)


def cli(args: list[str]) -> int:
    print(f"$ {' '.join(args)}")
    return subprocess.call(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true",
                        help="Create the dataset for the first time (PRIVATE)")
    parser.add_argument("--update", action="store_true",
                        help="Push a new version of an existing dataset")
    parser.add_argument("--note", type=str, default="updated",
                        help="Version note attached to --update")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage files only; do not call kaggle CLI")
    args = parser.parse_args()

    if not (args.create or args.update or args.dry_run):
        parser.error("pass --create, --update, or --dry-run")

    write_metadata()
    stage_files()

    if args.dry_run:
        print()
        print(f"Dry run done. Inspect {STAGE}/ then re-run without --dry-run.")
        return

    # Use --dir-mode tar instead of zip — empirically zip mode on Kaggle CLI
    # dropped top-level CSVs in our v3 push (only subdirs uploaded). tar mode
    # preserves all files at all levels.
    if args.create:
        print()
        print("Creating PRIVATE Kaggle dataset ...")
        rc = cli([
            "kaggle", "datasets", "create",
            "-p", str(STAGE),
            "--dir-mode", "tar",
            # No --public flag → DEFAULT PRIVATE
        ])
        if rc != 0:
            print("ERROR: kaggle datasets create failed.")
            print("  Check $env:USERPROFILE\\.kaggle\\kaggle.json is present and readable.")
            sys.exit(rc)
        print()
        print(f"Done. Dataset: https://www.kaggle.com/datasets/{DATASET_OWNER}/{DATASET_SLUG}")
        print("  (PRIVATE — only visible to you)")
    elif args.update:
        print()
        print(f"Updating Kaggle dataset (note: {args.note!r}) ...")
        rc = cli([
            "kaggle", "datasets", "version",
            "-p", str(STAGE),
            "-m", args.note,
            "--dir-mode", "tar",
        ])
        if rc != 0:
            sys.exit(rc)
        print()
        print(f"New version pushed.")


if __name__ == "__main__":
    main()
