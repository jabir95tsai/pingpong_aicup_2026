"""Build derived avg OOF components from per-seed source tags.

Skips any avg whose sources are not all present (logged, not fatal).
Used by orchestrate_deadline.ps1 finalizer.
"""
import os
import sys
import numpy as np

OOF = "C:/Users/jabir/Hacker_J/pingpong_aicup_2026/oof_predictions"

# (output_tag, [source_tags]) - all source tags must exist for build to happen
AVG_JOBS = [
    # Phase 2 results (built earlier; safe to rebuild idempotently)
    ("v11_mulminet_aug_oldtest_avg2",
     ["v11_mulminet_aug_oldtest", "v11_mulminet_aug_oldtest_seed31337"]),
    ("v16_testhist_aug_oldtest_avg3",
     ["v16_testhist_aug_oldtest", "v16_testhist_aug_oldtest_seed31337",
      "v16_testhist_aug_oldtest_seed51966"]),

    # New deadline-orchestrator targets
    ("v13_oldtest_avg2",
     ["v13_oldtest", "v13_oldtest_seed31337"]),
    ("v13_oldtest_avg3",
     ["v13_oldtest", "v13_oldtest_seed31337", "v13_oldtest_seed51966"]),
    ("v11_aug_oldtest_avg2",
     ["v11_aug_oldtest", "v11_aug_oldtest_seed31337"]),
    ("v11_aug_oldtest_avg3",
     ["v11_aug_oldtest", "v11_aug_oldtest_seed31337", "v11_aug_oldtest_seed51966"]),
    ("v11_mulminet_aug_oldtest_avg3",
     ["v11_mulminet_aug_oldtest", "v11_mulminet_aug_oldtest_seed31337",
      "v11_mulminet_aug_oldtest_seed51966"]),
    ("v16_testhist_aug_oldtest_avg5",
     ["v16_testhist_aug_oldtest", "v16_testhist_aug_oldtest_seed31337",
      "v16_testhist_aug_oldtest_seed51966", "v16_testhist_aug_oldtest_seed4",
      "v16_testhist_aug_oldtest_seed7"]),
    ("v11plus_oldtest_avg2",
     ["v11plus_oldtest", "v11plus_oldtest_seed31337"]),
    ("v14_oldtest_avg2",
     ["v14_seed0_oldtest", "v14_seed1_oldtest"]),
]

AVG_SUFFIXES = ["oof_act", "oof_pt", "oof_srv", "test_act", "test_pt", "test_srv"]
COPY_SUFFIXES = ["oof_y_act", "oof_y_pt", "oof_y_srv", "oof_mask", "oof_nsn",
                 "oof_pt_bin", "test_rally_uid"]


def avg_tags(out_tag, source_tags):
    """Average source OOF + test arrays into out_tag arrays."""
    # Verify all sources present
    for t in source_tags:
        if not os.path.exists(f"{OOF}/{t}_oof_act.npy"):
            return f"SKIP: missing source {t}"

    # Verify shapes (in case some are oldtest 72065 and others 69712)
    shapes = {}
    for s in AVG_SUFFIXES:
        arrs = [np.load(f"{OOF}/{t}_{s}.npy") for t in source_tags]
        shape0 = arrs[0].shape
        for t, a in zip(source_tags, arrs):
            if a.shape != shape0:
                return f"FAIL: shape mismatch {t}_{s} {a.shape} != {source_tags[0]}_{s} {shape0}"
        shapes[s] = shape0

    # Compute averages
    for s in AVG_SUFFIXES:
        arrs = [np.load(f"{OOF}/{t}_{s}.npy").astype(np.float32) for t in source_tags]
        avg = np.mean(np.stack(arrs, axis=0), axis=0)
        np.save(f"{OOF}/{out_tag}_{s}.npy", avg)

    # Copy y/mask/nsn/uid from first source (verify consistency where present)
    for s in COPY_SUFFIXES:
        path = f"{OOF}/{source_tags[0]}_{s}.npy"
        if not os.path.exists(path):
            continue
        arr = np.load(path)
        for t in source_tags[1:]:
            pt = f"{OOF}/{t}_{s}.npy"
            if os.path.exists(pt):
                a2 = np.load(pt)
                if not np.array_equal(arr, a2):
                    print(f"  WARN: {t}_{s} differs from {source_tags[0]}_{s}")
        np.save(f"{OOF}/{out_tag}_{s}.npy", arr)

    return f"OK shapes={shapes}"


def main():
    print("=" * 60)
    print(" _build_avg.py — building derived avg OOF components")
    print("=" * 60)
    built, skipped, failed = 0, 0, 0
    for out_tag, srcs in AVG_JOBS:
        result = avg_tags(out_tag, srcs)
        if result.startswith("OK"):
            built += 1
            print(f"  [BUILT]   {out_tag:<45} from {len(srcs)} sources")
        elif result.startswith("SKIP"):
            skipped += 1
            # Skipping is normal when sources haven't been trained yet
        else:
            failed += 1
            print(f"  [FAIL]    {out_tag:<45} {result}")

    print()
    print(f" Summary: built={built}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
