"""Build LB-ready submission from Bayes-refined R-052 weights.

Reads submissions/bayes_r052_search.json (produced by bayes_blend_search.py)
and writes:
  - submission_R055_bayes_r052.csv          (NONE-blend with Bayes weights)
  - submission_R055_bayes_r052_PLUS_RULE.csv (after rule_override post-process)

R-052 OOF (Bayes-refined): ~0.3844 (+0.0008 over Dirichlet-only 0.3836).
Predicted LB conservative (R-027 ratio 1.0035): ~0.3858
Predicted LB optimistic   (R-042 ratio 1.0142): ~0.3899
With rule_override (+0.0028 LB observed on R-042): up to ~0.3927.

USAGE:
    python -u src/build_bayes_r052_submission.py
"""
import json
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from config import PROJECT_ROOT, SUBMISSION_DIR  # noqa: E402
from analyze_oldtest_blend import load_components, write_submission  # noqa: E402


BAYES_JSON = os.path.join(SUBMISSION_DIR, "bayes_r052_search.json")
R055_TAG = "R055_bayes_r052"


def main() -> None:
    with open(BAYES_JSON) as f:
        bayes = json.load(f)
    subset = bayes["subset"]
    w_a = np.array(bayes["w_a"], dtype=np.float64)
    w_p = np.array(bayes["w_p"], dtype=np.float64)
    w_s = np.array(bayes["w_s"], dtype=np.float64)
    print(f" Bayes R-052 OOF: OV={bayes['R052_bayes']['OV']:.4f}")
    print(f" Subset ({len(subset)}): {subset}")
    print()
    print(f" w_a: {w_a.tolist()}")
    print(f" w_p: {w_p.tolist()}")
    print(f" w_s: {w_s.tolist()}")
    print()

    comp, y_a, y_p, y_s, _, test_uid = load_components(subset)

    test_act = np.stack([comp[t]["test_act"] for t in subset], axis=0)
    test_pt = np.stack([comp[t]["test_pt"] for t in subset], axis=0)
    test_srv = np.stack([comp[t]["test_srv"] for t in subset], axis=0)

    blend_a = (w_a[:, None, None] * test_act).sum(axis=0)
    blend_p = (w_p[:, None, None] * test_pt).sum(axis=0)
    blend_s = (w_s[:, None] * test_srv).sum(axis=0)
    pred_a = blend_a.argmax(axis=1)
    pred_p = blend_p.argmax(axis=1)

    fname = f"submission_{R055_TAG}.csv"
    out_path = write_submission(test_uid, pred_a, pred_p, blend_s, fname)
    print(f" Wrote: {out_path}")

    # Apply rule_override
    rule_script = os.path.join(PROJECT_ROOT, "src", "apply_rule_override.py")
    if not os.path.exists(rule_script):
        print(f" [warn] rule_override script not found at {rule_script}; skipping")
        return
    fname_rule = f"submission_{R055_TAG}_PLUS_RULE.csv"
    out_rule = os.path.join(SUBMISSION_DIR, fname_rule)
    train_csv = os.path.join(PROJECT_ROOT, "data", "train.csv")
    test_csv = os.path.join(PROJECT_ROOT, "data", "test_new.csv")
    cmd = ["python", "-u", rule_script,
           "--input", out_path,
           "--train", train_csv,
           "--test", test_csv,
           "--output", out_rule]
    print(f"\n Running rule_override: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("rule_override failed")
    print(f" Wrote: {out_rule}")


if __name__ == "__main__":
    main()
