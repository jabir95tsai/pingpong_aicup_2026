"""Quick OOF eval for R-052/R-053/R-054 + Bayes R-055 candidates.

Compares predicted LB ranges for the 4 ready-to-upload submissions.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from analyze_oldtest_blend import load_components, evaluate_subset_none

R052 = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a",
        "v16_avg3", "meta_stack_v2_logistic", "v11_mulminet_aug_avg3"]
R053 = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a",
        "v16_avg3", "meta_stack", "v11_mulminet_aug_avg3"]
R054 = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a",
        "v16_avg3", "meta_stack_v2_logistic", "v11_aug_big", "v14_recvprofile"]
R034 = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]

CONS = 1.0035   # R-027 PAIR origin (B-pure conservative)
R042_RATIO = 1.0142  # R-042 actual = R-034 0.3812 OOF -> 0.3866 LB
RULE_LIFT_LB = 0.0028  # R-042 observed lift from rule_override

def fmt(name, ov, fa, fp, auc):
    lo = ov * CONS
    hi = ov * R042_RATIO
    print(f"{name:<48} OV={ov:.4f}  F1a={fa:.4f} F1p={fp:.4f} AUC={auc:.4f}")
    print(f"{'':<48}   predLB no-rule: {lo:.4f}-{hi:.4f}")
    print(f"{'':<48}   predLB + RULE : {lo + RULE_LIFT_LB:.4f}-{hi + RULE_LIFT_LB:.4f}")

all_tags = list(set(R052 + R053 + R054 + R034))
comp, y_a, y_p, y_s, _, _ = load_components(all_tags)

print("=" * 90)
print(" Predicted-LB summary  (R-042 = 0.3866 current LB best)")
print("=" * 90)

for name, subset in [
    ("R034 5c LB anchor (0.3838)", R034),
    ("R052 7c +meta_v2 +mulminet_avg3", R052),
    ("R053 7c +meta_v1 +mulminet_avg3", R053),
    ("R054 8c +meta_v2 +v11_aug_big +recvprofile", R054),
]:
    m = evaluate_subset_none(subset, comp, y_a, y_p, y_s, optimize=True, n_samples=300, seed=20260522)
    fmt(name, m["OV"], m["F1_a"], m["F1_p"], m["AUC"])
    print()

bj = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "submissions", "bayes_r052_search.json")))
b = bj["R052_bayes"]
fmt("R055 7c R052 BAYES weights (500+COBYLA)", b["OV"], b["F1_a"], b["F1_p"], b["AUC"])
