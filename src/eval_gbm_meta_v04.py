"""Re-score Experiments A/B/C under candidate_goal v0.4 (theory-first)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidate_goal import score_candidate

CLEAN_GUARDS = {
    "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
    "sgp_derived_proxy": False, "forbidden_rally_uid_inference": False,
    "teammate_leak_artifact": False, "external_leak_data": False,
    "oof_test_alignment_validated": True,
    "codex_smoke_approved": False, "codex_5fold_approved": False,
    "rule_override_applied": True, "rule_override_applicable": True,
    "p11_directly_optimized": False,
}

candidates = [
    {
        "rid": "R-080-gbm-stack-v11prob-v1",
        "name": "GBM stack over V11/V14/V13/V16 softmax + entropy/margin/agreement",
        "tier": "T2-exploration",
        "class": "new-mechanism",
        "stage": "preflight",
        "oof_delta": {"OV": 0.005, "F1_a": 0.005, "F1_p": 0.006, "AUC": 0.003},
        "guards": CLEAN_GUARDS,
        "compute_cost_hours": 3.0,
        "novelty": "high",
        "theoretical_generalization_reason": (
            "Nonlinear GBM over model-output features (softmax probs, entropy, "
            "margin, cross-model agreement) captures WHEN to trust V11 vs V14 "
            "vs V16. Features are model-output-derived, not raw-data-derived, "
            "so they should be invariant to test_new distribution shift."
        ),
        "why_transfers_to_test_new": (
            "Confidence and disagreement signals are properties of the base "
            "models, not the test distribution. If the base models transfer, "
            "the meta-features transfer. Risk: GBM may learn OOF-specific "
            "rules that don't replicate (B-meta R-054r failure mode)."
        ),
        "lb_confirm_hypothesis": (
            "LB DeltaOV >= +0.002 => nonlinear meta-stacking with rich "
            "features beats linear zoo."
        ),
        "lb_reject_hypothesis": (
            "LB DeltaOV <= -0.003 => B-meta universal; even rich features "
            "do not rescue. 2nd LB datapoint that meta-stacking is dead."
        ),
    },
    {
        "rid": "R-081-gbm-corrector-v11zoo-v1",
        "name": "GBM conditional corrector on R-067cr (bounded override count)",
        "tier": "T2-component",
        "class": "new-mechanism",
        "stage": "preflight",
        "oof_delta": {"OV": 0.003, "F1_a": 0.004, "F1_p": 0.002, "AUC": 0.0},
        "guards": CLEAN_GUARDS,
        "compute_cost_hours": 2.0,
        "novelty": "medium",
        "theoretical_generalization_reason": (
            "Correct only low-confidence rows where GBM has a high-confidence "
            "alternative. Bounded override count (cap 50/task) caps risk. "
            "Mechanism closer to R-042 rule_override (1.0 transfer) than "
            "R-054r meta_stack (-0.0103 LB)."
        ),
        "why_transfers_to_test_new": (
            "Where blend is confident, we don't touch. Where uncertain, GBM "
            "uses agreement/entropy features that are distribution-invariant. "
            "Override cap means a wrong call loses at most ~R-072 magnitude."
        ),
        "lb_confirm_hypothesis": (
            "LB DeltaOV >= +0.001 => bounded conditional correction transfers."
        ),
        "lb_reject_hypothesis": (
            "LB DeltaOV <= -0.002 => even conditional correction overfits OOF."
        ),
    },
    {
        "rid": "R-082-v11-embed-gbm-smoke",
        "name": "V11 pooled embedding (192-d) -> GBM per task",
        "tier": "T2-exploration",
        "class": "new-mechanism",
        "stage": "preflight",
        "oof_delta": {"OV": 0.008, "F1_a": 0.010, "F1_p": 0.008, "AUC": 0.005},
        "guards": CLEAN_GUARDS,
        "compute_cost_hours": 6.0,
        "novelty": "high",
        "theoretical_generalization_reason": (
            "V11 softmax compresses 192-d pooled rep -> 15-d action logits. "
            "GBM on raw 192-d embedding accesses information the heads "
            "necessarily lose. New mechanism class."
        ),
        "why_transfers_to_test_new": (
            "Embeddings encode the same structural patterns V11's heads use. "
            "If V11 transfers (it does), embeddings transfer. OOF-safe by "
            "construction: val embeddings come from non-trained fold models."
        ),
        "lb_confirm_hypothesis": (
            "LB DeltaOV >= +0.005 => V11 embeddings carry signal beyond "
            "softmax outputs; new exploitable mechanism class."
        ),
        "lb_reject_hypothesis": (
            "LB DeltaOV <= -0.005 => V11 embeddings are OOF-overfit; no "
            "extractable signal beyond the heads."
        ),
    },
]

print("=" * 130)
print("  v0.4 verdict for proposed GBM-meta experiments (A / B / C)")
print("=" * 130)
print(f"{'rid':<32} {'priority':<10} {'action':<22} {'sanity':<6} {'probe?':<7} "
      f"{'expLB':>7} {'gen':>5}")
print("-" * 130)
for c in candidates:
    v = score_candidate(c)
    print(f"{v['rid']:<32} {v['priority']:<10} {v['recommended_action']:<22} "
          f"{('OK' if v['smoke_sanity_pass'] else 'FAIL'):<6} "
          f"{('YES' if v['lb_probe_worthy'] else 'no'):<7} "
          f"{v['expected_lb_delta']:+.4f} {v['generalization_score']:.2f}")
print()
print("Per-candidate report fields:")
for c in candidates:
    v = score_candidate(c)
    print(f"\n{v['rid']}:")
    print(f"  theory:  {v['theoretical_generalization_reason'][:200]}...")
    print(f"  transfer:{v['why_transfers_to_test_new'][:200]}...")
    print(f"  confirm: {v['lb_confirm_hypothesis']}")
    print(f"  reject:  {v['lb_reject_hypothesis']}")
