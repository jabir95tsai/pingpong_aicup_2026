# R-203 — V14 GBM with Focal CE + Cui Class-Balanced weights

**Status**: smoke RUNNING on Kaggle (`jabir95tsai/aicup-r-203-focal-fold1`), CPU kernel.
**Class**: B-feature (same architecture, new training objective).
**Novelty**: medium. **Predicted LB Δ**: +0.003 to +0.008.
**Created**: 2026-05-29 (autonomous /goal mode, while R-082 Phase 2 + R-203 smoke run remotely).

---

## Mechanism (what changed)

v14 trains the ACTION task with a LightGBM + XGBoost ensemble using standard
multiclass cross-entropy and hand-tuned per-class sample weights (`ACTION_CW`).
R-203 replaces the **LightGBM action objective** with:

1. **Focal CE** (Lin et al. 2017), γ=2:  `L = -α_y (1 - p_y)^γ log(p_y)`
   — down-weights already-easy examples, concentrates gradient on hard ones.
2. **Cui et al. 2019 Class-Balanced α**: `α_c = (1-β)/(1-β^{n_c})`, β=0.999,
   normalized to mean 1 over present classes — principled replacement for the
   hand-tuned `ACTION_CW` dict.
3. **Push/Loop boost ×1.5** on action ids {1, 5, 6, 13} per spec.

XGBoost stays on standard CE for this first smoke (single-axis change; if the
LGB-only swap shows signal, extend to XGB). The point and server tasks are
untouched, so any OV change is ≈ 0.4 × (action-macro-F1 change).

Gradient correctness verified by finite-difference check (rel err 5e-11) in
`src/r203_focal_obj.py` self-tests; synthetic LGB end-to-end learns
(val acc 0.56 vs 0.20 chance).

## Target-class prevalence (v14 OOF, action; n=69712)

| act | name | n | share | why targeted |
|---|---|---|---|---|
| 1 | Loop | 15435 | 22.1% | dominant attack class; focal should sharpen its boundary vs Cloop/Smash |
| 5 | Pushfast | 4192 | 6.0% | push family, under-represented, frequently confused |
| 6 | Push | 6635 | 9.5% | push family, control stroke |
| 13 | Block | 7848 | 11.3% | defensive; confused with push family at SN≥5 |

---

## v0.4 candidate report (6 fields)

**1. theoretical_generalization_reason** —
Focal loss and class-balanced weighting are well-established, distribution-level
regularizers that improve minority-class recall *without* memorizing any
instance-specific signal. They change only *how the loss is aggregated* over the
same train.csv labels and the same `features_v9` inputs. There is no new feature,
no new data source, no leakage vector — the decision boundary is re-shaped to pay
more attention to hard/rare classes. This is the textbook remedy for the macro-F1
class-imbalance problem the competition metric rewards.

**2. why_transfers_to_test_new** —
The training distribution is unchanged; only the objective's per-example and
per-class emphasis changes. The resulting model is the *same architecture* fit to
the *same data* with a loss that better matches the macro-F1 evaluation. Because
B-feature changes (same arch, same data, new objective/feature) have shown ~0.9
empirical LB transfer historically (R-034 family), and because focal/CB introduce
no test-specific or player-specific dependency, the improvement should carry to
test_new at roughly the OOF magnitude.

**3. smoke_sanity_pass** (criteria, evaluated by the running kernel) —
PASS iff **both**:
  - estimated OV delta = 0.4 × (action macro-F1 delta) ≥ **+0.003**, AND
  - push-family (act 5,6,13) mean F1 delta ≥ **+0.005**.
Guardrail: no canary class (especially the rare 8/Arch, 14/Lob) may drop
> −0.025; if a rare class collapses while push improves, treat as FAIL (the
boost merely traded classes, did not add skill).

**4. lb_probe_worthy** —
Only if smoke PASSES. A pass means the focal+CB action model is a *drop-in
improvement* on the existing v14 action OOF, so the natural LB probe is to swap
the R-203 action OOF into the R-034 PAIR / R-067cr blend slot currently held by a
v14-family action component, holding point + server fixed. This is the lowest-risk
way to LB-test (one component swap, everything else identical to the LB-best
0.3870095 submission). LB upload remains Jabir's manual decision.

**5. lb_confirm_hypothesis** —
LB Δ ≥ +0.003 ⇒ focal+CB training objective transfers to test_new. This opens
the "B-feature new-objective" track: extend focal+CB to XGBoost, to the point
task, and consider tuning γ / β. Validates that the macro-F1 imbalance was an
*optimization* problem (fixable by loss), not just a *capacity* problem.

**6. lb_reject_hypothesis** —
LB Δ ≤ −0.003 ⇒ focal+CB overfits the train-class frequencies in a way that does
not transfer (e.g., test_new push-family prevalence differs, or the boost
over-corrects). Close the GBM-focal route; record in GOAL_FUNCTION calibration log
as evidence that *loss-level imbalance correction is LB-neutral-to-toxic for this
dataset* (would join SoftF1-additive R-094 v2 as a second failed imbalance-attack,
strengthening the lesson that the imbalance is data-limited, not loss-limited).

---

## Leakage safety

- Same data as base v14 (train.csv, features_v9). No test SGP, no SGP proxy.
- No rally_uid / order inference, no player-profile features, no V15 hist/streak.
- No teammate parquet: the Kaggle dataset `aicup2026-r203-code` was built
  code+train+test only, explicitly excluding `test_history_pairs_new.parquet`.
- Class weights derived from *training* class counts only (computed per-fold
  inside the trainer from `y_a_act_combined`), never from test.

## Next actions (gated on smoke verdict)

- **GO**: full 5-fold R-203 (Kaggle CPU, ~6-8h), then build a single-component-swap
  candidate vs R-067cr, mark `ARTIFACT_READY_FOR_JABIR_UPLOAD_REVIEW`.
- **NO-GO**: close route, append calibration-log entry, do not LB-probe.
