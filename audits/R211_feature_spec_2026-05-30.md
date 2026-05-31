# R-211 Feature Spec — within-rally same-striker point-side history

**Goal:** give the point head explicit, ID-free evidence of the striker's own
side tendency so it can disambiguate FH vs BH zones (the dominant point-F1
failure mode). Signal validated at spread +0.147 (R-211 probe).

## Setup / indexing
- A rally is the ordered sequence of shots by `strikeNumber` (1..n).
- The **striker** of a shot alternates every shot, so the striker of the
  target shot at strikeNumber `t` is the same physical player as the strikers
  at `t-2, t-4, ...` (same parity). This is recoverable **positionally** — no
  `gamePlayerId` needed.
- At prediction time we see context shots `1..t-1` and predict shot `t`'s
  actionId/pointId/serverGetPoint.

## Feature definition (all computed from context shots < t ONLY)
Let `P = {pointId of prior shots j < t with (j % 2) == (t % 2)}` — the
striker's OWN prior shots this rally. Side map: FH={1,4,7}, BH={3,6,9},
MID={2,5,8}, MISS={0}.

1. `r211_n_own_prior`     = |P|                         (int, 0 if none)
2. `r211_own_fh`          = #(P in FH)                   (int)
3. `r211_own_bh`          = #(P in BH)                   (int)
4. `r211_own_fh_frac`     = own_fh / (own_fh+own_bh)     (0.5 if denom 0)
5. `r211_own_side_bias`   = (own_fh - own_bh)/max(1,|P|) (in [-1,1])
6. `r211_last_own_side`   = side of most-recent own shot (0=none,1=FH,2=BH,3=MID/MISS)
7. `r211_has_prior`       = 1 if |P|>0 else 0

(Optional, opponent-side mirror — the OTHER parity's prior side bias — captured
as `r211_opp_side_bias` analogously; can reveal rally geometry. Include if cheap.)

## Hard-rule leakage analysis
| Rule | Status | Why |
|---|---|---|
| no test SGP truth | CLEAN | feature is pointId-history only, never SGP |
| no SGP-derived proxy | CLEAN | no SGP used anywhere |
| no rally_uid/order inference | CLEAN | uses strikeNumber WITHIN a known rally (legitimate sequence position), does NOT infer cross-rally order or rally_uid identity |
| no teammate-leak artifacts | CLEAN | derived from the single rally's own shots |
| no player-profile features | CLEAN | NO gamePlayerId, NO cross-rally aggregation; parity-grouping is positional, resets every rally |
| no V15 hist/streak family | DISTINCT | V15 hist/streak aggregated a player's history ACROSS rallies (banned). R-211 is strictly WITHIN the current rally's context window — same data the sequence model already sees, only reorganized by striker parity |
| no target leakage | CLEAN | only shots j < t; never reads shot t's pointId/handId/etc |

**Transfer argument:** the feature is a deterministic function of the visible
context shots, identical in train and test_new construction. No fitted lookup,
no identity. If side-consistency holds in test (same sport), it transfers.

## A/B test design (V14 Fold-1, Kaggle GPU)
- Arm A (baseline): V14 unchanged.
- Arm B (+R211): V14 with the 7 features above appended to the point-path
  feature set ONLY (do not feed to action/server heads — isolates the effect).
- Metric: OOF Fold-1 macro-F1_p (+ per-class FH/BH-short), OV. GO if
  dF1_p >= +0.004 with no FH/BH-side class collapse.

## Open risk
The transformer/GBM already ingests prior pointId in-sequence; the explicit
same-striker grouping may add little. Smoke settles it before any full run.
