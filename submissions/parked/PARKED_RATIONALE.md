# Parked submission candidates (2026-05-23, extended 2026-05-24)

## R-054r (8-comp + rule_override) — LB-FAILED 2026-05-24
- Actual LB: **0.3762672** (−0.0103 vs R-042 0.3866)
- OOF: 0.3821, predicted LB+rule 0.3862-0.3903 → midpoint 0.3882
- OOF→LB ratio: 0.9848 (B-impure transfer territory)
- **Definitive evidence**: meta_stack_v2_logistic is LB-toxic INDEPENDENT of v11_mulminet
  (R-054r had no mulminet — clean B-meta isolation). Family BANNED.
- Per LESSONS_CHECKLIST B-meta rule (2026-05-24): DO NOT RE-UPLOAD.


## R-060r (v14_recvprofile swap + rule_override)
- OOF dOV vs R-034: +0.0003
- Predicted LB+rule: 0.3830-0.3870
- **Parked-hard reason**: same B-player-style risk class as R-062r (v14_recvprofile
  encodes per-receiver style features). R-062r LB-failed −0.0057 (2026-05-23).
  Predicted ratio ≤ 1.0 would put R-060r LB at ~0.3800-0.3835 — below R-042 0.3866.
- Per LESSONS_CHECKLIST B-player-style rule (2026-05-23): DO NOT UPLOAD.

## R-061r (v14_recvhand swap + rule_override)
- OOF dOV vs R-034: +0.0009
- Predicted LB+rule: 0.3837-0.3877
- **Parked-hard reason**: same B-player-style risk class. v14_recvhand encodes
  per-receiver hand-frequency features. Even though it's not literally per-player,
  the receiver-style aggregation collapses onto similar player-style information
  in a match-disjoint test.
- Per LESSONS_CHECKLIST B-player-style rule (2026-05-23): DO NOT UPLOAD.

## Recovery
If a future experiment proves these are actually B-feature class (not
B-player-style), retrieve from this folder and re-evaluate. Until then,
treat as PARKED-HARD per the 2026-05-23 R-062r post-mortem.
