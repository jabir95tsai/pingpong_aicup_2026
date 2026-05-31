# AutoGluon ensemble — Jabir component manifest (2026-05-31)

LB-best single: **R-067cr = 0.3870095**  (`submissions/submission_R067cr_alpha030_v22_blend_PLUS_RULE.csv`)
Nothing has beaten it; these are the components it blends, ready as AutoGluon meta-features.

All arrays in `oof_predictions/`. OOF rows = 69712 (train, fold-safe), test rows = 1845.
Per tag: `{tag}_oof_act.npy (N,15)`, `_oof_pt.npy (N,10)`, `_oof_srv.npy (N,)`, and `{tag}_test_*` likewise.
Shared labels: `v11_aug_oldtest_oof_y_act.npy / _y_pt.npy / _y_srv.npy` (align all tags by row index).

## Action + Point heads (15-cls / 10-cls)
- **v11_aug_oldtest** — bidir transformer (aug+old-test)  | oof(72065, 15) test(1845, 15)  ✓
- **v11plus** — transformer v11plus  | oof(69712, 15) test(1845, 15)  ✓
- **v13_oldtest** — transformer v13 (old-test)  | oof(72065, 19) test(1845, 19)  ✓
- **v14_seed2_v15feat_a** — GBM stack (v15feat_a)  | oof(69712, 19) test(1845, 19)  ✓
- **v16_avg3** — transformer v16 testhist-aug avg3  | oof(69712, 19) test(1845, 19)  ✓

## ServerGetPoint head (binary, AUC)
- **v22_causal_lm_v4_full** — causal-LM server head (AUC)  | oof(15833, 19) test(1845, 19)  ✓

## Notes for stacking
- Action eval classes 0-14 (serve 15-18 never appear as next-shot).
- pointId FH/BH axis is receiver-relative (handedness) — the residual hard bucket.
- OV = 0.4*F1_a + 0.4*F1_p + 0.2*AUC (macro-F1 for act/pt). Optimize the stacker per-task.
## ⚠️ Alignment gotchas (handle before stacking)
1. **Row count**: `_oldtest` tags (`v11_aug_oldtest`, `v13_oldtest`) have OOF
   rows = 72065 (old-test rows appended). SLICE the first **69712** rows to
   align with the standard index; the first 69712 labels match the others
   (verified in `src/analyze_oldtest_blend.py::load_components`).
2. **Action width**: some action arrays are (N,15), others (N,19) (serve cols
   15-18 are zeros). PAD to 19 or SLICE to 15 consistently
   (see `analyze_oldtest_blend.pad_act19`). Eval only over classes 0-14.
3. **v22 = SGP only**: use ONLY `v22_causal_lm_v4_full_{oof,test}_srv.npy` for
   the server head; its act/pt columns are unused. Its OOF srv covers a subset
   (15833 rows) — align via `v22_causal_lm_v4_full_oof_mask.npy`.
4. **Reference recipe**: `src/analyze_oldtest_blend.py` already loads + aligns
   all of these correctly (Dirichlet per-task blend). Reuse its
   `load_components()` to feed AutoGluon rather than re-deriving alignment.
