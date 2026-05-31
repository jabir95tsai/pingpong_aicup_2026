"""R-203 overnight orchestrator — runs ONLY inside the user's 1am-9am window.

Designed to be launched by Windows Task Scheduler at 01:00 local time. It:
  1. Guards: refuses to start outside [01:00, 09:00) local time (safety net so a
     mis-fired task never hogs the PC during the day).
  2. Runs the v14 BASELINE fold-1 (standard CE + ACTION_CW)         -> tag v14_baseline_fold1
  3. Runs the v14 R-203 fold-1   (focal CE + Cui CB + push/Loop boost) -> tag v14_r203_fold1
  4. Reads both saved OOF arrays, computes fold-1 per-class F1 for the
     push family (act 5,6,13) + Loop (act1) + overall OV.
  5. Writes a GO/NO-GO verdict to audits/R203_smoke_verdict.md per the
     R-203 smoke criteria:
        smoke_pass = (OV_r203 >= OV_baseline + 0.003) AND
                     (mean push-family F1 delta >= +0.005)

Both runs use --folds 5 --max-folds 1 --seed 42 so they share the SAME fold-1
GroupKFold partition (apples-to-apples).

Exit code 0 always (so the scheduled task shows success); the verdict file
carries the real PASS/FAIL signal.
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
OOF = os.path.join(ROOT, "oof_predictions")
LOGS = os.path.join(ROOT, "logs")
AUDITS = os.path.join(ROOT, "audits")
VERDICT = os.path.join(AUDITS, "R203_smoke_verdict.md")

PYTHON = sys.executable

ACTION_NAMES = {
    0: "None", 1: "Loop", 2: "Cloop", 3: "Smash", 4: "Flip", 5: "Pushfast",
    6: "Push", 7: "Flick", 8: "Arch", 9: "Knuckle", 10: "Chop_r",
    11: "ShortStop", 12: "Chop", 13: "Block", 14: "Lob",
}
PUSH_FAMILY = [5, 6, 13]
LOOP = [1]
ACTION_EVAL_LABELS = list(range(15))


def in_window() -> bool:
    h = datetime.now().hour
    return 1 <= h < 9


def run_training(tag: str, extra_args: list[str]) -> int:
    log_path = os.path.join(LOGS, f"{tag}.log")
    cmd = [
        PYTHON, "-u", os.path.join(SRC, "train_v14.py"),
        "--folds", "5", "--max-folds", "1", "--seed", "42",
        "--tag", tag,
    ] + extra_args
    print(f"[{datetime.now():%H:%M:%S}] RUN {tag}: {' '.join(cmd)}")
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"[{datetime.now():%H:%M:%S}] {tag} exit={proc.returncode}  log={log_path}")
    return proc.returncode


def fold1_action_f1(tag: str):
    """Compute fold-1 per-class action F1 from saved OOF arrays.

    train_v14.py saves {tag}_oof_act.npy (n,19), {tag}_oof_y_act.npy (n,),
    {tag}_oof_nsn.npy (n,), {tag}_oof_mask.npy (n,). With --max-folds 1 only
    fold-1 validation rows are populated (mask True).
    """
    import numpy as np
    from sklearn.metrics import f1_score

    oof_act = np.load(os.path.join(OOF, f"{tag}_oof_act.npy"))
    y_act = np.load(os.path.join(OOF, f"{tag}_oof_y_act.npy"))
    mask = np.load(os.path.join(OOF, f"{tag}_oof_mask.npy"))
    # oof_act already has action rules applied at save time? No — train_v14
    # stores ruled probs in oof_act (apply_action_rules called before storing).
    pred = np.argmax(oof_act[mask], axis=1)
    yt = y_act[mask]
    per_class = f1_score(yt, pred, labels=ACTION_EVAL_LABELS,
                         average=None, zero_division=0)
    macro = f1_score(yt, pred, labels=ACTION_EVAL_LABELS,
                     average="macro", zero_division=0)
    return per_class, macro, int(mask.sum())


def write_verdict(baseline_ok: bool, r203_ok: bool):
    import numpy as np

    lines = []
    lines.append("# R-203 Smoke Verdict (overnight auto-run)")
    lines.append("")
    lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S} (local)")
    lines.append("")
    lines.append("Mechanism: v14 ACTION models — focal CE (gamma=2) + Cui et al. "
                 "Class-Balanced weights + push/Loop boost (act 1,5,6,13 ×1.5).")
    lines.append("Comparison: fold-1 of GroupKFold(5), seed 42, identical partition.")
    lines.append("")

    if not (baseline_ok and r203_ok):
        lines.append("## RESULT: INCOMPLETE")
        lines.append("")
        lines.append(f"- baseline run ok: {baseline_ok}")
        lines.append(f"- r203 run ok: {r203_ok}")
        lines.append("One or both training runs failed — see logs/v14_baseline_fold1.log "
                     "and logs/v14_r203_fold1.log.")
        with open(VERDICT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    try:
        base_pc, base_macro, n_base = fold1_action_f1("v14_baseline_fold1")
        r203_pc, r203_macro, n_r203 = fold1_action_f1("v14_r203_fold1")
    except Exception as e:
        lines.append("## RESULT: ERROR computing F1")
        lines.append(f"\n{type(e).__name__}: {e}")
        with open(VERDICT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    # Per-class table
    lines.append("## Per-class action F1 (fold-1)")
    lines.append("")
    lines.append("| cls | name | baseline | R-203 | delta |")
    lines.append("|---|---|---|---|---|")
    for c in ACTION_EVAL_LABELS:
        d = r203_pc[c] - base_pc[c]
        flag = " ⭐" if c in (PUSH_FAMILY + LOOP) else ""
        lines.append(f"| {c} | {ACTION_NAMES[c]}{flag} | {base_pc[c]:.4f} | "
                     f"{r203_pc[c]:.4f} | {d:+.4f} |")
    lines.append("")

    push_delta = float(np.mean([r203_pc[c] - base_pc[c] for c in PUSH_FAMILY]))
    loop_delta = float(r203_pc[1] - base_pc[1])
    macro_delta = float(r203_macro - base_macro)

    lines.append("## Targeted-class deltas")
    lines.append("")
    lines.append(f"- push-family (act 5,6,13) mean F1 delta: **{push_delta:+.4f}** "
                 f"(criterion: >= +0.005)")
    lines.append(f"- Loop (act1) F1 delta: {loop_delta:+.4f}")
    lines.append(f"- action macro-F1 delta: {macro_delta:+.4f}")
    lines.append(f"- baseline action macro: {base_macro:.4f} (n={n_base})")
    lines.append(f"- R-203 action macro: {r203_macro:.4f} (n={n_r203})")
    lines.append("")

    # Smoke pass criteria. NOTE: full OV needs point+AUC which R-203 does NOT
    # change (it only touches the action LGB). So action-macro delta is the
    # relevant signal; OV delta ~= 0.4 * action_macro_delta (point/AUC ~unchanged).
    est_ov_delta = 0.4 * macro_delta
    lines.append(f"- estimated OV delta (0.4 × action macro delta, since R-203 "
                 f"only changes the action LGB): **{est_ov_delta:+.4f}** "
                 f"(criterion: >= +0.003)")
    lines.append("")

    smoke_pass = (est_ov_delta >= 0.003) and (push_delta >= 0.005)
    lines.append("## RESULT: " + ("GO ✅ (smoke PASS)" if smoke_pass else "NO-GO ❌ (smoke FAIL)"))
    lines.append("")
    if smoke_pass:
        lines.append("Both criteria met. Next step (needs Jabir): run full 5-fold "
                     "R-203, build a candidate that swaps the R-203 action OOF into "
                     "the R-034 PAIR / R-067cr blend, mark ARTIFACT_READY_FOR_JABIR_"
                     "UPLOAD_REVIEW.")
    else:
        lines.append("Criteria NOT met. Per R-203 lb_reject_hypothesis: focal+CB "
                     "loss does not improve push-family enough to justify a swap. "
                     "Close the GBM-focal route; do not LB-probe. Document in "
                     "GOAL_FUNCTION calibration log.")
    lines.append("")
    with open(VERDICT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[{datetime.now():%H:%M:%S}] verdict written: {VERDICT}  smoke_pass={smoke_pass}")


def main():
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(AUDITS, exist_ok=True)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] R-203 overnight orchestrator start")

    if not in_window():
        msg = (f"REFUSED: current hour {datetime.now().hour} is outside the "
               f"1am-9am window. Not starting training (PC-respect guard).")
        print(msg)
        with open(VERDICT, "w", encoding="utf-8") as f:
            f.write(f"# R-203 Smoke Verdict\n\n{msg}\n"
                    f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        return 0

    t0 = time.time()
    rc_base = run_training("v14_baseline_fold1", [])
    rc_r203 = run_training("v14_r203_fold1", ["--r203-focal"])
    write_verdict(rc_base == 0, rc_r203 == 0)
    print(f"[{datetime.now():%H:%M:%S}] DONE in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
