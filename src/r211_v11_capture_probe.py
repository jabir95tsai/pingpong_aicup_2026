"""R-211-on-V11 capture probe — does V11 ALREADY exploit same-striker prior
point-side, or is it ignoring recoverable in-context evidence?

No training. Reuses V11's existing OOF point predictions + rebuilt samples
(context shots carry prior pointId). For each FH/BH-side target we read the
target striker's OWN prior shots (context indices k-2,k-4,... — same parity)
and form a prior-side majority. Then we ask, on rows where prior evidence is
STRONG and CORRECT (prior majority == truth side):
  - does V11 already predict the correct side?  -> signal captured -> NO-GO
  - or does V11 still mispredict (esp. spam FH)? -> recoverable room -> GO

Alignment is self-validated: rebuilt y_point must equal the stored OOF y.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH
from data_cleaning import clean_data
from train_v11_transformer import build_samples

OOF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "oof_predictions")
V11_TAG = "v11_aug_oldtest"

FH_SIDE = {1, 4, 7}; BH_SIDE = {3, 6, 9}
def side_of(p):
    if p in FH_SIDE: return "FH"
    if p in BH_SIDE: return "BH"
    return None


def main():
    import pandas as pd
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, _test_df, player_map = clean_data(raw_train, raw_test)
    samples = build_samples(train_df, is_train=True, n_players=len(player_map))
    print(f" rebuilt samples: {len(samples)}")

    y_pt_mine = np.array([s["y_point"] for s in samples], dtype=np.int64)
    oof_y = np.load(f"{OOF_DIR}/{V11_TAG}_oof_y_pt.npy")
    oof_p = np.load(f"{OOF_DIR}/{V11_TAG}_oof_pt.npy")
    N = len(y_pt_mine)
    oof_y = oof_y[:N]; oof_p = oof_p[:N]
    # alignment self-check
    if not np.array_equal(y_pt_mine, oof_y):
        nmis = int((y_pt_mine != oof_y).sum())
        print(f" [ALIGN FAIL] {nmis}/{N} y mismatches -> abort (cannot trust join)")
        return
    print(" [ALIGN OK] rebuilt y_point matches stored OOF y exactly")
    v11_pred = oof_p.argmax(1)

    # per-sample same-striker prior-side majority
    rows = []
    for i, s in enumerate(samples):
        cs = s["cat_seq"]; k = len(cs)
        ss_pts = cs[max(k-2, 0)::-2, 4] if k >= 2 else np.array([], dtype=int)
        sides = [side_of(int(p)) for p in ss_pts]
        sides = [x for x in sides if x]
        nfh, nbh = sides.count("FH"), sides.count("BH")
        prior = "FH" if nfh > nbh else ("BH" if nbh > nfh else None)
        rows.append((side_of(int(oof_y[i])), prior, int(v11_pred[i]), nfh + nbh))
    rows = [r for r in rows if r[0] is not None]  # FH/BH-side truths only

    def side_pred(pid):
        return side_of(pid)

    print(f"\n FH/BH-side targets: {len(rows)}")
    # Focus: STRONG + CORRECT prior evidence (prior majority == truth side, >=1 prior)
    for label, filt in [
        ("ALL rows", lambda r: True),
        ("prior present (>=1)", lambda r: r[3] >= 1),
        ("prior CORRECT (majority==truth)", lambda r: r[1] is not None and r[1] == r[0]),
        ("prior CORRECT & strong (>=2)", lambda r: r[1] is not None and r[1] == r[0] and r[3] >= 2),
    ]:
        sub = [r for r in rows if filt(r)]
        if not sub:
            print(f"\n [{label}] n=0"); continue
        n = len(sub)
        v11_side_correct = sum(1 for r in sub if side_pred(r[2]) == r[0])
        # among these, how often does V11 spam the WRONG side
        wrong = [r for r in sub if side_pred(r[2]) is not None and side_pred(r[2]) != r[0]]
        print(f"\n [{label}] n={n}")
        print(f"   V11 predicts correct SIDE: {v11_side_correct/n:.3f}")
        print(f"   V11 predicts wrong side : {len(wrong)/n:.3f}")

    # Decisive: rows where prior CORRECT but V11 got side WRONG = recoverable room
    recoverable = [r for r in rows if r[1] is not None and r[1] == r[0] and r[3] >= 1
                   and side_pred(r[2]) is not None and side_pred(r[2]) != r[0]]
    correct_prior = [r for r in rows if r[1] is not None and r[1] == r[0] and r[3] >= 1]
    print("\n" + "=" * 60)
    print(" DECISIVE METRIC")
    print("=" * 60)
    if correct_prior:
        frac = len(recoverable) / len(correct_prior)
        print(f" rows with CORRECT prior evidence: {len(correct_prior)}")
        print(f"   of those, V11 STILL got side wrong: {len(recoverable)} ({frac:.3f})")
        print(f"   => recoverable rows = {len(recoverable)} / {len(rows)} FH-BH targets"
              f" = {len(recoverable)/len(rows):.3f} of all side-targets")
        print("   (high frac => V11 ignores recoverable evidence => R-211-on-V11 GO;")
        print("    low frac  => V11 already captures it => NO-GO)")


if __name__ == "__main__":
    main()
