"""R-211 signal probe — does within-rally same-striker point-side history
predict the next shot's point-side? (read-only on train.csv, no training)

Premise: pointId FH/BH axis is receiver-relative (handedness). The striker
alternates every shot, so a striker's OWN prior shots sit at strikeNumbers
n-2, n-4, ... (same parity) — recoverable purely positionally, with NO player
ID and NO cross-rally aggregation (so it is hard-rule clean and transfers to
de-identified test). If a striker's prior point-side strongly predicts their
next point-side, an explicit same-striker-history feature could disambiguate
the FH/BH confusion that the interleaved sequence model under-exploits.

We measure P(next side = FH) conditioned on the striker's own prior-shot side
majority, vs the base rate. A large shift => exploitable signal.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH

FH_SIDE = {1, 4, 7}   # FH short/half/long
BH_SIDE = {3, 6, 9}   # BH short/half/long


def side_of(pid):
    if pid in FH_SIDE:
        return "FH"
    if pid in BH_SIDE:
        return "BH"
    return None  # mid / miss


def main():
    df = pd.read_csv(TRAIN_PATH)
    # sort within rally by strike order
    df = df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)

    base_fh = base_bh = 0
    # conditional counts: prior-majority -> next side
    cond = {"FH": {"FH": 0, "BH": 0}, "BH": {"FH": 0, "BH": 0}, "tie/none": {"FH": 0, "BH": 0}}
    # also: immediate prev same-striker shot side -> next side
    prev1 = {"FH": {"FH": 0, "BH": 0}, "BH": {"FH": 0, "BH": 0}}
    n_targets = 0

    for uid, g in df.groupby("rally_uid", sort=False):
        pts = g["pointId"].to_numpy()
        sns = g["strikeNumber"].to_numpy()
        m = len(g)
        for i in range(m):
            tgt_side = side_of(int(pts[i]))
            if tgt_side is None:
                continue
            # same-striker prior shots: same parity of strikeNumber, earlier in rally
            par = sns[i] % 2
            prior_sides = [side_of(int(pts[j])) for j in range(i)
                           if sns[j] % 2 == par]
            prior_sides = [s for s in prior_sides if s is not None]
            n_targets += 1
            if tgt_side == "FH":
                base_fh += 1
            else:
                base_bh += 1
            if not prior_sides:
                key = "tie/none"
            else:
                nfh = prior_sides.count("FH"); nbh = prior_sides.count("BH")
                key = "FH" if nfh > nbh else ("BH" if nbh > nfh else "tie/none")
            cond[key][tgt_side] += 1
            # immediate previous same-striker side
            if prior_sides:
                prev1[prior_sides[-1]][tgt_side] += 1

    print("=" * 64)
    print(" R-211 within-rally same-striker point-side signal probe")
    print("=" * 64)
    tot = base_fh + base_bh
    p_fh = base_fh / tot
    print(f" targets (FH/BH-side only): {tot}")
    print(f" BASE RATE  P(FH)={p_fh:.3f}  P(BH)={1-p_fh:.3f}")
    print()
    print(" Conditioned on striker's OWN prior-shot side majority:")
    for key in ["FH", "BH", "tie/none"]:
        f, b = cond[key]["FH"], cond[key]["BH"]
        n = f + b
        if n:
            print(f"   prior majority {key:<8}: n={n:>6}  P(next=FH)={f/n:.3f}  "
                  f"(shift {f/n - p_fh:+.3f})")
    print()
    print(" Conditioned on striker's IMMEDIATE prior same-striker shot side:")
    for key in ["FH", "BH"]:
        f, b = prev1[key]["FH"], prev1[key]["BH"]
        n = f + b
        if n:
            print(f"   prev side {key:<3}: n={n:>6}  P(next=FH)={f/n:.3f}  (shift {f/n - p_fh:+.3f})")
    print()
    # crude signal verdict
    fh_maj = cond["FH"]; bh_maj = cond["BH"]
    pf = fh_maj["FH"]/(fh_maj["FH"]+fh_maj["BH"]) if (fh_maj["FH"]+fh_maj["BH"]) else p_fh
    pb = bh_maj["FH"]/(bh_maj["FH"]+bh_maj["BH"]) if (bh_maj["FH"]+bh_maj["BH"]) else p_fh
    print(f" SIGNAL SPREAD (P(FH|priorFH) - P(FH|priorBH)) = {pf - pb:+.3f}")
    print("   >0.15 = strong exploitable side-consistency; ~0 = no signal")


if __name__ == "__main__":
    main()
