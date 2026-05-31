"""T1 component-correlation analysis.

For each of the 14 menu components (GROUP_A/B/C/D/E + v14_recvhand) compute the
pairwise correlation of OOF probability arrays per task (action / point /
server). Reports:

- Per-task pairwise correlation matrices (heatmap-style ASCII).
- Per-task max off-diagonal correlation (the most-redundant pair per task).
- Per-task average correlation (proxy for "how much room is there for stacking
  to add value").

Rough rule of thumb:
- Mean correlation ≥ 0.95 → components are essentially redundant; stacking
  cannot extract much per-row signal.
- Mean correlation 0.85–0.95 → some room; stacking may add 0.001–0.005 OOF.
- Mean correlation ≤ 0.85 → meaningful decorrelation; stacking can plausibly
  add 0.005+ OOF.

This is a T1 read-only diagnostic. No new training, no submission, no schema
change.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")

TAGS = [
    # Group A (v16 family)
    "v16_testhist_aug", "v16_avg3", "v16_seed1", "v16_seed2",
    # Group B (v14 family) + v14_recvhand
    "v14_avg3", "v14_seed0", "v14_seed1", "v14_seed2", "v14_recvhand",
    # Group C
    "v12_5f",
    # Group D (transformers)
    "v11", "v11plus", "v11_aug",
    # Group E
    "v13",
]


def pad19(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] >= 19:
        return arr
    out = np.zeros((arr.shape[0], 19), dtype=arr.dtype)
    out[:, :arr.shape[1]] = arr
    return out


def corr_per_task(tags, mask):
    """Return three (n_tags x n_tags) corr matrices: action, point, server."""
    n = len(tags)
    act_flat = []
    pt_flat = []
    srv_flat = []
    for t in tags:
        a = pad19(np.load(f"{OOF_DIR}/{t}_oof_act.npy"))[mask]
        p = np.load(f"{OOF_DIR}/{t}_oof_pt.npy")[mask]
        s = np.load(f"{OOF_DIR}/{t}_oof_srv.npy")[mask]
        act_flat.append(a.reshape(-1))
        pt_flat.append(p.reshape(-1))
        srv_flat.append(s.reshape(-1))

    def corrmat(arrs):
        m = np.stack(arrs, axis=0)  # (n_tags, n_features)
        return np.corrcoef(m)

    return corrmat(act_flat), corrmat(pt_flat), corrmat(srv_flat)


def print_matrix(name, mat, tags):
    print(f"\n=== {name} pairwise correlation ===")
    n = len(tags)
    head = "                    " + "".join(f"{t[:10]:>11s}" for t in tags)
    print(head)
    for i, t in enumerate(tags):
        row = f"  {t:18s}"
        for j in range(n):
            v = mat[i, j]
            if i == j:
                row += "       1.00"
            else:
                row += f"     {v:.4f}"
        print(row)
    # Off-diag stats
    od = mat.copy()
    np.fill_diagonal(od, np.nan)
    odv = od[~np.isnan(od)]
    print(f"  off-diag: min={np.nanmin(od):.4f}  mean={np.nanmean(od):.4f}  "
          f"max={np.nanmax(od):.4f}  median={np.nanmedian(od):.4f}")
    # Top-5 most-redundant pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((mat[i, j], tags[i], tags[j]))
    pairs.sort(reverse=True)
    print(f"  top-5 most-redundant pairs:")
    for v, a, b in pairs[:5]:
        print(f"    {v:.4f}  {a:18s}  {b}")
    # Bottom-5 most-decorrelated pairs
    print(f"  top-5 most-decorrelated pairs:")
    for v, a, b in pairs[-5:][::-1]:
        print(f"    {v:.4f}  {a:18s}  {b}")


def main():
    # Reference tag for mask + y arrays
    ref = "v16_testhist_aug"
    mask = np.load(f"{OOF_DIR}/{ref}_oof_mask.npy")
    print(f"Mask sum: {mask.sum()} / {len(mask)} (must equal 69712)")

    tags = TAGS
    print(f"Components: {len(tags)}")
    for t in tags:
        if not os.path.exists(f"{OOF_DIR}/{t}_oof_act.npy"):
            raise FileNotFoundError(f"{OOF_DIR}/{t}_oof_act.npy")
    print("All component OOF files present.")

    # Hard-check mask alignment
    for t in tags:
        m = np.load(f"{OOF_DIR}/{t}_oof_mask.npy")
        assert np.array_equal(m, mask), f"mask mismatch: {t}"
    print("All masks byte-equal.")

    cm_a, cm_p, cm_s = corr_per_task(tags, mask)
    print_matrix("ACTION (19-dim probs)", cm_a, tags)
    print_matrix("POINT  (10-dim probs)", cm_p, tags)
    print_matrix("SERVER (scalar)", cm_s, tags)

    # Stacking-room verdict per task
    print("\n=== Stacking-room verdict ===")
    for name, mat in [("ACTION", cm_a), ("POINT", cm_p), ("SERVER", cm_s)]:
        od = mat.copy()
        np.fill_diagonal(od, np.nan)
        mu = np.nanmean(od)
        if mu >= 0.95:
            verdict = "LOW (≥0.95) — components nearly redundant; stacking ceiling tight"
        elif mu >= 0.85:
            verdict = "MEDIUM (0.85–0.95) — some room; expect +0.001 to +0.005"
        else:
            verdict = "HIGH (<0.85) — meaningful decorrelation; stacking plausible +0.005+"
        print(f"  {name:6s} mean off-diag corr = {mu:.4f}  →  {verdict}")


if __name__ == "__main__":
    main()
