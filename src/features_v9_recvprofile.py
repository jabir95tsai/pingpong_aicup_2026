"""features_v9_recvprofile — extends features_v9_recvhand with 4 receiver-mode axes.

R-011 implementation per Codex APPROVE_WITH_FIXES (2026-05-10):

1. **Encoding** (Codex fix #2 + #3):
   - `action` (15 valid classes 0..14): **one-hot mode** — 15 mode columns
     `recv_action_oh_0..14` plus 1 `recv_action_unknown` indicator.
     Tree splits like `recv_action_mode <= 7` would impose artificial
     label order; one-hot avoids this.
   - `point` (9 valid classes 1..9, cls0 dropped from mode by spec): **one-hot mode**
     — 9 mode columns `recv_point_oh_1..9` plus 1 `recv_point_unknown`.
   - `strength` (3 valid classes 1..3, 0 dropped): integer mode
     `recv_strength_mode` with **unknown=0**, valid range 1..3 (natural shift —
     cls0 is "none" so unknown coincides with the "no valid prior" indicator
     in the integer space).
   - `spin` (5 valid classes 1..5, 0 dropped): integer mode
     `recv_spin_mode` with **unknown=0**, valid range 1..5.
   - `hand` is unchanged via the imported recvhand baseline.

2. **Axis ablation toggle** (Codex fix #4): set `RECVPROFILE_AXES` env var
   before module load to subset axes, e.g. `"strength,spin"` for a
   conservative variant. Default = all 4. Trainer sets this from
   `--recvprofile-axes` flag.

3. **Prefix audit** (Codex fix #6): per build call (train/test), prints
   20 deterministic sample rows showing
   `(rally_uid, N, target_receiver_id, max_source_strikeNumber)` and a
   `[OK]/[VIOLATION]` flag confirming `max_src_sn < N` per row. Plus the
   aggregate assertion (any violation aborts the build).

4. **Distribution log per axis** (Codex fix #5): for train and test
   separately, print unknown% + top-3 mode classes with %.

5. **No count companion features** (Codex fix #7): only mode features +
   unknown indicators. NO `recv_n_prior_shots`, NO action/point counts,
   NO ratios.

6. **`recv_hand_est` unchanged** (Codex fix #5): inherited from
   features_v9_recvhand. Only NEW marginal axes added on top.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v9_recvhand import (
    build_features_v9_recvhand,
    compute_global_stats_v9_recvhand,
    get_feature_names_v9_recvhand,
)

# Axis registry. Each axis lists (lo, hi) for the inclusive valid range
# `lo..hi-1` (Python convention) used in mode computation.
ALL_AXES_ORDER = ["action", "point", "strength", "spin"]
AXIS_RANGES = {
    "action":   (0, 15),  # actionId 0..14 are train action classes
    "point":    (1, 10),  # pointId 1..9 (cls0 = miss/off-grid is dropped from mode)
    "strength": (1, 4),   # strengthId 1..3 (0 = "none" dropped)
    "spin":     (1, 6),   # spinId 1..5 (0 = "none" dropped)
}
ONE_HOT_AXES = {"action", "point"}    # high-cardinality → one-hot
INT_AXES     = {"strength", "spin"}   # low-cardinality → integer mode

# Re-export the global-stats function unchanged (no new tables needed).
compute_global_stats_v9_recvprofile = compute_global_stats_v9_recvhand


def get_active_axes() -> list:
    """Read RECVPROFILE_AXES env var; default to all 4 axes."""
    raw = os.environ.get("RECVPROFILE_AXES", ",".join(ALL_AXES_ORDER))
    requested = [a.strip() for a in raw.split(",") if a.strip()]
    # Preserve canonical order, validate names
    return [a for a in ALL_AXES_ORDER if a in requested]


def _axis_column_names(axis: str) -> list:
    """Return the column names this axis contributes."""
    lo, hi = AXIS_RANGES[axis]
    if axis in ONE_HOT_AXES:
        cols = [f"recv_{axis}_oh_{c}" for c in range(lo, hi)]
        cols.append(f"recv_{axis}_unknown")
        return cols
    else:  # int
        return [f"recv_{axis}_mode"]


def _mode_with_tie(values: np.ndarray):
    """Return (mode, is_tie) where mode is the most-frequent value (None on
    empty). is_tie=True if top-2 frequencies are equal — caller treats as
    unknown."""
    if len(values) == 0:
        return None, False
    vals, counts = np.unique(values, return_counts=True)
    order = np.argsort(-counts)
    top1_count = counts[order[0]]
    if len(counts) >= 2 and counts[order[1]] == top1_count:
        return None, True
    return int(vals[order[0]]), False


def _build_axis_features(target_rows: pd.DataFrame,
                         raw_df: pd.DataFrame,
                         axes: list,
                         label: str) -> tuple:
    """Per-row prefix-only mode features for the requested axes.

    Returns:
      out_columns: dict[str, np.ndarray] of new feature columns
      audit_samples: list of (rally_uid, N, target_receiver_id, max_src_sn)
    """
    cols_needed = ["rally_uid", "strikeNumber", "gamePlayerId",
                   "actionId", "pointId", "strengthId", "spinId"]
    raw = raw_df[cols_needed].copy()
    raw["strikeNumber"] = raw["strikeNumber"].astype(int)
    raw["gamePlayerId"] = raw["gamePlayerId"].astype(int)
    rally_to_rows = {rid: g.sort_values("strikeNumber").reset_index(drop=True)
                     for rid, g in raw.groupby("rally_uid", sort=False)}

    rally_uids = target_rows["rally_uid"].to_numpy()
    next_sns = target_rows["next_strikeNumber"].to_numpy(dtype=int)
    n = len(target_rows)

    # Allocate output arrays
    out_columns: dict = {}
    for axis in axes:
        for col in _axis_column_names(axis):
            out_columns[col] = np.zeros(n, dtype=np.int8)

    # Deterministic audit indices: every n//20-th row, capped at 20 rows
    step = max(n // 20, 1)
    audit_indices = set(range(0, n, step))
    audit_samples: list = []
    max_violations = 0

    for i in range(n):
        rid = rally_uids[i]
        N = int(next_sns[i])
        grp = rally_to_rows.get(rid)

        # Default unknown encoding
        if grp is None or len(grp) == 0:
            for axis in axes:
                if axis in ONE_HOT_AXES:
                    out_columns[f"recv_{axis}_unknown"][i] = 1
            if i in audit_indices and len(audit_samples) < 20:
                audit_samples.append((rid, N, -1, -1))
            continue

        prefix = grp[grp["strikeNumber"] < N]
        if len(prefix) > 0:
            max_src_sn = int(prefix["strikeNumber"].max())
            if max_src_sn >= N:
                max_violations += 1
        else:
            max_src_sn = -1

        # Identify target receiver = gamePlayerId at strike N-1
        recv_row = prefix[prefix["strikeNumber"] == N - 1]
        if len(recv_row) == 0:
            target_receiver_id = -1
            for axis in axes:
                if axis in ONE_HOT_AXES:
                    out_columns[f"recv_{axis}_unknown"][i] = 1
            if i in audit_indices and len(audit_samples) < 20:
                audit_samples.append((rid, N, target_receiver_id, max_src_sn))
            continue
        target_receiver_id = int(recv_row["gamePlayerId"].iloc[0])

        # Per-axis mode
        for axis in axes:
            lo, hi = AXIS_RANGES[axis]
            col_name = f"{axis}Id"
            mask = ((prefix["gamePlayerId"] == target_receiver_id) &
                    (prefix[col_name] >= lo) & (prefix[col_name] < hi))
            src_vals = prefix.loc[mask, col_name].to_numpy(dtype=int)
            mode_v, _is_tie = _mode_with_tie(src_vals)
            if axis in ONE_HOT_AXES:
                if mode_v is None:
                    out_columns[f"recv_{axis}_unknown"][i] = 1
                else:
                    out_columns[f"recv_{axis}_oh_{mode_v}"][i] = 1
            else:  # integer mode (already in [lo, hi-1] when not None)
                out_columns[f"recv_{axis}_mode"][i] = (
                    0 if mode_v is None else mode_v)

        if i in audit_indices and len(audit_samples) < 20:
            audit_samples.append((rid, N, target_receiver_id, max_src_sn))

    # Aggregate invariant assertion
    assert max_violations == 0, (
        f"recv_profile {label}: {max_violations} rows used a source "
        "strikeNumber >= next_strikeNumber. Prefix-only invariant violated.")

    return out_columns, audit_samples


def _print_axis_distribution(feat_df: pd.DataFrame, axes: list, label: str) -> None:
    """Per-axis log: unknown% + top-3 mode classes with %."""
    n = len(feat_df)
    for axis in axes:
        lo, hi = AXIS_RANGES[axis]
        if axis in ONE_HOT_AXES:
            unk_col = f"recv_{axis}_unknown"
            unk_pct = feat_df[unk_col].mean() * 100.0
            mode_pcts = []
            for c in range(lo, hi):
                col = f"recv_{axis}_oh_{c}"
                pct = feat_df[col].mean() * 100.0
                mode_pcts.append((c, pct))
            mode_pcts.sort(key=lambda x: -x[1])
            top3 = ", ".join(f"cls{c}={p:.1f}%" for c, p in mode_pcts[:3])
            print(f"  [recvprofile] {label} {axis} (one-hot): "
                  f"unknown={unk_pct:.1f}%  top3=[{top3}]")
        else:
            mode_col = f"recv_{axis}_mode"
            unk_pct = (feat_df[mode_col] == 0).mean() * 100.0
            counts = feat_df[mode_col].value_counts()
            mode_pcts = sorted(
                [(int(k), v / n * 100.0) for k, v in counts.items() if k > 0],
                key=lambda x: -x[1])
            top3 = ", ".join(f"v{c}={p:.1f}%" for c, p in mode_pcts[:3])
            print(f"  [recvprofile] {label} {axis} (int): "
                  f"unknown={unk_pct:.1f}%  top3=[{top3}]")


def build_features_v9_recvprofile(df: pd.DataFrame, is_train: bool,
                                  global_stats_v9: dict,
                                  raw_df: pd.DataFrame = None) -> pd.DataFrame:
    """v9 + recv_hand_est (R-001) + 4 new receiver-mode axes (R-011)."""
    feat_df = build_features_v9_recvhand(df, is_train=is_train,
                                          global_stats_v9=global_stats_v9,
                                          raw_df=raw_df)
    if raw_df is None:
        raw_df = df

    axes = get_active_axes()
    label = "train" if is_train else "test"
    print(f"  [recvprofile] {label}: building axes={axes}  "
          f"(env RECVPROFILE_AXES = "
          f"{os.environ.get('RECVPROFILE_AXES', '<unset, default all>')})")

    target_rows = feat_df[["rally_uid", "next_strikeNumber"]]
    new_cols, audit_samples = _build_axis_features(
        target_rows, raw_df, axes, label)

    # Append columns
    for col_name, arr in new_cols.items():
        feat_df[col_name] = arr.astype(np.int8)

    # Per-axis distribution
    _print_axis_distribution(feat_df, axes, label)

    # Deterministic prefix audit (Codex fix #6)
    print(f"  [recvprofile] {label} prefix audit ({len(audit_samples)} "
          "deterministic samples; expect all [OK]):")
    for rid, N, recv_id, max_src in audit_samples[:20]:
        ok = "[OK]" if max_src < N else "[VIOLATION]"
        print(f"    rally={str(rid)[:32]:32s}  N={N:3d}  receiver={recv_id:4d}  "
              f"max_src_sn={max_src:3d}  N>max? {ok}")

    return feat_df


def get_feature_names_v9_recvprofile(feat_df: pd.DataFrame) -> list:
    base = list(get_feature_names_v9_recvhand(feat_df))
    base_set = set(base)
    axes = get_active_axes()
    extra = []
    for axis in axes:
        for col in _axis_column_names(axis):
            if col in feat_df.columns and col not in base_set:
                extra.append(col)
    return base + extra
