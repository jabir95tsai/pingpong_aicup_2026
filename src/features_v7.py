"""Feature engineering V7: extends V6 with action-point grammar priors.

V6 baseline: 1138 features (V5 + 4 extra one-hot lags).
V7 adds ~30 features that encode physical / tactical compatibility:

  Group 1 — depth/side/valid priors conditioned on (prev_action, phase):
    P(point_depth=short | prev_action, phase) → v7_p_depth_short
    P(point_depth=half  | prev_action, phase) → v7_p_depth_half
    P(point_depth=long  | prev_action, phase) → v7_p_depth_long
    P(point_depth=none  | prev_action, phase) → v7_p_depth_none
    P(point_side=FH     | prev_action, phase) → v7_p_side_fh
    P(point_side=mid    | prev_action, phase) → v7_p_side_mid
    P(point_side=BH     | prev_action, phase) → v7_p_side_bh
    P(is_valid          | prev_action, phase) → v7_p_valid

  Group 2 — refined priors conditioned on (prev_action, prev_point, phase):
    P(is_valid       | prev_action, prev_point, phase) → v7_p_valid_refined
    P(point_depth_d  | prev_action, prev_point, phase) → v7_p_depth_*_refined

  Group 3 — third-order Markov (trigram + phase):
    P(next_action | prev2, prev1, phase) → v7_trigram_top_action,
        v7_trigram_top_prob, v7_trigram_entropy, v7_trigram_top1_minus_top2

  Group 4 — SN-specific grammar (SN=2 receive-action prior):
    P(receive_action | serve_action_lag1, sex) → v7_recv_top_action,
        v7_recv_top_prob, v7_recv_entropy

  Group 5 — terminal prior:
    P(pt=0 | prev_action, phase) → v7_p_terminal_pa_phase
    P(pt=0 | prev_action, prev_point) → v7_p_terminal_papp

All probability tables are computed on training fold data (fold-safe).
Falls back to uniform when (key) not seen.
"""
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v6 import (
    build_features_v6, compute_global_stats_v6, get_feature_names_v6,
)

# pointId classes:  0=miss, 1-3=short(FH/mid/BH), 4-6=half, 7-9=long
DEPTH_BUCKET = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3}
# 0=none, 1=short, 2=half, 3=long
SIDE_BUCKET = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 1, 8: 2, 9: 3}
# 0=none, 1=FH, 2=mid, 3=BH

N_DEPTH = 4
N_SIDE = 4
N_ACT = 19
N_PT = 10


def _phase_of(sn):
    """Phase bucket: 1=serve, 2=receive, 3=early(3-4), 4=mid(5-8), 5=late(9+)."""
    if sn == 1:
        return 1
    if sn == 2:
        return 2
    if sn <= 4:
        return 3
    if sn <= 8:
        return 4
    return 5


def _build_grammar_tables(train_df: pd.DataFrame) -> dict:
    """Build all conditional probability tables from training fold data.

    Returns dict with keys:
      depth_pa_phase[(prev_a, phase)]  -> [4]    P(depth | prev_a, phase)
      side_pa_phase[(prev_a, phase)]   -> [4]    P(side  | prev_a, phase)
      valid_pa_phase[(prev_a, phase)]  -> float  P(valid | prev_a, phase)
      depth_papp[(prev_a, prev_p, phase)] -> [4] P(depth | prev_a, prev_p, phase)
      valid_papp[(prev_a, prev_p, phase)] -> float
      trigram[(prev2_a, prev1_a, phase)] -> [N_ACT] P(next_a | trigram_ctx)
      recv_serve[(serve_a, sex)]       -> [N_ACT] P(receive_a | serve_a, sex)
      term_pa_phase[(prev_a, phase)]   -> float  P(pt=0 | prev_a, phase)
      term_papp[(prev_a, prev_p)]      -> float  P(pt=0 | prev_a, prev_p)
    """
    tables = {
        "depth_pa_phase": {}, "side_pa_phase": {}, "valid_pa_phase": {},
        "depth_papp": {}, "valid_papp": {},
        "trigram": {}, "recv_serve": {},
        "term_pa_phase": {}, "term_papp": {},
    }

    rallies = train_df.groupby("rally_uid", sort=False)
    for _, group in rallies:
        group = group.sort_values("strikeNumber")
        sns  = group["strikeNumber"].values.astype(int)
        acts = group["actionId"].values.astype(int)
        pts  = group["pointId"].values.astype(int)
        sex  = int(group["sex"].iloc[0]) if "sex" in group.columns else 1
        n = len(group)

        for i in range(1, n):  # current shot uses i-1, i-2 as context
            cur_a = acts[i] if acts[i] < N_ACT else -1
            cur_p = pts[i]  if pts[i]  < N_PT  else -1
            if cur_a < 0 or cur_p < 0:
                continue
            prev_a = acts[i-1] if acts[i-1] < N_ACT else -1
            prev_p = pts[i-1]  if pts[i-1]  < N_PT  else -1
            phase = _phase_of(sns[i])
            depth = DEPTH_BUCKET[cur_p]
            side  = SIDE_BUCKET[cur_p]
            is_valid = int(cur_p != 0)
            is_term  = int(cur_p == 0)

            # --- (prev_a, phase) ---
            key = (prev_a, phase)
            t = tables["depth_pa_phase"].setdefault(key, np.zeros(N_DEPTH))
            t[depth] += 1
            t = tables["side_pa_phase"].setdefault(key, np.zeros(N_SIDE))
            t[side] += 1
            tv = tables["valid_pa_phase"].setdefault(key, [0, 0])
            tv[0] += is_valid; tv[1] += 1
            tt = tables["term_pa_phase"].setdefault(key, [0, 0])
            tt[0] += is_term; tt[1] += 1

            # --- (prev_a, prev_p, phase) ---
            key3 = (prev_a, prev_p, phase)
            t = tables["depth_papp"].setdefault(key3, np.zeros(N_DEPTH))
            t[depth] += 1
            tv = tables["valid_papp"].setdefault(key3, [0, 0])
            tv[0] += is_valid; tv[1] += 1

            # --- term refined by (prev_a, prev_p) ---
            key2 = (prev_a, prev_p)
            tt = tables["term_papp"].setdefault(key2, [0, 0])
            tt[0] += is_term; tt[1] += 1

            # --- trigram: (prev2_a, prev1_a, phase) → next_a ---
            if i >= 2:
                prev2_a = acts[i-2] if acts[i-2] < N_ACT else -1
                tkey = (prev2_a, prev_a, phase)
                t = tables["trigram"].setdefault(tkey, np.zeros(N_ACT))
                t[cur_a] += 1

            # --- receive prior: SN=2 conditioned on (serve_action_lag1, sex) ---
            if sns[i] == 2:
                rkey = (prev_a, sex)
                t = tables["recv_serve"].setdefault(rkey, np.zeros(N_ACT))
                t[cur_a] += 1

    # Normalize all distributions
    for key in tables["depth_pa_phase"]:
        s = tables["depth_pa_phase"][key].sum()
        if s > 0:
            tables["depth_pa_phase"][key] /= s
    for key in tables["side_pa_phase"]:
        s = tables["side_pa_phase"][key].sum()
        if s > 0:
            tables["side_pa_phase"][key] /= s
    for key in tables["depth_papp"]:
        s = tables["depth_papp"][key].sum()
        if s > 0:
            tables["depth_papp"][key] /= s
    for key in tables["trigram"]:
        s = tables["trigram"][key].sum()
        if s > 0:
            tables["trigram"][key] /= s
    for key in tables["recv_serve"]:
        s = tables["recv_serve"][key].sum()
        if s > 0:
            tables["recv_serve"][key] /= s
    # valid / terminal: convert to single rate
    for key in list(tables["valid_pa_phase"]):
        v, n = tables["valid_pa_phase"][key]
        tables["valid_pa_phase"][key] = v / n if n > 0 else 0.5
    for key in list(tables["valid_papp"]):
        v, n = tables["valid_papp"][key]
        tables["valid_papp"][key] = v / n if n > 0 else 0.5
    for key in list(tables["term_pa_phase"]):
        v, n = tables["term_pa_phase"][key]
        tables["term_pa_phase"][key] = v / n if n > 0 else 0.2
    for key in list(tables["term_papp"]):
        v, n = tables["term_papp"][key]
        tables["term_papp"][key] = v / n if n > 0 else 0.2

    return tables


def _entropy(p, eps=1e-9):
    p = np.asarray(p) + eps
    p = p / p.sum()
    return -float((p * np.log(p)).sum())


def add_grammar_features(feat_df: pd.DataFrame, raw_df: pd.DataFrame,
                          tables: dict) -> pd.DataFrame:
    """Look up grammar prior features for each (prev_a, prev_p, phase, ...) row."""
    df = feat_df.copy()
    n = len(df)

    nsn = df["next_strikeNumber"].values.astype(int)
    phases = np.array([_phase_of(int(s)) for s in nsn])

    # Locate prev1 / prev2 actions and points by looking up raw_df at sn = nsn-1, nsn-2
    shot_lookup = raw_df[["rally_uid", "strikeNumber",
                          "actionId", "pointId"]].copy()
    shot_lookup["strikeNumber"] = shot_lookup["strikeNumber"].astype(int)

    def fetch_lag(lag):
        target_sn = nsn - lag
        merge_left = pd.DataFrame({
            "rally_uid": df["rally_uid"].values,
            "strikeNumber": target_sn,
        })
        merged = merge_left.merge(shot_lookup, on=["rally_uid", "strikeNumber"], how="left")
        a = merged["actionId"].fillna(-1).astype(int).values
        p = merged["pointId"].fillna(-1).astype(int).values
        return a, p

    prev1_a, prev1_p = fetch_lag(1)
    prev2_a, _       = fetch_lag(2)

    # sex per row
    sex_arr = df["sex"].values.astype(int) if "sex" in df.columns else np.ones(n, dtype=int)

    # ---------- Group 1 ----------
    p_depth = np.zeros((n, N_DEPTH), dtype=np.float32)
    p_side  = np.zeros((n, N_SIDE),  dtype=np.float32)
    p_valid = np.full(n, 0.5, dtype=np.float32)
    p_term_pa = np.full(n, 0.2, dtype=np.float32)
    for i in range(n):
        key = (int(prev1_a[i]), int(phases[i]))
        if key in tables["depth_pa_phase"]:
            p_depth[i] = tables["depth_pa_phase"][key]
        if key in tables["side_pa_phase"]:
            p_side[i] = tables["side_pa_phase"][key]
        if key in tables["valid_pa_phase"]:
            p_valid[i] = tables["valid_pa_phase"][key]
        if key in tables["term_pa_phase"]:
            p_term_pa[i] = tables["term_pa_phase"][key]

    df["v7_p_depth_none"]  = p_depth[:, 0]
    df["v7_p_depth_short"] = p_depth[:, 1]
    df["v7_p_depth_half"]  = p_depth[:, 2]
    df["v7_p_depth_long"]  = p_depth[:, 3]
    df["v7_p_side_none"]   = p_side[:, 0]
    df["v7_p_side_fh"]     = p_side[:, 1]
    df["v7_p_side_mid"]    = p_side[:, 2]
    df["v7_p_side_bh"]     = p_side[:, 3]
    df["v7_p_valid"]       = p_valid
    df["v7_p_terminal_pa_phase"] = p_term_pa

    # ---------- Group 2: refined depth on (prev_a, prev_p, phase) ----------
    p_depth_r = np.zeros((n, N_DEPTH), dtype=np.float32)
    p_valid_r = np.full(n, 0.5, dtype=np.float32)
    p_term_pp = np.full(n, 0.2, dtype=np.float32)
    for i in range(n):
        key3 = (int(prev1_a[i]), int(prev1_p[i]), int(phases[i]))
        if key3 in tables["depth_papp"]:
            p_depth_r[i] = tables["depth_papp"][key3]
        if key3 in tables["valid_papp"]:
            p_valid_r[i] = tables["valid_papp"][key3]
        kpp = (int(prev1_a[i]), int(prev1_p[i]))
        if kpp in tables["term_papp"]:
            p_term_pp[i] = tables["term_papp"][kpp]

    df["v7_p_depth_none_r"]  = p_depth_r[:, 0]
    df["v7_p_depth_short_r"] = p_depth_r[:, 1]
    df["v7_p_depth_half_r"]  = p_depth_r[:, 2]
    df["v7_p_depth_long_r"]  = p_depth_r[:, 3]
    df["v7_p_valid_refined"] = p_valid_r
    df["v7_p_terminal_papp"] = p_term_pp

    # ---------- Group 3: trigram (prev2, prev1, phase) → action ----------
    tg_top  = np.zeros(n, dtype=np.float32)
    tg_top1 = np.zeros(n, dtype=np.int32)
    tg_ent  = np.zeros(n, dtype=np.float32)
    tg_marg = np.zeros(n, dtype=np.float32)
    for i in range(n):
        key = (int(prev2_a[i]), int(prev1_a[i]), int(phases[i]))
        if key in tables["trigram"]:
            d = tables["trigram"][key]
            top1 = int(np.argmax(d))
            top1_p = float(d[top1])
            d2 = d.copy(); d2[top1] = -1
            top2_p = float(d2.max())
            tg_top[i]  = top1_p
            tg_top1[i] = top1
            tg_ent[i]  = _entropy(d)
            tg_marg[i] = top1_p - max(top2_p, 0.0)
    df["v7_trigram_top_prob"]   = tg_top
    df["v7_trigram_top_action"] = tg_top1
    df["v7_trigram_entropy"]    = tg_ent
    df["v7_trigram_margin"]     = tg_marg

    # ---------- Group 4: receive prior (SN=2) ----------
    rv_top  = np.zeros(n, dtype=np.float32)
    rv_top1 = np.zeros(n, dtype=np.int32)
    rv_ent  = np.zeros(n, dtype=np.float32)
    is_recv = (nsn == 2).astype(np.float32)
    for i in range(n):
        if nsn[i] != 2:
            continue
        key = (int(prev1_a[i]), int(sex_arr[i]))
        if key in tables["recv_serve"]:
            d = tables["recv_serve"][key]
            top1 = int(np.argmax(d))
            rv_top[i]  = float(d[top1])
            rv_top1[i] = top1
            rv_ent[i]  = _entropy(d)
    df["v7_recv_top_prob"]   = rv_top
    df["v7_recv_top_action"] = rv_top1
    df["v7_recv_entropy"]    = rv_ent
    df["v7_is_recv_phase"]   = is_recv  # redundant w/ V5 but keep for clarity

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_global_stats_v7(train_df: pd.DataFrame) -> dict:
    """Compute V6 stats + V7 grammar tables (fold-safe; pass training fold only)."""
    stats = compute_global_stats_v6(train_df)
    stats["v7_grammar"] = _build_grammar_tables(train_df)
    return stats


def build_features_v7(df: pd.DataFrame, is_train: bool,
                       global_stats_v7: dict,
                       raw_df: pd.DataFrame = None) -> pd.DataFrame:
    """Full V7 features: V6 + grammar priors."""
    feat_df = build_features_v6(df, is_train=is_train,
                                 global_stats_v6=global_stats_v7,
                                 raw_df=raw_df)
    if raw_df is None:
        raw_df = df
    feat_df = add_grammar_features(feat_df, raw_df, global_stats_v7["v7_grammar"])
    return feat_df


def get_feature_names_v7(feat_df: pd.DataFrame) -> list:
    """Same exclusion rules as V6 — V7 features are all SGP-clean."""
    return get_feature_names_v6(feat_df)
