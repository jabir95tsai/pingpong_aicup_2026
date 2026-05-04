"""Feature engineering V8: extends V7 with point-grammar priors.

V7 baseline: 1145 features (V6 + 24 action-grammar priors).
V8 adds ~34 features encoding how point landing zones chain through the rally:

  Group 1 — point-side transition (where does ball go given prev landing + action?):
    P(pt_side=FH  | prev_pt_side, prev_action, phase) → v8_ps_side_fh
    P(pt_side=mid | prev_pt_side, prev_action, phase) → v8_ps_side_mid
    P(pt_side=BH  | prev_pt_side, prev_action, phase) → v8_ps_side_bh
    P(pt_side=none| ...)                              → v8_ps_side_none

  Group 2 — point-depth transition:
    P(pt_depth=short | prev_pt_depth, prev_action, phase) → v8_pd_depth_short
    P(pt_depth=half  | ...)                               → v8_pd_depth_half
    P(pt_depth=long  | ...)                               → v8_pd_depth_long
    P(pt_depth=none  | ...)                               → v8_pd_depth_none

  Group 3 — ball physics → point (strength × spin → landing zone):
    P(pt_depth=* | prev_strength, prev_spin) → v8_phys_depth_*   (4 features)
    P(pt_side=*  | prev_strength, prev_spin) → v8_phys_side_*    (4 features)

  Group 4 — receive-shot point prior (SN=2):
    P(recv_pt_side=* | serve_action, sex)  → v8_recv_pt_side_*   (4 features)
    P(recv_pt_depth=*| serve_action, sex)  → v8_recv_pt_depth_*  (4 features)
    entropy(recv_pt_side | serve_action, sex) → v8_recv_pt_side_ent
    entropy(recv_pt_depth| ...)               → v8_recv_pt_depth_ent

  Group 5 — action-trigram for point (prev2_a, prev1_a, phase) → point:
    top-point, prob, entropy, margin  → v8_pt_tg_top_pt, v8_pt_tg_top_prob,
                                         v8_pt_tg_entropy, v8_pt_tg_margin

All tables computed on training fold only (fold-safe).
"""
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v7 import (
    build_features_v7, compute_global_stats_v7, get_feature_names_v7,
    _phase_of, _entropy,
    DEPTH_BUCKET, SIDE_BUCKET,
    N_DEPTH, N_SIDE, N_ACT, N_PT,
)

# Spin values: 0=none, 1=top, 2=back, 3=no-spin, 4=side-top, 5=side-back
N_SPIN   = 6
# Strength values: 0=none, 1=strong, 2=mid, 3=weak
N_STR    = 4


def _build_point_grammar_tables(train_df: pd.DataFrame) -> dict:
    """Build point-grammar conditional probability tables from training fold data."""
    tables = {
        "side_transition":   {},   # (prev_pt_side, prev_action, phase) → [N_SIDE]
        "depth_transition":  {},   # (prev_pt_depth, prev_action, phase) → [N_DEPTH]
        "phys_depth":        {},   # (prev_strength, prev_spin) → [N_DEPTH]
        "phys_side":         {},   # (prev_strength, prev_spin) → [N_SIDE]
        "recv_pt_side":      {},   # (serve_action, sex) → [N_SIDE]
        "recv_pt_depth":     {},   # (serve_action, sex) → [N_DEPTH]
        "pt_trigram":        {},   # (prev2_action, prev1_action, phase) → [N_PT]
    }

    rallies = train_df.groupby("rally_uid", sort=False)
    for _, group in rallies:
        group = group.sort_values("strikeNumber")
        sns    = group["strikeNumber"].values.astype(int)
        acts   = group["actionId"].values.astype(int)
        pts    = group["pointId"].values.astype(int)
        strs   = group["strengthId"].values.astype(int) \
                 if "strengthId" in group.columns else np.zeros(len(group), int)
        spins  = group["spinId"].values.astype(int) \
                 if "spinId" in group.columns else np.zeros(len(group), int)
        sex    = int(group["sex"].iloc[0]) if "sex" in group.columns else 1
        n = len(group)

        for i in range(1, n):
            cur_p = pts[i]
            if cur_p < 0 or cur_p >= N_PT:
                continue
            prev_a = int(acts[i-1]) if acts[i-1] < N_ACT else -1
            prev_p = int(pts[i-1])  if pts[i-1]  < N_PT  else -1
            phase  = _phase_of(sns[i])
            cur_depth = DEPTH_BUCKET[cur_p]
            cur_side  = SIDE_BUCKET[cur_p]

            # --- Group 1: side transition ---
            if prev_p >= 0:
                prev_side  = SIDE_BUCKET[prev_p]
                prev_depth = DEPTH_BUCKET[prev_p]
                skey = (prev_side, prev_a, phase)
                t = tables["side_transition"].setdefault(skey, np.zeros(N_SIDE))
                t[cur_side] += 1
                # --- Group 2: depth transition ---
                dkey = (prev_depth, prev_a, phase)
                t = tables["depth_transition"].setdefault(dkey, np.zeros(N_DEPTH))
                t[cur_depth] += 1

            # --- Group 3: physics (strength × spin) → point ---
            prev_str  = int(strs[i-1])  if strs[i-1]  < N_STR  else 0
            prev_spin = int(spins[i-1]) if spins[i-1] < N_SPIN else 0
            pkey = (prev_str, prev_spin)
            t = tables["phys_depth"].setdefault(pkey, np.zeros(N_DEPTH))
            t[cur_depth] += 1
            t = tables["phys_side"].setdefault(pkey, np.zeros(N_SIDE))
            t[cur_side] += 1

            # --- Group 4: receive point prior (SN=2) ---
            if sns[i] == 2:
                rkey = (prev_a, sex)
                t = tables["recv_pt_side"].setdefault(rkey, np.zeros(N_SIDE))
                t[cur_side] += 1
                t = tables["recv_pt_depth"].setdefault(rkey, np.zeros(N_DEPTH))
                t[cur_depth] += 1

            # --- Group 5: action-trigram → point ---
            if i >= 2:
                prev2_a = int(acts[i-2]) if acts[i-2] < N_ACT else -1
                tkey = (prev2_a, prev_a, phase)
                t = tables["pt_trigram"].setdefault(tkey, np.zeros(N_PT))
                t[cur_p] += 1

    # Normalize
    for tbl_name in ["side_transition", "depth_transition",
                     "phys_depth", "phys_side",
                     "recv_pt_side", "recv_pt_depth", "pt_trigram"]:
        for key in tables[tbl_name]:
            s = tables[tbl_name][key].sum()
            if s > 0:
                tables[tbl_name][key] /= s

    return tables


def add_point_grammar_features(feat_df: pd.DataFrame, raw_df: pd.DataFrame,
                                tables: dict) -> pd.DataFrame:
    """Look up point-grammar prior features for each row."""
    df  = feat_df.copy()
    n   = len(df)
    nsn = df["next_strikeNumber"].values.astype(int)
    phases = np.array([_phase_of(int(s)) for s in nsn])

    # Fetch lag-1 and lag-2 shots
    shot_lookup = raw_df[["rally_uid", "strikeNumber",
                          "actionId", "pointId",
                          "strengthId", "spinId"]].copy()
    for c in ["actionId", "pointId", "strengthId", "spinId"]:
        if c not in shot_lookup.columns:
            shot_lookup[c] = 0
    shot_lookup["strikeNumber"] = shot_lookup["strikeNumber"].astype(int)

    def fetch_lag(lag):
        target_sn = nsn - lag
        left = pd.DataFrame({
            "rally_uid":    df["rally_uid"].values,
            "strikeNumber": target_sn,
        })
        merged = left.merge(shot_lookup, on=["rally_uid", "strikeNumber"], how="left")
        a   = merged["actionId"].fillna(-1).astype(int).values
        p   = merged["pointId"].fillna(-1).astype(int).values
        str_= merged["strengthId"].fillna(0).astype(int).values
        spn = merged["spinId"].fillna(0).astype(int).values
        return a, p, str_, spn

    prev1_a, prev1_p, prev1_str, prev1_spn = fetch_lag(1)
    prev2_a, _,       _,         _          = fetch_lag(2)

    sex_arr = df["sex"].values.astype(int) if "sex" in df.columns else np.ones(n, int)

    # ── Group 1: side transition ──────────────────────────────────────────────
    side_trans = np.zeros((n, N_SIDE), dtype=np.float32)
    for i in range(n):
        if prev1_p[i] < 0 or prev1_p[i] >= N_PT:
            continue
        prev_side = SIDE_BUCKET[prev1_p[i]]
        key = (prev_side, int(prev1_a[i]), int(phases[i]))
        if key in tables["side_transition"]:
            side_trans[i] = tables["side_transition"][key]
        else:
            side_trans[i] = 0.25  # uniform
    df["v8_ps_side_none"] = side_trans[:, 0]
    df["v8_ps_side_fh"]   = side_trans[:, 1]
    df["v8_ps_side_mid"]  = side_trans[:, 2]
    df["v8_ps_side_bh"]   = side_trans[:, 3]

    # ── Group 2: depth transition ─────────────────────────────────────────────
    depth_trans = np.zeros((n, N_DEPTH), dtype=np.float32)
    for i in range(n):
        if prev1_p[i] < 0 or prev1_p[i] >= N_PT:
            continue
        prev_depth = DEPTH_BUCKET[prev1_p[i]]
        key = (prev_depth, int(prev1_a[i]), int(phases[i]))
        if key in tables["depth_transition"]:
            depth_trans[i] = tables["depth_transition"][key]
        else:
            depth_trans[i] = 0.25
    df["v8_pd_depth_none"]  = depth_trans[:, 0]
    df["v8_pd_depth_short"] = depth_trans[:, 1]
    df["v8_pd_depth_half"]  = depth_trans[:, 2]
    df["v8_pd_depth_long"]  = depth_trans[:, 3]

    # ── Group 3: physics (strength × spin) ───────────────────────────────────
    phys_d = np.zeros((n, N_DEPTH), dtype=np.float32)
    phys_s = np.zeros((n, N_SIDE),  dtype=np.float32)
    for i in range(n):
        str_  = int(prev1_str[i]) if prev1_str[i] < N_STR  else 0
        spin_ = int(prev1_spn[i]) if prev1_spn[i] < N_SPIN else 0
        pkey = (str_, spin_)
        if pkey in tables["phys_depth"]:
            phys_d[i] = tables["phys_depth"][pkey]
        if pkey in tables["phys_side"]:
            phys_s[i] = tables["phys_side"][pkey]
    df["v8_phys_depth_none"]  = phys_d[:, 0]
    df["v8_phys_depth_short"] = phys_d[:, 1]
    df["v8_phys_depth_half"]  = phys_d[:, 2]
    df["v8_phys_depth_long"]  = phys_d[:, 3]
    df["v8_phys_side_none"]   = phys_s[:, 0]
    df["v8_phys_side_fh"]     = phys_s[:, 1]
    df["v8_phys_side_mid"]    = phys_s[:, 2]
    df["v8_phys_side_bh"]     = phys_s[:, 3]

    # ── Group 4: receive point prior (SN=2 specific) ─────────────────────────
    rv_side   = np.zeros((n, N_SIDE),  dtype=np.float32)
    rv_depth  = np.zeros((n, N_DEPTH), dtype=np.float32)
    rv_s_ent  = np.full(n, _entropy(np.ones(N_SIDE)), dtype=np.float32)
    rv_d_ent  = np.full(n, _entropy(np.ones(N_DEPTH)), dtype=np.float32)
    is_recv   = (nsn == 2)
    for i in np.where(is_recv)[0]:
        key = (int(prev1_a[i]), int(sex_arr[i]))
        if key in tables["recv_pt_side"]:
            d = tables["recv_pt_side"][key]
            rv_side[i] = d
            rv_s_ent[i] = _entropy(d)
        if key in tables["recv_pt_depth"]:
            d = tables["recv_pt_depth"][key]
            rv_depth[i] = d
            rv_d_ent[i] = _entropy(d)
    df["v8_recv_pt_side_none"]  = rv_side[:, 0]
    df["v8_recv_pt_side_fh"]    = rv_side[:, 1]
    df["v8_recv_pt_side_mid"]   = rv_side[:, 2]
    df["v8_recv_pt_side_bh"]    = rv_side[:, 3]
    df["v8_recv_pt_depth_none"] = rv_depth[:, 0]
    df["v8_recv_pt_depth_short"]= rv_depth[:, 1]
    df["v8_recv_pt_depth_half"] = rv_depth[:, 2]
    df["v8_recv_pt_depth_long"] = rv_depth[:, 3]
    df["v8_recv_pt_side_ent"]   = rv_s_ent
    df["v8_recv_pt_depth_ent"]  = rv_d_ent

    # ── Group 5: action-trigram for point ─────────────────────────────────────
    tg_top_pt = np.zeros(n, dtype=np.int32)
    tg_top_pb = np.zeros(n, dtype=np.float32)
    tg_ent    = np.full(n, _entropy(np.ones(N_PT)), dtype=np.float32)
    tg_marg   = np.zeros(n, dtype=np.float32)
    for i in range(n):
        key = (int(prev2_a[i]), int(prev1_a[i]), int(phases[i]))
        if key in tables["pt_trigram"]:
            d    = tables["pt_trigram"][key]
            top1 = int(np.argmax(d))
            top1_p = float(d[top1])
            d2 = d.copy(); d2[top1] = -1
            top2_p = float(max(d2.max(), 0.0))
            tg_top_pt[i] = top1
            tg_top_pb[i] = top1_p
            tg_ent[i]    = _entropy(d)
            tg_marg[i]   = top1_p - top2_p
    df["v8_pt_tg_top_pt"]   = tg_top_pt
    df["v8_pt_tg_top_prob"] = tg_top_pb
    df["v8_pt_tg_entropy"]  = tg_ent
    df["v8_pt_tg_margin"]   = tg_marg

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_global_stats_v8(train_df: pd.DataFrame) -> dict:
    """Compute V7 stats + V8 point-grammar tables (fold-safe)."""
    stats = compute_global_stats_v7(train_df)
    stats["v8_pt_grammar"] = _build_point_grammar_tables(train_df)
    return stats


def build_features_v8(df: pd.DataFrame, is_train: bool,
                      global_stats_v8: dict,
                      raw_df: pd.DataFrame = None) -> pd.DataFrame:
    """Full V8 features: V7 + point-grammar priors."""
    feat_df = build_features_v7(df, is_train=is_train,
                                 global_stats_v7=global_stats_v8,
                                 raw_df=raw_df)
    if raw_df is None:
        raw_df = df
    feat_df = add_point_grammar_features(
        feat_df, raw_df, global_stats_v8["v8_pt_grammar"])
    return feat_df


def get_feature_names_v8(feat_df: pd.DataFrame) -> list:
    """Same SGP-clean exclusion as V6/V7."""
    return get_feature_names_v7(feat_df)
