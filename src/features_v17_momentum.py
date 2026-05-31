"""features_v17_momentum — rally momentum / initiative / pressure-state features.

R-015 implementation per Codex APPROVE_WITH_FIXES (2026-05-11) with 1
documented Claude pushback (Group 4 per-side aggregates kept; only the
parity-bit redundancy with `next_is_server` was removed).

Wraps `features_v9_recvhand` and appends prefix-only momentum features in
5 groups, selectable via `--momentum-groups core|all` (env var
`MOMENTUM_GROUPS_ACTIVE`). Default = `core` per Codex P2.3.

Group composition (post-fixes):
  - Group 1  (4 features): action-group lags
  - Group 2 (12 features): recent-window ratios
  - Group 3 (10 features): streaks & transitions
  - Group 4 (10 features): per-side initiative (parity bit dropped per
                            Codex overlap with `next_is_server`)
  - Group 5  (5 features): pressure derivatives (simplified pressure
                            scalar, fixed-constant per Codex P2.4)

Codex P3.5: explicit `SOURCE_COLS` list + 4 build-time assertions.
"""
from __future__ import annotations

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

# ─── Action-group taxonomy (per CLAUDE.md) ───────────────────────────────────
# 0 = none (actionId 0)
# 1 = attack  (actionId 1..7  : Loop / Cloop / Smash / Flip / Pushfast / Push / Flick)
# 2 = control (actionId 8..11 : Arch / Knuckle / Chop_r / ShortStop)
# 3 = defense (actionId 12..14: Chop / Block / Lob)
# 4 = serve   (actionId 15..18: serve types — never appear as next-shot targets,
#                                only as prefix shots at strikeNumber=1)
ATTACK_ACTIONS = {1, 2, 3, 4, 5, 6, 7}
CONTROL_ACTIONS = {8, 9, 10, 11}
DEFENSE_ACTIONS = {12, 13, 14}
SERVE_ACTIONS = {15, 16, 17, 18}

ACTION_GROUP_MAP = np.zeros(20, dtype=np.int8)  # actionId 0..18 + safety pad
for _a in ATTACK_ACTIONS:
    ACTION_GROUP_MAP[_a] = 1
for _a in CONTROL_ACTIONS:
    ACTION_GROUP_MAP[_a] = 2
for _a in DEFENSE_ACTIONS:
    ACTION_GROUP_MAP[_a] = 3
for _a in SERVE_ACTIONS:
    ACTION_GROUP_MAP[_a] = 4

GROUP_ATTACK = 1
GROUP_CONTROL = 2
GROUP_DEFENSE = 3

# ─── Codex P3.5: source-column allow-list + forbidden-field guard ─────────────
SOURCE_COLS = [
    "rally_uid",      # groupby key only — NEVER an embedding/feature input
    "strikeNumber",   # per-shot, prefix-only
    "actionId",       # per-shot
    "strengthId",     # per-shot (used by simplified pressure scalar)
]
FORBIDDEN_SOURCE = {
    "serverGetPoint", "match", "gamePlayerId", "gamePlayerOtherId",
}
assert FORBIDDEN_SOURCE.isdisjoint(set(SOURCE_COLS)), (
    f"VIOLATION (build-time, 8.D): SOURCE_COLS contains forbidden: "
    f"{FORBIDDEN_SOURCE & set(SOURCE_COLS)}"
)


# ─── Pressure scalar (Codex P2.4: fixed-constant only, no fold stats) ────────
def pressure_array(actions: np.ndarray, strengths: np.ndarray) -> np.ndarray:
    """Per-shot pressure = is_attack(actionId) × strength_factor(strengthId).

    Fixed constants:
      strength_factor: 1=strong→1.5, 2=mid→1.0, 3=weak→0.5, 0/other→1.0
      is_attack:       actionId in {1..7} → 1, else 0

    Range: [0.0, 1.5].
    """
    is_attack = ((actions >= 1) & (actions <= 7)).astype(np.float32)
    sf = np.ones_like(actions, dtype=np.float32)
    sf[strengths == 1] = 1.5
    sf[strengths == 3] = 0.5
    return is_attack * sf


# ─── Feature group registry ──────────────────────────────────────────────────
GROUP_1_FEATURES = [
    "v17m_prev1_action_group",
    "v17m_prev2_action_group",
    "v17m_prev1_is_attack",
    "v17m_prev1_is_defense",
]
GROUP_2_FEATURES = [
    "v17m_recent3_attack_ratio", "v17m_recent3_control_ratio",
    "v17m_recent3_defense_ratio", "v17m_recent3_attack_count",
    "v17m_recent5_attack_ratio", "v17m_recent5_control_ratio",
    "v17m_recent5_defense_ratio", "v17m_recent5_attack_count",
    "v17m_recent3_initiative_score", "v17m_recent5_initiative_score",
    "v17m_recent3_pressure_score", "v17m_recent5_pressure_score",
]
GROUP_3_FEATURES = [
    "v17m_attack_streak_len", "v17m_defense_streak_len", "v17m_control_streak_len",
    "v17m_n_attacks_total", "v17m_n_defenses_total", "v17m_n_controls_total",
    "v17m_transitions_atk_to_def", "v17m_transitions_def_to_atk",
    "v17m_transitions_ctl_to_atk",
    "v17m_n_action_group_changes",
]
GROUP_4_FEATURES = [
    # Per-side aggregates (NEW info; not redundant with existing next_is_server)
    "v17m_server_side_attack_count", "v17m_returner_side_attack_count",
    "v17m_server_side_attack_ratio", "v17m_returner_side_attack_ratio",
    "v17m_server_side_avg_pressure", "v17m_returner_side_avg_pressure",
    "v17m_pressure_imbalance", "v17m_attack_imbalance",
    "v17m_target_hitter_recent_was_attacking",
    "v17m_target_hitter_no_prior_own_shot",
    # NOTE: dropped `v17m_target_hitter_is_server_side` per Codex —
    # redundant with existing `next_is_server` / `next_sn_parity` from features_v3.
]
GROUP_5_FEATURES = [
    # Simplified pressure derivatives (Codex P2.4: fixed-constant scalar only)
    "v17m_prev1_pressure", "v17m_prev2_pressure", "v17m_prev3_pressure",
    "v17m_pressure_delta_1_2",
    "v17m_target_hitter_under_pressure",
    # NOTE: dropped slope, max, min, delta_2_3 per Claude self-review
    # (redundant linear combos of the kept features).
]
ALL_GROUPS = {
    1: GROUP_1_FEATURES,
    2: GROUP_2_FEATURES,
    3: GROUP_3_FEATURES,
    4: GROUP_4_FEATURES,
    5: GROUP_5_FEATURES,
}
GROUP_PRESETS = {
    "core": [1, 2, 3],          # 26 features — first smoke per Codex P2.3
    "all":  [1, 2, 3, 4, 5],    # 41 features — second smoke if core passes
}

CAP_STREAK = 5
CAP_TOTAL = 20

# Re-export global stats unchanged (Codex: no new fold tables permitted).
compute_global_stats_v17_momentum = compute_global_stats_v9_recvhand


def get_active_groups() -> tuple[list, str]:
    raw = os.environ.get("MOMENTUM_GROUPS_ACTIVE", "core").strip()
    if raw not in GROUP_PRESETS:
        raise ValueError(
            f"MOMENTUM_GROUPS_ACTIVE='{raw}' invalid; must be one of "
            f"{list(GROUP_PRESETS.keys())}")
    return GROUP_PRESETS[raw], raw


def get_active_features() -> list:
    groups, _ = get_active_groups()
    feats = []
    for g in groups:
        feats.extend(ALL_GROUPS[g])
    return feats


def get_feature_names_v17_momentum(feat_df: pd.DataFrame) -> list:
    """Base v9_recvhand features + the active v17m features present in feat_df."""
    base = list(get_feature_names_v9_recvhand(feat_df))
    base_set = set(base)
    extra = []
    for f in get_active_features():
        if f in feat_df.columns and f not in base_set:
            extra.append(f)
    return base + extra


# ─── Per-rally array precomputation ──────────────────────────────────────────

def _per_rally_arrays(rally_grp: pd.DataFrame) -> dict:
    """Compute per-shot arrays for one rally (sn-sorted)."""
    grp = rally_grp.sort_values("strikeNumber").reset_index(drop=True)
    sn = grp["strikeNumber"].to_numpy(dtype=np.int32)
    action = grp["actionId"].to_numpy(dtype=np.int32)
    strength = grp["strengthId"].to_numpy(dtype=np.int32)
    group = ACTION_GROUP_MAP[np.clip(action, 0, 19)]
    pressure = pressure_array(action, strength)
    # side: 0 = server side (odd SN: 1, 3, 5, ...)
    #       1 = returner side (even SN: 2, 4, 6, ...)
    side = ((sn % 2) == 0).astype(np.int8)
    return {
        "sn": sn,
        "action": action,
        "group": group,
        "pressure": pressure,
        "side": side,
    }


def _row_features(rally_arrays: dict, N: int, active_groups: list,
                   cap_hits_acc: dict) -> dict:
    """Compute the active features for a target row at next_strikeNumber=N.

    Returns dict {feature_name: value}.

    Updates `cap_hits_acc` in place: counts rows whose streak/total caps fired.
    """
    sn = rally_arrays["sn"]
    group = rally_arrays["group"]
    pressure = rally_arrays["pressure"]
    side = rally_arrays["side"]

    prefix_mask = sn < N
    p_sn = sn[prefix_mask]
    p_group = group[prefix_mask]
    p_pressure = pressure[prefix_mask]
    p_side = side[prefix_mask]
    n_prefix = len(p_sn)

    out = {}
    row_streak_cap = False
    row_total_cap = False

    # ─── Group 1 — action-group lags ──────────────────────────────────────────
    if 1 in active_groups:
        out["v17m_prev1_action_group"] = int(p_group[-1]) if n_prefix >= 1 else 0
        out["v17m_prev2_action_group"] = int(p_group[-2]) if n_prefix >= 2 else 0
        out["v17m_prev1_is_attack"] = int(p_group[-1] == GROUP_ATTACK) if n_prefix >= 1 else 0
        out["v17m_prev1_is_defense"] = int(p_group[-1] == GROUP_DEFENSE) if n_prefix >= 1 else 0

    # ─── Group 2 — recent-window ratios ──────────────────────────────────────
    if 2 in active_groups:
        for k in (3, 5):
            window = p_group[-k:] if n_prefix >= 1 else np.array([], dtype=np.int8)
            press_window = p_pressure[-k:] if n_prefix >= 1 else \
                           np.array([], dtype=np.float32)
            n_w = max(len(window), 1)
            n_atk = int((window == GROUP_ATTACK).sum())
            n_ctl = int((window == GROUP_CONTROL).sum())
            n_def = int((window == GROUP_DEFENSE).sum())
            out[f"v17m_recent{k}_attack_ratio"] = n_atk / n_w
            out[f"v17m_recent{k}_control_ratio"] = n_ctl / n_w
            out[f"v17m_recent{k}_defense_ratio"] = n_def / n_w
            out[f"v17m_recent{k}_attack_count"] = n_atk
            out[f"v17m_recent{k}_initiative_score"] = (n_atk - n_def) / n_w
            out[f"v17m_recent{k}_pressure_score"] = (
                float(press_window.mean()) if len(press_window) > 0 else 0.0)

    # ─── Group 3 — streaks & transitions ─────────────────────────────────────
    if 3 in active_groups:
        # Streaks ending at last shot (cap CAP_STREAK)
        def _streak_at_end(target_g: int) -> int:
            nonlocal row_streak_cap
            if n_prefix == 0 or int(p_group[-1]) != target_g:
                return 0
            sl = 0
            for g in p_group[::-1]:
                if int(g) == target_g:
                    sl += 1
                    if sl >= CAP_STREAK:
                        row_streak_cap = True
                        return CAP_STREAK
                else:
                    break
            return sl

        out["v17m_attack_streak_len"] = _streak_at_end(GROUP_ATTACK)
        out["v17m_defense_streak_len"] = _streak_at_end(GROUP_DEFENSE)
        out["v17m_control_streak_len"] = _streak_at_end(GROUP_CONTROL)

        # Totals (capped CAP_TOTAL)
        n_atk_total = int((p_group == GROUP_ATTACK).sum())
        n_def_total = int((p_group == GROUP_DEFENSE).sum())
        n_ctl_total = int((p_group == GROUP_CONTROL).sum())
        if max(n_atk_total, n_def_total, n_ctl_total) >= CAP_TOTAL:
            row_total_cap = True
        out["v17m_n_attacks_total"] = min(n_atk_total, CAP_TOTAL)
        out["v17m_n_defenses_total"] = min(n_def_total, CAP_TOTAL)
        out["v17m_n_controls_total"] = min(n_ctl_total, CAP_TOTAL)

        # Transitions (vectorized)
        if n_prefix >= 2:
            g_prev = p_group[:-1]
            g_cur = p_group[1:]
            n_atd = int(((g_prev == GROUP_ATTACK) & (g_cur == GROUP_DEFENSE)).sum())
            n_dta = int(((g_prev == GROUP_DEFENSE) & (g_cur == GROUP_ATTACK)).sum())
            n_cta = int(((g_prev == GROUP_CONTROL) & (g_cur == GROUP_ATTACK)).sum())
            n_chg = int((g_prev != g_cur).sum())
        else:
            n_atd = n_dta = n_cta = n_chg = 0
        out["v17m_transitions_atk_to_def"] = min(n_atd, CAP_TOTAL)
        out["v17m_transitions_def_to_atk"] = min(n_dta, CAP_TOTAL)
        out["v17m_transitions_ctl_to_atk"] = min(n_cta, CAP_TOTAL)
        out["v17m_n_action_group_changes"] = min(n_chg, CAP_TOTAL)

    # ─── Group 4 — per-side initiative ───────────────────────────────────────
    if 4 in active_groups:
        srv_mask = (p_side == 0)   # server side = odd SN = side 0
        rcv_mask = (p_side == 1)   # returner side = even SN = side 1
        n_srv = int(srv_mask.sum())
        n_rcv = int(rcv_mask.sum())
        srv_atk = int((p_group[srv_mask] == GROUP_ATTACK).sum())
        rcv_atk = int((p_group[rcv_mask] == GROUP_ATTACK).sum())
        srv_avg = float(p_pressure[srv_mask].mean()) if n_srv > 0 else 0.0
        rcv_avg = float(p_pressure[rcv_mask].mean()) if n_rcv > 0 else 0.0

        out["v17m_server_side_attack_count"] = srv_atk
        out["v17m_returner_side_attack_count"] = rcv_atk
        out["v17m_server_side_attack_ratio"] = srv_atk / max(n_srv, 1)
        out["v17m_returner_side_attack_ratio"] = rcv_atk / max(n_rcv, 1)
        out["v17m_server_side_avg_pressure"] = srv_avg
        out["v17m_returner_side_avg_pressure"] = rcv_avg
        out["v17m_pressure_imbalance"] = srv_avg - rcv_avg
        out["v17m_attack_imbalance"] = srv_atk - rcv_atk

        # Target-hitter aggression (own-side most-recent)
        target_is_server = (N % 2 == 1)
        own_mask = srv_mask if target_is_server else rcv_mask
        own_groups = p_group[own_mask]
        if len(own_groups) > 0:
            out["v17m_target_hitter_recent_was_attacking"] = int(own_groups[-1] == GROUP_ATTACK)
            out["v17m_target_hitter_no_prior_own_shot"] = 0
        else:
            out["v17m_target_hitter_recent_was_attacking"] = 0
            out["v17m_target_hitter_no_prior_own_shot"] = 1

    # ─── Group 5 — pressure derivatives ──────────────────────────────────────
    if 5 in active_groups:
        out["v17m_prev1_pressure"] = float(p_pressure[-1]) if n_prefix >= 1 else 0.0
        out["v17m_prev2_pressure"] = float(p_pressure[-2]) if n_prefix >= 2 else 0.0
        out["v17m_prev3_pressure"] = float(p_pressure[-3]) if n_prefix >= 3 else 0.0
        if n_prefix >= 2:
            out["v17m_pressure_delta_1_2"] = float(p_pressure[-1] - p_pressure[-2])
        else:
            out["v17m_pressure_delta_1_2"] = 0.0
        # target_hitter_under_pressure = pressure of opponent's most recent own shot
        target_is_server = (N % 2 == 1)
        opp_side_value = 1 if target_is_server else 0  # opponent of server is returner
        opp_mask = (p_side == opp_side_value)
        opp_pressures = p_pressure[opp_mask]
        if len(opp_pressures) > 0:
            out["v17m_target_hitter_under_pressure"] = float(opp_pressures[-1])
        else:
            out["v17m_target_hitter_under_pressure"] = 0.0

    if row_streak_cap:
        cap_hits_acc["streak_rows"] += 1
    if row_total_cap:
        cap_hits_acc["total_rows"] += 1

    return out


def build_features_v17_momentum(df: pd.DataFrame, is_train: bool,
                                 global_stats_v9: dict,
                                 raw_df: pd.DataFrame = None) -> pd.DataFrame:
    """v9 + recv_hand_est + 26/41 momentum features (R-015)."""
    feat_df = build_features_v9_recvhand(df, is_train=is_train,
                                          global_stats_v9=global_stats_v9,
                                          raw_df=raw_df)
    if raw_df is None:
        raw_df = df

    active_groups, group_label = get_active_groups()
    active_features = get_active_features()
    label = "train" if is_train else "test"
    print(f"  [v17_momentum] {label}: building groups={active_groups} "
          f"({group_label}) → {len(active_features)} features  "
          f"(env MOMENTUM_GROUPS_ACTIVE={os.environ.get('MOMENTUM_GROUPS_ACTIVE', 'core')})")

    # ─── Per-rally array precomputation ──────────────────────────────────────
    rally_arrays_cache: dict = {}
    needed_cols = [c for c in SOURCE_COLS if c != "rally_uid"]
    for c in needed_cols:
        assert c in raw_df.columns, (
            f"VIOLATION: SOURCE_COLS requires '{c}' but raw_df has columns "
            f"{list(raw_df.columns)[:20]}…")
    for rid, grp in raw_df.groupby("rally_uid", sort=False):
        rally_arrays_cache[rid] = _per_rally_arrays(grp)

    # ─── Per-row computation ─────────────────────────────────────────────────
    n = len(feat_df)
    rally_uids = feat_df["rally_uid"].to_numpy()
    nsns = feat_df["next_strikeNumber"].to_numpy(dtype=np.int32)

    out_data = {f: np.zeros(n, dtype=np.float32) for f in active_features}
    cap_hits = {"streak_rows": 0, "total_rows": 0}
    max_src_violations = 0

    for i in range(n):
        rid = rally_uids[i]
        N = int(nsns[i])
        rally_arrays = rally_arrays_cache.get(rid)
        if rally_arrays is None:
            continue  # default zeros — should never fire if raw_df covers feat_df rallies
        prefix_mask = rally_arrays["sn"] < N
        if prefix_mask.any():
            if int(rally_arrays["sn"][prefix_mask].max()) >= N:
                max_src_violations += 1
        feats = _row_features(rally_arrays, N, active_groups, cap_hits)
        for k, v in feats.items():
            out_data[k][i] = v

    # Append columns
    for f in active_features:
        feat_df[f] = out_data[f]

    # ─── Codex P3.5 — assertions ─────────────────────────────────────────────
    # 8.B-equivalent: max source SN < target N, per row.
    assert max_src_violations == 0, (
        f"VIOLATION (8.B): {max_src_violations} rows used source "
        "strikeNumber >= next_strikeNumber. Prefix-only invariant violated.")

    # 8.D-equivalent: emitted feature names contain no forbidden identifiers.
    forbidden_in_names = []
    for f in active_features:
        f_low = f.lower()
        for forbidden in FORBIDDEN_SOURCE:
            if forbidden.lower() in f_low:
                forbidden_in_names.append((f, forbidden))
    assert not forbidden_in_names, (
        f"VIOLATION (8.D): feature names reference forbidden: {forbidden_in_names}")

    # No-NaN-or-inf check.
    for f in active_features:
        arr = feat_df[f].to_numpy()
        assert np.isfinite(arr).all(), (
            f"VIOLATION (no NaN/inf): {f} has "
            f"{int((~np.isfinite(arr)).sum())} non-finite cells")

    # Pressure-scalar bound check (only if Group 5 active).
    if 5 in active_groups:
        for f in ("v17m_prev1_pressure", "v17m_prev2_pressure",
                   "v17m_prev3_pressure", "v17m_target_hitter_under_pressure"):
            arr = feat_df[f].to_numpy()
            assert (arr >= 0.0).all() and (arr <= 1.5).all(), (
                f"VIOLATION (pressure bound): {f} out of [0.0, 1.5]: "
                f"min={arr.min()}, max={arr.max()}")

    # ─── Diagnostic logging ──────────────────────────────────────────────────
    pct_streak = (cap_hits["streak_rows"] / max(n, 1)) * 100
    pct_total = (cap_hits["total_rows"] / max(n, 1)) * 100
    print(f"  [v17_momentum] {label} cap-hit rates: streak={pct_streak:.2f}%, "
          f"total={pct_total:.2f}%  (caps: streak={CAP_STREAK}, total={CAP_TOTAL})")

    print(f"  [v17_momentum] {label} active features = {len(active_features)} "
          f"({group_label}); first 6 stats:")
    for f in active_features[:6]:
        arr = feat_df[f].to_numpy()
        print(f"    {f:42s} mean={arr.mean():7.3f}  std={arr.std():6.3f}  "
              f"min={arr.min():6.2f}  max={arr.max():6.2f}")

    return feat_df
