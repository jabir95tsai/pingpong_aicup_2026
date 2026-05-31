"""P2A external action-prior features.

P2A labels only overlap the AICUP target space on hand / serve / action type.
This module therefore uses P2A as an external transition prior for `actionId`
only. It does not create synthetic AICUP training rows and it does not infer
`pointId` or `serverGetPoint`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

N_ACTION_CLASSES = 19
_EPS = 1e-12

P2A_PRIOR_PROB_COLUMNS = [f"p2a_prior_action_p{c}" for c in range(N_ACTION_CLASSES)]
P2A_SN_PRIOR_PROB_COLUMNS = [f"p2a_prior_sn_action_p{c}" for c in range(N_ACTION_CLASSES)]
P2A_PRIOR_SUMMARY_COLUMNS = [
    "p2a_prior_entropy",
    "p2a_prior_confidence",
    "p2a_prior_top_action",
    "p2a_prior_context_count",
    "p2a_prior_source_code",
]
P2A_PRIOR_COLUMNS = (
    P2A_PRIOR_PROB_COLUMNS
    + P2A_SN_PRIOR_PROB_COLUMNS
    + P2A_PRIOR_SUMMARY_COLUMNS
)


def _empty_counts() -> np.ndarray:
    return np.zeros(N_ACTION_CLASSES, dtype=np.float64)


def _sn_bucket(sn: int) -> int:
    """Bucket target strike number for sparse external priors."""
    if sn <= 1:
        return 1
    if sn <= 7:
        return int(sn)
    return 8


def _normalise(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Return smoothed probability vector for actionId 0..18."""
    arr = np.asarray(counts, dtype=np.float64)
    smoothed = arr + float(alpha)
    total = float(smoothed.sum())
    if total <= 0.0:
        return np.full(N_ACTION_CLASSES, 1.0 / N_ACTION_CLASSES, dtype=np.float32)
    return (smoothed / total).astype(np.float32)


def _entropy(prob: np.ndarray) -> float:
    p = np.asarray(prob, dtype=np.float64)
    p = p[p > 0.0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p + _EPS)))


def _apply_aicup_action_constraints(prob: np.ndarray, next_sn: int) -> np.ndarray:
    """Constrain prior mass to AICUP-valid action classes for the target shot."""
    out = np.asarray(prob, dtype=np.float32).copy()
    if next_sn == 1:
        mask = np.zeros(N_ACTION_CLASSES, dtype=np.float32)
        mask[[0, 15, 16, 17, 18]] = 1.0
        out *= mask
    else:
        out[[15, 16, 17, 18]] = 0.0
    total = float(out.sum())
    if total <= 0.0:
        out[:] = 1.0 / N_ACTION_CLASSES
    else:
        out /= total
    return out


def _add_count(table: dict[Any, np.ndarray], key: Any, target_action: int) -> None:
    if not (0 <= int(target_action) < N_ACTION_CLASSES):
        return
    if key not in table:
        table[key] = _empty_counts()
    table[key][int(target_action)] += 1.0


def _table_to_prob_and_count(table: dict[Any, np.ndarray], alpha: float) -> dict[Any, tuple[np.ndarray, int]]:
    out = {}
    for key, counts in table.items():
        out[key] = (_normalise(counts, alpha), int(np.asarray(counts).sum()))
    return out


def build_p2a_prior_tables(p2a_flat_path: str | Path, alpha: float = 1.0) -> dict[str, Any]:
    """Build external transition-prior tables from flattened P2A labels.

    Args:
        p2a_flat_path: CSV created by `python src/aicup_analyzer.py p2a`.
        alpha: additive smoothing mass per action class.

    Returns:
        Pickle-friendly dict consumed by `add_p2a_prior_features`.
    """
    p = Path(p2a_flat_path)
    if not p.exists():
        raise FileNotFoundError(
            f"P2A flat CSV not found: {p}. Run `python src/aicup_analyzer.py p2a` first."
        )
    df = pd.read_csv(p)
    required = {
        "version",
        "video_id",
        "p2a_rally_id",
        "p2a_strikeNumber",
        "handId",
        "mapped_actionId",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"P2A flat CSV missing required columns: {missing}")

    df = df[df["mapped_actionId"].between(0, N_ACTION_CLASSES - 1)].copy()
    df["mapped_actionId"] = df["mapped_actionId"].astype(int)
    df["handId"] = df["handId"].fillna(0).astype(int)
    df["p2a_strikeNumber"] = df["p2a_strikeNumber"].fillna(0).astype(int)

    global_counts = _empty_counts()
    by_prev_action: dict[int, np.ndarray] = {}
    by_prev_action_hand: dict[tuple[int, int], np.ndarray] = {}
    by_receive_after_serve: dict[int, np.ndarray] = {}
    by_sn_bucket: dict[int, np.ndarray] = {}

    for action in df["mapped_actionId"].to_numpy(dtype=np.int32):
        global_counts[int(action)] += 1.0

    for sn, target_action in zip(
        df["p2a_strikeNumber"].to_numpy(dtype=np.int32),
        df["mapped_actionId"].to_numpy(dtype=np.int32),
    ):
        _add_count(by_sn_bucket, _sn_bucket(int(sn)), int(target_action))

    group_cols = ["version", "video_id", "p2a_rally_id"]
    sort_cols = group_cols + ["p2a_strikeNumber", "start_sec"]
    if "start_sec" not in df.columns:
        sort_cols = group_cols + ["p2a_strikeNumber"]

    for _, grp in df.sort_values(sort_cols).groupby(group_cols, sort=False):
        actions = grp["mapped_actionId"].to_numpy(dtype=np.int32)
        hands = grp["handId"].to_numpy(dtype=np.int32)
        sns = grp["p2a_strikeNumber"].to_numpy(dtype=np.int32)
        if len(actions) < 2:
            continue
        for i in range(1, len(actions)):
            prev_action = int(actions[i - 1])
            prev_hand = int(hands[i - 1])
            target_action = int(actions[i])
            target_sn = int(sns[i])
            _add_count(by_prev_action, prev_action, target_action)
            _add_count(by_prev_action_hand, (prev_action, prev_hand), target_action)
            if target_sn == 2:
                _add_count(by_receive_after_serve, prev_action, target_action)

    tables = {
        "alpha": float(alpha),
        "global": (_normalise(global_counts, alpha), int(global_counts.sum())),
        "by_prev_action": _table_to_prob_and_count(by_prev_action, alpha),
        "by_prev_action_hand": _table_to_prob_and_count(by_prev_action_hand, alpha),
        "by_receive_after_serve": _table_to_prob_and_count(by_receive_after_serve, alpha),
        "by_sn_bucket": _table_to_prob_and_count(by_sn_bucket, alpha),
        "source_codes": {
            "global": 0,
            "prev_action": 1,
            "prev_action_hand": 2,
            "receive_after_serve": 3,
        },
        "source_path": str(p),
        "mapped_rows": int(len(df)),
    }
    return tables


def _lookup_context_prior(row: pd.Series, tables: dict[str, Any]) -> tuple[np.ndarray, int, int]:
    """Pick the most specific P2A context prior available for a feature row."""
    source_codes = tables["source_codes"]
    global_prob, global_count = tables["global"]
    next_sn = int(row.get("next_strikeNumber", 0))

    if next_sn == 2:
        serve_action = int(row.get("serve_actionId", row.get("lag1_actionId", -1)))
        hit = tables["by_receive_after_serve"].get(serve_action)
        if hit is not None:
            return hit[0], hit[1], source_codes["receive_after_serve"]

    prev_action = int(row.get("lag1_actionId", -1))
    prev_hand = int(row.get("lag1_handId", -1))
    hit = tables["by_prev_action_hand"].get((prev_action, prev_hand))
    if hit is not None:
        return hit[0], hit[1], source_codes["prev_action_hand"]

    hit = tables["by_prev_action"].get(prev_action)
    if hit is not None:
        return hit[0], hit[1], source_codes["prev_action"]

    return global_prob, global_count, source_codes["global"]


def _lookup_sn_prior(next_sn: int, tables: dict[str, Any]) -> np.ndarray:
    hit = tables["by_sn_bucket"].get(_sn_bucket(int(next_sn)))
    if hit is not None:
        return hit[0]
    return tables["global"][0]


def add_p2a_prior_features(feat_df: pd.DataFrame, tables: dict[str, Any]) -> pd.DataFrame:
    """Append P2A prior feature columns to an existing AICUP feature frame."""
    out = feat_df.copy()
    n = len(out)
    ctx_probs = np.zeros((n, N_ACTION_CLASSES), dtype=np.float32)
    sn_probs = np.zeros((n, N_ACTION_CLASSES), dtype=np.float32)
    entropy = np.zeros(n, dtype=np.float32)
    confidence = np.zeros(n, dtype=np.float32)
    top_action = np.zeros(n, dtype=np.int16)
    context_count = np.zeros(n, dtype=np.float32)
    source_code = np.zeros(n, dtype=np.int8)

    for i, (_, row) in enumerate(out.iterrows()):
        next_sn = int(row.get("next_strikeNumber", 0))
        ctx_prob, ctx_count, src = _lookup_context_prior(row, tables)
        ctx_prob = _apply_aicup_action_constraints(ctx_prob, next_sn)
        sn_prob = _apply_aicup_action_constraints(_lookup_sn_prior(next_sn, tables), next_sn)

        ctx_probs[i] = ctx_prob
        sn_probs[i] = sn_prob
        entropy[i] = _entropy(ctx_prob)
        confidence[i] = float(np.max(ctx_prob))
        top_action[i] = int(np.argmax(ctx_prob))
        context_count[i] = np.log1p(float(ctx_count))
        source_code[i] = int(src)

    for c in range(N_ACTION_CLASSES):
        out[f"p2a_prior_action_p{c}"] = ctx_probs[:, c]
    for c in range(N_ACTION_CLASSES):
        out[f"p2a_prior_sn_action_p{c}"] = sn_probs[:, c]

    out["p2a_prior_entropy"] = entropy
    out["p2a_prior_confidence"] = confidence
    out["p2a_prior_top_action"] = top_action
    out["p2a_prior_context_count"] = context_count
    out["p2a_prior_source_code"] = source_code
    return out

