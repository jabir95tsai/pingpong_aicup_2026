"""
Feature engineering for table tennis sequential data.

Task structure (confirmed from data analysis):
- train.csv: full rally sequences with labels. We train on predicting shot i using shots 1..i-1.
- test.csv: partial rally sequences (the "history"). We predict the NEXT shot after all test shots.
- Submission: 1 row per unique rally_uid in test (1236 rows).

Key data facts:
- actionId: 19 classes (0-18), 0 is valid (serve-related)
- pointId: 10 classes (0-9), 0 = out-of-bounds / no landing point
- serverGetPoint: binary (0/1), rally-level label (same value for all shots in a rally)
- strikeId: values {1, 2, 4} (not consecutive, treat as categorical)
- positionId: values {0,1,2,3}, 0 is the dominant class (72% of data), valid state
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns from history shots to use as sequential features
# Excludes serverGetPoint (rally-level label, not shot-level feature)
SEQ_COLS = ["actionId", "pointId", "handId", "strengthId", "spinId", "positionId", "strikeId"]

# Columns where numeric aggregations (mean, std) have no meaning because
# the IDs are pure nominal categories. strengthId + positionId are arguably
# ordinal so we still aggregate them numerically.
NOMINAL_SEQ_COLS = {"actionId", "pointId", "handId", "spinId", "strikeId"}


TARGET_COLS = ["actionId", "pointId", "serverGetPoint"]

# Columns that are NOMINAL categories (not ordinal/numeric). AutoGluon must
# treat them as categorical so it doesn't try to find numeric thresholds on
# IDs that have no meaningful ordering. Built dynamically from SEQ_COLS plus
# the raw context columns that are also nominal.
CATEGORICAL_FEATURE_COLS: tuple[str, ...] = (
    "sex",
    "numberGame",
    "gamePlayerId",
    "gamePlayerOtherId",
    "is_deuce",
    "match_point_self",
    "match_point_other",
    "is_serve_side",
    "rally_phase",
    "last_action_point_combo",
    "prev2_action_point_combo",
    *(f"last_{c}" for c in SEQ_COLS),
    *(f"prev2_{c}" for c in SEQ_COLS),
    *(f"hist_mode_{c}" for c in SEQ_COLS),
)


def _extract_features(context_shot: pd.Series, history: pd.DataFrame, shot_idx: int) -> dict:
    """
    Build features for predicting the next shot.

    Args:
        context_shot: The most recent known shot (provides score, player, etc.)
        history: All shots before the one being predicted (used for sequential features)
        shot_idx: How many shots have been seen (= strikeNumber of next shot - 1)
    """
    row: dict = {}

    # --- Context: match state at prediction time ---
    row["sex"] = context_shot["sex"]
    row["numberGame"] = context_shot["numberGame"]
    row["scoreSelf"] = context_shot["scoreSelf"]
    row["scoreOther"] = context_shot["scoreOther"]
    row["score_diff"] = int(context_shot["scoreSelf"]) - int(context_shot["scoreOther"])
    next_sn = shot_idx + 1  # the shot number we're predicting
    row["next_strikeNumber"] = next_sn
    row["gamePlayerId"] = context_shot["gamePlayerId"]
    row["gamePlayerOtherId"] = context_shot["gamePlayerOtherId"]

    # --- Serve-side & rally-phase ---
    # Odd strikeNumbers = server's turn, even = receiver's.
    # Action distributions are radically different between the two.
    row["is_serve_side"] = int(next_sn % 2 == 1)
    # Rally phases have distinct tactical patterns:
    # 1=serve, 2=return, 3=third-ball attack, 4=fourth-ball, 5+=open rally
    row["rally_phase"] = min(next_sn, 5)

    # --- Score pressure flags ---
    s_self = int(context_shot["scoreSelf"])
    s_other = int(context_shot["scoreOther"])
    row["total_points"] = s_self + s_other
    row["is_deuce"] = int(s_self >= 10 and s_other >= 10)
    row["match_point_self"] = int(s_self >= 10 and s_self - s_other >= 0)
    row["match_point_other"] = int(s_other >= 10 and s_other - s_self >= 0)

    # --- Last & second-to-last shot features ---
    for col in SEQ_COLS:
        if len(history) > 0:
            row[f"last_{col}"] = history.iloc[-1][col]
        else:
            row[f"last_{col}"] = -1

        if len(history) > 1:
            row[f"prev2_{col}"] = history.iloc[-2][col]
        else:
            row[f"prev2_{col}"] = -1

    # --- Action-Point interaction combos ---
    # The combination of action type + landing zone is highly predictive of next shot.
    if len(history) > 0:
        last_a = int(history.iloc[-1]["actionId"])
        last_p = int(history.iloc[-1]["pointId"])
        row["last_action_point_combo"] = last_a * 10 + last_p
    else:
        row["last_action_point_combo"] = -1
    if len(history) > 1:
        prev2_a = int(history.iloc[-2]["actionId"])
        prev2_p = int(history.iloc[-2]["pointId"])
        row["prev2_action_point_combo"] = prev2_a * 10 + prev2_p
    else:
        row["prev2_action_point_combo"] = -1

    # --- History shot count ---
    row["hist_shot_count"] = len(history)

    # --- Aggregate features over full history ---
    # For nominal categorical IDs (actionId, pointId, handId, spinId, strikeId)
    # we only keep hist_mode and hist_nunique — these have real meaning.
    # hist_mean/std/last3_mean of a nominal category ID is nonsense and was
    # adding noise for the model to split on.
    for col in SEQ_COLS:
        is_nominal = col in NOMINAL_SEQ_COLS
        if len(history) > 0:
            vals = history[col].values
            row[f"hist_mode_{col}"] = int(pd.Series(vals).mode().iloc[0])
            row[f"hist_nunique_{col}"] = int(pd.Series(vals).nunique())
            if not is_nominal:
                row[f"hist_mean_{col}"] = float(vals.mean())
                row[f"hist_std_{col}"] = float(vals.std()) if len(vals) > 1 else 0.0
                row[f"hist_last3_mean_{col}"] = float(vals[-3:].mean())
        else:
            row[f"hist_mode_{col}"] = -1
            row[f"hist_nunique_{col}"] = -1
            if not is_nominal:
                row[f"hist_mean_{col}"] = -1.0
                row[f"hist_std_{col}"] = -1.0
                row[f"hist_last3_mean_{col}"] = -1.0

    # --- Per-class frequency features for actionId and pointId ---
    # Give the model direct access to how often each class appeared in history.
    # These are critical for macro F1 on multi-class targets with rare classes.
    _N_ACTIONS = 19
    _N_POINTS = 10
    if len(history) > 0:
        n_hist = len(history)
        action_vals = history["actionId"].values.astype(int)
        point_vals = history["pointId"].values.astype(int)
        action_counts = np.bincount(action_vals, minlength=_N_ACTIONS)
        point_counts = np.bincount(point_vals, minlength=_N_POINTS)
        for c in range(_N_ACTIONS):
            row[f"hist_action_freq_{c}"] = float(action_counts[c]) / n_hist
        for c in range(_N_POINTS):
            row[f"hist_point_freq_{c}"] = float(point_counts[c]) / n_hist
        # Entropy of action/point distributions (diversity of shot selection)
        row["hist_action_entropy"] = float(_entropy(action_counts))
        row["hist_point_entropy"] = float(_entropy(point_counts))
        # Dominance: frequency of the most common class
        row["hist_action_dominance"] = float(action_counts.max()) / n_hist
        row["hist_point_dominance"] = float(point_counts.max()) / n_hist
    else:
        for c in range(_N_ACTIONS):
            row[f"hist_action_freq_{c}"] = 0.0
        for c in range(_N_POINTS):
            row[f"hist_point_freq_{c}"] = 0.0
        row["hist_action_entropy"] = 0.0
        row["hist_point_entropy"] = 0.0
        row["hist_action_dominance"] = 0.0
        row["hist_point_dominance"] = 0.0

    # --- Streak features ---
    # How many consecutive identical values at tail of history (repetition signal)
    if len(history) > 0:
        row["streak_action"] = _count_consecutive_tail(
            history["actionId"].values, int(history.iloc[-1]["actionId"]))
        row["streak_point"] = _count_consecutive_tail(
            history["pointId"].values, int(history.iloc[-1]["pointId"]))
    else:
        row["streak_action"] = 0
        row["streak_point"] = 0

    # --- Player alternation pattern ---
    if len(history) > 0:
        player_seq = history["gamePlayerId"].values
        row["consecutive_same_player"] = _count_consecutive_tail(player_seq, player_seq[-1])
    else:
        row["consecutive_same_player"] = 0

    # --- Score / rally context features ---
    # These numeric features are cheap and don't introduce sparse splits.
    row["score_lead_abs"] = abs(int(context_shot["scoreSelf"]) - int(context_shot["scoreOther"]))
    row["points_to_win_self"] = max(0, 11 - s_self)
    row["points_to_win_other"] = max(0, 11 - s_other)

    return row


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a count vector (base-e). Zeros are ignored."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs)))


def _count_consecutive_tail(arr: np.ndarray, val: int) -> int:
    count = 0
    for x in reversed(arr):
        if x == val:
            count += 1
        else:
            break
    return count


REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "rally_uid",
    "strikeNumber",
    "sex",
    "numberGame",
    "scoreSelf",
    "scoreOther",
    "gamePlayerId",
    "gamePlayerOtherId",
    "actionId",
    "pointId",
    "handId",
    "strengthId",
    "spinId",
    "positionId",
    "strikeId",
    "serverGetPoint",
)


def validate_raw_dataframe(df: pd.DataFrame, *, kind: str) -> None:
    """Validate that a raw shot-level DataFrame has the columns we need.

    Args:
        df: parsed CSV
        kind: "train" or "test" — test data is allowed to omit target columns.

    Raises:
        ValueError: with a human-readable message if validation fails.
    """
    if df is None or len(df) == 0:
        raise ValueError(f"{kind} dataframe is empty")

    optional_for_test = {"actionId", "pointId", "serverGetPoint"}
    required = set(REQUIRED_RAW_COLUMNS)
    if kind == "test":
        required = required - optional_for_test

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{kind} dataframe is missing required columns: {missing}. "
            f"Got columns: {sorted(df.columns)}"
        )
    if df["rally_uid"].isna().any():
        raise ValueError(f"{kind} dataframe has NaN in rally_uid")
    if df["strikeNumber"].isna().any():
        raise ValueError(f"{kind} dataframe has NaN in strikeNumber")


def _cast_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Cast nominal-ID columns to pandas `category` dtype.

    AutoGluon's tree models will then use proper categorical splits
    instead of looking for numeric thresholds on category IDs that
    have no ordinal meaning.
    """
    for col in CATEGORICAL_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def build_train_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build training features. For each shot i in each rally, predict shot i
    using shots 0..i-1 as history. This yields maximum training samples.

    Skips the first shot of each rally (no history to learn from).
    Returns a DataFrame with feature columns + rally_uid + TARGET_COLS.
    """
    df = df.copy().sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)

    feature_rows = []
    for rally_uid, group in df.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber").reset_index(drop=True)

        for i in range(1, len(group)):  # start at 1: need at least 1 history shot
            history = group.iloc[:i]
            current = group.iloc[i]  # shot to predict

            row = _extract_features(
                context_shot=history.iloc[-1],
                history=history,
                shot_idx=i,
            )
            row["rally_uid"] = rally_uid
            row["_match_id"] = current.get("match", -1)  # NEW: for match-CV grouping
            row["actionId"] = current["actionId"]
            row["pointId"] = current["pointId"]
            row["serverGetPoint"] = current["serverGetPoint"]
            feature_rows.append(row)

    return _cast_categorical(pd.DataFrame(feature_rows))


def build_test_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build test features. For each test rally, use ALL test shots as history
    and predict the NEXT shot (which is not in the CSV).

    Returns 1 row per unique rally_uid (= 1236 rows for the given test set).
    """
    df = df.copy().sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)

    feature_rows = []
    for rally_uid, group in df.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber").reset_index(drop=True)

        history = group  # all test shots are history
        last = group.iloc[-1]  # context comes from the last known shot

        row = _extract_features(
            context_shot=last,
            history=history,
            shot_idx=len(group),  # next strikeNumber = len(group) + 1 - 1 (0-indexed)
        )
        row["rally_uid"] = rally_uid
        feature_rows.append(row)

    return _cast_categorical(pd.DataFrame(feature_rows))


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    # _match_id is for GroupKFold splitting only, NOT a model input
    exclude = {"rally_uid", "_match_id"} | set(TARGET_COLS)
    return [c for c in df.columns if c not in exclude]


# --- Player profile (target encoding) ---
# Cross-rally player statistics provide information the per-rally history
# features cannot: how a player behaves *in general* across all matches.
#
# Tested rollback: expanding to all 19 actions / 10 points with serve-side
# stratification (~179 new features vs the current 31) makes CV slightly
# worse and adds noise faster than signal. Stick with these top-k tuples.

_PROFILE_ACTION_TOPK = (0, 1, 2, 5, 6, 10, 13, 15)  # most common action classes
_PROFILE_POINT_TOPK = (0, 4, 5, 8, 9)                # most common point classes


def compute_player_profiles(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-player aggregate statistics from raw shot-level data.

    Returns a DataFrame indexed by player ID with columns like
    ``player_win_rate``, ``player_action_0_rate``, etc.

    Only uses shot-level data (not feature matrix), so it can be computed
    from any subset of raw data without feature-engineering dependency.
    """
    rows = []
    for pid, grp in raw_df.groupby("gamePlayerId"):
        n_shots = len(grp)
        # Win rate: serverGetPoint when this player is the server (odd strikeNumber)
        rallies = grp.drop_duplicates("rally_uid")
        n_rallies = len(rallies)
        win_rate = float(rallies["serverGetPoint"].mean()) if n_rallies > 0 else 0.5

        # Action distribution (this player's shots only)
        action_counts = np.bincount(grp["actionId"].values.astype(int), minlength=19)
        point_counts = np.bincount(grp["pointId"].values.astype(int), minlength=10)

        row = {"player_id": pid, "player_n_rallies": n_rallies, "player_win_rate": win_rate}
        for c in _PROFILE_ACTION_TOPK:
            row[f"player_action_{c}_rate"] = float(action_counts[c]) / max(n_shots, 1)
        for c in _PROFILE_POINT_TOPK:
            row[f"player_point_{c}_rate"] = float(point_counts[c]) / max(n_shots, 1)

        rows.append(row)

    return pd.DataFrame(rows).set_index("player_id")


def merge_player_profiles(
    X: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Merge player profile features into feature matrix X.

    Adds columns for both the current player (``gamePlayerId``) and the
    opponent (``gamePlayerOtherId``).  Unknown players get 0.5 win rate
    and 0 for all action/point rates.
    """
    profile_cols = list(profiles.columns)

    # Defaults for unknown players: 0.5 for win-rate, 0 for distribution rates
    defaults = pd.Series(
        {c: (0.5 if "win_rate" in c else 0.0) for c in profile_cols},
    )

    player_ids = X["gamePlayerId"].astype(int).values
    opp_ids = X["gamePlayerOtherId"].astype(int).values

    # Reindex profile rows to the per-row player IDs in one shot, then
    # concat once — avoids the fragmented-DataFrame penalty from inserting
    # ~120 columns individually.
    p_df = profiles.reindex(player_ids).fillna(defaults).reset_index(drop=True)
    opp_df = profiles.reindex(opp_ids).fillna(defaults).reset_index(drop=True)
    p_df.columns = [f"p_{c}" for c in profile_cols]
    opp_df.columns = [f"opp_{c}" for c in profile_cols]

    merged = pd.concat(
        [X.reset_index(drop=True), p_df, opp_df], axis=1, copy=False,
    )
    merged["win_rate_diff"] = merged["p_player_win_rate"] - merged["opp_player_win_rate"]
    return merged


PLAYER_PROFILE_COLS: list[str] = []  # populated at import time below

def _init_profile_cols() -> list[str]:
    """Return the list of column names that merge_player_profiles adds."""
    base = ["player_n_rallies", "player_win_rate"]
    base += [f"player_action_{c}_rate" for c in _PROFILE_ACTION_TOPK]
    base += [f"player_point_{c}_rate" for c in _PROFILE_POINT_TOPK]
    cols = [f"p_{c}" for c in base] + [f"opp_{c}" for c in base]
    cols.append("win_rate_diff")
    return cols

PLAYER_PROFILE_COLS = _init_profile_cols()


# --- Transition matrix features ---
# Empirical conditional distributions of next shot given the last shot context.
# These give the model a strong "prior" that captures global patterns across
# all rallies, complementing the within-rally history features which are noisy
# for short rallies (median test_new rally has only 2 history shots).

_N_ACTIONS = 19
_N_POINTS = 10


def compute_transition_tables(raw_df: pd.DataFrame) -> dict:
    """Compute transition probability tables from raw shot-level data.

    Returns a dict with:
      - ``action``: {(last_action, next_is_serve_side): ndarray(19,)}
      - ``point``:  {(last_action, last_point): ndarray(10,)}
      - ``action_global``: ndarray(19,) — marginal action distribution
      - ``point_global``:  ndarray(10,) — marginal point distribution
    """
    df = raw_df.sort_values(["rally_uid", "strikeNumber"]).copy()

    # Build next-shot columns within each rally
    df["next_actionId"] = df.groupby("rally_uid")["actionId"].shift(-1)
    df["next_pointId"] = df.groupby("rally_uid")["pointId"].shift(-1)
    df["next_strikeNumber"] = df.groupby("rally_uid")["strikeNumber"].shift(-1)

    trans = df.dropna(subset=["next_actionId", "next_pointId"]).copy()
    trans["next_actionId"] = trans["next_actionId"].astype(int)
    trans["next_pointId"] = trans["next_pointId"].astype(int)
    trans["next_is_serve_side"] = (trans["next_strikeNumber"].astype(int) % 2 == 1).astype(int)

    # Global marginals (fallback for unseen contexts)
    action_global = np.bincount(trans["next_actionId"].values, minlength=_N_ACTIONS).astype(float)
    action_global /= max(action_global.sum(), 1)
    point_global = np.bincount(trans["next_pointId"].values, minlength=_N_POINTS).astype(float)
    point_global /= max(point_global.sum(), 1)

    # Action transition: P(next_action | last_action, next_is_serve_side)
    action_table: dict[tuple[int, int], np.ndarray] = {}
    for (last_a, serve_side), grp in trans.groupby(["actionId", "next_is_serve_side"]):
        counts = np.bincount(grp["next_actionId"].values, minlength=_N_ACTIONS).astype(float)
        total = counts.sum()
        if total > 0:
            action_table[(int(last_a), int(serve_side))] = counts / total

    # Point transition: P(next_point | last_action, last_point)
    point_table: dict[tuple[int, int], np.ndarray] = {}
    for (last_a, last_p), grp in trans.groupby(["actionId", "pointId"]):
        counts = np.bincount(grp["next_pointId"].values, minlength=_N_POINTS).astype(float)
        total = counts.sum()
        if total > 0:
            point_table[(int(last_a), int(last_p))] = counts / total

    return {
        "action": action_table,
        "point": point_table,
        "action_global": action_global,
        "point_global": point_global,
    }


def merge_transition_features(
    X: pd.DataFrame,
    transition_tables: dict,
) -> pd.DataFrame:
    """Merge transition probability features into feature matrix X.

    For each row, looks up the empirical distribution of next_action and
    next_point given the row's (last_actionId, is_serve_side) and
    (last_actionId, last_pointId) context. Adds 29 prior-probability
    columns plus 4 summary statistics.
    """
    action_table = transition_tables["action"]
    point_table = transition_tables["point"]
    action_global = transition_tables["action_global"]
    point_global = transition_tables["point_global"]

    last_a = X["last_actionId"].astype(int).values
    last_p = X["last_pointId"].astype(int).values
    serve_side = X["is_serve_side"].astype(int).values

    # Vectorized lookup via list comprehension (dict lookups are O(1))
    action_keys = list(zip(last_a, serve_side))
    point_keys = list(zip(last_a, last_p))

    action_priors = np.array([
        action_table.get(k, action_global) for k in action_keys
    ])
    point_priors = np.array([
        point_table.get(k, point_global) for k in point_keys
    ])

    # Build columns
    cols = {}
    for c in range(_N_ACTIONS):
        cols[f"trans_action_prior_{c}"] = action_priors[:, c]
    for c in range(_N_POINTS):
        cols[f"trans_point_prior_{c}"] = point_priors[:, c]

    # Summary statistics: entropy and dominance of the prior
    cols["trans_action_entropy"] = np.array([
        float(_entropy((p * 1000).astype(int))) for p in action_priors
    ])
    cols["trans_point_entropy"] = np.array([
        float(_entropy((p * 1000).astype(int))) for p in point_priors
    ])
    cols["trans_action_top1"] = action_priors.max(axis=1)
    cols["trans_point_top1"] = point_priors.max(axis=1)

    prior_df = pd.DataFrame(cols, index=X.index)
    return pd.concat([X, prior_df], axis=1, copy=False)


TRANSITION_FEATURE_COLS: list[str] = (
    [f"trans_action_prior_{c}" for c in range(_N_ACTIONS)]
    + [f"trans_point_prior_{c}" for c in range(_N_POINTS)]
    + ["trans_action_entropy", "trans_point_entropy",
       "trans_action_top1", "trans_point_top1"]
)


def prepare_train_test(
    train_path: str,
    test_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Returns:
        X_train: feature DataFrame
        y_train: target DataFrame (actionId, pointId, serverGetPoint)
        groups:  rally_uid Series for GroupKFold (aligned with X_train)
        X_test:  feature DataFrame (1236 rows, one per test rally)
        test_rally_uids: Series of rally_uids for submission
    """
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    validate_raw_dataframe(train_raw, kind="train")
    validate_raw_dataframe(test_raw, kind="test")

    logger.info("Train raw: %d shots, %d rallies", len(train_raw), train_raw["rally_uid"].nunique())
    logger.info("Test raw:  %d shots, %d rallies", len(test_raw), test_raw["rally_uid"].nunique())
    print(f"Train raw: {len(train_raw)} shots, {train_raw['rally_uid'].nunique()} rallies")
    print(f"Test raw:  {len(test_raw)} shots, {test_raw['rally_uid'].nunique()} rallies")

    train_feat = build_train_features(train_raw)
    test_feat = build_test_features(test_raw)

    logger.info("Train features: %d samples", len(train_feat))
    logger.info("Test features:  %d samples (one per test rally)", len(test_feat))
    print(f"Train features: {len(train_feat)} samples")
    print(f"Test features:  {len(test_feat)} samples (one per test rally)")

    feature_cols = get_feature_cols(train_feat)

    X_train = train_feat[feature_cols].reset_index(drop=True)
    y_train = train_feat[TARGET_COLS].reset_index(drop=True)
    # MATCH-CV (honest): group by match_id so train/val matches disjoint
    # This better simulates Private LB (cluster 2 = unseen matches)
    # Fallback to rally_uid if _match_id missing (older feature build)
    if "_match_id" in train_feat.columns and (train_feat["_match_id"] >= 0).all():
        groups = train_feat["_match_id"].reset_index(drop=True)
        logger.info("Using MATCH-CV grouping (%d unique matches)", groups.nunique())
        print(f"Using MATCH-CV grouping ({groups.nunique()} unique matches)")
    else:
        groups = train_feat["rally_uid"].reset_index(drop=True)
        logger.info("Using RALLY-CV grouping (%d unique rallies)", groups.nunique())
        print(f"Using RALLY-CV grouping ({groups.nunique()} unique rallies) [fallback]")

    X_test = test_feat[feature_cols].reset_index(drop=True)
    test_rally_uids = test_feat["rally_uid"].reset_index(drop=True)

    return X_train, y_train, groups, X_test, test_rally_uids
