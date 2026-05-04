"""Feature selection policy shared across training pipelines.

The official competition notice on 2026-04-17 explicitly recommended removing
``serverGetPoint`` features to improve generalization after the public-test
leakage issue. We interpret this conservatively and drop any feature that is
derived from rally outcome / server win-rate statistics.
"""


SERVER_TARGET_FEATURE_SUBSTRINGS = (
    "serverGetPoint",
    "te_game_sgp",
    "te_sd_bin_sgp",
    "player_score_sit_wr",
    "serve_type_wr",
    "rally_len_wr",
    "sgp_pred",
    "matchup_winrate",
    "score_diff_hist_winrate",
    "hitter_win_rate",
    "receiver_win_rate",
    "hitter_wr",
    "_x_hwr",
    "wr_diff",
    "wr_product",
)


def is_server_target_feature(feature_name: str) -> bool:
    """Return True when a feature is derived from serverGetPoint targets."""
    return any(token in feature_name for token in SERVER_TARGET_FEATURE_SUBSTRINGS)


def filter_server_target_features(feature_names):
    """Drop features derived from serverGetPoint / rally outcome statistics."""
    return [name for name in feature_names if not is_server_target_feature(name)]
