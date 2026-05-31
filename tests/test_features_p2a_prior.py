import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features_p2a_prior import (  # noqa: E402
    N_ACTION_CLASSES,
    P2A_PRIOR_COLUMNS,
    add_p2a_prior_features,
    build_p2a_prior_tables,
)


def test_p2a_prior_build_and_append(tmp_path):
    flat = pd.DataFrame(
        [
            {
                "version": "v1",
                "video_id": "a.mp4",
                "p2a_rally_id": 1,
                "p2a_strikeNumber": 1,
                "start_sec": 0.0,
                "handId": 1,
                "mapped_actionId": 15,
            },
            {
                "version": "v1",
                "video_id": "a.mp4",
                "p2a_rally_id": 1,
                "p2a_strikeNumber": 2,
                "start_sec": 1.0,
                "handId": 2,
                "mapped_actionId": 4,
            },
            {
                "version": "v1",
                "video_id": "a.mp4",
                "p2a_rally_id": 1,
                "p2a_strikeNumber": 3,
                "start_sec": 2.0,
                "handId": 1,
                "mapped_actionId": 1,
            },
            {
                "version": "v1",
                "video_id": "b.mp4",
                "p2a_rally_id": 1,
                "p2a_strikeNumber": 1,
                "start_sec": 0.0,
                "handId": 1,
                "mapped_actionId": -1,
            },
        ]
    )
    p = tmp_path / "p2a_flat.csv"
    flat.to_csv(p, index=False)

    tables = build_p2a_prior_tables(p, alpha=0.1)
    feat = pd.DataFrame(
        [
            {
                "rally_uid": 1,
                "next_strikeNumber": 2,
                "serve_actionId": 15,
                "lag1_actionId": 15,
                "lag1_handId": 1,
            },
            {
                "rally_uid": 2,
                "next_strikeNumber": 3,
                "serve_actionId": 15,
                "lag1_actionId": 4,
                "lag1_handId": 2,
            },
        ]
    )
    out = add_p2a_prior_features(feat, tables)

    for col in P2A_PRIOR_COLUMNS:
        assert col in out.columns

    prob_cols = [f"p2a_prior_action_p{i}" for i in range(N_ACTION_CLASSES)]
    np.testing.assert_allclose(out[prob_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-6)

    # Non-serve target shots should not put mass on serve classes after the
    # AICUP legality constraint is applied to the prior vector.
    assert (out[[f"p2a_prior_action_p{i}" for i in (15, 16, 17, 18)]].sum(axis=1) == 0.0).all()
    assert out.loc[0, "p2a_prior_top_action"] == 4
    assert out.loc[1, "p2a_prior_top_action"] == 1

