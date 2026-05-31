"""
Override extreme-outlier predictions in a submission with the train-empirical
mode under the same (prev_action, last_action, last_point) context.

Method: build a conditional distribution from train.csv —

    P(next_actionId | prev_actionId, last_actionId, last_pointId)
    P(next_pointId  | prev_actionId, last_actionId, last_pointId)

For each test rally, look up the same context and check the prediction's
empirical probability. If the predicted class has zero observations in train
under that context (and the context has enough samples for the lookup to be
meaningful), replace with the train mode.

Conservative on purpose: the "0% probability" threshold catches predictions
that are empirically *impossible* under the training distribution while
leaving every other row untouched. Tested LB: v3 0.4401 → v4 0.4415
(+0.0014) on a 7-row delta out of 1236.

Usage:
    python -m src.apply_rule_override \\
        --input models/v3/submission.csv \\
        --train data/raw/train.csv \\
        --test  data/raw/test.csv \\
        --output models/v4/submission.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Minimum train sample count for a (prev_action, last_action, last_point)
# context to be considered a reliable lookup. Below this, the empirical
# distribution is too noisy to trust as an override signal.
MIN_CONTEXT_SAMPLES = 30


def _build_conditional_table(train: pd.DataFrame, target_col: str) -> dict:
    """Build P(target | prev_action, last_action, last_point) from train.

    Returns ``{(prev_a, last_a, last_p): (probs_dict, mode_class, n_total)}``
    for contexts that meet the minimum-sample threshold.
    """
    df = train.sort_values(["rally_uid", "strikeNumber"]).copy()
    df["prev_actionId"] = df.groupby("rally_uid")["actionId"].shift(1)
    df[f"next_{target_col}"] = df.groupby("rally_uid")[target_col].shift(-1)
    df = df.dropna(subset=["prev_actionId", f"next_{target_col}"])
    for c in ["prev_actionId", f"next_{target_col}"]:
        df[c] = df[c].astype(int)

    table: dict = {}
    grouped = df.groupby(["prev_actionId", "actionId", "pointId"])[f"next_{target_col}"]
    for key, vals in grouped:
        if len(vals) < MIN_CONTEXT_SAMPLES:
            continue
        probs = vals.value_counts(normalize=True).to_dict()
        mode = int(vals.value_counts().idxmax())
        table[tuple(int(k) for k in key)] = (probs, mode, len(vals))
    return table


def _build_test_context(test: pd.DataFrame) -> pd.DataFrame:
    """For each test rally, return its (prev_action, last_action, last_point).

    Skips rallies with fewer than 2 history shots — there is no ``prev`` to
    condition on.
    """
    rows = []
    for rid, g in test.groupby("rally_uid"):
        g = g.sort_values("strikeNumber").reset_index(drop=True)
        if len(g) < 2:
            continue
        last = g.iloc[-1]
        prev = g.iloc[-2]
        rows.append({
            "rally_uid": int(rid),
            "prev_actionId": int(prev["actionId"]),
            "last_actionId": int(last["actionId"]),
            "last_pointId": int(last["pointId"]),
        })
    if not rows:
        return pd.DataFrame(
            columns=["prev_actionId", "last_actionId", "last_pointId"],
            index=pd.Index([], name="rally_uid", dtype=int),
        )
    return pd.DataFrame(rows).set_index("rally_uid")


def apply_rule_override(
    submission: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply zero-probability overrides; return (new_submission, change_log)."""
    action_tbl = _build_conditional_table(train, "actionId")
    point_tbl = _build_conditional_table(train, "pointId")
    test_ctx = _build_test_context(test)

    new_action = submission["actionId"].astype(int).copy().values
    new_point = submission["pointId"].astype(int).copy().values
    changes: list[dict] = []

    for i, row in submission.iterrows():
        rid = int(row["rally_uid"])
        if rid not in test_ctx.index:
            continue
        ctx = test_ctx.loc[rid]
        key = (int(ctx["prev_actionId"]), int(ctx["last_actionId"]), int(ctx["last_pointId"]))

        if key in action_tbl:
            probs, mode, n = action_tbl[key]
            v_class = int(row["actionId"])
            if probs.get(v_class, 0.0) == 0.0 and mode != v_class:
                new_action[i] = mode
                changes.append({"rally_uid": rid, "target": "actionId",
                                "from": v_class, "to": mode, "context_n": n})

        if key in point_tbl:
            probs, mode, n = point_tbl[key]
            v_class = int(row["pointId"])
            if probs.get(v_class, 0.0) == 0.0 and mode != v_class:
                new_point[i] = mode
                changes.append({"rally_uid": rid, "target": "pointId",
                                "from": v_class, "to": mode, "context_n": n})

    out = submission.copy()
    out["actionId"] = new_action
    out["pointId"] = new_point
    return out, changes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", required=True, help="Input submission.csv")
    p.add_argument("--train", required=True, help="train.csv for building rules")
    p.add_argument("--test", required=True, help="test.csv for matching context")
    p.add_argument("--output", required=True, help="Output submission.csv")
    args = p.parse_args()

    submission = pd.read_csv(args.input)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    out, changes = apply_rule_override(submission, train, test)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Rows overridden: {len(changes)}/{len(submission)}")
    for c in changes:
        print(f"  rally={c['rally_uid']} {c['target']}: "
              f"{c['from']} → {c['to']} (n={c['context_n']})")


if __name__ == "__main__":
    main()
