from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "eda_output"
TABLES = OUT / "tables"
FIGS = OUT / "figures"


ACTION_LABELS = {
    0: "other",
    1: "loop",
    2: "counter_loop",
    3: "smash",
    4: "banana",
    5: "drive",
    6: "push_press",
    7: "flick",
    8: "arch",
    9: "block_chop",
    10: "chop",
    11: "short_push",
    12: "def_chop",
    13: "block",
    14: "lob",
    15: "serve_trad",
    16: "serve_hook",
    17: "serve_reverse",
    18: "serve_squat",
}

POINT_LABELS = {
    0: "off_grid",
    1: "FH_short",
    2: "mid_short",
    3: "BH_short",
    4: "FH_half",
    5: "mid_half",
    6: "BH_half",
    7: "FH_long",
    8: "mid_long",
    9: "BH_long",
}


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def sn_bucket(sn: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(sn)
    return pd.cut(
        s,
        bins=[-np.inf, 2, 4, 8, 12, np.inf],
        labels=["2", "3-4", "5-8", "9-12", "13+"],
        right=True,
    ).astype(str)


def dist_table(series: pd.Series, labels: dict[int, str] | None = None) -> pd.DataFrame:
    counts = series.value_counts(dropna=False).sort_index()
    df = pd.DataFrame({"class": counts.index, "count": counts.values})
    df["pct"] = df["count"] / df["count"].sum()
    if labels:
        df["label"] = df["class"].map(labels).fillna(df["class"].astype(str))
    return df


def categorical_shift(train_s: pd.Series, test_s: pd.Series) -> tuple[float, float]:
    cats = sorted(set(train_s.dropna().astype(str)) | set(test_s.dropna().astype(str)))
    if not cats:
        return 0.0, 0.0
    p = train_s.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(float)
    q = test_s.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(float)
    tv = 0.5 * np.abs(p - q).sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(tv), float(js)


def binary_auc_rank(y: pd.Series, score: pd.Series) -> float:
    y_arr = y.to_numpy(int)
    s = pd.Series(score.to_numpy(float))
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s.rank(method="average").to_numpy(float)
    sum_pos = ranks[y_arr == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, rotate: int = 0) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.ylabel(y)
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_grouped_bar(df: pd.DataFrame, category_col: str, value_cols: list[str], title: str, path: Path) -> None:
    x = np.arange(len(df))
    width = 0.8 / len(value_cols)
    plt.figure(figsize=(10, 4.8))
    for i, col in enumerate(value_cols):
        plt.bar(x + (i - (len(value_cols) - 1) / 2) * width, df[col], width=width, label=col)
    plt.xticks(x, df[category_col].astype(str))
    plt.title(title)
    plt.ylabel("pct")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def transition_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    ordered = df.sort_values(["rally_uid", "strikeNumber"]).copy()
    ordered[f"prev_{col}"] = ordered.groupby("rally_uid")[col].shift(1)
    trans = ordered.dropna(subset=[f"prev_{col}"]).copy()
    trans[f"prev_{col}"] = trans[f"prev_{col}"].astype(int)
    out = (
        trans.groupby([f"prev_{col}", col])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    out["pct"] = out["count"] / out["count"].sum()
    return out


def main() -> None:
    ensure_dirs()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    train = pd.read_csv(DATA / "train.csv")
    test_new = pd.read_csv(DATA / "test_new.csv")
    test_old_path = DATA / "test.csv"
    test_old = pd.read_csv(test_old_path) if test_old_path.exists() else None
    sample = pd.read_csv(DATA / "sample_submission.csv")

    train_targets = train[train["strikeNumber"] >= 2].copy()
    train_rally = (
        train.groupby("rally_uid")
        .agg(
            rows=("strikeNumber", "size"),
            max_sn=("strikeNumber", "max"),
            min_sn=("strikeNumber", "min"),
            match=("match", "first"),
            sex=("sex", "first"),
            serverGetPoint=("serverGetPoint", "first"),
            scoreSelf_first=("scoreSelf", "first"),
            scoreOther_first=("scoreOther", "first"),
        )
        .reset_index()
    )
    test_rally = (
        test_new.groupby("rally_uid")
        .agg(
            rows=("strikeNumber", "size"),
            max_sn=("strikeNumber", "max"),
            min_sn=("strikeNumber", "min"),
            match=("match", "first"),
            sex=("sex", "first"),
            scoreSelf_first=("scoreSelf", "first"),
            scoreOther_first=("scoreOther", "first"),
        )
        .reset_index()
    )
    test_rally["target_next_sn"] = test_rally["max_sn"] + 1

    old_overlap_info = {}
    if test_old is not None:
        old_uids = set(test_old["rally_uid"].unique())
        new_uids = set(test_new["rally_uid"].unique())
        overlap = sorted(old_uids & new_uids)
        shared_cols = [c for c in test_new.columns if c in test_old.columns]
        old_part = test_old[test_old["rally_uid"].isin(overlap)].sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
        new_part = test_new[test_new["rally_uid"].isin(overlap)].sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
        old_overlap_info = {
            "old_test_rows": len(test_old),
            "old_test_rallies": test_old["rally_uid"].nunique(),
            "overlap_rallies": len(overlap),
            "new_only_rallies": test_new["rally_uid"].nunique() - len(overlap),
            "overlap_rows": len(new_part),
            "overlap_histories_equal_on_shared_cols": bool(new_part[shared_cols].equals(old_part[shared_cols])),
            "old_has_serverGetPoint": "serverGetPoint" in test_old.columns,
        }

    train_players = set(train["gamePlayerId"]) | set(train["gamePlayerOtherId"])
    test_players = set(test_new["gamePlayerId"]) | set(test_new["gamePlayerOtherId"])
    player_overlap = sorted(train_players & test_players)

    overview_rows = [
        {"dataset": "train", "rows": len(train), "rallies": train["rally_uid"].nunique(), "matches": train["match"].nunique(), "columns": train.shape[1]},
        {"dataset": "test_new", "rows": len(test_new), "rallies": test_new["rally_uid"].nunique(), "matches": test_new["match"].nunique(), "columns": test_new.shape[1]},
        {"dataset": "sample_submission", "rows": len(sample), "rallies": sample["rally_uid"].nunique() if "rally_uid" in sample else 0, "matches": np.nan, "columns": sample.shape[1]},
    ]
    if test_old is not None:
        overview_rows.append({"dataset": "old_test", "rows": len(test_old), "rallies": test_old["rally_uid"].nunique(), "matches": test_old["match"].nunique(), "columns": test_old.shape[1]})
    overview = pd.DataFrame(overview_rows)
    overview.to_csv(TABLES / "dataset_overview.csv", index=False)

    missing = pd.DataFrame(
        {
            "column": sorted(set(train.columns) | set(test_new.columns)),
            "train_missing": [train[c].isna().sum() if c in train.columns else np.nan for c in sorted(set(train.columns) | set(test_new.columns))],
            "test_new_missing": [test_new[c].isna().sum() if c in test_new.columns else np.nan for c in sorted(set(train.columns) | set(test_new.columns))],
        }
    )
    missing.to_csv(TABLES / "missing_values.csv", index=False)

    action_dist = dist_table(train_targets["actionId"], ACTION_LABELS)
    point_dist = dist_table(train_targets["pointId"], POINT_LABELS)
    server_dist = dist_table(train_rally["serverGetPoint"])
    test_visible_action_dist = dist_table(test_new["actionId"], ACTION_LABELS)
    test_visible_point_dist = dist_table(test_new["pointId"], POINT_LABELS)
    action_dist.to_csv(TABLES / "train_target_action_distribution.csv", index=False)
    point_dist.to_csv(TABLES / "train_target_point_distribution.csv", index=False)
    server_dist.to_csv(TABLES / "train_rally_server_distribution.csv", index=False)
    test_visible_action_dist.to_csv(TABLES / "test_visible_action_distribution.csv", index=False)
    test_visible_point_dist.to_csv(TABLES / "test_visible_point_distribution.csv", index=False)

    train_target_sn = sn_bucket(train_targets["strikeNumber"]).value_counts(normalize=True).sort_index()
    test_target_sn = sn_bucket(test_rally["target_next_sn"]).value_counts(normalize=True).sort_index()
    sn_compare = pd.DataFrame({"sn_bucket": ["2", "3-4", "5-8", "9-12", "13+"]})
    sn_compare["train_target_pct"] = sn_compare["sn_bucket"].map(train_target_sn).fillna(0.0)
    sn_compare["test_next_target_pct"] = sn_compare["sn_bucket"].map(test_target_sn).fillna(0.0)
    sn_compare["delta_test_minus_train"] = sn_compare["test_next_target_pct"] - sn_compare["train_target_pct"]
    sn_compare.to_csv(TABLES / "target_next_sn_distribution.csv", index=False)

    rally_summary = pd.DataFrame(
        [
            {"dataset": "train", **train_rally["max_sn"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()},
            {"dataset": "test_new_history", **test_rally["max_sn"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()},
            {"dataset": "test_new_next_target", **test_rally["target_next_sn"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()},
        ]
    )
    rally_summary.to_csv(TABLES / "rally_length_summary.csv", index=False)

    raw_compare_cols = [
        "sex",
        "numberGame",
        "strikeId",
        "handId",
        "strengthId",
        "spinId",
        "positionId",
        "actionId",
        "pointId",
    ]
    shift_rows = []
    train_tmp = train.copy()
    test_tmp = test_new.copy()
    train_tmp["sn_bucket"] = sn_bucket(train_tmp["strikeNumber"])
    test_tmp["sn_bucket"] = sn_bucket(test_tmp["strikeNumber"])
    train_tmp["score_diff_bin"] = pd.cut(train_tmp["scoreSelf"] - train_tmp["scoreOther"], bins=[-30, -5, -2, -1, 0, 1, 2, 5, 30]).astype(str)
    test_tmp["score_diff_bin"] = pd.cut(test_tmp["scoreSelf"] - test_tmp["scoreOther"], bins=[-30, -5, -2, -1, 0, 1, 2, 5, 30]).astype(str)
    for col in raw_compare_cols + ["sn_bucket", "score_diff_bin"]:
        tv, js = categorical_shift(train_tmp[col], test_tmp[col])
        shift_rows.append({"column": col, "total_variation": tv, "js_divergence": js})
    shift = pd.DataFrame(shift_rows).sort_values("total_variation", ascending=False)
    shift.to_csv(TABLES / "train_test_raw_shift.csv", index=False)

    player_table = pd.DataFrame(
        [
            {"metric": "train_unique_players", "value": len(train_players)},
            {"metric": "test_new_unique_players", "value": len(test_players)},
            {"metric": "overlap_unique_players", "value": len(player_overlap)},
            {"metric": "test_player_overlap_rate", "value": len(player_overlap) / max(1, len(test_players))},
        ]
    )
    player_table.to_csv(TABLES / "player_overlap.csv", index=False)

    sn_action = (
        train_targets.assign(sn_bucket=sn_bucket(train_targets["strikeNumber"]))
        .groupby(["sn_bucket", "actionId"])
        .size()
        .reset_index(name="count")
    )
    sn_action["pct_within_bucket"] = sn_action["count"] / sn_action.groupby("sn_bucket")["count"].transform("sum")
    sn_action.to_csv(TABLES / "train_action_by_target_sn_bucket.csv", index=False)

    sn_point = (
        train_targets.assign(sn_bucket=sn_bucket(train_targets["strikeNumber"]))
        .groupby(["sn_bucket", "pointId"])
        .size()
        .reset_index(name="count")
    )
    sn_point["pct_within_bucket"] = sn_point["count"] / sn_point.groupby("sn_bucket")["count"].transform("sum")
    sn_point.to_csv(TABLES / "train_point_by_target_sn_bucket.csv", index=False)

    cooc = (
        train_targets.groupby(["actionId", "pointId"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    cooc["pct"] = cooc["count"] / cooc["count"].sum()
    cooc.to_csv(TABLES / "train_target_action_point_cooccurrence.csv", index=False)

    rare_action = action_dist.sort_values("count").head(10).assign(task="actionId")
    rare_point = point_dist.sort_values("count").head(10).assign(task="pointId")
    rare = pd.concat(
        [
            rare_action.rename(columns={"class": "class_id"})[["task", "class_id", "label", "count", "pct"]],
            rare_point.rename(columns={"class": "class_id"})[["task", "class_id", "label", "count", "pct"]],
        ],
        ignore_index=True,
    )
    rare.to_csv(TABLES / "rare_target_classes.csv", index=False)

    train_action_trans = transition_table(train, "actionId")
    test_action_trans = transition_table(test_new, "actionId")
    train_point_trans = transition_table(train, "pointId")
    test_point_trans = transition_table(test_new, "pointId")
    train_action_trans.to_csv(TABLES / "train_action_transition.csv", index=False)
    test_action_trans.to_csv(TABLES / "test_new_visible_action_transition.csv", index=False)
    train_point_trans.to_csv(TABLES / "train_point_transition.csv", index=False)
    test_point_trans.to_csv(TABLES / "test_new_visible_point_transition.csv", index=False)

    # Leakage diagnostic: full rally length parity is forbidden for prediction, but useful
    # here to document why any terminal-length feature is unsafe for SGP.
    parity_score = (train_rally["max_sn"] % 2).astype(float)
    parity_auc = binary_auc_rank(train_rally["serverGetPoint"], parity_score)
    parity_auc_best_direction = max(parity_auc, 1.0 - parity_auc)
    parity_table = (
        train_rally.assign(length_parity=train_rally["max_sn"] % 2)
        .groupby("length_parity")["serverGetPoint"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "serverGetPoint_rate"})
    )
    parity_table["auc_best_direction"] = parity_auc_best_direction
    parity_table.to_csv(TABLES / "leakage_rally_length_parity_vs_sgp.csv", index=False)

    save_bar(action_dist.assign(class_label=action_dist["class"].astype(str) + "_" + action_dist["label"]), "class_label", "pct", "Train target actionId distribution", FIGS / "train_target_action_distribution.png", rotate=45)
    save_bar(point_dist.assign(class_label=point_dist["class"].astype(str) + "_" + point_dist["label"]), "class_label", "pct", "Train target pointId distribution", FIGS / "train_target_point_distribution.png", rotate=45)
    save_bar(test_visible_action_dist.assign(class_label=test_visible_action_dist["class"].astype(str) + "_" + test_visible_action_dist["label"]), "class_label", "pct", "Test visible actionId distribution", FIGS / "test_visible_action_distribution.png", rotate=45)
    save_bar(test_visible_point_dist.assign(class_label=test_visible_point_dist["class"].astype(str) + "_" + test_visible_point_dist["label"]), "class_label", "pct", "Test visible pointId distribution", FIGS / "test_visible_point_distribution.png", rotate=45)
    save_grouped_bar(sn_compare, "sn_bucket", ["train_target_pct", "test_next_target_pct"], "Target next-strikeNumber bucket: train vs test_new", FIGS / "target_next_sn_distribution.png")
    rally_plot = pd.concat(
        [
            train_rally[["max_sn"]].assign(dataset="train"),
            test_rally[["max_sn"]].assign(dataset="test_new_history"),
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(10, 4.8))
    for name, grp in rally_plot.groupby("dataset"):
        bins = np.arange(1, max(30, int(rally_plot["max_sn"].max())) + 2) - 0.5
        plt.hist(grp["max_sn"], bins=bins, alpha=0.55, density=True, label=name)
    plt.title("Rally history length distribution")
    plt.xlabel("max strikeNumber in observed rows")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "rally_length_distribution.png", dpi=160)
    plt.close()
    save_bar(shift.head(12), "column", "total_variation", "Top train vs test_new raw distribution shifts", FIGS / "train_test_shift_top.png", rotate=45)
    save_bar(parity_table.assign(length_parity=parity_table["length_parity"].astype(str)), "length_parity", "serverGetPoint_rate", "Leakage diagnostic: full rally length parity vs SGP", FIGS / "leakage_rally_length_parity_vs_sgp.png")
    heat = cooc.pivot(index="actionId", columns="pointId", values="count").reindex(index=range(19), columns=range(10), fill_value=0)
    plt.figure(figsize=(9, 7))
    plt.imshow(np.log1p(heat.to_numpy()), aspect="auto", cmap="viridis")
    plt.title("Train target actionId x pointId co-occurrence (log1p count)")
    plt.xlabel("pointId")
    plt.ylabel("actionId")
    plt.xticks(range(10), range(10))
    plt.yticks(range(19), range(19))
    plt.colorbar(label="log1p(count)")
    plt.tight_layout()
    plt.savefig(FIGS / "train_action_point_cooccurrence_heatmap.png", dpi=160)
    plt.close()

    def md_table(df: pd.DataFrame, max_rows: int = 12, float_fmt: str = ".4f") -> str:
        sub = df.head(max_rows).copy()
        for col in sub.columns:
            if pd.api.types.is_float_dtype(sub[col]):
                sub[col] = sub[col].map(lambda x: "" if pd.isna(x) else format(float(x), float_fmt))
        return sub.to_markdown(index=False)

    train_target_count = len(train_targets)
    test_history_pairs = int((test_new["strikeNumber"] >= 2).sum())
    report = f"""# Current EDA Report

Generated: {generated_at}

Data source: `data/train.csv` and active `data/test_new.csv`.

## Executive Findings

1. The active test set is `test_new.csv` with {len(test_new):,} visible rows and {test_new['rally_uid'].nunique():,} rallies. All modeling and submission checks must target this file.
2. Train provides {len(train):,} raw shot rows and {train_target_count:,} supervised next-shot target rows (`strikeNumber >= 2`).
3. Test-history augmentation can legally produce {test_history_pairs:,} visible action/point history pairs from `test_new.csv`, but no SGP labels.
4. Old `test.csv` overlaps {old_overlap_info.get('overlap_rallies', 0):,} of {test_new['rally_uid'].nunique():,} new-test rallies. The visible histories are identical on shared columns: `{old_overlap_info.get('overlap_histories_equal_on_shared_cols', 'n/a')}`. Because old test contains SGP, it is a leakage audit source only.
5. Full-rally length parity is a strong SGP leak diagnostic on train (`best-direction AUC={parity_auc_best_direction:.4f}`). Do not use total rally length, terminal parity, or any full-rally aggregate for server prediction.
6. The strongest raw train-vs-test shifts are listed below. Use them as feature/model-risk hints, not as hard rules.

## Dataset Overview

{md_table(overview)}

## Old-Test Overlap / Leakage Audit

{md_table(pd.DataFrame([old_overlap_info]) if old_overlap_info else pd.DataFrame([{'note': 'old test not found'}]))}

## Missing Values

{md_table(missing)}

## Train Target Class Distributions

### actionId

{md_table(action_dist, max_rows=25)}

### pointId

{md_table(point_dist, max_rows=20)}

### serverGetPoint by rally

{md_table(server_dist)}

## Test Visible History Class Distributions

These are observed history rows only, not target labels for the hidden next shot.

### visible actionId

{md_table(test_visible_action_dist, max_rows=25)}

### visible pointId

{md_table(test_visible_point_dist, max_rows=20)}

## Target next-strikeNumber Distribution

Train target rows use each observed train shot with `strikeNumber >= 2`.
Test target rows are one per rally, with target position `max(strikeNumber) + 1`.

{md_table(sn_compare)}

## Rally Length Summary

{md_table(rally_summary)}

## Train vs Test Raw Distribution Shift

Total variation is easier to read: 0 means identical marginal distribution, 1 means no overlap.

{md_table(shift, max_rows=20)}

## Player-ID Overlap

Player IDs are de-identified and should not be treated as stable identity priors. This table is diagnostic only.

{md_table(player_table)}

## Rare Target Classes

{md_table(rare, max_rows=20)}

## Top action/point co-occurrences

{md_table(cooc, max_rows=20)}

## Top visible transition pairs

### train action transitions

{md_table(train_action_trans, max_rows=15)}

### test_new visible action transitions

{md_table(test_action_trans, max_rows=15)}

### train point transitions

{md_table(train_point_trans, max_rows=15)}

### test_new visible point transitions

{md_table(test_point_trans, max_rows=15)}

## SGP Leakage Diagnostic

This is not a feature recommendation. It documents why terminal/full-rally aggregates are forbidden for SGP.

{md_table(parity_table)}

## Generated Tables

- `tables/dataset_overview.csv`
- `tables/missing_values.csv`
- `tables/train_target_action_distribution.csv`
- `tables/train_target_point_distribution.csv`
- `tables/train_rally_server_distribution.csv`
- `tables/test_visible_action_distribution.csv`
- `tables/test_visible_point_distribution.csv`
- `tables/target_next_sn_distribution.csv`
- `tables/rally_length_summary.csv`
- `tables/train_test_raw_shift.csv`
- `tables/player_overlap.csv`
- `tables/train_action_by_target_sn_bucket.csv`
- `tables/train_point_by_target_sn_bucket.csv`
- `tables/train_target_action_point_cooccurrence.csv`
- `tables/rare_target_classes.csv`
- `tables/train_action_transition.csv`
- `tables/test_new_visible_action_transition.csv`
- `tables/train_point_transition.csv`
- `tables/test_new_visible_point_transition.csv`
- `tables/leakage_rally_length_parity_vs_sgp.csv`

## Generated Figures

- `figures/train_target_action_distribution.png`
- `figures/train_target_point_distribution.png`
- `figures/test_visible_action_distribution.png`
- `figures/test_visible_point_distribution.png`
- `figures/target_next_sn_distribution.png`
- `figures/rally_length_distribution.png`
- `figures/train_test_shift_top.png`
- `figures/train_action_point_cooccurrence_heatmap.png`
- `figures/leakage_rally_length_parity_vs_sgp.png`

## Modeling Implications

- Treat `test_new.csv` as the only active submission target.
- Keep legal test-history augmentation, but verify `aug_rows_in_server_loss == 0`.
- Focus pointId improvements on rare/low-support classes, especially short/half zones.
- Avoid same-family seed averaging as a standalone thesis unless it adds transfer evidence.
- Never import old-test-SGP-trained external caches or submissions into the legal zoo.
- Any new SGP head must be audited for terminal-length and parity leakage before training.
"""

    (OUT / "EDA_CURRENT.md").write_text(report, encoding="utf-8", newline="\n")
    (OUT / "README.md").write_text(
        "# EDA Output\n\n"
        "Current report: `EDA_CURRENT.md`\n\n"
        "This directory was regenerated for `data/test_new.csv`. Old 2026-04-05 plots were removed to avoid stale pre-reset conclusions.\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Wrote {OUT / 'EDA_CURRENT.md'}")
    print(f"Wrote {len(list(TABLES.glob('*.csv')))} tables and {len(list(FIGS.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
