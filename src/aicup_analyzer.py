"""AI CUP 2026 table-tennis analysis CLI.

This script is intentionally conservative:
- report/p2a only inspect and convert data into auditable artifacts.
- train/predict use the in-repo tabular competition data only.
- P2A labels are mapped for analysis, not mixed into supervised training.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    MODEL_DIR,
    N_ACTION_CLASSES,
    N_POINT_CLASSES,
    PROJECT_ROOT,
    RETURN_FORBIDDEN_ACTIONS,
    SAMPLE_SUB_PATH,
    SERVE_ACTION_IDS,
    SUBMISSION_DIR,
    TEST_PATH,
    TRAIN_PATH,
)
from features import build_features, compute_player_stats, get_feature_names  # noqa: E402
from features_p2a_prior import (  # noqa: E402
    add_p2a_prior_features,
    build_p2a_prior_tables,
)
from models import apply_action_constraints  # noqa: E402

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - fallback is for minimal environments
    lgb = None


TARGETS = ["actionId", "pointId", "serverGetPoint"]
DEFAULT_P2A_ROOT = Path(r"D:\P2A_dataset\dataset")
DEFAULT_ANALYZER_ARTIFACT_DIR = Path("artifacts/aicup_analyzer")
DEFAULT_P2A_FLAT_PATH = DEFAULT_ANALYZER_ARTIFACT_DIR / "p2a_actions_flat.csv"

P2A_ACTION_MAP = {
    "拉": 1,
    "侧身拉": 1,
    "擰": 4,
    "拧": 4,
    "挑": 7,
    "摆短": 11,
    "擺短": 11,
    "劈长": 10,
    "劈長": 10,
    "普通": 15,
    "转不转": 15,
    "轉不轉": 15,
    "侧旋": 15,
    "側旋": 15,
    "逆旋转": 17,
    "逆旋轉": 17,
    "勾球": 16,
    "下蹲": 18,
}

P2A_UNRESOLVED = {"控制", "中性", ""}
HAND_MAP = {"正手": 1, "反手": 2}
SERVE_MAP = {"是": 1, "否": 0}


def project_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(PROJECT_ROOT) / p


def ensure_dir(path: str | Path) -> Path:
    p = project_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_csv(path: str | Path) -> pd.DataFrame:
    p = project_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def value_counts_frame(series: pd.Series, name: str) -> pd.DataFrame:
    counts = series.value_counts(dropna=False).sort_index()
    out = counts.rename("count").reset_index()
    out.columns = [name, "count"]
    out["rate"] = out["count"] / max(int(out["count"].sum()), 1)
    return out


def top_transitions(df: pd.DataFrame, col: str, top_n: int = 30) -> pd.DataFrame:
    rows = []
    for _, grp in df.groupby("rally_uid", sort=False):
        vals = grp.sort_values("strikeNumber")[col].astype(int).to_numpy()
        for a, b in zip(vals[:-1], vals[1:]):
            rows.append((a, b))
    if not rows:
        return pd.DataFrame(columns=[f"prev_{col}", f"next_{col}", "count", "rate"])
    counts = Counter(rows)
    total = sum(counts.values())
    data = [
        {
            f"prev_{col}": prev,
            f"next_{col}": nxt,
            "count": count,
            "rate": count / total,
        }
        for (prev, nxt), count in counts.most_common(top_n)
    ]
    return pd.DataFrame(data)


def class_distribution_block(df: pd.DataFrame, title: str, cols: Iterable[str]) -> list[str]:
    lines = [f"### {title}", ""]
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False).sort_index()
        compact = ", ".join(f"{int(k) if pd.notna(k) else 'NA'}:{int(v)}" for k, v in vc.items())
        lines.append(f"- `{col}`: {compact}")
    lines.append("")
    return lines


def dataset_overview(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in [("train", train_df), ("test", test_df)]:
        rally_len = df.groupby("rally_uid").size()
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "rallies": df["rally_uid"].nunique(),
                "matches": df["match"].nunique() if "match" in df.columns else np.nan,
                "players": pd.concat([df["gamePlayerId"], df["gamePlayerOtherId"]]).nunique()
                if {"gamePlayerId", "gamePlayerOtherId"}.issubset(df.columns)
                else np.nan,
                "rally_len_mean": round(float(rally_len.mean()), 3),
                "rally_len_median": round(float(rally_len.median()), 3),
                "rally_len_max": int(rally_len.max()),
                "has_targets": all(c in df.columns for c in TARGETS),
            }
        )
    return pd.DataFrame(rows)


def rule_checks(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    checks = []
    if {"strikeNumber", "strikeId"}.issubset(df.columns):
        checks.append(
            {
                "dataset": dataset_name,
                "check": "strikeNumber=1 has strikeId=1",
                "violations": int(((df["strikeNumber"] == 1) & (df["strikeId"] != 1)).sum()),
            }
        )
    if {"strikeNumber", "actionId"}.issubset(df.columns):
        serve_mask = df["strikeNumber"] == 1
        legal_serve_actions = set(SERVE_ACTION_IDS)
        checks.append(
            {
                "dataset": dataset_name,
                "check": "strikeNumber=1 actionId in serve-compatible set",
                "violations": int((serve_mask & ~df["actionId"].isin(legal_serve_actions)).sum()),
            }
        )
        return_mask = df["strikeNumber"] == 2
        checks.append(
            {
                "dataset": dataset_name,
                "check": "strikeNumber=2 actionId not serve action",
                "violations": int((return_mask & df["actionId"].isin(set(RETURN_FORBIDDEN_ACTIONS))).sum()),
            }
        )
    if {"rally_uid", "serverGetPoint"}.issubset(df.columns):
        nunique = df.groupby("rally_uid")["serverGetPoint"].nunique()
        checks.append(
            {
                "dataset": dataset_name,
                "check": "serverGetPoint constant within rally",
                "violations": int((nunique > 1).sum()),
            }
        )
    return pd.DataFrame(checks)


def report_p2a_inventory(p2a_root: Path) -> dict:
    out = {
        "root": str(p2a_root),
        "exists": p2a_root.exists(),
        "versions": {},
    }
    for version in ["v1", "v2"]:
        label_path = p2a_root / "label" / f"{version}_renamed.json"
        video_dir = p2a_root / "video" / version
        version_info = {
            "label_exists": label_path.exists(),
            "video_dir_exists": video_dir.exists(),
            "mp4_count": len(list(video_dir.glob("*.mp4"))) if video_dir.exists() else 0,
            "partial_count": len(list(video_dir.glob("*.downloading"))) if video_dir.exists() else 0,
            "label_items": None,
            "actions": None,
            "missing_videos": None,
        }
        if label_path.exists():
            data = json.loads(label_path.read_text(encoding="utf-8"))
            expected = {item.get("url") for item in data}
            actual = {p.name for p in video_dir.glob("*.mp4")} if video_dir.exists() else set()
            version_info["label_items"] = len(data)
            version_info["actions"] = sum(len(item.get("actions") or []) for item in data)
            version_info["missing_videos"] = len(expected - actual)
        out["versions"][version] = version_info
    return out


def make_report(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(args.output_dir)
    train_df = load_csv(args.train)
    test_df = load_csv(args.test)

    overview = dataset_overview(train_df, test_df)
    write_csv(overview, out_dir / "dataset_overview.csv")

    checks = pd.concat(
        [rule_checks(train_df, "train"), rule_checks(test_df, "test")],
        ignore_index=True,
    )
    write_csv(checks, out_dir / "rule_checks.csv")

    write_csv(value_counts_frame(train_df["actionId"], "actionId"), out_dir / "train_action_distribution.csv")
    write_csv(value_counts_frame(train_df["pointId"], "pointId"), out_dir / "train_point_distribution.csv")
    write_csv(
        value_counts_frame(train_df["serverGetPoint"], "serverGetPoint"),
        out_dir / "train_server_distribution.csv",
    )

    if "actionId" in test_df.columns:
        write_csv(value_counts_frame(test_df["actionId"], "actionId"), out_dir / "test_visible_action_distribution.csv")
    if "pointId" in test_df.columns:
        write_csv(value_counts_frame(test_df["pointId"], "pointId"), out_dir / "test_visible_point_distribution.csv")

    train_rally_len = train_df.groupby("rally_uid").size().rename("rally_len").reset_index()
    write_csv(train_rally_len["rally_len"].describe().to_frame("train_rally_len").T, out_dir / "rally_length_summary.csv")
    write_csv(top_transitions(train_df, "actionId"), out_dir / "train_action_transitions_top.csv")
    write_csv(top_transitions(train_df, "pointId"), out_dir / "train_point_transitions_top.csv")

    test_next_sn = (
        test_df.groupby("rally_uid")["strikeNumber"]
        .max()
        .add(1)
        .rename("next_strikeNumber")
        .reset_index()
    )
    write_csv(
        value_counts_frame(test_next_sn["next_strikeNumber"], "next_strikeNumber"),
        out_dir / "test_next_strikeNumber_distribution.csv",
    )

    p2a_info = report_p2a_inventory(project_path(args.p2a_root))
    (out_dir / "p2a_inventory.json").write_text(
        json.dumps(p2a_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# AI CUP Analyzer Report",
        "",
        "## Dataset Overview",
        "",
        overview.to_markdown(index=False),
        "",
        "## Rule Checks",
        "",
        checks.to_markdown(index=False),
        "",
    ]
    lines += class_distribution_block(train_df, "Train Target Distributions", TARGETS)
    lines += class_distribution_block(test_df, "Test Visible History Distributions", ["actionId", "pointId"])
    lines += [
        "## P2A Inventory",
        "",
        f"- root: `{p2a_info['root']}`",
        f"- exists: `{p2a_info['exists']}`",
    ]
    for version, info in p2a_info["versions"].items():
        lines.append(
            f"- {version}: labels={info['label_items']}, actions={info['actions']}, "
            f"mp4={info['mp4_count']}, partial={info['partial_count']}, missing={info['missing_videos']}"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- P2A is kept as an analysis/transition-prior resource here; this script does not train on it automatically.",
        "- Use `python src/aicup_analyzer.py p2a` to flatten P2A labels into auditable CSV files.",
        "- Use `python src/aicup_analyzer.py train` for an isolated tabular baseline under `models/aicup_analyzer/`.",
        "- Use `python src/aicup_analyzer.py train --use-p2a-prior` to add P2A transition-prior features without adding P2A rows to training.",
        "",
    ]

    report_path = out_dir / "AICUP_ANALYZER_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {report_path}")
    return report_path


def segment_p2a_actions(actions: list[dict], gap_seconds: float) -> list[dict]:
    flat = []
    last_end = None
    p2a_rally_id = 0
    strike_number = 0
    for action in sorted(actions, key=lambda a: float(a.get("start_id") or 0.0)):
        labels = list(action.get("label_names") or [])
        hand_label = labels[0] if len(labels) > 0 else ""
        serve_label = labels[1] if len(labels) > 1 else ""
        action_label = labels[2] if len(labels) > 2 else ""
        start_sec = float(action.get("start_id") or 0.0)
        end_sec = float(action.get("end_id") or start_sec)
        is_serve = SERVE_MAP.get(serve_label, -1)

        new_rally = False
        if last_end is None:
            new_rally = True
        elif start_sec - last_end > gap_seconds:
            new_rally = True
        elif is_serve == 1 and strike_number > 0:
            new_rally = True

        if new_rally:
            p2a_rally_id += 1
            strike_number = 1
        else:
            strike_number += 1

        mapped_action = P2A_ACTION_MAP.get(action_label, -1)
        if action_label in P2A_UNRESOLVED:
            mapped_action = -1
        inferred_strike_id = 1 if is_serve == 1 else (2 if strike_number == 2 else 4)

        flat.append(
            {
                "p2a_rally_id": p2a_rally_id,
                "p2a_strikeNumber": strike_number,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": max(0.0, end_sec - start_sec),
                "hand_label": hand_label,
                "handId": HAND_MAP.get(hand_label, 0),
                "is_serve_label": serve_label,
                "is_serve": is_serve,
                "p2a_action": action_label,
                "mapped_actionId": mapped_action,
                "mapping_status": "mapped" if mapped_action >= 0 else "unresolved",
                "inferred_strikeId": inferred_strike_id,
            }
        )
        last_end = end_sec
    return flat


def flatten_p2a(args: argparse.Namespace) -> Path:
    p2a_root = project_path(args.p2a_root)
    if not p2a_root.exists():
        raise FileNotFoundError(f"P2A root not found: {p2a_root}")
    out_dir = ensure_dir(args.output_dir)

    records = []
    video_rows = []
    for version in args.versions:
        label_path = p2a_root / "label" / f"{version}_renamed.json"
        video_dir = p2a_root / "video" / version
        if not label_path.exists():
            raise FileNotFoundError(f"Missing P2A label file: {label_path}")
        data = json.loads(label_path.read_text(encoding="utf-8"))
        available_videos = {p.name for p in video_dir.glob("*.mp4")} if video_dir.exists() else set()

        for item in data:
            video_id = item.get("url")
            actions = item.get("actions") or []
            video_rows.append(
                {
                    "version": version,
                    "video_id": video_id,
                    "action_count": len(actions),
                    "video_exists": int(video_id in available_videos),
                }
            )
            for action_row in segment_p2a_actions(actions, args.gap_seconds):
                action_row["version"] = version
                action_row["video_id"] = video_id
                records.append(action_row)

    flat_df = pd.DataFrame(records)
    if not flat_df.empty:
        flat_df = flat_df[
            [
                "version",
                "video_id",
                "p2a_rally_id",
                "p2a_strikeNumber",
                "start_sec",
                "end_sec",
                "duration_sec",
                "hand_label",
                "handId",
                "is_serve_label",
                "is_serve",
                "p2a_action",
                "mapped_actionId",
                "mapping_status",
                "inferred_strikeId",
            ]
        ]

    flat_path = out_dir / "p2a_actions_flat.csv"
    video_path = out_dir / "p2a_video_inventory.csv"
    write_csv(flat_df, flat_path)
    write_csv(pd.DataFrame(video_rows), video_path)

    summary = {
        "p2a_root": str(p2a_root),
        "versions": list(args.versions),
        "gap_seconds": args.gap_seconds,
        "rows": int(len(flat_df)),
        "videos": int(len(video_rows)),
        "mapped_rows": int((flat_df["mapped_actionId"] >= 0).sum()) if not flat_df.empty else 0,
        "unresolved_rows": int((flat_df["mapped_actionId"] < 0).sum()) if not flat_df.empty else 0,
        "action_counts": flat_df["p2a_action"].value_counts().to_dict() if not flat_df.empty else {},
        "mapped_action_counts": flat_df["mapped_actionId"].value_counts().sort_index().to_dict()
        if not flat_df.empty
        else {},
    }
    (out_dir / "p2a_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# P2A Flatten Summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- videos: `{summary['videos']}`",
        f"- mapped rows: `{summary['mapped_rows']}`",
        f"- unresolved rows: `{summary['unresolved_rows']}`",
        f"- gap_seconds: `{summary['gap_seconds']}`",
        "",
        "## Top P2A Actions",
        "",
    ]
    if not flat_df.empty:
        lines.append(flat_df["p2a_action"].value_counts().head(20).to_markdown())
        lines += ["", "## Mapped actionId Counts", ""]
        lines.append(flat_df["mapped_actionId"].value_counts().sort_index().to_markdown())
    lines.append("")
    (out_dir / "P2A_FLAT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote P2A flat labels to {flat_path}")
    return flat_path


def lgb_params(objective: str, n_classes: int | None, seed: int) -> dict:
    params = {
        "objective": objective,
        "metric": "multi_logloss" if objective == "multiclass" else "auc",
        "learning_rate": 0.06,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "feature_fraction": 0.9,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "seed": seed,
        "num_threads": -1,
        "verbose": -1,
    }
    if objective == "multiclass":
        params["num_class"] = n_classes
    return params


def fit_lgb_classifier(X_train, y_train, X_val, y_val, objective: str, n_classes: int | None, args):
    if lgb is None:
        raise ImportError("lightgbm is required for aicup_analyzer train/predict")
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    callbacks = [lgb.log_evaluation(0)]
    if args.early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(args.early_stopping_rounds, verbose=False))
    model = lgb.train(
        lgb_params(objective, n_classes, args.seed),
        dtrain,
        num_boost_round=args.n_estimators,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return model


def multiclass_proba(model, X, n_classes: int) -> np.ndarray:
    proba = model.predict(X, num_iteration=model.best_iteration)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    out = np.zeros((len(X), n_classes), dtype=np.float32)
    width = min(proba.shape[1], n_classes)
    out[:, :width] = proba[:, :width]
    row_sum = out.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 0
    if np.any(bad):
        out[bad, :] = 1.0 / n_classes
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def default_model_dir(use_p2a_prior: bool) -> Path:
    name = "aicup_analyzer_p2a_prior" if use_p2a_prior else "aicup_analyzer"
    return Path(MODEL_DIR) / name


def default_model_path(use_p2a_prior: bool) -> Path:
    return default_model_dir(use_p2a_prior) / "analyzer_models.pkl"


def default_submission_path(use_p2a_prior: bool) -> Path:
    name = "submission_aicup_analyzer_p2a_prior.csv" if use_p2a_prior else "submission_aicup_analyzer.csv"
    return Path(SUBMISSION_DIR) / name


def maybe_add_p2a_prior_features(feat_df: pd.DataFrame, args: argparse.Namespace, bundle: dict | None = None):
    use_p2a = bool(getattr(args, "use_p2a_prior", False))
    tables = None
    if bundle is not None:
        use_p2a = use_p2a or bool(bundle.get("use_p2a_prior", False))
        tables = bundle.get("p2a_prior_tables")
    if not use_p2a:
        return feat_df, None
    if tables is None:
        tables = build_p2a_prior_tables(project_path(args.p2a_flat_path), alpha=args.p2a_alpha)
    print("Adding P2A prior features...")
    return add_p2a_prior_features(feat_df, tables), tables


def train_analyzer(args: argparse.Namespace) -> Path:
    train_df = load_csv(args.train)
    model_dir = ensure_dir(args.model_dir or default_model_dir(args.use_p2a_prior))
    if args.max_rallies and args.max_rallies > 0:
        keep_rallies = train_df["rally_uid"].drop_duplicates().head(args.max_rallies)
        train_df = train_df[train_df["rally_uid"].isin(keep_rallies)].copy()
        print(f"Using smoke subset: {train_df['rally_uid'].nunique()} rallies, {len(train_df)} rows")

    print("Computing player stats and feature matrix...")
    player_stats = compute_player_stats(train_df)
    feat_df = build_features(train_df, is_train=True, player_stats=player_stats)
    feat_df, p2a_prior_tables = maybe_add_p2a_prior_features(feat_df, args)
    feature_names = get_feature_names(feat_df)

    X = feat_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(-1)
    y_action = feat_df["y_actionId"].astype(int).to_numpy()
    y_point = feat_df["y_pointId"].astype(int).to_numpy()
    y_server = feat_df["y_serverGetPoint"].astype(int).to_numpy()
    next_sn = feat_df["next_strikeNumber"].astype(int).to_numpy()
    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = feat_df["rally_uid"].map(rally_to_match).to_numpy()

    n_splits = min(args.folds, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    models = {"action": [], "point": [], "server": []}
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, groups=groups), start=1):
        print(f"Fold {fold}/{n_splits}: training action/point/server models")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]

        action_model = fit_lgb_classifier(
            X_tr, y_action[tr_idx], X_va, y_action[va_idx], "multiclass", N_ACTION_CLASSES, args
        )
        action_proba = multiclass_proba(action_model, X_va, N_ACTION_CLASSES)
        action_proba = apply_action_constraints(action_proba, next_sn[va_idx])
        action_pred = np.argmax(action_proba, axis=1)
        action_f1 = f1_score(
            y_action[va_idx],
            action_pred,
            labels=list(range(N_ACTION_CLASSES)),
            average="macro",
            zero_division=0,
        )

        point_model = fit_lgb_classifier(
            X_tr, y_point[tr_idx], X_va, y_point[va_idx], "multiclass", N_POINT_CLASSES, args
        )
        point_proba = multiclass_proba(point_model, X_va, N_POINT_CLASSES)
        point_pred = np.argmax(point_proba, axis=1)
        point_f1 = f1_score(
            y_point[va_idx],
            point_pred,
            labels=list(range(N_POINT_CLASSES)),
            average="macro",
            zero_division=0,
        )

        server_model = fit_lgb_classifier(
            X_tr, y_server[tr_idx], X_va, y_server[va_idx], "binary", None, args
        )
        server_proba = server_model.predict(X_va, num_iteration=server_model.best_iteration)
        server_auc = roc_auc_score(y_server[va_idx], server_proba)

        models["action"].append(action_model)
        models["point"].append(point_model)
        models["server"].append(server_model)
        fold_rows.append(
            {
                "fold": fold,
                "action_macro_f1": action_f1,
                "point_macro_f1": point_f1,
                "server_auc": server_auc,
                "composite": 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc,
                "val_rows": len(va_idx),
            }
        )
        print(
            f"  action={action_f1:.4f} point={point_f1:.4f} "
            f"server={server_auc:.4f} composite={fold_rows[-1]['composite']:.4f}"
        )

    cv_df = pd.DataFrame(fold_rows)
    write_csv(cv_df, model_dir / "cv_scores.csv")
    summary = {
        "folds": n_splits,
        "n_estimators": args.n_estimators,
        "use_p2a_prior": bool(args.use_p2a_prior),
        "p2a_flat_path": str(project_path(args.p2a_flat_path)) if args.use_p2a_prior else None,
        "p2a_alpha": float(args.p2a_alpha) if args.use_p2a_prior else None,
        "feature_count": len(feature_names),
        "train_feature_rows": len(feat_df),
        "mean_action_macro_f1": float(cv_df["action_macro_f1"].mean()),
        "mean_point_macro_f1": float(cv_df["point_macro_f1"].mean()),
        "mean_server_auc": float(cv_df["server_auc"].mean()),
        "mean_composite": float(cv_df["composite"].mean()),
    }
    (model_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (model_dir / "analyzer_models.pkl").open("wb") as f:
        pickle.dump(
            {
                "models": models,
                "feature_names": feature_names,
                "player_stats": player_stats,
                "use_p2a_prior": bool(args.use_p2a_prior),
                "p2a_prior_tables": p2a_prior_tables,
                "summary": summary,
            },
            f,
        )
    print(f"Saved analyzer models to {model_dir / 'analyzer_models.pkl'}")
    return model_dir / "analyzer_models.pkl"


def predict_analyzer(args: argparse.Namespace) -> Path:
    test_df = load_csv(args.test)
    model_path = project_path(args.model_path or default_model_path(args.use_p2a_prior))
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model bundle: {model_path}")
    with model_path.open("rb") as f:
        bundle = pickle.load(f)

    feat_df = build_features(test_df, is_train=False, player_stats=bundle["player_stats"])
    feat_df, _ = maybe_add_p2a_prior_features(feat_df, args, bundle=bundle)
    feature_names = bundle["feature_names"]
    X = feat_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(-1)
    next_sn = feat_df["next_strikeNumber"].astype(int).to_numpy()

    action_proba = np.zeros((len(X), N_ACTION_CLASSES), dtype=np.float32)
    for model in bundle["models"]["action"]:
        action_proba += multiclass_proba(model, X, N_ACTION_CLASSES)
    action_proba /= max(len(bundle["models"]["action"]), 1)
    action_proba = apply_action_constraints(action_proba, next_sn)
    action_pred = np.argmax(action_proba, axis=1).astype(int)

    point_proba = np.zeros((len(X), N_POINT_CLASSES), dtype=np.float32)
    for model in bundle["models"]["point"]:
        point_proba += multiclass_proba(model, X, N_POINT_CLASSES)
    point_proba /= max(len(bundle["models"]["point"]), 1)
    point_pred = np.argmax(point_proba, axis=1).astype(int)

    server_proba = np.zeros(len(X), dtype=np.float32)
    for model in bundle["models"]["server"]:
        server_proba += model.predict(X, num_iteration=model.best_iteration)
    server_proba /= max(len(bundle["models"]["server"]), 1)
    server_pred = (server_proba >= args.server_threshold).astype(int)

    submission = pd.DataFrame(
        {
            "rally_uid": feat_df["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred,
            "pointId": point_pred,
            "serverGetPoint": server_pred,
        }
    )
    sample_path = project_path(args.sample_submission)
    if sample_path.exists():
        sample_cols = list(pd.read_csv(sample_path, nrows=0).columns)
        if sample_cols:
            submission = submission[sample_cols]

    output_path = args.output or default_submission_path(bool(bundle.get("use_p2a_prior", False) or args.use_p2a_prior))
    out_path = project_path(output_path)
    write_csv(submission, out_path)
    print(f"Wrote analyzer submission to {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI CUP 2026 analysis CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    common_train = {"default": TRAIN_PATH, "help": "Path to train.csv"}
    common_test = {"default": TEST_PATH, "help": "Path to test/test_new CSV"}

    p_report = sub.add_parser("report", help="Create EDA and integrity report")
    p_report.add_argument("--train", **common_train)
    p_report.add_argument("--test", **common_test)
    p_report.add_argument("--p2a-root", default=str(DEFAULT_P2A_ROOT), help="P2A dataset root")
    p_report.add_argument("--output-dir", default=str(DEFAULT_ANALYZER_ARTIFACT_DIR), help="Report output directory")
    p_report.set_defaults(func=make_report)

    p_p2a = sub.add_parser("p2a", help="Flatten P2A label JSON into auditable CSV")
    p_p2a.add_argument("--p2a-root", default=str(DEFAULT_P2A_ROOT), help="P2A dataset root")
    p_p2a.add_argument("--versions", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])
    p_p2a.add_argument("--gap-seconds", type=float, default=4.0, help="Gap threshold for rally segmentation")
    p_p2a.add_argument("--output-dir", default=str(DEFAULT_ANALYZER_ARTIFACT_DIR), help="Output directory")
    p_p2a.set_defaults(func=flatten_p2a)

    p_train = sub.add_parser("train", help="Train isolated tabular baseline")
    p_train.add_argument("--train", **common_train)
    p_train.add_argument("--model-dir", default=None)
    p_train.add_argument("--folds", type=int, default=3)
    p_train.add_argument("--n-estimators", type=int, default=300)
    p_train.add_argument("--early-stopping-rounds", type=int, default=30)
    p_train.add_argument("--max-rallies", type=int, default=0, help="Optional smoke-test cap on training rallies")
    p_train.add_argument("--use-p2a-prior", action="store_true", help="Append P2A external action-prior features")
    p_train.add_argument("--p2a-flat-path", default=str(DEFAULT_P2A_FLAT_PATH), help="Flattened P2A CSV path")
    p_train.add_argument("--p2a-alpha", type=float, default=1.0, help="Additive smoothing for P2A prior tables")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=train_analyzer)

    p_predict = sub.add_parser("predict", help="Predict submission from baseline")
    p_predict.add_argument("--test", **common_test)
    p_predict.add_argument("--model-path", default=None)
    p_predict.add_argument("--sample-submission", default=SAMPLE_SUB_PATH)
    p_predict.add_argument("--output", default=None)
    p_predict.add_argument("--server-threshold", type=float, default=0.5)
    p_predict.add_argument("--use-p2a-prior", action="store_true", help="Use P2A-prior default model/output paths")
    p_predict.add_argument("--p2a-flat-path", default=str(DEFAULT_P2A_FLAT_PATH), help="Flattened P2A CSV path")
    p_predict.add_argument("--p2a-alpha", type=float, default=1.0, help="Additive smoothing if bundle lacks P2A tables")
    p_predict.set_defaults(func=predict_analyzer)

    p_all = sub.add_parser("all", help="Run report and P2A flattening")
    p_all.add_argument("--train", **common_train)
    p_all.add_argument("--test", **common_test)
    p_all.add_argument("--p2a-root", default=str(DEFAULT_P2A_ROOT), help="P2A dataset root")
    p_all.add_argument("--output-dir", default=str(DEFAULT_ANALYZER_ARTIFACT_DIR), help="Output directory")
    p_all.add_argument("--versions", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])
    p_all.add_argument("--gap-seconds", type=float, default=4.0)

    def run_all(ns: argparse.Namespace):
        make_report(ns)
        flatten_p2a(ns)

    p_all.set_defaults(func=run_all)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
