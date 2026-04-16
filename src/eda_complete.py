"""
AI CUP 2026 Table Tennis Prediction - Complete EDA Script
Covers all 23 items (Sections A through I)
"""

import os
import sys
import warnings
import textwrap
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm
from collections import Counter

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "eda_output")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_PATH  = os.path.join(DATA_DIR, "train.csv")
TEST_PATH   = os.path.join(DATA_DIR, "test.csv")

# ──────────────────────────────────────────────
# Font – try Chinese, fallback to default
# ──────────────────────────────────────────────
def setup_font():
    chinese_candidates = [
        "Microsoft YaHei", "Microsoft JhengHei", "SimHei", "Noto Sans CJK TC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in chinese_candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            print(f"[font] Using Chinese font: {name}")
            return True
    print("[font] No Chinese font found – using English labels")
    return False

USE_CHINESE = setup_font()
matplotlib.rcParams.update({
    "axes.unicode_minus": False,
    "figure.dpi": 100,
    "savefig.dpi": 120,
    "savefig.bbox": "tight",
})

# ──────────────────────────────────────────────
# Label maps (Chinese if available else English)
# ──────────────────────────────────────────────
ACTION_LABELS_ZH = {
    0: "無/其他", 1: "拉球", 2: "反拉", 3: "殺球", 4: "擰球",
    5: "快帶", 6: "推擠", 7: "挑撥", 8: "拱球", 9: "磕球",
    10: "搓球", 11: "擺短", 12: "削球", 13: "擋球", 14: "放高球",
    15: "傳統發", 16: "勾手發", 17: "逆旋轉發", 18: "下蹲式發",
}
ACTION_LABELS_EN = {
    0: "None/Other", 1: "Loop", 2: "Counter", 3: "Smash", 4: "Flip",
    5: "Fast-block", 6: "Push", 7: "Flick", 8: "Arc", 9: "Poke",
    10: "Chop(ctrl)", 11: "Short", 12: "Chop(def)", 13: "Block", 14: "Lob",
    15: "Serve-trad", 16: "Serve-hook", 17: "Serve-rev", 18: "Serve-squat",
}
POINT_LABELS_ZH = {
    0: "無/出界", 1: "正短", 2: "中短", 3: "反短",
    4: "正半台", 5: "中半台", 6: "反半台",
    7: "正長", 8: "中長", 9: "反長",
}
POINT_LABELS_EN = {
    0: "None/Out", 1: "FH-short", 2: "Mid-short", 3: "BH-short",
    4: "FH-mid", 5: "Mid-mid", 6: "BH-mid",
    7: "FH-long", 8: "Mid-long", 9: "BH-long",
}
STRIKE_LABELS_EN = {1: "Serve", 2: "Return", 4: "3rd+", 8: "No-vid", 16: "Pause"}
HAND_LABELS_EN   = {0: "None", 1: "FH", 2: "BH"}
STR_LABELS_EN    = {0: "None", 1: "Strong", 2: "Medium", 3: "Weak"}
SPIN_LABELS_EN   = {0: "None", 1: "Topspin", 2: "Backspin", 3: "No-spin", 4: "Side-top", 5: "Side-back"}
POS_LABELS_EN    = {0: "None", 1: "Left", 2: "Center", 3: "Right"}
SEX_LABELS_EN    = {1: "Male", 2: "Female"}

ACTION_LABELS = ACTION_LABELS_ZH if USE_CHINESE else ACTION_LABELS_EN
POINT_LABELS  = POINT_LABELS_ZH  if USE_CHINESE else POINT_LABELS_EN

# ──────────────────────────────────────────────
# Summary log
# ──────────────────────────────────────────────
LOG_LINES = []

def log(msg=""):
    print(msg)
    LOG_LINES.append(msg)

def save_log():
    path = os.path.join(OUT_DIR, "eda_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    print(f"\n[saved] {path}")

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path)
    plt.close("all")
    print(f"[saved] {path}")

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
log("=" * 70)
log("Loading data …")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
log(f"Train shape : {train.shape}")
log(f"Test  shape : {test.shape}")

# SN groups helper
def sn_group(sn):
    if sn == 1:   return "SN=1"
    if sn == 2:   return "SN=2"
    if sn <= 4:   return "SN=3-4"
    if sn <= 8:   return "SN=5-8"
    if sn <= 12:  return "SN=9-12"
    return "SN=13+"

SN_ORDER = ["SN=1", "SN=2", "SN=3-4", "SN=5-8", "SN=9-12", "SN=13+"]
train["sn_group"] = train["strikeNumber"].apply(sn_group)

# ══════════════════════════════════════════════
# A. DATA OVERVIEW
# ══════════════════════════════════════════════
log("\n" + "=" * 70)
log("A. DATA OVERVIEW")
log("=" * 70)

# Item 1 – Basic data check
log("\n--- Item 1: Basic data check ---")
log(f"\nTrain dtypes:\n{train.dtypes.to_string()}")
log(f"\nTrain missing values:\n{train.isnull().sum().to_string()}")
log(f"\nTest  missing values:\n{test.isnull().sum().to_string()}")

log("\nUnique value counts (train):")
for col in train.columns:
    log(f"  {col:30s} nunique={train[col].nunique():6d}  "
        f"range=[{train[col].min()}, {train[col].max()}]")

log("\nTrain describe:\n" + train.describe().to_string())

# Item 2 – Column role classification
log("\n--- Item 2: Column role classification ---")
roles = {
    "ID"     : ["rally_uid", "match", "rally_id"],
    "State"  : ["sex", "numberGame", "strikeNumber", "scoreSelf", "scoreOther",
                 "gamePlayerId", "gamePlayerOtherId", "strikeId"],
    "Action" : ["handId", "strengthId", "spinId", "positionId"],
    "Target" : ["actionId", "pointId", "serverGetPoint"],
}
for role, cols in roles.items():
    log(f"  {role:8s}: {cols}")

# Quick visual: missing heatmap (simple bar)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, df, title in [(axes[0], train, "Train"), (axes[1], test, "Test")]:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f"{title}: Missing Values")
    else:
        missing.plot(kind="bar", ax=ax, color="tomato")
        ax.set_title(f"{title}: Missing Values")
        ax.set_ylabel("Count")
plt.tight_layout()
savefig("A1_missing_values.png")

# ══════════════════════════════════════════════
# B. TARGET DISTRIBUTIONS
# ══════════════════════════════════════════════
log("\n" + "=" * 70)
log("B. TARGET DISTRIBUTIONS")
log("=" * 70)

# Item 3 – actionId distribution
log("\n--- Item 3: actionId distribution ---")
action_counts = train["actionId"].value_counts().sort_index()
log(action_counts.to_string())

fig, ax = plt.subplots(figsize=(16, 5))
x = action_counts.index.tolist()
y = action_counts.values
colors = []
for i in x:
    if i == 0:              colors.append("#aaaaaa")
    elif 1 <= i <= 7:       colors.append("#e74c3c")   # attack
    elif 8 <= i <= 11:      colors.append("#2ecc71")   # control
    elif 12 <= i <= 14:     colors.append("#3498db")   # defensive
    else:                   colors.append("#f39c12")   # serve
bars = ax.bar([str(v) for v in x], y, color=colors)
ax.set_xlabel("actionId")
ax.set_ylabel("Count")
ax.set_title("actionId Distribution (Attack=red, Control=green, Def=blue, Serve=orange)")
for bar, val in zip(bars, y):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{val:,}", ha="center", va="bottom", fontsize=7)
# Labels
ax_labels = [ACTION_LABELS.get(i, str(i)) for i in x]
ax.set_xticks(range(len(x)))
ax.set_xticklabels([f"{i}\n{lbl}" for i, lbl in zip(x, ax_labels)], fontsize=7)
plt.tight_layout()
savefig("B3_actionId_distribution.png")

# Item 4 – pointId distribution
log("\n--- Item 4: pointId distribution ---")
point_counts = train["pointId"].value_counts().sort_index()
log(point_counts.to_string())

fig, ax = plt.subplots(figsize=(12, 4))
x = point_counts.index.tolist()
y = point_counts.values
ax.bar([str(v) for v in x], y, color="#9b59b6", edgecolor="white")
ax.set_xlabel("pointId")
ax.set_ylabel("Count")
ax.set_title("pointId Distribution (0=out/net, 1-3=short, 4-6=mid, 7-9=long)")
ax_labels = [POINT_LABELS.get(i, str(i)) for i in x]
ax.set_xticks(range(len(x)))
ax.set_xticklabels([f"{i}\n{lbl}" for i, lbl in zip(x, ax_labels)], fontsize=8)
for bar, val in zip(ax.patches, y):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{val:,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
savefig("B4_pointId_distribution.png")

# Item 5 – serverGetPoint distribution
log("\n--- Item 5: serverGetPoint distribution ---")
sgp_counts = train["serverGetPoint"].value_counts().sort_index()
log(sgp_counts.to_string())
log(f"Ratio (1/0): {sgp_counts.get(1,0)/sgp_counts.get(0,1):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(["No (0)", "Yes (1)"], [sgp_counts.get(0,0), sgp_counts.get(1,0)],
            color=["#e74c3c", "#2ecc71"])
axes[0].set_title("serverGetPoint Distribution (count)")
axes[0].set_ylabel("Count")
for bar, val in zip(axes[0].patches, [sgp_counts.get(0,0), sgp_counts.get(1,0)]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{val:,}", ha="center", va="bottom")
axes[1].pie([sgp_counts.get(0,0), sgp_counts.get(1,0)],
            labels=["Server loses (0)", "Server wins (1)"],
            colors=["#e74c3c", "#2ecc71"], autopct="%1.1f%%", startangle=90)
axes[1].set_title("serverGetPoint Proportion")
plt.tight_layout()
savefig("B5_serverGetPoint_distribution.png")

# ══════════════════════════════════════════════
# C. TEMPORAL STRUCTURE
# ══════════════════════════════════════════════
log("\n" + "=" * 70)
log("C. TEMPORAL STRUCTURE")
log("=" * 70)

# Item 6 – strikeNumber distribution
log("\n--- Item 6: strikeNumber distribution ---")
sn_counts = train["strikeNumber"].value_counts().sort_index()
log(f"SN stats: min={train['strikeNumber'].min()}, max={train['strikeNumber'].max()}, "
    f"mean={train['strikeNumber'].mean():.2f}, median={train['strikeNumber'].median()}")
sn_group_counts = train["sn_group"].value_counts().reindex(SN_ORDER)
log(f"\nSN group distribution:\n{sn_group_counts.to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
# Overall
axes[0].bar(sn_counts.index, sn_counts.values, color="#3498db", width=0.8)
axes[0].set_xlabel("Strike Number")
axes[0].set_ylabel("Count")
axes[0].set_title("Strike Number Distribution (all)")
axes[0].set_xlim(0, min(50, sn_counts.index.max() + 1))
# By group
sn_group_counts.plot(kind="bar", ax=axes[1], color="#e67e22")
axes[1].set_title("Strike Number Distribution by Group")
axes[1].set_ylabel("Count")
axes[1].set_xlabel("SN Group")
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
savefig("C6_strikeNumber_distribution.png")

# Item 7 – Rally length distribution
log("\n--- Item 7: Rally length distribution ---")
rally_len = train.groupby("rally_uid")["strikeNumber"].max()
log(f"Rally length stats:\n{rally_len.describe().to_string()}")
log(f"  Rallies with len=1 : {(rally_len==1).sum()}")
log(f"  Rallies with len>=10: {(rally_len>=10).sum()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(rally_len.values, bins=range(1, min(60, rally_len.max()+2)),
             color="#1abc9c", edgecolor="white")
axes[0].set_xlabel("Rally Length (strikes)")
axes[0].set_ylabel("Count")
axes[0].set_title("Rally Length Distribution")
axes[0].axvline(rally_len.mean(), color="red", linestyle="--",
                label=f"Mean={rally_len.mean():.1f}")
axes[0].legend()
# CDF
sorted_len = np.sort(rally_len.values)
cdf = np.arange(1, len(sorted_len)+1) / len(sorted_len)
axes[1].plot(sorted_len, cdf, color="#8e44ad")
axes[1].set_xlabel("Rally Length")
axes[1].set_ylabel("CDF")
axes[1].set_title("Rally Length CDF")
axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Median")
axes[1].axhline(0.9, color="orange", linestyle="--", alpha=0.5, label="90th pct")
axes[1].legend()
plt.tight_layout()
savefig("C7_rally_length_distribution.png")

# ══════════════════════════════════════════════
# D. TARGET vs STRIKENUMBER
# ══════════════════════════════════════════════
log("\n" + "=" * 70)
log("D. TARGET vs STRIKENUMBER")
log("=" * 70)

# Item 8 – actionId distribution at different SN groups
log("\n--- Item 8: actionId vs SN groups ---")
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()
for idx, grp in enumerate(SN_ORDER):
    subset = train[train["sn_group"] == grp]
    counts = subset["actionId"].value_counts().sort_index()
    all_ids = range(0, 19)
    y = [counts.get(i, 0) for i in all_ids]
    colors = []
    for i in all_ids:
        if i == 0:          colors.append("#aaaaaa")
        elif 1 <= i <= 7:   colors.append("#e74c3c")
        elif 8 <= i <= 11:  colors.append("#2ecc71")
        elif 12 <= i <= 14: colors.append("#3498db")
        else:               colors.append("#f39c12")
    axes[idx].bar([str(i) for i in all_ids], y, color=colors)
    axes[idx].set_title(f"actionId | {grp} (n={len(subset):,})")
    axes[idx].set_xlabel("actionId")
    axes[idx].set_ylabel("Count")
    log(f"\n  {grp}: top-5 actionId = "
        f"{subset['actionId'].value_counts().head().to_dict()}")
plt.suptitle("actionId Distribution by Strike Number Group", fontsize=14, y=1.01)
plt.tight_layout()
savefig("D8_actionId_by_SN_group.png")

# Item 9 – pointId distribution at different SN groups
log("\n--- Item 9: pointId vs SN groups ---")
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.flatten()
for idx, grp in enumerate(SN_ORDER):
    subset = train[train["sn_group"] == grp]
    counts = subset["pointId"].value_counts().sort_index()
    all_ids = range(0, 10)
    y = [counts.get(i, 0) for i in all_ids]
    axes[idx].bar([str(i) for i in all_ids], y, color="#9b59b6")
    axes[idx].set_title(f"pointId | {grp} (n={len(subset):,})")
    axes[idx].set_xlabel("pointId")
    axes[idx].set_ylabel("Count")
    log(f"\n  {grp}: top-5 pointId = "
        f"{subset['pointId'].value_counts().head().to_dict()}")
plt.suptitle("pointId Distribution by Strike Number Group", fontsize=14, y=1.01)
plt.tight_layout()
savefig("D9_pointId_by_SN_group.png")

# Item 10 – serverGetPoint vs strikeNumber
log("\n--- Item 10: serverGetPoint vs strikeNumber ---")
# Same serverGetPoint for whole rally – look at SN=1 row per rally
rally_sgp = train[train["strikeNumber"] == 1][["rally_uid", "serverGetPoint"]].copy()
rally_sgp_counts = rally_sgp["serverGetPoint"].value_counts().sort_index()
log(f"Rally-level serverGetPoint:\n{rally_sgp_counts.to_string()}")

sgp_by_sn = train.groupby("sn_group")["serverGetPoint"].mean().reindex(SN_ORDER)
log(f"\nMean serverGetPoint by SN group:\n{sgp_by_sn.to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sgp_by_sn.plot(kind="bar", ax=axes[0], color="#e67e22")
axes[0].set_title("Mean serverGetPoint by SN Group")
axes[0].set_ylabel("Mean (proportion of server winning)")
axes[0].set_xlabel("SN Group")
axes[0].tick_params(axis="x", rotation=30)
axes[0].axhline(0.5, color="red", linestyle="--", alpha=0.6)

sgp_sn_raw = train.groupby("strikeNumber")["serverGetPoint"].mean().head(30)
axes[1].plot(sgp_sn_raw.index, sgp_sn_raw.values, marker="o", color="#2c3e50")
axes[1].set_title("Mean serverGetPoint vs Strike Number (first 30)")
axes[1].set_xlabel("Strike Number")
axes[1].set_ylabel("Proportion server wins")
axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.6)
plt.tight_layout()
savefig("D10_serverGetPoint_vs_SN.png")

# ══════════════════════════════════════════════
# E. CONDITIONAL DISTRIBUTIONS
# ══════════════════════════════════════════════
log("\n" + "=" * 70)
log("E. CONDITIONAL DISTRIBUTIONS")
log("=" * 70)

# Item 11 – sex effect
log("\n--- Item 11: sex effect on targets ---")
for target in ["actionId", "pointId", "serverGetPoint"]:
    log(f"\n  {target} by sex:")
    log(train.groupby("sex")[target].value_counts(normalize=True).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, target in zip(axes, ["actionId", "pointId", "serverGetPoint"]):
    grouped = train.groupby(["sex", target]).size().unstack(fill_value=0)
    grouped_norm = grouped.div(grouped.sum(axis=1), axis=0)
    grouped_norm.index = [SEX_LABELS_EN.get(i, str(i)) for i in grouped_norm.index]
    grouped_norm.plot(kind="bar", ax=ax, legend=(target == "serverGetPoint"))
    ax.set_title(f"sex vs {target} (proportion)")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Proportion")
    ax.tick_params(axis="x", rotation=0)
plt.suptitle("Sex Effect on Targets", fontsize=13)
plt.tight_layout()
savefig("E11_sex_vs_targets.png")

# Item 12 – handId / strengthId / spinId / positionId vs targets
log("\n--- Item 12: feature effects on targets ---")
feature_cols = ["handId", "strengthId", "spinId", "positionId"]
feature_label_maps = {
    "handId"    : HAND_LABELS_EN,
    "strengthId": STR_LABELS_EN,
    "spinId"    : SPIN_LABELS_EN,
    "positionId": POS_LABELS_EN,
}

for feat in feature_cols:
    log(f"\n  {feat} value counts: {train[feat].value_counts().to_dict()}")

fig, axes = plt.subplots(4, 3, figsize=(22, 20))
for row_idx, feat in enumerate(feature_cols):
    lmap = feature_label_maps[feat]
    for col_idx, target in enumerate(["actionId", "pointId", "serverGetPoint"]):
        ax = axes[row_idx][col_idx]
        grouped = train.groupby([feat, target]).size().unstack(fill_value=0)
        grouped_norm = grouped.div(grouped.sum(axis=1), axis=0)
        grouped_norm.index = [lmap.get(i, str(i)) for i in grouped_norm.index]
        grouped_norm.plot(kind="bar", ax=ax, legend=False)
        ax.set_title(f"{feat} vs {target}")
        ax.set_xlabel(feat)
        ax.set_ylabel("Proportion")
        ax.tick_params(axis="x", rotation=30)
plt.suptitle("Feature Effects on Targets (handId/strengthId/spinId/positionId)", fontsize=13)
plt.tight_layout()
savefig("E12_feature_effects_on_targets.png")

# Item 13 – score effect
log("\n--- Item 13: score effect on targets ---")
train["score_diff"] = train["scoreSelf"] - train["scoreOther"]
log(f"score_diff stats:\n{train['score_diff'].describe().to_string()}")

# Bin score_diff into categories
def score_diff_bin(d):
    if d <= -3:  return "Behind>=3"
    if d == -2:  return "Behind 2"
    if d == -1:  return "Behind 1"
    if d == 0:   return "Tied"
    if d == 1:   return "Ahead 1"
    if d == 2:   return "Ahead 2"
    return "Ahead>=3"

SCORE_BIN_ORDER = ["Behind>=3", "Behind 2", "Behind 1", "Tied", "Ahead 1", "Ahead 2", "Ahead>=3"]
train["score_diff_bin"] = train["score_diff"].apply(score_diff_bin)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, target in zip(axes, ["actionId", "pointId", "serverGetPoint"]):
    grouped = train.groupby("score_diff_bin")[target].mean().reindex(SCORE_BIN_ORDER)
    grouped.plot(kind="bar", ax=ax, color="#e74c3c")
    ax.set_title(f"Mean {target} by Score Diff")
    ax.set_xlabel("Score Difference (self - other)")
    ax.set_ylabel(f"Mean {target}")
    ax.tick_params(axis="x", rotation=30)
plt.suptitle("Score Difference Effect on Targets", fontsize=13)
plt.tight_layout()
savefig("E13_score_diff_vs_targets.png")

# ══════════════════════════════════════════════
# F. PLAYER ANALYSIS
# ══════════════════════════════════════════════
log("\n" + "=" * 70)
log("F. PLAYER ANALYSIS")
log("=" * 70)

# Item 14 – Player distribution
log("\n--- Item 14: Player distribution ---")
n_players = train["gamePlayerId"].nunique()
log(f"Number of unique players (train): {n_players}")
player_counts = train["gamePlayerId"].value_counts()
log(f"Rows per player – stats:\n{player_counts.describe().to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(player_counts.values, bins=30, color="#16a085", edgecolor="white")
axes[0].set_title(f"Rows per Player (n={n_players} players)")
axes[0].set_xlabel("Number of rows")
axes[0].set_ylabel("Players")
top15 = player_counts.head(15)
axes[1].barh(range(len(top15)), top15.values, color="#2980b9")
axes[1].set_yticks(range(len(top15)))
axes[1].set_yticklabels([f"P{i}" for i in top15.index], fontsize=8)
axes[1].set_title("Top 15 Players by Row Count")
axes[1].set_xlabel("Row count")
plt.tight_layout()
savefig("F14_player_distribution.png")

# Item 15 – Player tactical styles
log("\n--- Item 15: Player tactical style differences ---")
top_players = player_counts.head(10).index.tolist()
player_action_style = (
    train[train["gamePlayerId"].isin(top_players)]
    .groupby("gamePlayerId")["actionId"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)
player_point_style = (
    train[train["gamePlayerId"].isin(top_players)]
    .groupby("gamePlayerId")["pointId"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)
log(f"\nTop-10 player action style (proportion):\n{player_action_style.to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
im1 = axes[0].imshow(player_action_style.values, aspect="auto", cmap="Blues")
axes[0].set_title("Top-10 Players: actionId Proportion Heatmap")
axes[0].set_xticks(range(player_action_style.shape[1]))
axes[0].set_xticklabels([ACTION_LABELS.get(c, str(c)) for c in player_action_style.columns],
                         rotation=75, fontsize=7)
axes[0].set_yticks(range(len(top_players)))
axes[0].set_yticklabels([f"P{p}" for p in top_players], fontsize=8)
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(player_point_style.values, aspect="auto", cmap="Greens")
axes[1].set_title("Top-10 Players: pointId Proportion Heatmap")
axes[1].set_xticks(range(player_point_style.shape[1]))
axes[1].set_xticklabels([POINT_LABELS.get(c, str(c)) for c in player_point_style.columns],
                         rotation=45, fontsize=8)
axes[1].set_yticks(range(len(top_players)))
axes[1].set_yticklabels([f"P{p}" for p in top_players], fontsize=8)
plt.colorbar(im2, ax=axes[1])
plt.tight_layout()
savefig("F15_player_tactical_styles.png")

# ══════════════════════════════════════════════
# G. BOTTLENECK ANALYSIS
# ══════════════════════════════════════════════
log("\n" + "=" * 70)
log("G. BOTTLENECK ANALYSIS")
log("=" * 70)

# Item 16 – Rare class analysis
log("\n--- Item 16: Rare class analysis ---")
action_freq = train["actionId"].value_counts().sort_values()
point_freq  = train["pointId"].value_counts().sort_values()
log(f"\nactionId (ascending frequency):\n{action_freq.to_string()}")
log(f"\npointId (ascending frequency):\n{point_freq.to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
# log scale
bars1 = axes[0].barh(
    [f"{i}: {ACTION_LABELS.get(i,'?')}" for i in action_freq.index],
    action_freq.values, color="#c0392b"
)
axes[0].set_xscale("log")
axes[0].set_title("actionId Frequency Ranking (log scale)")
axes[0].set_xlabel("Count (log)")
axes[0].axvline(100, color="black", linestyle="--", alpha=0.5, label="n=100")
axes[0].legend()

bars2 = axes[1].barh(
    [f"{i}: {POINT_LABELS.get(i,'?')}" for i in point_freq.index],
    point_freq.values, color="#8e44ad"
)
axes[1].set_xscale("log")
axes[1].set_title("pointId Frequency Ranking (log scale)")
axes[1].set_xlabel("Count (log)")
plt.tight_layout()
savefig("G16_rare_class_analysis.png")

# Item 17 – Co-occurrence / confusion patterns
log("\n--- Item 17: Co-occurrence patterns ---")
# action vs point co-occurrence heatmap
cooccur = pd.crosstab(train["actionId"], train["pointId"])
cooccur_norm = cooccur.div(cooccur.sum(axis=1), axis=0)
log(f"\naction->point co-occurrence (row-normalized):\n{cooccur_norm.to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
im1 = axes[0].imshow(cooccur.values, cmap="YlOrRd", aspect="auto",
                      norm=LogNorm(vmin=max(1, cooccur.values.min()),
                                   vmax=cooccur.values.max()))
axes[0].set_title("actionId vs pointId Co-occurrence (raw count, log color)")
axes[0].set_xlabel("pointId")
axes[0].set_ylabel("actionId")
axes[0].set_xticks(range(len(cooccur.columns)))
axes[0].set_xticklabels([POINT_LABELS.get(c, str(c)) for c in cooccur.columns],
                         rotation=45, fontsize=7)
axes[0].set_yticks(range(len(cooccur.index)))
axes[0].set_yticklabels([f"{i}:{ACTION_LABELS.get(i,'?')}" for i in cooccur.index],
                         fontsize=7)
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(cooccur_norm.values, cmap="Blues", aspect="auto",
                      vmin=0, vmax=1)
axes[1].set_title("actionId -> pointId (row-normalized proportion)")
axes[1].set_xlabel("pointId")
axes[1].set_ylabel("actionId")
axes[1].set_xticks(range(len(cooccur_norm.columns)))
axes[1].set_xticklabels([POINT_LABELS.get(c, str(c)) for c in cooccur_norm.columns],
                         rotation=45, fontsize=7)
axes[1].set_yticks(range(len(cooccur_norm.index)))
axes[1].set_yticklabels([f"{i}:{ACTION_LABELS.get(i,'?')}" for i in cooccur_norm.index],
                         fontsize=7)
plt.colorbar(im2, ax=axes[1])
plt.tight_layout()
savefig("G17_cooccurrence_action_point.png")

# Item 18 – Performance by SN groups
log("\n--- Item 18: Data distribution by SN groups ---")
sn_stats = train.groupby("sn_group").agg(
    n_rows=("strikeNumber", "count"),
    n_rallies=("rally_uid", "nunique"),
    mean_action=("actionId", "mean"),
    mean_point=("pointId", "mean"),
    mean_sgp=("serverGetPoint", "mean"),
).reindex(SN_ORDER)
log(f"\n{sn_stats.to_string()}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sn_stats["n_rows"].plot(kind="bar", ax=axes[0], color="#2980b9")
axes[0].set_title("Rows per SN Group")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=30)

sn_stats["mean_action"].plot(kind="bar", ax=axes[1], color="#e74c3c")
axes[1].set_title("Mean actionId per SN Group")
axes[1].set_ylabel("Mean actionId")
axes[1].tick_params(axis="x", rotation=30)

sn_stats["mean_point"].plot(kind="bar", ax=axes[2], color="#8e44ad")
axes[2].set_title("Mean pointId per SN Group")
axes[2].set_ylabel("Mean pointId")
axes[2].tick_params(axis="x", rotation=30)
plt.tight_layout()
savefig("G18_sn_group_stats.png")

# Item 19 – Per-task difficulty at different SN ranges
log("\n--- Item 19: Per-task difficulty at different SN ranges ---")
# Approximate difficulty = entropy of class distribution
from scipy.stats import entropy

def class_entropy(series):
    counts = series.value_counts()
    probs = counts / counts.sum()
    return entropy(probs, base=2)

difficulty_rows = []
for grp in SN_ORDER:
    subset = train[train["sn_group"] == grp]
    difficulty_rows.append({
        "sn_group": grp,
        "n_rows": len(subset),
        "actionId_entropy": class_entropy(subset["actionId"]),
        "pointId_entropy" : class_entropy(subset["pointId"]),
        "sgp_entropy"     : class_entropy(subset["serverGetPoint"]),
        "actionId_nclass" : subset["actionId"].nunique(),
        "pointId_nclass"  : subset["pointId"].nunique(),
    })
difficulty_df = pd.DataFrame(difficulty_rows).set_index("sn_group")
log(f"\n{difficulty_df.to_string()}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
difficulty_df["actionId_entropy"].plot(kind="bar", ax=axes[0], color="#e74c3c")
axes[0].set_title("actionId Entropy per SN Group\n(higher = harder)")
axes[0].set_ylabel("Entropy (bits)")
axes[0].tick_params(axis="x", rotation=30)

difficulty_df["pointId_entropy"].plot(kind="bar", ax=axes[1], color="#8e44ad")
axes[1].set_title("pointId Entropy per SN Group\n(higher = harder)")
axes[1].set_ylabel("Entropy (bits)")
axes[1].tick_params(axis="x", rotation=30)

difficulty_df["sgp_entropy"].plot(kind="bar", ax=axes[2], color="#27ae60")
axes[2].set_title("serverGetPoint Entropy per SN Group")
axes[2].set_ylabel("Entropy (bits)")
axes[2].tick_params(axis="x", rotation=30)
plt.tight_layout()
savefig("G19_per_task_difficulty.png")

# ══════════════════════════════════════════════
# H. TRAIN / TEST DISTRIBUTION SHIFT
# ══════════════════════════════════════════════
log("\n" + "=" * 70)
log("H. TRAIN / TEST DISTRIBUTION SHIFT")
log("=" * 70)

# Item 20 – Compare train vs test distributions
log("\n--- Item 20: Train vs Test distribution comparison ---")

feature_compare = [
    "sex", "strikeNumber", "scoreSelf", "scoreOther",
    "strikeId", "handId", "strengthId", "spinId", "positionId",
    "numberGame",
]

# Test has target columns too (filled from submission context) – check
log(f"\nTest columns: {list(test.columns)}")
log(f"Train target in test: {[c for c in ['actionId','pointId','serverGetPoint'] if c in test.columns]}")

fig, axes = plt.subplots(4, 3, figsize=(20, 22))
axes = axes.flatten()
for idx, col in enumerate(feature_compare):
    ax = axes[idx]
    if col not in train.columns or col not in test.columns:
        ax.set_visible(False)
        continue
    # Normalize
    train_vc = train[col].value_counts(normalize=True).sort_index()
    test_vc  = test[col].value_counts(normalize=True).sort_index()
    all_vals = sorted(set(train_vc.index) | set(test_vc.index))
    tr = [train_vc.get(v, 0) for v in all_vals]
    te = [test_vc.get(v, 0) for v in all_vals]
    x = np.arange(len(all_vals))
    w = 0.35
    ax.bar(x - w/2, tr, w, label="Train", color="#3498db", alpha=0.8)
    ax.bar(x + w/2, te, w, label="Test",  color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in all_vals], fontsize=6)
    ax.set_title(f"{col} distribution")
    ax.set_ylabel("Proportion")
    ax.legend(fontsize=7)
    # KL divergence (approx)
    tr_arr = np.array(tr) + 1e-10
    te_arr = np.array(te) + 1e-10
    kl = np.sum(tr_arr * np.log(tr_arr / te_arr))
    ax.set_xlabel(f"KL(train||test)={kl:.4f}")

for idx in range(len(feature_compare), len(axes)):
    axes[idx].set_visible(False)
plt.suptitle("Train vs Test Feature Distributions", fontsize=14, y=1.005)
plt.tight_layout()
savefig("H20_train_test_distribution_shift.png")

# Summary KL stats
log("\nKL(train||test) per feature:")
from scipy.special import kl_div
for col in feature_compare:
    if col not in train.columns or col not in test.columns:
        continue
    train_vc = train[col].value_counts(normalize=True).sort_index()
    test_vc  = test[col].value_counts(normalize=True).sort_index()
    all_vals = sorted(set(train_vc.index) | set(test_vc.index))
    tr_arr = np.array([train_vc.get(v, 0) for v in all_vals]) + 1e-10
    te_arr = np.array([test_vc.get(v, 0) for v in all_vals])  + 1e-10
    tr_arr /= tr_arr.sum()
    te_arr /= te_arr.sum()
    kl = float(np.sum(tr_arr * np.log(tr_arr / te_arr)))
    log(f"  {col:20s}: KL = {kl:.5f}")

# ══════════════════════════════════════════════
# I. FEATURE ENGINEERING SUPPORT
# ══════════════════════════════════════════════
log("\n" + "=" * 70)
log("I. FEATURE ENGINEERING SUPPORT")
log("=" * 70)

# Sort data to ensure correct temporal order
train_sorted = train.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)

# Item 21 – Transition matrices
log("\n--- Item 21: Transition matrices ---")

def build_transition_matrix(df, from_col, to_col, n_from, n_to):
    """Build a transition matrix from consecutive rows within same rally."""
    df2 = df.copy()
    df2["_next_rally"] = df2["rally_uid"].shift(-1)
    df2["_next_val"]   = df2[to_col].shift(-1)
    # Only keep within same rally
    valid = df2["rally_uid"] == df2["_next_rally"]
    pairs = df2.loc[valid, [from_col, "_next_val"]]
    pairs.columns = ["from", "to"]
    mat = pd.crosstab(pairs["from"], pairs["to"])
    # Reindex to full range
    mat = mat.reindex(index=range(n_from), columns=range(n_to), fill_value=0)
    return mat

# action -> action
aa_mat = build_transition_matrix(train_sorted, "actionId", "actionId", 19, 19)
# point -> point
pp_mat = build_transition_matrix(train_sorted, "pointId", "pointId", 10, 10)
# action -> point
ap_mat = build_transition_matrix(train_sorted, "actionId", "pointId", 19, 10)

log(f"\naction->action transition matrix shape: {aa_mat.shape}")
log(f"\nTop transitions (action->action):")
aa_norm = aa_mat.div(aa_mat.sum(axis=1).replace(0, 1), axis=0)
log(aa_norm.to_string())

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, mat, title, row_labels, col_labels in [
    (axes[0], aa_norm,
     "action -> action transition\n(row-normalized)",
     [f"{i}:{ACTION_LABELS.get(i,'?')}" for i in aa_norm.index],
     [str(c) for c in aa_norm.columns]),
    (axes[1], pp_mat.div(pp_mat.sum(axis=1).replace(0, 1), axis=0),
     "point -> point transition\n(row-normalized)",
     [f"{i}:{POINT_LABELS.get(i,'?')}" for i in pp_mat.index],
     [f"{c}:{POINT_LABELS.get(c,'?')}" for c in pp_mat.columns]),
    (axes[2], ap_mat.div(ap_mat.sum(axis=1).replace(0, 1), axis=0),
     "action -> point transition\n(row-normalized)",
     [f"{i}:{ACTION_LABELS.get(i,'?')}" for i in ap_mat.index],
     [f"{c}:{POINT_LABELS.get(c,'?')}" for c in ap_mat.columns]),
]:
    im = ax.imshow(mat.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(col_labels, rotation=75, fontsize=5)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=5)
    plt.colorbar(im, ax=ax)
plt.suptitle("Transition Matrices", fontsize=13)
plt.tight_layout()
savefig("I21_transition_matrices.png")

# Item 22 – Bigram / Trigram frequency tables
log("\n--- Item 22: Bigram/Trigram frequency tables ---")

def build_ngrams(df, col, n=2):
    """Build n-gram sequences within each rally."""
    ngrams = []
    for _, grp in df.groupby("rally_uid"):
        seq = grp.sort_values("strikeNumber")[col].tolist()
        for i in range(len(seq) - n + 1):
            ngrams.append(tuple(seq[i:i+n]))
    return Counter(ngrams)

action_bigrams  = build_ngrams(train_sorted, "actionId", 2)
action_trigrams = build_ngrams(train_sorted, "actionId", 3)
point_bigrams   = build_ngrams(train_sorted, "pointId", 2)

top_ab = action_bigrams.most_common(20)
top_at = action_trigrams.most_common(15)
top_pb = point_bigrams.most_common(20)

log("\nTop-20 action bigrams:")
for pair, cnt in top_ab:
    lbl = " -> ".join([ACTION_LABELS.get(p, str(p)) for p in pair])
    log(f"  {pair}: {cnt:6d}  ({lbl})")

log("\nTop-15 action trigrams:")
for tri, cnt in top_at:
    lbl = " -> ".join([ACTION_LABELS.get(p, str(p)) for p in tri])
    log(f"  {tri}: {cnt:6d}  ({lbl})")

log("\nTop-20 point bigrams:")
for pair, cnt in top_pb:
    lbl = " -> ".join([POINT_LABELS.get(p, str(p)) for p in pair])
    log(f"  {pair}: {cnt:6d}  ({lbl})")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
# Action bigrams
ab_labels = [f"{a}->{b}" for a, b in [x[0] for x in top_ab]]
ab_counts = [x[1] for x in top_ab]
axes[0].barh(ab_labels[::-1], ab_counts[::-1], color="#e74c3c")
axes[0].set_title("Top-20 Action Bigrams")
axes[0].set_xlabel("Count")
axes[0].tick_params(axis="y", labelsize=7)

# Action trigrams
at_labels = [f"{a}->{b}->{c}" for a, b, c in [x[0] for x in top_at]]
at_counts = [x[1] for x in top_at]
axes[1].barh(at_labels[::-1], at_counts[::-1], color="#e67e22")
axes[1].set_title("Top-15 Action Trigrams")
axes[1].set_xlabel("Count")
axes[1].tick_params(axis="y", labelsize=7)

# Point bigrams
pb_labels = [f"{a}->{b}" for a, b in [x[0] for x in top_pb]]
pb_counts = [x[1] for x in top_pb]
axes[2].barh(pb_labels[::-1], pb_counts[::-1], color="#8e44ad")
axes[2].set_title("Top-20 Point Bigrams")
axes[2].set_xlabel("Count")
axes[2].tick_params(axis="y", labelsize=7)
plt.suptitle("Bigram / Trigram Frequencies", fontsize=13)
plt.tight_layout()
savefig("I22_bigram_trigram_frequency.png")

# Item 23 – Transition patterns at different SN ranges
log("\n--- Item 23: Transition patterns at different SN ranges ---")

sn_transitions = {}
for grp in SN_ORDER:
    subset = train_sorted[train_sorted["sn_group"] == grp]
    mat = build_transition_matrix(subset, "actionId", "actionId", 19, 19)
    mat_norm = mat.div(mat.sum(axis=1).replace(0, 1), axis=0)
    sn_transitions[grp] = mat_norm

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
axes = axes.flatten()
for idx, grp in enumerate(SN_ORDER):
    ax = axes[idx]
    mat = sn_transitions[grp]
    im = ax.imshow(mat.values, cmap="Blues", aspect="auto", vmin=0, vmax=0.8)
    ax.set_title(f"action->action transition | {grp}", fontsize=9)
    ax.set_xlabel("Next actionId")
    ax.set_ylabel("Current actionId")
    ax.set_xticks(range(0, 19, 2))
    ax.set_xticklabels(range(0, 19, 2), fontsize=7)
    ax.set_yticks(range(0, 19, 2))
    ax.set_yticklabels(range(0, 19, 2), fontsize=7)
    plt.colorbar(im, ax=ax)
plt.suptitle("Action->Action Transition by SN Group", fontsize=14, y=1.005)
plt.tight_layout()
savefig("I23a_action_transition_by_SN_group.png")

# Also: point->point transition by SN group
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
axes = axes.flatten()
for idx, grp in enumerate(SN_ORDER):
    ax = axes[idx]
    subset = train_sorted[train_sorted["sn_group"] == grp]
    mat = build_transition_matrix(subset, "pointId", "pointId", 10, 10)
    mat_norm = mat.div(mat.sum(axis=1).replace(0, 1), axis=0)
    im = ax.imshow(mat_norm.values, cmap="Greens", aspect="auto", vmin=0, vmax=0.8)
    ax.set_title(f"point->point transition | {grp}", fontsize=9)
    ax.set_xlabel("Next pointId")
    ax.set_ylabel("Current pointId")
    ax.set_xticks(range(10))
    ax.set_xticklabels([POINT_LABELS.get(c, str(c)) for c in range(10)],
                        rotation=45, fontsize=6)
    ax.set_yticks(range(10))
    ax.set_yticklabels([POINT_LABELS.get(c, str(c)) for c in range(10)], fontsize=6)
    plt.colorbar(im, ax=ax)
plt.suptitle("Point->Point Transition by SN Group", fontsize=14, y=1.005)
plt.tight_layout()
savefig("I23b_point_transition_by_SN_group.png")

# ──────────────────────────────────────────────
# Final summary
# ──────────────────────────────────────────────
log("\n" + "=" * 70)
log("SUMMARY – KEY FINDINGS")
log("=" * 70)

# Class imbalance metrics
def imbalance_ratio(series):
    vc = series.value_counts()
    return vc.max() / vc.min()

log(f"\nClass imbalance ratios (max/min count):")
log(f"  actionId         : {imbalance_ratio(train['actionId']):.1f}x")
log(f"  pointId          : {imbalance_ratio(train['pointId']):.1f}x")
log(f"  serverGetPoint   : {imbalance_ratio(train['serverGetPoint']):.2f}x")

log(f"\nTotal rallies in train: {train['rally_uid'].nunique():,}")
log(f"Total rallies in test : {test['rally_uid'].nunique():,}")
log(f"Unique matches train  : {train['match'].nunique():,}")
log(f"Unique matches test   : {test['match'].nunique():,}")
log(f"Unique players train  : {train['gamePlayerId'].nunique():,}")
log(f"Unique players test   : {test['gamePlayerId'].nunique():,}")

log(f"\nClass counts for actionId (train):")
ac = train["actionId"].value_counts().sort_index()
for i, cnt in ac.items():
    log(f"  {i:2d} {ACTION_LABELS.get(i,'?'):20s}: {cnt:6d}  ({100*cnt/len(train):.2f}%)")

log(f"\nClass counts for pointId (train):")
pc = train["pointId"].value_counts().sort_index()
for i, cnt in pc.items():
    log(f"  {i:2d} {POINT_LABELS.get(i,'?'):20s}: {cnt:6d}  ({100*cnt/len(train):.2f}%)")

log(f"\nOutput files saved to: {OUT_DIR}")
log("\nEDA complete.")

save_log()
print(f"\nAll outputs saved to: {OUT_DIR}")
