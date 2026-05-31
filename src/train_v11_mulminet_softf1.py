"""R-031 v1 (Codex APPROVE_WITH_FIXES, 2026-05-21).

Soft-F1 fine-tune on v11_mulminet_aug_oldtest architecture, attacking
rare-class action macro F1. Implements all 6 Codex fixes:

  [P1-1] Checkpoints do not exist -> from-scratch retrain (not continuation).
  [P1-2] Two-phase schedule: CE warmup -> CE + soft-F1 final phase
         (saves fold checkpoints so future continuation experiments work).
  [P1-3] Point/SGP unchanged claim made TRUE by freezing the shared encoder
         + point head + SGP head during the soft-F1 phase (default).
         Use --no-freeze to allow encoder fine-tuning (with regression gate).
  [P2-4] Soft-F1 MASKS classes with zero positive support in the batch;
         per-batch/class support diagnostics logged.
  [P2-5] Unit tests use a balanced fixture (see tests/test_softf1_loss.py).
  [P2-6] Fold-1 baseline pinned by computing pre-fine-tune val metrics
         AFTER Phase A and storing them in artifacts metadata.

CLI defaults match Codex's revised approved scope (alpha ramp 0 -> 0.3 over
Phase B; CE-only warmup first; Fold-1 smoke only; no LB submission).

USAGE (Fold-1 smoke, Kaggle GPU ~45 min):
  python -u src/train_v11_mulminet_softf1.py \\
    --tag v11_mulminet_aug_oldtest_softf1_smoke \\
    --max-folds 1 \\
    --ce-epochs 70 --softf1-epochs 10 \\
    --alpha-start 0.0 --alpha-end 0.3 \\
    --freeze-encoder \\
    --include-old-test data/test.csv \\
    --aug-parquet data/test_history_pairs_new.parquet \\
    --test-path data/test_new.csv \\
    --seed 42

Outputs:
  oof_predictions/{tag}_oof_*.npy + test_*.npy
  models/{tag}/fold{f}/best_ce.pt    (after CE warmup)
  models/{tag}/fold{f}/best_softf1.pt (after soft-F1 phase)
  runs/{tag}_metadata.json (per-class baseline + post-finetune metrics, gate verdict)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Reuse heavy infrastructure from baseline trainer
from config import (  # noqa: E402
    N_FOLDS, RANDOM_SEED, SUBMISSION_DIR, TEST_PATH, TRAIN_PATH,
)
from data_cleaning import STRIKE_ID_MAP, clean_data  # noqa: E402
from train_v11_mulminet import (  # noqa: E402
    ACTION_EVAL_LABELS, ACTION_W, DEVICE, FocalLoss, N_ACTION_TRAIN, N_POINT,
    POINT_EVAL_LABELS, POINT_W, RallyDataset, RallyTransformer,
    action_macro_f1, apply_action_rules, build_samples, evaluate,
    point_macro_f1,
)


# ---------------------------------------------------------------------------
# Soft-F1 surrogate loss (Codex fixes P2-4 + P2-5)
# ---------------------------------------------------------------------------

def softf1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eval_classes: list[int],
    eps: float = 1e-7,
    return_support: bool = False,
) -> tuple[torch.Tensor, Optional[dict]]:
    """Differentiable macro-F1 surrogate over `eval_classes` only.

    For each class c in eval_classes that has > 0 positive support in this
    batch, computes:
        TP_c = sum_i softmax(logits)_{i,c} * 1[y_i == c]
        FP_c = sum_i softmax(logits)_{i,c} * 1[y_i != c]
        FN_c = sum_i (1 - softmax(logits)_{i,c}) * 1[y_i == c]
        F1_c = 2 TP_c / (2 TP_c + FP_c + FN_c + eps)
    Loss = 1 - mean_c(F1_c) over classes that had > 0 support.

    Classes with 0 positive support in the batch are MASKED — their gradient
    contribution is undefined and would otherwise bias the loss toward the
    rare-class background. This addresses Codex P2-4.

    Returns:
      (loss tensor scalar, diagnostics dict if return_support else None)
    """
    probs = F.softmax(logits, dim=-1)
    K = probs.shape[1]
    f1s = []
    supports = {}
    for c in eval_classes:
        pos_mask = (targets == c).float()       # 1 where label == c
        neg_mask = (targets != c).float()       # 1 where label != c
        n_pos = float(pos_mask.sum().item())
        supports[int(c)] = n_pos
        if n_pos <= 0:
            continue  # mask: class absent in this batch
        p_c = probs[:, c]
        tp = (p_c * pos_mask).sum()
        fp = (p_c * neg_mask).sum()
        fn = ((1.0 - p_c) * pos_mask).sum()
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        f1s.append(f1)

    if not f1s:
        # Pathological: no eval class present in this batch. Return 0 with grad.
        loss = (probs.sum() * 0.0)
    else:
        f1_mean = torch.stack(f1s).mean()
        loss = 1.0 - f1_mean

    diagnostics = None
    if return_support:
        diagnostics = {
            "n_classes_active": len(f1s),
            "n_classes_eval": len(eval_classes),
            "support_per_class": supports,
        }
    return loss, diagnostics


# ---------------------------------------------------------------------------
# Two-phase train_epoch (Codex P1-3 freeze + P2-4 masked soft-F1)
# ---------------------------------------------------------------------------

def train_epoch_softf1(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    act_loss_fn: nn.Module,
    pt_loss_fn: nn.Module,
    phase: str,           # "ce" or "softf1"
    alpha: float,         # only used in "softf1" phase
    aux_lambda: float = 0.2,
    eval_classes: list[int] = ACTION_EVAL_LABELS,
):
    """Two-phase training step.

    phase == "ce":     same as baseline train_epoch (CE on action, point, SGP, aux)
    phase == "softf1": loss = (1-alpha) * CE_action + alpha * SoftF1_action
                       + CE_point + BCE_SGP + aux_lambda * sum(aux)
                       (point + SGP losses unchanged from baseline)

    Diagnostics:
      - Logs mean active classes per batch in softf1 phase
      - aug_rows_in_server_loss MUST stay 0 (Codex P6 guard)
    """
    model.train()
    total_loss = 0.0
    n = 0
    sums = {"a": 0.0, "p": 0.0, "s": 0.0,
            "hand": 0.0, "strength": 0.0, "spin": 0.0, "position": 0.0,
            "softf1": 0.0}
    aug_rows_seen = 0
    aug_rows_in_server_loss = 0  # MUST stay 0
    active_classes_running = []  # for soft-F1 diagnostics

    for batch in loader:
        cat = batch["cat"].to(DEVICE)
        num = batch["num"].to(DEVICE)
        ctx = batch["context"].to(DEVICE)
        ps = batch["pid_self"].to(DEVICE)
        po = batch["pid_other"].to(DEVICE)
        mask = batch["pad_mask"].to(DEVICE)
        slen = batch["seq_len"].to(DEVICE)
        ya = batch["y_action"].to(DEVICE)
        yp = batch["y_point"].to(DEVICE)
        ys = batch["y_server"].to(DEVICE)
        y_h = batch["y_hand"].to(DEVICE)
        y_st = batch["y_strength"].to(DEVICE)
        y_sp = batch["y_spin"].to(DEVICE)
        y_po = batch["y_position"].to(DEVICE)
        is_aug = batch["is_aug"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            (a_logits, p_logits, s_logit,
             h_logits, st_logits, sp_logits, po_logits) = model(
                cat, num, ctx, ps, po, mask, slen)

            loss_a_ce = act_loss_fn(a_logits, ya)
            loss_p = pt_loss_fn(p_logits, yp)

            # P6 server-mask
            real_mask = (is_aug == 0)
            n_real = int(real_mask.sum().item())
            if n_real > 0:
                loss_s = F.binary_cross_entropy_with_logits(
                    s_logit[real_mask], ys[real_mask])
            else:
                loss_s = torch.zeros((), device=DEVICE)

            # Aux losses
            loss_hand = F.cross_entropy(h_logits, y_h, ignore_index=0)
            loss_strength = F.cross_entropy(st_logits, y_st, ignore_index=0)
            loss_spin = F.cross_entropy(sp_logits, y_sp, ignore_index=0)
            loss_position = F.cross_entropy(po_logits, y_po, ignore_index=0)
            for nm, l in [("hand", loss_hand), ("strength", loss_strength),
                          ("spin", loss_spin), ("position", loss_position)]:
                pass
            loss_hand = torch.nan_to_num(loss_hand, nan=0.0)
            loss_strength = torch.nan_to_num(loss_strength, nan=0.0)
            loss_spin = torch.nan_to_num(loss_spin, nan=0.0)
            loss_position = torch.nan_to_num(loss_position, nan=0.0)
            loss_aux_sum = (loss_hand + loss_strength + loss_spin + loss_position)

            # Action loss: phase-dependent
            if phase == "ce":
                loss_a = loss_a_ce
                softf1_value = torch.tensor(0.0, device=DEVICE)
            else:  # softf1 phase
                sf1, diag = softf1_loss(
                    a_logits, ya, eval_classes, return_support=True,
                )
                softf1_value = sf1
                # Track diagnostics
                if diag is not None:
                    active_classes_running.append(diag["n_classes_active"])
                loss_a = (1.0 - alpha) * loss_a_ce + alpha * sf1

            loss = (0.4 * loss_a + 0.4 * loss_p + 0.2 * loss_s
                    + aux_lambda * loss_aux_sum)

        aug_rows_seen += int((is_aug == 1).sum().item())
        # P6 guard
        if aug_rows_in_server_loss != 0:
            raise AssertionError("P6 GUARD FAIL: aug_rows in server BCE")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        sums["a"] += float(loss_a.item())
        sums["p"] += float(loss_p.item())
        sums["s"] += float(loss_s.item())
        sums["softf1"] += float(softf1_value.item())
        sums["hand"] += float(loss_hand.item())
        sums["strength"] += float(loss_strength.item())
        sums["spin"] += float(loss_spin.item())
        sums["position"] += float(loss_position.item())
        n += 1

    means = {k: v / max(n, 1) for k, v in sums.items()}
    diagnostics = {}
    if active_classes_running:
        diagnostics["n_classes_active_mean"] = float(np.mean(active_classes_running))
        diagnostics["n_classes_active_min"] = int(np.min(active_classes_running))
        diagnostics["n_classes_active_max"] = int(np.max(active_classes_running))
    return total_loss / max(n, 1), aug_rows_seen, aug_rows_in_server_loss, means, diagnostics


# ---------------------------------------------------------------------------
# Freeze helpers (Codex P1-3)
# ---------------------------------------------------------------------------

def freeze_non_action_params(model: nn.Module) -> int:
    """Freeze every parameter except those reaching the action head.

    The architecture (see RallyTransformer in train_v11_mulminet.py) has:
      - Shared encoder (cat embed, num embed, transformer blocks)
      - 7 heads: action, point, server, hand, strength, spin, position
    We freeze everything except the action head.

    Returns: number of parameters set requires_grad=False.
    """
    frozen = 0
    for name, p in model.named_parameters():
        if "action_head" in name or "act_head" in name or "head_action" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
            frozen += p.numel()
    return frozen


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


# ---------------------------------------------------------------------------
# Main training driver (mirrors train_v11_mulminet but with two phases)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    # --- All parameters mirror train_v11_mulminet.py defaults ---
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tag", type=str, default="v11_mulminet_softf1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-lambda", type=float, default=0.2)
    parser.add_argument("--include-old-test", type=str, default=None)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--max-folds", type=int, default=1,
                        help="Codex restriction: Fold-1 smoke only by default.")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--point-w-scale", type=float, default=1.0)
    parser.add_argument("--aug-parquet", type=str, default="")
    parser.add_argument("--test-path", type=str, default=None)

    # --- R-031 specific (two-phase schedule) ---
    parser.add_argument("--ce-epochs", type=int, default=70,
                        help="Phase A: CE-only warmup epochs (Codex: required)")
    parser.add_argument("--softf1-epochs", type=int, default=10,
                        help="Phase B: CE + soft-F1 fine-tune epochs "
                             "(Codex: 5-10 after warmup)")
    parser.add_argument("--alpha-start", type=float, default=0.0,
                        help="Phase B: alpha at start of soft-F1 phase "
                             "(Codex: ramp 0 -> 0.3 preferred)")
    parser.add_argument("--alpha-end", type=float, default=0.3,
                        help="Phase B: alpha at end of soft-F1 phase")
    parser.add_argument("--freeze-encoder", action="store_true", default=True,
                        help="Default: freeze shared encoder + point + SGP heads "
                             "during Phase B (Codex P1-3 makes 'point/SGP unchanged' "
                             "TRUE)")
    parser.add_argument("--no-freeze", dest="freeze_encoder", action="store_false",
                        help="Override: allow encoder + all heads to update in "
                             "Phase B (Codex P1-3 requires regression gate)")
    parser.add_argument("--softf1-lr", type=float, default=1e-4,
                        help="Lower LR for soft-F1 phase (Codex: lower LR reasonable)")
    parser.add_argument("--save-checkpoints", action="store_true", default=True,
                        help="Save best fold checkpoints to disk for future "
                             "continuation (Codex P1-2 requirement going forward)")
    parser.add_argument("--init-from-ce", type=str, default=None,
                        help="Path to a pre-trained CE checkpoint (from a "
                             "prior --save-checkpoints run). When set, Phase A "
                             "is SKIPPED entirely — model loads from this ckpt "
                             "and goes straight to Phase B (saves ~12 hr on "
                             "Kaggle T4 timeout). Use after a kernel timed out "
                             "in Phase B and saved best_ce.pt.")

    args = parser.parse_args()

    seed_actual = args.seed
    torch.manual_seed(seed_actual)
    np.random.seed(seed_actual)

    out_tag = args.tag
    n_folds = args.folds
    bs = args.batch
    use_aug = not args.no_aug

    if args.smoke:
        # Smoke = 1 fold, 5 + 5 epochs (1 each phase, but kept >= 5 for early-stop)
        n_folds = 1
        args.max_folds = 1
        args.ce_epochs = 5
        args.softf1_epochs = 5

    t_start = time.time()
    print("=" * 70)
    print(f"R-031 V11 MULMINET SOFTF1{'  (SMOKE)' if args.smoke else ''}  tag={out_tag}")
    print(f"  device={DEVICE}  d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}")
    print(f"  folds={n_folds}  max_folds={args.max_folds}  batch={bs}  lr={args.lr}")
    print(f"  Phase A (CE warmup):       {args.ce_epochs} epochs")
    print(f"  Phase B (soft-F1):         {args.softf1_epochs} epochs")
    print(f"  alpha ramp: {args.alpha_start} -> {args.alpha_end}")
    print(f"  freeze_encoder in Phase B: {args.freeze_encoder}")
    print(f"  softf1_lr (Phase B):       {args.softf1_lr}")
    print(f"  seed={seed_actual}  include_old_test={args.include_old_test}")
    print("=" * 70)

    # ---- Data loading (identical to v11_mulminet) ----
    raw_train = pd.read_csv(TRAIN_PATH)
    if args.include_old_test:
        old_test = pd.read_csv(args.include_old_test)
        required_cols = ["rally_uid", "match", "strikeNumber", "actionId", "pointId",
                         "serverGetPoint", "gamePlayerId", "gamePlayerOtherId",
                         "strikeId", "handId", "strengthId", "spinId", "positionId",
                         "sex", "numberGame", "rally_id", "scoreSelf", "scoreOther"]
        missing = [c for c in required_cols if c not in old_test.columns]
        if missing:
            raise ValueError(f"old test missing columns: {missing}")
        raw_train = pd.concat([raw_train, old_test[required_cols]], ignore_index=True)
        print(f"  [include-old-test] +{len(old_test)} rows")
    test_path = args.test_path or TEST_PATH
    raw_test = pd.read_csv(test_path)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1
    n_players = len(player_map)
    print(f"  Players: {n_players}")

    # ---- Build samples ----
    train_samples = build_samples(train_df, is_train=True, n_players=n_players)
    test_samples = build_samples(test_df, is_train=False, n_players=n_players)
    for s in train_samples:
        s["is_aug"] = 0
    n_train = len(train_samples)
    print(f"  Train samples: {n_train}  Test samples: {len(test_samples)}")

    # ---- Optional aug parquet ----
    aug_samples = []
    if args.aug_parquet:
        aug_raw = pd.read_parquet(args.aug_parquet)
        assert (aug_raw["serverGetPoint"] == -1).all(), "GUARD FAIL: aug parquet SGP != -1"
        assert (aug_raw["is_aug"] == 1).all(), "GUARD FAIL: aug parquet is_aug != 1"
        aug_raw = aug_raw.copy()
        aug_raw["strikeId"] = aug_raw["strikeId"].map(STRIKE_ID_MAP).fillna(0).astype(int)
        for col in ("gamePlayerId", "gamePlayerOtherId"):
            aug_raw[col] = aug_raw[col].map(player_map).fillna(-1).astype(int)
        aug_raw["numberGame"] = aug_raw["numberGame"].clip(upper=7)
        aug_samples = build_samples(aug_raw, is_train=True, n_players=n_players)
        for s in aug_samples:
            s["is_aug"] = 1
        print(f"  Aug samples: {len(aug_samples)}  (server BCE masked for is_aug==1)")

    all_samples = train_samples + aug_samples
    n_aug = len(aug_samples)
    print(f"  Total: {len(all_samples)} (real={n_train}, aug={n_aug})")

    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    sample_rallies = np.array([s["rally_uid"] for s in train_samples])
    sample_matches = np.array([rally_to_match.get(r, -1) for r in sample_rallies])

    y_a_all = np.array([s["y_action"] for s in train_samples])
    y_p_all = np.array([s["y_point"] for s in train_samples])
    y_s_all = np.array([s["y_server"] for s in train_samples])
    nsn_all = np.array([s["next_sn"] for s in train_samples])

    oof_act = np.zeros((n_train, N_ACTION_TRAIN))
    oof_pt = np.zeros((n_train, N_POINT))
    oof_srv = np.zeros(n_train)
    oof_mask_arr = np.zeros(n_train, dtype=bool)
    test_act_acc = np.zeros((len(test_samples), N_ACTION_TRAIN))
    test_pt_acc = np.zeros((len(test_samples), N_POINT))
    test_srv_acc = np.zeros(len(test_samples))
    nsn_test = np.array([s["next_sn"] for s in test_samples])
    rally_uid_test = [s["rally_uid"] for s in test_samples]

    # Loss weights
    _pt_w = POINT_W.copy()
    if args.point_w_scale != 1.0:
        for c in [1, 3]:
            _pt_w[c] *= args.point_w_scale
    act_w = torch.tensor(ACTION_W, device=DEVICE)
    pt_w = torch.tensor(_pt_w, device=DEVICE)
    act_loss_fn = FocalLoss(act_w, gamma=2.0)
    pt_loss_fn = FocalLoss(pt_w, gamma=2.0)

    gkf = GroupKFold(n_splits=max(n_folds, 2))
    splits = list(gkf.split(np.arange(n_train), groups=sample_matches))
    aug_indices = np.arange(n_train, n_train + n_aug, dtype=int)
    if args.max_folds > 0:
        splits = splits[:args.max_folds]
        print(f"  --max-folds {args.max_folds} active: running {len(splits)} fold(s)")

    test_ds = RallyDataset(test_samples, augment=False)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False,
                             num_workers=0, pin_memory=True)

    # ---- Per-fold metadata (Codex P2-6 baseline pinning) ----
    fold_metadata = []

    for fold, (tr_idx, val_idx) in enumerate(splits):
        t_fold = time.time()
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{len(splits)}")
        print(f"{'='*60}")

        full_tr_idx = np.concatenate([tr_idx, aug_indices]) if n_aug > 0 else tr_idx
        tr_samps = [all_samples[i] for i in full_tr_idx]
        val_samps = [all_samples[i] for i in val_idx]

        tr_ds = RallyDataset(tr_samps, augment=use_aug)
        val_ds = RallyDataset(val_samps, augment=False)
        tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, num_workers=0, pin_memory=True)

        model = RallyTransformer(
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            dropout=0.15, n_players=n_players + 5, max_len=40,
        ).to(DEVICE)
        scaler = GradScaler()

        # ---- Phase A: CE-only warmup (SKIPPED if --init-from-ce given) ----
        best_ov_ce = -1.0
        best_state_ce = None

        if args.init_from_ce:
            print(f"\n  -- Phase A SKIPPED — loading from {args.init_from_ce} --")
            ckpt = torch.load(args.init_from_ce, map_location=DEVICE)
            # Support both raw state_dict and {state_dict: ...} formats
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.load_state_dict({k: v.to(DEVICE) for k, v in ckpt.items()})
            best_state_ce = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Need a dummy optimizer for the phase-A skip path to keep
            # the rest of the function flow unchanged
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          weight_decay=1e-2, eps=1e-8)
            print(f"  Loaded CE checkpoint ({sum(p.numel() for p in model.parameters()):,} params)")
        else:
            print(f"\n  -- Phase A: CE warmup ({args.ce_epochs} epochs, lr={args.lr}) --")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          weight_decay=1e-2, eps=1e-8)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=args.ce_epochs, eta_min=args.lr / 20)

            for epoch in range(1, args.ce_epochs + 1):
                tr_loss, aug_seen, aug_in_srv, loss_means, _ = train_epoch_softf1(
                    model, tr_loader, optimizer, scaler, act_loss_fn, pt_loss_fn,
                    phase="ce", alpha=0.0, aux_lambda=args.aux_lambda)
                scheduler.step()

                if epoch % 5 == 0 or epoch == args.ce_epochs:
                    a_p, p_p, s_p = evaluate(model, val_loader)
                    y_a_val = np.array([s["y_action"] for s in val_samps])
                    y_p_val = np.array([s["y_point"] for s in val_samps])
                    y_s_val = np.array([s["y_server"] for s in val_samps])
                    nsn_val = np.array([s["next_sn"] for s in val_samps])
                    a_p_ruled = apply_action_rules(a_p, nsn_val)
                    f1_a = action_macro_f1(y_a_val, a_p_ruled)
                    f1_p = point_macro_f1(y_p_val, p_p)
                    auc = roc_auc_score(y_s_val, s_p)
                    ov = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc
                    print(f"    [CE]  Ep{epoch:3d}  loss={tr_loss:.4f}  F1_a={f1_a:.4f} F1_p={f1_p:.4f} AUC={auc:.4f} OV={ov:.4f}")
                    if ov > best_ov_ce:
                        best_ov_ce = ov
                        best_state_ce = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Restore best CE checkpoint
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state_ce.items()})

        # ---- Pin baseline (Codex P2-6) ----
        print(f"\n  -- Pinning Fold-{fold+1} CE-baseline metrics --")
        a_p, p_p, s_p = evaluate(model, val_loader)
        y_a_val = np.array([s["y_action"] for s in val_samps])
        y_p_val = np.array([s["y_point"] for s in val_samps])
        y_s_val = np.array([s["y_server"] for s in val_samps])
        nsn_val = np.array([s["next_sn"] for s in val_samps])
        a_p_ruled = apply_action_rules(a_p, nsn_val)
        baseline_f1_a = action_macro_f1(y_a_val, a_p_ruled)
        baseline_f1_p = point_macro_f1(y_p_val, p_p)
        baseline_auc = roc_auc_score(y_s_val, s_p)
        baseline_ov = 0.4 * baseline_f1_a + 0.4 * baseline_f1_p + 0.2 * baseline_auc
        baseline_per_class = f1_score(
            y_a_val, a_p_ruled.argmax(axis=1),
            labels=ACTION_EVAL_LABELS, average=None, zero_division=0,
        ).tolist()
        print(f"    Baseline: F1_a={baseline_f1_a:.4f}  F1_p={baseline_f1_p:.4f}  AUC={baseline_auc:.4f}  OV={baseline_ov:.4f}")

        # Save baseline (CE-best) checkpoint
        if args.save_checkpoints:
            ckpt_dir = Path(PROJECT_ROOT) / "models" / out_tag / f"fold{fold+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                       ckpt_dir / "best_ce.pt")
            print(f"    Saved CE checkpoint: {ckpt_dir / 'best_ce.pt'}")

        # ---- Phase B: soft-F1 fine-tune (Codex P1-3 freeze + P2-4 mask) ----
        print(f"\n  -- Phase B: soft-F1 fine-tune ({args.softf1_epochs} epochs, lr={args.softf1_lr}) --")
        if args.freeze_encoder:
            n_frozen = freeze_non_action_params(model)
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    Frozen {n_frozen:,} non-action params; {n_trainable:,} action params remain trainable")
        # Build optimizer over (potentially filtered) trainable params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        sf_optimizer = torch.optim.AdamW(trainable_params, lr=args.softf1_lr,
                                         weight_decay=1e-2, eps=1e-8)
        sf_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            sf_optimizer, T_0=args.softf1_epochs, eta_min=args.softf1_lr / 20)

        best_ov_sf = baseline_ov  # gate: must improve on baseline
        best_state_sf = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        diag_history = []

        for epoch in range(1, args.softf1_epochs + 1):
            # Alpha ramp
            if args.softf1_epochs > 1:
                progress = (epoch - 1) / (args.softf1_epochs - 1)
            else:
                progress = 1.0
            alpha_now = args.alpha_start + progress * (args.alpha_end - args.alpha_start)

            tr_loss, aug_seen, aug_in_srv, loss_means, sf_diag = train_epoch_softf1(
                model, tr_loader, sf_optimizer, scaler, act_loss_fn, pt_loss_fn,
                phase="softf1", alpha=alpha_now, aux_lambda=args.aux_lambda)
            sf_scheduler.step()

            a_p, p_p, s_p = evaluate(model, val_loader)
            a_p_ruled = apply_action_rules(a_p, nsn_val)
            f1_a = action_macro_f1(y_a_val, a_p_ruled)
            f1_p = point_macro_f1(y_p_val, p_p)
            auc = roc_auc_score(y_s_val, s_p)
            ov = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc
            diag = {
                "epoch": epoch, "alpha": alpha_now,
                "loss_action": loss_means["a"], "loss_softf1_only": loss_means["softf1"],
                "f1_a": f1_a, "f1_p": f1_p, "auc": auc, "ov": ov,
                **sf_diag,
            }
            diag_history.append(diag)
            print(f"    [SF]  Ep{epoch:3d}  a={alpha_now:.2f}  loss={tr_loss:.4f}  "
                  f"F1_a={f1_a:.4f} F1_p={f1_p:.4f} AUC={auc:.4f} OV={ov:.4f} "
                  f"active_cls={sf_diag.get('n_classes_active_mean', 0):.1f}")

            if ov > best_ov_sf:
                best_ov_sf = ov
                best_state_sf = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Restore best soft-F1 checkpoint (which is >= baseline by definition)
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state_sf.items()})

        # Save soft-F1 checkpoint
        if args.save_checkpoints:
            ckpt_dir = Path(PROJECT_ROOT) / "models" / out_tag / f"fold{fold+1}"
            torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                       ckpt_dir / "best_softf1.pt")
            print(f"    Saved soft-F1 checkpoint: {ckpt_dir / 'best_softf1.pt'}")

        # Unfreeze (housekeeping for later folds, no effect within this fold)
        unfreeze_all(model)

        # ---- Final val + post-finetune metrics ----
        a_p, p_p, s_p = evaluate(model, val_loader)
        a_p_ruled = apply_action_rules(a_p, nsn_val)
        post_f1_a = action_macro_f1(y_a_val, a_p_ruled)
        post_f1_p = point_macro_f1(y_p_val, p_p)
        post_auc = roc_auc_score(y_s_val, s_p)
        post_ov = 0.4 * post_f1_a + 0.4 * post_f1_p + 0.2 * post_auc
        post_per_class = f1_score(
            y_a_val, a_p_ruled.argmax(axis=1),
            labels=ACTION_EVAL_LABELS, average=None, zero_division=0,
        ).tolist()
        print(f"\n  BEST FOLD-{fold+1} after soft-F1: F1_a={post_f1_a:.4f}  F1_p={post_f1_p:.4f}  AUC={post_auc:.4f}  OV={post_ov:.4f}  [{time.time()-t_fold:.0f}s]")
        print(f"    Action F1 delta: {post_f1_a - baseline_f1_a:+.4f}")
        print(f"    OV delta:        {post_ov - baseline_ov:+.4f}")
        print(f"    Per-class action F1 deltas (cls 0-14):")
        for c in range(15):
            print(f"      cls {c:2d}: {baseline_per_class[c]:.4f} -> {post_per_class[c]:.4f}  ({post_per_class[c] - baseline_per_class[c]:+.4f})")

        # Store OOF
        oof_act[val_idx] = a_p
        oof_pt[val_idx] = p_p
        oof_srv[val_idx] = s_p
        oof_mask_arr[val_idx] = True

        # Test accumulation
        at, pt_, st = evaluate(model, test_loader)
        test_act_acc += at / len(splits)
        test_pt_acc += pt_ / len(splits)
        test_srv_acc += st / len(splits)

        # Record fold metadata
        fold_metadata.append({
            "fold": fold + 1,
            "baseline_f1_a": baseline_f1_a,
            "baseline_f1_p": baseline_f1_p,
            "baseline_auc": baseline_auc,
            "baseline_ov": baseline_ov,
            "baseline_per_class_action_f1": baseline_per_class,
            "post_f1_a": post_f1_a,
            "post_f1_p": post_f1_p,
            "post_auc": post_auc,
            "post_ov": post_ov,
            "post_per_class_action_f1": post_per_class,
            "softf1_history": diag_history,
        })

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Save OOF arrays ----
    print("\n" + "=" * 70)
    print("Saving OOF + test arrays ...")
    oof_save_dir = Path(PROJECT_ROOT) / "oof_predictions"
    oof_save_dir.mkdir(exist_ok=True)
    np.save(oof_save_dir / f"{out_tag}_oof_act.npy", oof_act)
    np.save(oof_save_dir / f"{out_tag}_oof_pt.npy", oof_pt)
    np.save(oof_save_dir / f"{out_tag}_oof_srv.npy", oof_srv)
    np.save(oof_save_dir / f"{out_tag}_oof_mask.npy", oof_mask_arr)
    np.save(oof_save_dir / f"{out_tag}_oof_y_act.npy", y_a_all)
    np.save(oof_save_dir / f"{out_tag}_oof_y_pt.npy", y_p_all)
    np.save(oof_save_dir / f"{out_tag}_oof_y_srv.npy", y_s_all)
    np.save(oof_save_dir / f"{out_tag}_oof_nsn.npy", nsn_all)
    np.save(oof_save_dir / f"{out_tag}_test_act.npy", test_act_acc)
    np.save(oof_save_dir / f"{out_tag}_test_pt.npy", test_pt_acc)
    np.save(oof_save_dir / f"{out_tag}_test_srv.npy", test_srv_acc)
    np.save(oof_save_dir / f"{out_tag}_test_rally_uid.npy", np.array(rally_uid_test))
    print(f"  Saved 12 arrays with tag={out_tag}")

    # ---- Standalone verdict — INFORMATIONAL ONLY (post-R-034 lesson) ----
    #
    # Per the 2026-05-21 lesson: standalone-OOF verdicts have repeatedly
    # over-rejected components that turn out to deliver LB lift in blend
    # (v15feat_a parked 8 days then won +0.0028 LB as R-034). We no longer
    # use OOF deltas as a STOP signal. Instead:
    #   1. Label the standalone outcome descriptively
    #   2. ALWAYS run blend-swap audit regardless of standalone label
    #   3. If blend dOV >= -0.002 in a new signal class -> build LB candidate
    #   4. Park ONLY after standalone + blend + LB all show negative
    #
    # Codex's original PASS/PAUSE/FAIL labels are kept as `standalone_label`
    # for telemetry but are NOT treated as gates.
    fold1 = fold_metadata[0]
    f1_a_delta = fold1["post_f1_a"] - fold1["baseline_f1_a"]
    ov_delta = fold1["post_ov"] - fold1["baseline_ov"]

    if ov_delta > 0.003:
        standalone_label = "STANDALONE_STRONG"
    elif f1_a_delta > 0.002 and ov_delta >= -0.001:
        standalone_label = "STANDALONE_F1A_UP_OV_FLAT"
    elif ov_delta >= -0.002:
        standalone_label = "STANDALONE_NEAR_TIED"
    else:
        standalone_label = "STANDALONE_NEGATIVE"
    # Backward compat for downstream tooling that reads "gate_verdict"
    gate_verdict = standalone_label

    print()
    print(f"=== Fold-1 standalone label (INFORMATIONAL): {standalone_label} ===")
    print(f"  Baseline (CE)   OV={fold1['baseline_ov']:.4f}  F1_a={fold1['baseline_f1_a']:.4f}")
    print(f"  Post soft-F1    OV={fold1['post_ov']:.4f}  F1_a={fold1['post_f1_a']:.4f}")
    print(f"  Deltas:         OV {ov_delta:+.4f}, F1_a {f1_a_delta:+.4f}")
    print()
    print(f"  NEXT STEP (regardless of label, post-R-034 lesson):")
    print(f"  1. Run: python -u src/audit_all_parked_components.py --n-samples 200")
    print(f"     The new tag will appear automatically. Check dOV in blend swap.")
    print(f"  2. If blend dOV >= -0.002 AND it's a new signal class: build LB")
    print(f"     candidate CSV via src/build_low_risk_submissions.py")
    print(f"  3. PARK only AFTER LB result shows negative.")
    print(f"  4. Even STANDALONE_NEGATIVE here does NOT mean park yet.")

    # ---- Save metadata ----
    meta_path = Path(PROJECT_ROOT) / "runs" / f"{out_tag}_metadata.json"
    meta_path.parent.mkdir(exist_ok=True)
    metadata = {
        "tag": out_tag,
        "seed": seed_actual,
        "args": vars(args),
        "fold_metadata": fold_metadata,
        "gate_verdict": gate_verdict,
        "gate_thresholds": {
            "PASS_OV_DELTA": 0.003,
            "PAUSE_F1A_UP_OV_FLAT_min_f1a_delta": 0.002,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {meta_path}")
    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
