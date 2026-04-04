"""Train Transformer V1 with improved settings and save OOF predictions for blending with GBDT."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data
from transformer_model import PingPongTransformer, PingPongDataset, prepare_sequences

N_ACTION, N_POINT = 19, 10
SERVE_OK = {0, 15, 16, 17, 18}
SERVE_FORBIDDEN = {15, 16, 17, 18}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def macro_f1(y_true, y_probs, n_classes):
    y_pred = np.argmax(y_probs, axis=1)
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def apply_action_rules(probs, next_sns):
    preds = probs.copy()
    for i in range(len(preds)):
        sn = next_sns[i]
        if sn == 1:
            mask = np.zeros(preds.shape[1])
            for a in SERVE_OK:
                if a < preds.shape[1]: mask[a] = 1.0
            preds[i] *= mask
        elif sn == 2:
            for a in SERVE_FORBIDDEN:
                if a < preds.shape[1]: preds[i, a] = 0.0
        total = preds[i].sum()
        if total > 0: preds[i] /= total
        else: preds[i] = np.ones(preds.shape[1]) / preds.shape[1]
    return preds


def compute_class_weights(y, n_classes):
    """Compute inverse-frequency class weights."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1  # avoid div by 0
    weights = 1.0 / np.sqrt(counts)
    weights /= weights.sum() / n_classes  # normalize so mean weight = 1
    return torch.FloatTensor(weights).to(DEVICE)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        p = torch.exp(-ce)
        loss = (1 - p) ** self.gamma * ce
        return loss.mean()


def train_one_fold(model, train_loader, val_loader, val_indices, val_next_sn,
                   y_act_val, y_pt_val, y_srv_val,
                   act_weights, pt_weights, n_epochs=25, lr=1e-3):
    """Train one fold with cosine annealing and class-weighted focal loss."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    act_loss_fn = FocalLoss(weight=act_weights, gamma=2.0)
    pt_loss_fn = FocalLoss(weight=pt_weights, gamma=2.0)
    srv_loss_fn = nn.BCEWithLogitsLoss()

    best_ov = -1
    best_oof = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            cat_seq = batch["cat_seq"].to(DEVICE)
            num_seq = batch["num_seq"].to(DEVICE)
            context = batch["context"].to(DEVICE)
            player_ids = batch["player_ids"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            y_a = batch["y_action"].to(DEVICE)
            y_p = batch["y_point"].to(DEVICE)
            y_s = torch.tensor(batch["y_server"], dtype=torch.float32).to(DEVICE)

            optimizer.zero_grad()
            act_logits, pt_logits, srv_logits = model(cat_seq, num_seq, context, player_ids, mask)

            loss = (0.4 * act_loss_fn(act_logits, y_a) +
                    0.4 * pt_loss_fn(pt_logits, y_p) +
                    0.2 * srv_loss_fn(srv_logits, y_s))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Evaluate
        model.eval()
        all_act_probs = []
        all_pt_probs = []
        all_srv_probs = []

        with torch.no_grad():
            for batch in val_loader:
                cat_seq = batch["cat_seq"].to(DEVICE)
                num_seq = batch["num_seq"].to(DEVICE)
                context = batch["context"].to(DEVICE)
                player_ids = batch["player_ids"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)

                act_logits, pt_logits, srv_logits = model(cat_seq, num_seq, context, player_ids, mask)
                all_act_probs.append(F.softmax(act_logits, dim=1).cpu().numpy())
                all_pt_probs.append(F.softmax(pt_logits, dim=1).cpu().numpy())
                all_srv_probs.append(torch.sigmoid(srv_logits).cpu().numpy())

        act_probs = np.concatenate(all_act_probs)
        pt_probs = np.concatenate(all_pt_probs)
        srv_probs = np.concatenate(all_srv_probs)

        act_ruled = apply_action_rules(act_probs, val_next_sn)
        f1a = macro_f1(y_act_val, act_ruled, N_ACTION)
        f1p = macro_f1(y_pt_val, pt_probs, N_POINT)
        auc = roc_auc_score(y_srv_val, srv_probs)
        ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
        avg_loss = total_loss / n_batches

        if ov > best_ov:
            best_ov = ov
            best_oof = {"act": act_probs.copy(), "pt": pt_probs.copy(), "srv": srv_probs.copy()}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Ep {epoch+1:2d}: loss={avg_loss:.4f} F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    print(f"    Best OV: {best_ov:.4f}")
    model.load_state_dict(best_state)
    return best_oof, model


def main():
    t_start = time.time()
    print("=" * 70)
    print("TRANSFORMER V1 RETRAIN + OOF FOR BLENDING")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    n_players = len(player_map)

    print("\nPreparing sequences...")
    t0 = time.time()
    train_seqs = prepare_sequences(train_df, is_train=True)
    test_seqs = prepare_sequences(test_df, is_train=False)
    print(f"  Done in {time.time()-t0:.1f}s: {len(train_seqs)} train, {len(test_seqs)} test")

    # Extract rally_uids, targets, and next_sn for each sample
    train_rally_uids = np.array([s["rally_uid"] for s in train_seqs])
    y_act = np.array([s["y_action"] for s in train_seqs])
    y_pt = np.array([s["y_point"] for s in train_seqs])
    y_srv = np.array([s["y_server"] for s in train_seqs])

    # Compute next_strikeNumber for each sample
    next_sn = np.array([len(s["cat_seq"]) + 1 for s in train_seqs])
    test_next_sn = np.array([len(s["cat_seq"]) + 1 for s in test_seqs])

    # Compute class weights
    act_weights = compute_class_weights(y_act, N_ACTION)
    pt_weights = compute_class_weights(y_pt, N_POINT)
    print(f"  Action weights: {act_weights.cpu().numpy().round(2)}")

    # Group by match for CV
    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = np.array([rally_to_match.get(uid, 0) for uid in train_rally_uids])

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(gkf.split(np.arange(len(train_seqs)), groups=groups))

    # OOF predictions
    oof_act = np.zeros((len(train_seqs), N_ACTION))
    oof_pt = np.zeros((len(train_seqs), N_POINT))
    oof_srv = np.zeros(len(train_seqs))
    test_act = np.zeros((len(test_seqs), N_ACTION))
    test_pt = np.zeros((len(test_seqs), N_POINT))
    test_srv = np.zeros(len(test_seqs))

    test_ds = PingPongDataset(test_seqs, max_seq_len=50)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{N_FOLDS}")
        print(f"{'='*50}")

        train_fold = [train_seqs[i] for i in tr_idx]
        val_fold = [train_seqs[i] for i in val_idx]

        train_ds = PingPongDataset(train_fold, max_seq_len=50)
        val_ds = PingPongDataset(val_fold, max_seq_len=50)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

        model = PingPongTransformer(
            d_model=128, nhead=8, n_layers=3, dropout=0.15,
            n_action_classes=N_ACTION, n_point_classes=N_POINT,
            n_players=n_players
        ).to(DEVICE)

        best_oof, model = train_one_fold(
            model, train_loader, val_loader, val_idx, next_sn[val_idx],
            y_act[val_idx], y_pt[val_idx], y_srv[val_idx],
            act_weights, pt_weights, n_epochs=30, lr=1e-3
        )

        oof_act[val_idx] = best_oof["act"]
        oof_pt[val_idx] = best_oof["pt"]
        oof_srv[val_idx] = best_oof["srv"]

        # Test predictions
        model.eval()
        fold_test_act = []
        fold_test_pt = []
        fold_test_srv = []
        with torch.no_grad():
            for batch in test_loader:
                cat_seq = batch["cat_seq"].to(DEVICE)
                num_seq = batch["num_seq"].to(DEVICE)
                context = batch["context"].to(DEVICE)
                player_ids = batch["player_ids"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                a, p, s = model(cat_seq, num_seq, context, player_ids, mask)
                fold_test_act.append(F.softmax(a, dim=1).cpu().numpy())
                fold_test_pt.append(F.softmax(p, dim=1).cpu().numpy())
                fold_test_srv.append(torch.sigmoid(s).cpu().numpy())

        test_act += np.concatenate(fold_test_act) / N_FOLDS
        test_pt += np.concatenate(fold_test_pt) / N_FOLDS
        test_srv += np.concatenate(fold_test_srv) / N_FOLDS

    # Overall OOF
    act_ruled = apply_action_rules(oof_act, next_sn)
    f1a = macro_f1(y_act, act_ruled, N_ACTION)
    f1p = macro_f1(y_pt, oof_pt, N_POINT)
    auc = roc_auc_score(y_srv, oof_srv)
    ov = 0.4*f1a + 0.4*f1p + 0.2*auc
    print(f"\n{'='*50}")
    print(f"TRANSFORMER V1 OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")
    print(f"{'='*50}")

    # Save OOF for blending
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(os.path.join(MODEL_DIR, "oof_transformer.npz"),
             oof_act=oof_act, oof_pt=oof_pt, oof_srv=oof_srv,
             y_act=y_act, y_pt=y_pt, y_srv=y_srv, next_sn=next_sn)
    np.savez(os.path.join(MODEL_DIR, "test_transformer.npz"),
             test_act=test_act, test_pt=test_pt, test_srv=test_srv,
             test_next_sn=test_next_sn,
             rally_uids=np.array([s["rally_uid"] for s in test_seqs]))

    # Now blend with GBDT
    print("\n--- BLENDING TRANSFORMER + GBDT ---")
    v2_path = os.path.join(MODEL_DIR, "oof_v2_fast.npz")
    if os.path.exists(v2_path):
        dv = np.load(v2_path)
        gbdt_act = 0.6 * dv["catboost_act"] + 0.4 * dv["xgboost_act"]
        gbdt_pt = 0.6 * dv["catboost_pt"] + 0.3 * dv["xgboost_pt"] + 0.1 * dv["lightgbm_pt"]
        gbdt_srv = 0.3 * dv["catboost_srv"] + 0.4 * dv["xgboost_srv"] + 0.3 * dv["lightgbm_srv"]
        gbdt_next_sn = dv["next_sn"]
        gbdt_y_act = dv["y_act"]
        gbdt_y_pt = dv["y_pt"]
        gbdt_y_srv = dv["y_srv"]

        # Note: Transformer OOF and GBDT OOF may have different sample orders
        # Both are built from the same data expansion, so they should match
        if len(gbdt_act) == len(oof_act):
            print("  Same size — blending directly")
            for w in np.arange(0, 1.05, 0.1):
                blend_act = w * oof_act + (1-w) * gbdt_act
                blend_pt = w * oof_pt + (1-w) * gbdt_pt
                blend_srv = w * oof_srv + (1-w) * gbdt_srv

                ba_ruled = apply_action_rules(blend_act, next_sn)
                f1a = macro_f1(y_act, ba_ruled, N_ACTION)
                f1p = macro_f1(y_pt, blend_pt, N_POINT)
                auc = roc_auc_score(y_srv, blend_srv)
                ov = 0.4*f1a + 0.4*f1p + 0.2*auc
                print(f"  w_tfm={w:.1f}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")
        else:
            print(f"  Size mismatch: Transformer={len(oof_act)}, GBDT={len(gbdt_act)}")
            print("  Cannot blend directly — saving separately for manual blend")

    # Also generate Transformer-only submission
    test_act_ruled = apply_action_rules(test_act, test_next_sn)
    submission = pd.DataFrame({
        "rally_uid": np.array([s["rally_uid"] for s in test_seqs]).astype(int),
        "actionId": np.argmax(test_act_ruled, axis=1).astype(int),
        "pointId": np.argmax(test_pt, axis=1).astype(int),
        "serverGetPoint": (test_srv >= 0.5).astype(int),
    })
    out_path = os.path.join(SUBMISSION_DIR, "submission_transformer_v1r.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nTransformer submission: {out_path}")

    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
