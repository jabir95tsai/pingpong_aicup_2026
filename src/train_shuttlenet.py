"""Train ShuttleNet-style autoregressive model."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, N_FOLDS
from data_cleaning import clean_data
from shuttlenet import (ShuttleNetModel, ShuttleNetDataset,
                        prepare_autoregressive_data, N_ACTION, N_POINT)

import torch
import torch.nn.functional as F

SERVE_OK = {0, 15, 16, 17, 18}
SERVE_FORBIDDEN = {15, 16, 17, 18}


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


def train_one_fold(tr_rallies, val_rallies, n_players, device, config):
    max_seq = 50
    train_ds = ShuttleNetDataset(tr_rallies, max_seq_len=max_seq)
    val_ds = ShuttleNetDataset(val_rallies, max_seq_len=max_seq)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["batch_size"], num_workers=0, pin_memory=True)

    model = ShuttleNetModel(
        d_model=config["d_model"], nhead=config["nhead"],
        n_layers=config["n_layers"], dropout=config["dropout"],
        n_players=n_players,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    warmup_epochs = 3
    total_epochs = config["epochs"]
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Class weights
    all_actions = []
    all_points = []
    for r in tr_rallies:
        if "y_actions" in r:
            all_actions.extend(r["y_actions"].tolist())
            all_points.extend(r["y_points"].tolist())
    act_counts = np.bincount(all_actions, minlength=N_ACTION).astype(float)
    act_w = torch.FloatTensor(np.sqrt(1.0/(act_counts+1))).to(device)
    act_w = act_w / act_w.sum() * N_ACTION
    pt_counts = np.bincount(all_points, minlength=N_POINT).astype(float)
    pt_w = torch.FloatTensor(np.sqrt(1.0/(pt_counts+1))).to(device)
    pt_w = pt_w / pt_w.sum() * N_POINT

    best_score = -1
    best_state = None
    patience, wait = 10, 0

    for epoch in range(total_epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_batch = 0

        for batch in train_loader:
            cat_seq = batch["cat_seq"].to(device)
            num_seq = batch["num_seq"].to(device)
            ctx = batch["context"].to(device)
            pids = batch["player_ids"].to(device)
            mask = batch["mask"].to(device)
            y_act = batch["y_actions"].to(device)   # (B, T)
            y_pt = batch["y_points"].to(device)      # (B, T)
            y_srv = batch["y_server"].float().to(device)  # (B,)

            act_logits, pt_logits, srv_logits = model(cat_seq, num_seq, ctx, pids, mask)
            # act_logits: (B, T, N_ACTION)

            # Autoregressive loss: only on valid (non-padded, non -1) positions
            B, T, C = act_logits.shape
            act_flat = act_logits.reshape(-1, N_ACTION)
            pt_flat = pt_logits.reshape(-1, N_POINT)
            ya_flat = y_act.reshape(-1)
            yp_flat = y_pt.reshape(-1)

            # Valid mask: not padding and target != -1
            valid = ya_flat >= 0
            if valid.sum() > 0:
                loss_act = F.cross_entropy(act_flat[valid], ya_flat[valid], weight=act_w)
                loss_pt = F.cross_entropy(pt_flat[valid], yp_flat[valid], weight=pt_w)
            else:
                loss_act = torch.tensor(0.0, device=device)
                loss_pt = torch.tensor(0.0, device=device)

            loss_srv = F.binary_cross_entropy_with_logits(srv_logits, y_srv)
            loss = 0.4 * loss_act + 0.4 * loss_pt + 0.2 * loss_srv

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        scheduler.step()

        # Evaluate: for each validation rally, predict the LAST position's next strike
        model.eval()
        all_act_probs = []
        all_pt_probs = []
        all_srv_probs = []
        all_ya = []
        all_yp = []
        all_ys = []
        all_next_sn = []

        with torch.no_grad():
            for batch in val_loader:
                cat_seq = batch["cat_seq"].to(device)
                num_seq = batch["num_seq"].to(device)
                ctx = batch["context"].to(device)
                pids = batch["player_ids"].to(device)
                bmask = batch["mask"].to(device)
                seq_lens = batch["seq_len"]

                act_logits, pt_logits, srv_logits = model(cat_seq, num_seq, ctx, pids, bmask)

                B = cat_seq.shape[0]
                for i in range(B):
                    T = seq_lens[i].item()
                    # Last position's prediction = next strike
                    last_act = F.softmax(act_logits[i, T-1], dim=-1).cpu().numpy()
                    last_pt = F.softmax(pt_logits[i, T-1], dim=-1).cpu().numpy()
                    srv_prob = torch.sigmoid(srv_logits[i]).cpu().item()

                    all_act_probs.append(last_act)
                    all_pt_probs.append(last_pt)
                    all_srv_probs.append(srv_prob)

                    # Target: what comes after the last position
                    ya = batch["y_actions"][i]
                    yp = batch["y_points"][i]
                    if T-1 < len(ya) and ya[T-1].item() >= 0:
                        all_ya.append(ya[T-1].item())
                        all_yp.append(yp[T-1].item())
                    else:
                        # No valid target for this position (shouldn't happen in train)
                        all_ya.append(0)
                        all_yp.append(0)
                    all_ys.append(batch["y_server"][i].item())
                    all_next_sn.append(T + 1)

        act_probs = np.array(all_act_probs)
        pt_probs = np.array(all_pt_probs)
        srv_probs = np.array(all_srv_probs)
        ya = np.array(all_ya)
        yp = np.array(all_yp)
        ys = np.array(all_ys)

        # Apply rules
        next_sns = np.array(all_next_sn)
        act_probs = apply_action_rules(act_probs, next_sns)

        f1a = macro_f1(ya, act_probs, N_ACTION)
        f1p = macro_f1(yp, pt_probs, N_POINT)
        auc = roc_auc_score(ys, srv_probs) if len(np.unique(ys)) > 1 else 0.5
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc
        elapsed = time.time() - t0

        improved = ""
        if ov > best_score:
            best_score = ov
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            improved = " *"
        else:
            wait += 1

        if (epoch+1) % 5 == 0 or epoch == 0 or improved:
            print(f"  Ep {epoch+1:2d}: loss={total_loss/n_batch:.4f} F1a={f1a:.4f} F1p={f1p:.4f} "
                  f"AUC={auc:.4f} OV={ov:.4f} ({elapsed:.1f}s){improved}")

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, best_score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading data...")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    n_players = len(player_map)

    print("\nPreparing autoregressive data...")
    train_rallies = prepare_autoregressive_data(train_df, is_train=True)
    test_rallies = prepare_autoregressive_data(test_df, is_train=False)
    print(f"  Train: {len(train_rallies)} rallies, Test: {len(test_rallies)} rallies")

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = np.array([rally_to_match.get(r["rally_uid"], 0) for r in train_rallies])

    config = {
        "d_model": 128,
        "nhead": 8,
        "n_layers": 2,
        "dropout": 0.1,
        "epochs": 60,
        "batch_size": 128,
        "lr": 5e-4,
    }
    print(f"Config: {config}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    all_scores = []
    all_models = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_rallies)), groups=groups)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{N_FOLDS}")
        print(f"{'='*60}")

        tr = [train_rallies[i] for i in tr_idx]
        val = [train_rallies[i] for i in val_idx]

        model, best_ov = train_one_fold(tr, val, n_players, device, config)
        print(f"  >> Fold {fold+1} Best OV: {best_ov:.4f}")
        all_scores.append(best_ov)
        all_models.append(model.cpu())

    mean_ov = np.mean(all_scores)
    std_ov = np.std(all_scores)
    print(f"\n{'='*60}")
    print(f"ShuttleNet CV: {mean_ov:.4f} ± {std_ov:.4f}")
    print(f"{'='*60}")

    # Generate submission
    print("\nGenerating submission...")
    max_seq = 50
    test_ds = ShuttleNetDataset(test_rallies, max_seq_len=max_seq)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256)

    all_act = np.zeros((len(test_rallies), N_ACTION))
    all_pt = np.zeros((len(test_rallies), N_POINT))
    all_srv = np.zeros(len(test_rallies))

    for model in all_models:
        model.to(device).eval()
        batch_act, batch_pt, batch_srv = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                act_logits, pt_logits, srv_logits = model(
                    batch["cat_seq"].to(device), batch["num_seq"].to(device),
                    batch["context"].to(device), batch["player_ids"].to(device),
                    batch["mask"].to(device))

                B = batch["cat_seq"].shape[0]
                for i in range(B):
                    T = batch["seq_len"][i].item()
                    batch_act.append(F.softmax(act_logits[i, T-1].float(), dim=-1).cpu().numpy())
                    batch_pt.append(F.softmax(pt_logits[i, T-1].float(), dim=-1).cpu().numpy())
                    batch_srv.append(torch.sigmoid(srv_logits[i].float()).cpu().item())

        all_act += np.array(batch_act) / len(all_models)
        all_pt += np.array(batch_pt) / len(all_models)
        all_srv += np.array(batch_srv) / len(all_models)
        model.cpu()

    # Apply rules
    test_next_sn = np.array([len(r["cat_seq"]) + 1 for r in test_rallies])
    all_act = apply_action_rules(all_act, test_next_sn)

    submission = pd.DataFrame({
        "rally_uid": [r["rally_uid"] for r in test_rallies],
        "actionId": np.argmax(all_act, axis=1).astype(int),
        "pointId": np.argmax(all_pt, axis=1).astype(int),
        "serverGetPoint": (all_srv >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission_shuttlenet.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"Saved: {out_path} ({submission.shape})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    for i, m in enumerate(all_models):
        torch.save(m.state_dict(), os.path.join(MODEL_DIR, f"shuttlenet_fold{i}.pt"))
    print("Models saved.")


if __name__ == "__main__":
    main()
