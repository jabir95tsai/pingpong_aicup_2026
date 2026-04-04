"""Train Transformer V2 (improved architecture) on GPU."""
import sys, os, time, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, N_FOLDS
from data_cleaning import clean_data
from transformer_model import prepare_sequences  # reuse data prep
from transformer_v2 import PingPongTransformerV2, PingPongDatasetV2

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

N_ACTION, N_POINT = 19, 10
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


def train_one_fold(tr_samples, val_samples, n_players, device, config):
    max_seq = 50
    train_ds = PingPongDatasetV2(tr_samples, max_seq_len=max_seq)
    val_ds = PingPongDatasetV2(val_samples, max_seq_len=max_seq)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["batch_size"], num_workers=0, pin_memory=True)

    model = PingPongTransformerV2(
        d_model=config["d_model"], nhead=config["nhead"],
        n_layers=config["n_layers"], dropout=config["dropout"],
        n_players=n_players,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    # Warmup + cosine schedule
    warmup_epochs = 3
    total_epochs = config["epochs"]
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Class weights (sqrt inverse frequency for smoother weighting)
    act_counts = np.bincount([s["y_action"] for s in tr_samples], minlength=N_ACTION).astype(float)
    act_w = torch.FloatTensor(np.sqrt(1.0/(act_counts+1))).to(device)
    act_w = act_w / act_w.sum() * N_ACTION

    pt_counts = np.bincount([s["y_point"] for s in tr_samples], minlength=N_POINT).astype(float)
    pt_w = torch.FloatTensor(np.sqrt(1.0/(pt_counts+1))).to(device)
    pt_w = pt_w / pt_w.sum() * N_POINT

    # Mixed precision
    scaler = GradScaler()

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
            y_a = batch["y_action"].long().to(device)
            y_p = batch["y_point"].long().to(device)
            y_s = batch["y_server"].float().to(device)

            with autocast(dtype=torch.float16):
                al, pl, sl = model(cat_seq, num_seq, ctx, pids, mask)
                loss = 0.4*F.cross_entropy(al, y_a, weight=act_w) + \
                       0.4*F.cross_entropy(pl, y_p, weight=pt_w) + \
                       0.2*F.binary_cross_entropy_with_logits(sl, y_s)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batch += 1

        scheduler.step()

        # Eval
        model.eval()
        a_probs, p_probs, s_probs, ya, yp, ys = [], [], [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                with autocast(dtype=torch.float16):
                    al, pl, sl = model(
                        batch["cat_seq"].to(device), batch["num_seq"].to(device),
                        batch["context"].to(device), batch["player_ids"].to(device),
                        batch["mask"].to(device))
                a_probs.append(F.softmax(al.float(), dim=-1).cpu().numpy())
                p_probs.append(F.softmax(pl.float(), dim=-1).cpu().numpy())
                s_probs.append(torch.sigmoid(sl.float()).cpu().numpy())
                ya.extend(batch["y_action"].numpy())
                yp.extend(batch["y_point"].numpy())
                ys.extend(batch["y_server"].numpy())

        a_arr = np.concatenate(a_probs)
        p_arr = np.concatenate(p_probs)
        s_arr = np.concatenate(s_probs)
        f1a = macro_f1(np.array(ya), a_arr, N_ACTION)
        f1p = macro_f1(np.array(yp), p_arr, N_POINT)
        auc = roc_auc_score(np.array(ys), s_arr)
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
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Ep {epoch+1:2d}: loss={total_loss/n_batch:.4f} F1a={f1a:.4f} F1p={f1p:.4f} "
                  f"AUC={auc:.4f} OV={ov:.4f} lr={lr_now:.1e} ({elapsed:.1f}s){improved}")

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, best_score


def main():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load and clean
    print("\nLoading data...")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    n_players = len(player_map)

    print("\nBuilding sequences...")
    train_samples = prepare_sequences(train_df, is_train=True)
    test_samples = prepare_sequences(test_df, is_train=False)
    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = np.array([rally_to_match.get(s["rally_uid"], 0) for s in train_samples])

    # Model config
    config = {
        "d_model": 256,
        "nhead": 8,
        "n_layers": 3,
        "dropout": 0.1,
        "epochs": 60,
        "batch_size": 256,
        "lr": 5e-4,
    }
    print(f"\nConfig: {config}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    all_scores = []
    all_models = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_samples)), groups=groups)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{N_FOLDS} (train={len(tr_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")

        tr_s = [train_samples[i] for i in tr_idx]
        val_s = [train_samples[i] for i in val_idx]

        model, best_ov = train_one_fold(tr_s, val_s, n_players, device, config)

        # Re-evaluate with action rules
        max_seq = 50
        val_ds = PingPongDatasetV2(val_s, max_seq_len=max_seq)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256)
        model.to(device).eval()
        a_probs, p_probs, s_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                al, pl, sl = model(
                    batch["cat_seq"].to(device), batch["num_seq"].to(device),
                    batch["context"].to(device), batch["player_ids"].to(device),
                    batch["mask"].to(device))
                a_probs.append(F.softmax(al.float(), dim=-1).cpu().numpy())
                p_probs.append(F.softmax(pl.float(), dim=-1).cpu().numpy())
                s_probs.append(torch.sigmoid(sl.float()).cpu().numpy())

        a_arr = np.concatenate(a_probs)
        p_arr = np.concatenate(p_probs)
        s_arr = np.concatenate(s_probs)

        val_next_sn = np.array([len(s["cat_seq"]) + 1 for s in val_s])
        a_arr = apply_action_rules(a_arr, val_next_sn)

        ya = np.array([s["y_action"] for s in val_s])
        yp = np.array([s["y_point"] for s in val_s])
        ys = np.array([s["y_server"] for s in val_s])
        f1a = macro_f1(ya, a_arr, N_ACTION)
        f1p = macro_f1(yp, p_arr, N_POINT)
        auc = roc_auc_score(ys, s_arr)
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc

        print(f"\n  >> Fold {fold+1} Final: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")
        all_scores.append({"f1a": f1a, "f1p": f1p, "auc": auc, "ov": ov})
        all_models.append(model.cpu())

    # Summary
    print(f"\n{'='*60}")
    print("CV RESULTS - Transformer V2")
    print(f"{'='*60}")
    mean_ov = np.mean([s["ov"] for s in all_scores])
    std_ov = np.std([s["ov"] for s in all_scores])
    print(f"  F1_action:  {np.mean([s['f1a'] for s in all_scores]):.4f}")
    print(f"  F1_point:   {np.mean([s['f1p'] for s in all_scores]):.4f}")
    print(f"  AUC_server: {np.mean([s['auc'] for s in all_scores]):.4f}")
    print(f"  Overall:    {mean_ov:.4f} ± {std_ov:.4f}")

    # Generate submission
    print(f"\nGenerating submission...")
    max_seq = 50
    test_ds = PingPongDatasetV2(test_samples, max_seq_len=max_seq)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256)

    all_act = np.zeros((len(test_samples), N_ACTION))
    all_pt = np.zeros((len(test_samples), N_POINT))
    all_srv = np.zeros(len(test_samples))

    for model in all_models:
        model.to(device).eval()
        act_l, pt_l, srv_l = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                al, pl, sl = model(
                    batch["cat_seq"].to(device), batch["num_seq"].to(device),
                    batch["context"].to(device), batch["player_ids"].to(device),
                    batch["mask"].to(device))
                act_l.append(F.softmax(al.float(), dim=-1).cpu().numpy())
                pt_l.append(F.softmax(pl.float(), dim=-1).cpu().numpy())
                srv_l.append(torch.sigmoid(sl.float()).cpu().numpy())
        all_act += np.concatenate(act_l)
        all_pt += np.concatenate(pt_l)
        all_srv += np.concatenate(srv_l)
        model.cpu()

    all_act /= len(all_models)
    all_pt /= len(all_models)
    all_srv /= len(all_models)

    test_next_sn = np.array([len(s["cat_seq"]) + 1 for s in test_samples])
    all_act = apply_action_rules(all_act, test_next_sn)

    submission = pd.DataFrame({
        "rally_uid": [s["rally_uid"] for s in test_samples],
        "actionId": np.argmax(all_act, axis=1).astype(int),
        "pointId": np.argmax(all_pt, axis=1).astype(int),
        "serverGetPoint": (all_srv >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission_v2.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"Saved: {out_path} ({submission.shape})")

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    for i, model in enumerate(all_models):
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"transformer_v2_fold{i}.pt"))
    print("Models saved.")


if __name__ == "__main__":
    main()
