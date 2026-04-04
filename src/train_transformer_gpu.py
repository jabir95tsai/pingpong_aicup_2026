"""Full 5-fold Transformer training on GPU + generate submission."""
import sys, os, time, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, N_FOLDS
from data_cleaning import clean_data
from transformer_model import prepare_sequences, PingPongTransformer, PingPongDataset

import torch
import torch.nn.functional as F

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


def train_one_fold(tr_samples, val_samples, n_players, device,
                   d_model=128, nhead=8, n_layers=2, epochs=50, batch_size=256, lr=1e-3):
    max_seq = 50
    train_ds = PingPongDataset(tr_samples, max_seq_len=max_seq)
    val_ds = PingPongDataset(val_samples, max_seq_len=max_seq)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=0, pin_memory=True)

    model = PingPongTransformer(d_model=d_model, nhead=nhead, n_layers=n_layers,
                                 n_players=n_players).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Inverse frequency class weights
    act_counts = np.bincount([s["y_action"] for s in tr_samples], minlength=N_ACTION).astype(float)
    act_w = torch.FloatTensor((1.0/(act_counts+1)) / (1.0/(act_counts+1)).sum() * N_ACTION).to(device)
    pt_counts = np.bincount([s["y_point"] for s in tr_samples], minlength=N_POINT).astype(float)
    pt_w = torch.FloatTensor((1.0/(pt_counts+1)) / (1.0/(pt_counts+1)).sum() * N_POINT).to(device)

    best_score = -1
    best_state = None
    best_results = None
    patience, wait = 8, 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_batch = 0
        for batch in train_loader:
            al, pl, sl = model(
                batch["cat_seq"].to(device), batch["num_seq"].to(device),
                batch["context"].to(device), batch["player_ids"].to(device),
                batch["mask"].to(device))
            y_a = batch["y_action"].long().to(device)
            y_p = batch["y_point"].long().to(device)
            y_s = batch["y_server"].float().to(device)

            loss = 0.4*F.cross_entropy(al, y_a, weight=act_w) + \
                   0.4*F.cross_entropy(pl, y_p, weight=pt_w) + \
                   0.2*F.binary_cross_entropy_with_logits(sl, y_s)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        scheduler.step()

        # Eval
        model.eval()
        a_probs, p_probs, s_probs, ya, yp, ys = [], [], [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                al, pl, sl = model(
                    batch["cat_seq"].to(device), batch["num_seq"].to(device),
                    batch["context"].to(device), batch["player_ids"].to(device),
                    batch["mask"].to(device))
                a_probs.append(F.softmax(al, dim=-1).cpu().numpy())
                p_probs.append(F.softmax(pl, dim=-1).cpu().numpy())
                s_probs.append(torch.sigmoid(sl).cpu().numpy())
                ya.extend(batch["y_action"].cpu().numpy())
                yp.extend(batch["y_point"].cpu().numpy())
                ys.extend(batch["y_server"].cpu().numpy())

        a_probs_arr = np.concatenate(a_probs)
        p_probs_arr = np.concatenate(p_probs)
        s_probs_arr = np.concatenate(s_probs)
        f1a = macro_f1(np.array(ya), a_probs_arr, N_ACTION)
        f1p = macro_f1(np.array(yp), p_probs_arr, N_POINT)
        auc = roc_auc_score(np.array(ys), s_probs_arr)
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc

        elapsed = time.time() - t0
        if (epoch+1) % 5 == 0 or epoch == 0 or ov > best_score:
            print(f"  Ep {epoch+1:2d}: loss={total_loss/n_batch:.4f} F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f} ({elapsed:.1f}s)")

        if ov > best_score:
            best_score = ov
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_results = (a_probs_arr.copy(), p_probs_arr.copy(), s_probs_arr.copy())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, best_results, best_score


def main():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Load and clean
    print("\nLoading data...")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    n_players = len(player_map)

    # Build sequences
    print("\nBuilding train sequences...")
    train_samples = prepare_sequences(train_df, is_train=True)
    print(f"  {len(train_samples)} train samples")

    print("Building test sequences...")
    test_samples = prepare_sequences(test_df, is_train=False)
    print(f"  {len(test_samples)} test samples")

    # Groups for fold split
    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = np.array([rally_to_match.get(s["rally_uid"], 0) for s in train_samples])

    # Get next_strikeNumber for action rules
    def get_next_sn(samples):
        return np.array([len(s["cat_seq"]) + 1 for s in samples])

    # 5-Fold CV
    gkf = GroupKFold(n_splits=N_FOLDS)
    all_scores = []
    all_models = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_samples)), groups=groups)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{N_FOLDS} (train={len(tr_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")

        tr_s = [train_samples[i] for i in tr_idx]
        val_s = [train_samples[i] for i in val_idx]

        model, (act_p, pt_p, srv_p), best_ov = train_one_fold(
            tr_s, val_s, n_players, device,
            d_model=128, nhead=8, n_layers=2, epochs=50, batch_size=256, lr=1e-3,
        )

        # Apply action rules
        val_next_sn = get_next_sn(val_s)
        act_p = apply_action_rules(act_p, val_next_sn)

        ya = np.array([s["y_action"] for s in val_s])
        yp = np.array([s["y_point"] for s in val_s])
        ys = np.array([s["y_server"] for s in val_s])

        f1a = macro_f1(ya, act_p, N_ACTION)
        f1p = macro_f1(yp, pt_p, N_POINT)
        auc = roc_auc_score(ys, srv_p)
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc

        print(f"\n  Fold {fold+1} Final (with rules): F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")
        all_scores.append({"f1a": f1a, "f1p": f1p, "auc": auc, "ov": ov})
        all_models.append(model)

    # Summary
    print(f"\n{'='*60}")
    print("CV RESULTS SUMMARY")
    print(f"{'='*60}")
    mean_f1a = np.mean([s["f1a"] for s in all_scores])
    mean_f1p = np.mean([s["f1p"] for s in all_scores])
    mean_auc = np.mean([s["auc"] for s in all_scores])
    mean_ov = np.mean([s["ov"] for s in all_scores])
    std_ov = np.std([s["ov"] for s in all_scores])
    print(f"  F1_action:  {mean_f1a:.4f}")
    print(f"  F1_point:   {mean_f1p:.4f}")
    print(f"  AUC_server: {mean_auc:.4f}")
    print(f"  Overall:    {mean_ov:.4f} ± {std_ov:.4f}")
    print(f"  Baseline:   0.2800")

    # Generate submission with ensemble of all fold models
    print(f"\nGenerating submission...")
    max_seq = 50
    test_ds = PingPongDataset(test_samples, max_seq_len=max_seq)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, num_workers=0)

    all_act = np.zeros((len(test_samples), N_ACTION))
    all_pt = np.zeros((len(test_samples), N_POINT))
    all_srv = np.zeros(len(test_samples))

    for model in all_models:
        model.to(device).eval()
        act_list, pt_list, srv_list = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                al, pl, sl = model(
                    batch["cat_seq"].to(device), batch["num_seq"].to(device),
                    batch["context"].to(device), batch["player_ids"].to(device),
                    batch["mask"].to(device))
                act_list.append(F.softmax(al, dim=-1).cpu().numpy())
                pt_list.append(F.softmax(pl, dim=-1).cpu().numpy())
                srv_list.append(torch.sigmoid(sl).cpu().numpy())
        all_act += np.concatenate(act_list)
        all_pt += np.concatenate(pt_list)
        all_srv += np.concatenate(srv_list)
        model.cpu()  # free GPU memory

    all_act /= len(all_models)
    all_pt /= len(all_models)
    all_srv /= len(all_models)

    # Apply rules
    test_next_sn = get_next_sn(test_samples)
    all_act = apply_action_rules(all_act, test_next_sn)

    act_preds = np.argmax(all_act, axis=1)
    pt_preds = np.argmax(all_pt, axis=1)
    srv_preds = (all_srv >= 0.5).astype(int)

    rally_uids = [s["rally_uid"] for s in test_samples]
    submission = pd.DataFrame({
        "rally_uid": rally_uids,
        "actionId": act_preds.astype(int),
        "pointId": pt_preds.astype(int),
        "serverGetPoint": srv_preds.astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission_transformer.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"Submission saved to {out_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  actionId: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId: {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  serverGetPoint: {submission.serverGetPoint.value_counts().to_dict()}")

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    for i, model in enumerate(all_models):
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"transformer_fold{i}.pt"))
    print(f"Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
