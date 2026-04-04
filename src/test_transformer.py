"""Quick test of Transformer model only."""
import sys, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data
from transformer_model import prepare_sequences, PingPongTransformer, PingPongDataset

import torch
import torch.nn.functional as F

N_ACTION, N_POINT = 19, 10

def macro_f1(y_true, y_probs, n_classes):
    y_pred = np.argmax(y_probs, axis=1)
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)

def main():
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(os.path.join(os.path.dirname(TRAIN_PATH), "test.csv"))
    train_df, _, player_map = clean_data(raw_train, raw_test)
    n_players = len(player_map)

    print("Building sequences...")
    samples = prepare_sequences(train_df, is_train=True)
    print(f"  {len(samples)} samples")

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = np.array([rally_to_match.get(s["rally_uid"], 0) for s in samples])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    all_ov = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(samples)), groups=groups)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        tr_samples = [samples[i] for i in tr_idx]
        val_samples = [samples[i] for i in val_idx]

        max_seq = max(len(s["cat_seq"]) for s in tr_samples + val_samples)
        max_seq = min(max_seq, 50)

        train_ds = PingPongDataset(tr_samples, max_seq_len=max_seq)
        val_ds = PingPongDataset(val_samples, max_seq_len=max_seq)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256)

        model = PingPongTransformer(d_model=128, nhead=8, n_layers=2,
                                     n_players=n_players).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        # Class weights
        act_counts = np.bincount([s["y_action"] for s in tr_samples], minlength=N_ACTION).astype(float)
        act_w = torch.FloatTensor((1.0/(act_counts+1)) / (1.0/(act_counts+1)).sum() * N_ACTION).to(device)
        pt_counts = np.bincount([s["y_point"] for s in tr_samples], minlength=N_POINT).astype(float)
        pt_w = torch.FloatTensor((1.0/(pt_counts+1)) / (1.0/(pt_counts+1)).sum() * N_POINT).to(device)

        best_score = -1
        best_state = None
        patience, wait = 5, 0

        t0 = time.time()
        for epoch in range(30):
            model.train()
            for batch in train_loader:
                cat_seq = batch["cat_seq"].to(device)
                num_seq = batch["num_seq"].to(device)
                ctx = batch["context"].to(device)
                pids = batch["player_ids"].to(device)
                mask = batch["mask"].to(device)
                y_a = torch.tensor(batch["y_action"], dtype=torch.long).to(device)
                y_p = torch.tensor(batch["y_point"], dtype=torch.long).to(device)
                y_s = torch.tensor(batch["y_server"], dtype=torch.float32).to(device)

                al, pl, sl = model(cat_seq, num_seq, ctx, pids, mask)
                loss = 0.4*F.cross_entropy(al, y_a, weight=act_w) + \
                       0.4*F.cross_entropy(pl, y_p, weight=pt_w) + \
                       0.2*F.binary_cross_entropy_with_logits(sl, y_s)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
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
                    ya.extend(batch["y_action"].numpy())
                    yp.extend(batch["y_point"].numpy())
                    ys.extend(batch["y_server"].numpy())

            a_probs = np.concatenate(a_probs)
            p_probs = np.concatenate(p_probs)
            s_probs = np.concatenate(s_probs)
            f1a = macro_f1(np.array(ya), a_probs, N_ACTION)
            f1p = macro_f1(np.array(yp), p_probs, N_POINT)
            auc = roc_auc_score(np.array(ys), s_probs)
            ov = 0.4*f1a + 0.4*f1p + 0.2*auc

            if ov > best_score:
                best_score = ov
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f"  Ep {epoch+1}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

        print(f"  Best OV: {best_score:.4f} ({time.time()-t0:.1f}s)")
        all_ov.append(best_score)

    print(f"\n{'='*50}")
    print(f"Transformer CV: {np.mean(all_ov):.4f} ± {np.std(all_ov):.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
