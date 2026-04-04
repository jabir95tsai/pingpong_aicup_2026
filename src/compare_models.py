"""Compare all models: LightGBM, XGBoost, CatBoost, Transformer."""
import sys
import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data

N_ACTION = 19
N_POINT = 10


# =============================================================================
# Evaluation helpers
# =============================================================================
def macro_f1(y_true, y_probs, n_classes):
    y_pred = np.argmax(y_probs, axis=1)
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def apply_action_rules(probs, next_strike_numbers):
    """Constrain actionId based on strike type."""
    SERVE_OK = {0, 15, 16, 17, 18}
    SERVE_FORBIDDEN_ON_RETURN = {15, 16, 17, 18}
    preds = probs.copy()
    for i in range(len(preds)):
        sn = next_strike_numbers[i]
        if sn == 1:
            mask = np.zeros(preds.shape[1])
            for a in SERVE_OK:
                if a < preds.shape[1]: mask[a] = 1.0
            preds[i] *= mask
        elif sn == 2:
            for a in SERVE_FORBIDDEN_ON_RETURN:
                if a < preds.shape[1]: preds[i, a] = 0.0
        total = preds[i].sum()
        if total > 0:
            preds[i] /= total
        else:
            preds[i] = np.ones(preds.shape[1]) / preds.shape[1]
    return preds


# =============================================================================
# Feature extraction (for GBDT models)
# =============================================================================
from features import build_features, compute_player_stats, get_feature_names


# =============================================================================
# Model 1: LightGBM
# =============================================================================
def run_lightgbm(X_tr, y_tr, X_val, y_val, task, **kwargs):
    import lightgbm as lgb
    if task == "multiclass":
        n_classes = kwargs["n_classes"]
        params = {
            "objective": "multiclass", "num_class": n_classes,
            "metric": "multi_logloss", "boosting_type": "gbdt",
            "learning_rate": 0.05, "num_leaves": 127, "max_depth": -1,
            "min_child_samples": 10, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "lambda_l1": 0.05, "lambda_l2": 0.05,
            "seed": RANDOM_SEED, "verbose": -1, "is_unbalance": True,
            "num_threads": 4,
        }
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=2000,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        probs = model.predict(X_val, num_iteration=model.best_iteration)
        return model, probs
    else:  # binary
        params = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "learning_rate": 0.05, "num_leaves": 127, "max_depth": -1,
            "min_child_samples": 10, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "lambda_l1": 0.05, "lambda_l2": 0.05,
            "seed": RANDOM_SEED, "verbose": -1, "is_unbalance": True,
            "num_threads": 4,
        }
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=2000,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        probs = model.predict(X_val, num_iteration=model.best_iteration)
        return model, probs


# =============================================================================
# Model 2: XGBoost
# =============================================================================
def run_xgboost(X_tr, y_tr, X_val, y_val, task, **kwargs):
    import xgboost as xgb
    if task == "multiclass":
        n_classes = kwargs["n_classes"]
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            "objective": "multi:softprob", "num_class": n_classes,
            "eval_metric": "mlogloss", "tree_method": "hist",
            "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 10,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda": 0.1, "alpha": 0.05,
            "seed": RANDOM_SEED, "verbosity": 0,
        }
        model = xgb.train(params, dtrain, num_boost_round=2000,
                          evals=[(dval, "val")],
                          early_stopping_rounds=100, verbose_eval=0)
        probs = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        return model, probs
    else:
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "tree_method": "hist", "learning_rate": 0.05,
            "max_depth": 8, "min_child_weight": 10,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda": 0.1, "alpha": 0.05,
            "seed": RANDOM_SEED, "verbosity": 0,
        }
        model = xgb.train(params, dtrain, num_boost_round=2000,
                          evals=[(dval, "val")],
                          early_stopping_rounds=100, verbose_eval=0)
        probs = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        return model, probs


# =============================================================================
# Model 3: CatBoost
# =============================================================================
def run_catboost(X_tr, y_tr, X_val, y_val, task, **kwargs):
    from catboost import CatBoostClassifier
    if task == "multiclass":
        n_classes = kwargs["n_classes"]
        model = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=8,
            loss_function="MultiClass", classes_count=n_classes,
            auto_class_weights="Balanced",
            early_stopping_rounds=100, verbose=0,
            random_seed=RANDOM_SEED,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        probs = model.predict_proba(X_val)
        return model, probs
    else:
        model = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=8,
            loss_function="Logloss", auto_class_weights="Balanced",
            early_stopping_rounds=100, verbose=0,
            random_seed=RANDOM_SEED,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        probs = model.predict_proba(X_val)[:, 1]
        return model, probs


# =============================================================================
# Model 4: Transformer
# =============================================================================
def run_transformer_fold(train_samples, val_samples, d_model=128, nhead=8,
                         n_layers=2, n_players=200, epochs=30, batch_size=256, lr=1e-3):
    import torch
    from transformer_model import PingPongTransformer, PingPongDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_seq = max(len(s["cat_seq"]) for s in train_samples + val_samples)
    max_seq = min(max_seq, 50)

    train_ds = PingPongDataset(train_samples, max_seq_len=max_seq)
    val_ds = PingPongDataset(val_samples, max_seq_len=max_seq)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    model = PingPongTransformer(
        d_model=d_model, nhead=nhead, n_layers=n_layers,
        n_action_classes=N_ACTION, n_point_classes=N_POINT, n_players=n_players,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weights for action and point
    action_counts = np.bincount([s["y_action"] for s in train_samples], minlength=N_ACTION).astype(float)
    action_weights = 1.0 / (action_counts + 1)
    action_weights = torch.FloatTensor(action_weights / action_weights.sum() * N_ACTION).to(device)

    point_counts = np.bincount([s["y_point"] for s in train_samples], minlength=N_POINT).astype(float)
    point_weights = 1.0 / (point_counts + 1)
    point_weights = torch.FloatTensor(point_weights / point_weights.sum() * N_POINT).to(device)

    best_score = -1
    best_state = None
    patience = 5
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            cat_seq = batch["cat_seq"].to(device)
            num_seq = batch["num_seq"].to(device)
            context = batch["context"].to(device)
            player_ids = batch["player_ids"].to(device)
            mask = batch["mask"].to(device)
            y_act = torch.tensor(batch["y_action"], dtype=torch.long).to(device)
            y_pt = torch.tensor(batch["y_point"], dtype=torch.long).to(device)
            y_srv = torch.tensor(batch["y_server"], dtype=torch.float32).to(device)

            act_logits, pt_logits, srv_logits = model(cat_seq, num_seq, context, player_ids, mask)

            loss_act = F.cross_entropy(act_logits, y_act, weight=action_weights)
            loss_pt = F.cross_entropy(pt_logits, y_pt, weight=point_weights)
            loss_srv = F.binary_cross_entropy_with_logits(srv_logits, y_srv)
            loss = 0.4 * loss_act + 0.4 * loss_pt + 0.2 * loss_srv

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        all_act_probs, all_pt_probs, all_srv_probs = [], [], []
        all_y_act, all_y_pt, all_y_srv = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                cat_seq = batch["cat_seq"].to(device)
                num_seq = batch["num_seq"].to(device)
                context = batch["context"].to(device)
                player_ids = batch["player_ids"].to(device)
                mask = batch["mask"].to(device)

                act_logits, pt_logits, srv_logits = model(cat_seq, num_seq, context, player_ids, mask)
                all_act_probs.append(F.softmax(act_logits, dim=-1).cpu().numpy())
                all_pt_probs.append(F.softmax(pt_logits, dim=-1).cpu().numpy())
                all_srv_probs.append(torch.sigmoid(srv_logits).cpu().numpy())
                all_y_act.extend(batch["y_action"])
                all_y_pt.extend(batch["y_point"])
                all_y_srv.extend(batch["y_server"])

        act_probs = np.concatenate(all_act_probs)
        pt_probs = np.concatenate(all_pt_probs)
        srv_probs = np.concatenate(all_srv_probs)
        y_act = np.array(all_y_act)
        y_pt = np.array(all_y_pt)
        y_srv = np.array(all_y_srv)

        f1_act = macro_f1(y_act, act_probs, N_ACTION)
        f1_pt = macro_f1(y_pt, pt_probs, N_POINT)
        auc_srv = roc_auc_score(y_srv, srv_probs) if len(np.unique(y_srv)) > 1 else 0.5
        score = 0.4 * f1_act + 0.4 * f1_pt + 0.2 * auc_srv

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_results = (act_probs, pt_probs, srv_probs)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f} "
                  f"F1_act={f1_act:.4f} F1_pt={f1_pt:.4f} AUC={auc_srv:.4f} OV={score:.4f}")

    model.load_state_dict(best_state)
    return model, best_results, best_score


# =============================================================================
# Main comparison
# =============================================================================
def main():
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Load and clean data
    print("\nLoading data...")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    n_players = len(player_map)

    # Build GBDT features
    print("\nBuilding GBDT features...")
    player_stats = compute_player_stats(train_df)
    feat_df = build_features(train_df, is_train=True, player_stats=player_stats)
    feature_names = get_feature_names(feat_df)
    X = feat_df[feature_names].values
    y_action = feat_df["y_actionId"].values
    y_point = feat_df["y_pointId"].values
    y_server = feat_df["y_serverGetPoint"].values
    next_sn = feat_df["next_strikeNumber"].values
    print(f"  GBDT features: {X.shape}")

    # Build Transformer sequences
    print("Building Transformer sequences...")
    from transformer_model import prepare_sequences
    seq_samples = prepare_sequences(train_df, is_train=True)
    print(f"  Transformer samples: {len(seq_samples)}")

    # Groups for fold splitting (by match)
    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups_feat = feat_df["rally_uid"].map(rally_to_match).values
    # For transformer, group by rally_uid -> match
    seq_groups = np.array([rally_to_match.get(s["rally_uid"], 0) for s in seq_samples])

    gkf = GroupKFold(n_splits=N_FOLDS)

    # Results storage
    results = {
        "LightGBM": {"act": [], "pt": [], "srv": [], "ov": []},
        "XGBoost": {"act": [], "pt": [], "srv": [], "ov": []},
        "CatBoost": {"act": [], "pt": [], "srv": [], "ov": []},
        "Transformer": {"act": [], "pt": [], "srv": [], "ov": []},
    }

    # We need consistent fold splits for GBDT and Transformer
    # Use GBDT's groups for fold generation, then map to transformer indices
    fold_splits_gbdt = list(gkf.split(X, groups=groups_feat))

    # For transformer, create matching folds by match
    fold_splits_tf = list(gkf.split(np.arange(len(seq_samples)), groups=seq_groups))

    for fold in range(N_FOLDS):
        print(f"\n{'='*70}")
        print(f"FOLD {fold+1}/{N_FOLDS}")
        print(f"{'='*70}")

        tr_idx, val_idx = fold_splits_gbdt[fold]
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_act_tr, y_act_val = y_action[tr_idx], y_action[val_idx]
        y_pt_tr, y_pt_val = y_point[tr_idx], y_point[val_idx]
        y_srv_tr, y_srv_val = y_server[tr_idx], y_server[val_idx]
        val_sn = next_sn[val_idx]

        # --- LightGBM ---
        print("\n[LightGBM]")
        t0 = time.time()
        _, act_probs = run_lightgbm(X_tr, y_act_tr, X_val, y_act_val, "multiclass", n_classes=N_ACTION)
        act_probs = apply_action_rules(act_probs, val_sn)
        _, pt_probs = run_lightgbm(X_tr, y_pt_tr, X_val, y_pt_val, "multiclass", n_classes=N_POINT)
        _, srv_probs = run_lightgbm(X_tr, y_srv_tr, X_val, y_srv_val, "binary")
        f1_a = macro_f1(y_act_val, act_probs, N_ACTION)
        f1_p = macro_f1(y_pt_val, pt_probs, N_POINT)
        auc_s = roc_auc_score(y_srv_val, srv_probs)
        ov = 0.4*f1_a + 0.4*f1_p + 0.2*auc_s
        results["LightGBM"]["act"].append(f1_a)
        results["LightGBM"]["pt"].append(f1_p)
        results["LightGBM"]["srv"].append(auc_s)
        results["LightGBM"]["ov"].append(ov)
        print(f"  F1_act={f1_a:.4f} F1_pt={f1_p:.4f} AUC={auc_s:.4f} OV={ov:.4f} ({time.time()-t0:.1f}s)")

        # --- XGBoost ---
        print("\n[XGBoost]")
        t0 = time.time()
        _, act_probs = run_xgboost(X_tr, y_act_tr, X_val, y_act_val, "multiclass", n_classes=N_ACTION)
        act_probs = apply_action_rules(act_probs, val_sn)
        _, pt_probs = run_xgboost(X_tr, y_pt_tr, X_val, y_pt_val, "multiclass", n_classes=N_POINT)
        _, srv_probs = run_xgboost(X_tr, y_srv_tr, X_val, y_srv_val, "binary")
        f1_a = macro_f1(y_act_val, act_probs, N_ACTION)
        f1_p = macro_f1(y_pt_val, pt_probs, N_POINT)
        auc_s = roc_auc_score(y_srv_val, srv_probs)
        ov = 0.4*f1_a + 0.4*f1_p + 0.2*auc_s
        results["XGBoost"]["act"].append(f1_a)
        results["XGBoost"]["pt"].append(f1_p)
        results["XGBoost"]["srv"].append(auc_s)
        results["XGBoost"]["ov"].append(ov)
        print(f"  F1_act={f1_a:.4f} F1_pt={f1_p:.4f} AUC={auc_s:.4f} OV={ov:.4f} ({time.time()-t0:.1f}s)")

        # --- CatBoost ---
        print("\n[CatBoost]")
        t0 = time.time()
        _, act_probs = run_catboost(X_tr, y_act_tr, X_val, y_act_val, "multiclass", n_classes=N_ACTION)
        act_probs = apply_action_rules(act_probs, val_sn)
        _, pt_probs = run_catboost(X_tr, y_pt_tr, X_val, y_pt_val, "multiclass", n_classes=N_POINT)
        _, srv_probs = run_catboost(X_tr, y_srv_tr, X_val, y_srv_val, "binary")
        f1_a = macro_f1(y_act_val, act_probs, N_ACTION)
        f1_p = macro_f1(y_pt_val, pt_probs, N_POINT)
        auc_s = roc_auc_score(y_srv_val, srv_probs)
        ov = 0.4*f1_a + 0.4*f1_p + 0.2*auc_s
        results["CatBoost"]["act"].append(f1_a)
        results["CatBoost"]["pt"].append(f1_p)
        results["CatBoost"]["srv"].append(auc_s)
        results["CatBoost"]["ov"].append(ov)
        print(f"  F1_act={f1_a:.4f} F1_pt={f1_p:.4f} AUC={auc_s:.4f} OV={ov:.4f} ({time.time()-t0:.1f}s)")

        # --- Transformer ---
        print("\n[Transformer]")
        t0 = time.time()
        tr_idx_tf, val_idx_tf = fold_splits_tf[fold]
        tr_samples = [seq_samples[i] for i in tr_idx_tf]
        val_samples = [seq_samples[i] for i in val_idx_tf]

        tf_model, (act_probs, pt_probs, srv_probs), best_ov = run_transformer_fold(
            tr_samples, val_samples, d_model=128, nhead=8, n_layers=2,
            n_players=n_players, epochs=30, batch_size=256,
        )

        y_act_tf = np.array([s["y_action"] for s in val_samples])
        y_pt_tf = np.array([s["y_point"] for s in val_samples])
        y_srv_tf = np.array([s["y_server"] for s in val_samples])

        f1_a = macro_f1(y_act_tf, act_probs, N_ACTION)
        f1_p = macro_f1(y_pt_tf, pt_probs, N_POINT)
        auc_s = roc_auc_score(y_srv_tf, srv_probs) if len(np.unique(y_srv_tf)) > 1 else 0.5
        ov = 0.4*f1_a + 0.4*f1_p + 0.2*auc_s
        results["Transformer"]["act"].append(f1_a)
        results["Transformer"]["pt"].append(f1_p)
        results["Transformer"]["srv"].append(auc_s)
        results["Transformer"]["ov"].append(ov)
        print(f"  F1_act={f1_a:.4f} F1_pt={f1_p:.4f} AUC={auc_s:.4f} OV={ov:.4f} ({time.time()-t0:.1f}s)")

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print(f"\n{'='*70}")
    print("FINAL RESULTS (mean ± std across folds)")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'F1_action':>12} {'F1_point':>12} {'AUC_server':>12} {'Overall':>12}")
    print("-" * 65)

    best_model = None
    best_ov = -1
    for name, r in results.items():
        m_act = np.mean(r["act"])
        m_pt = np.mean(r["pt"])
        m_srv = np.mean(r["srv"])
        m_ov = np.mean(r["ov"])
        s_ov = np.std(r["ov"])
        print(f"{name:<15} {m_act:>10.4f}   {m_pt:>10.4f}   {m_srv:>10.4f}   {m_ov:.4f}±{s_ov:.4f}")
        if m_ov > best_ov:
            best_ov = m_ov
            best_model = name

    print(f"\nBest model: {best_model} (OV={best_ov:.4f})")
    print(f"Baseline: 0.2800")


if __name__ == "__main__":
    main()
