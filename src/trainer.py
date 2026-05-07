"""
trainer.py
=========================================
Unified training pipeline for LBVS models:

Supports:
- DNN (PyTorch)
- XGBoost
- Random Forest

Features:
- Shared scaffold-aware CV (from splitter.py)
- Fold-safe preprocessing (from preprocessing.py)
- Optuna hyperparameter optimization
- Consistent evaluation interface
- Optional OOF collection for stacking / analysis
"""

import numpy as np
import torch
import optuna

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

from torch.utils.data import DataLoader, TensorDataset
from src.preprocessing import scale_fold
from src.models import LBVSDNN   # your DNN module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ─────────────────────────────────────────────
# DNN training per fold
# ─────────────────────────────────────────────

def train_dnn_fold(model, train_loader, val_loader, optimizer, loss_fn, epochs=50):
    best_rmse = float("inf")

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
                trues.append(yb.numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        score = rmse(trues, preds)
        best_rmse = min(best_rmse, score)

    return best_rmse


# ─────────────────────────────────────────────
# OPTUNA: DNN
# ─────────────────────────────────────────────

def dnn_objective(trial, bundle, folds):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    n_layers = trial.suggest_int("n_layers", 2, 4)

    hidden_dims = [
        trial.suggest_int(f"h{i}", 64, 512, step=64)
        for i in range(n_layers)
    ]

    batch_size = trial.suggest_categorical("batch", [32, 64, 128])

    fold_scores = []

    for train_idx, val_idx in folds:

        X_train, X_val, _ = scale_fold(
            bundle.X, train_idx, val_idx, bundle.desc_mask
        )

        y_train = bundle.y[train_idx]
        y_val = bundle.y[val_idx]

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )

        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = LBVSDNN(
            input_dim=bundle.X.shape[1],
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = torch.nn.HuberLoss()

        score = train_dnn_fold(
            model, train_loader, val_loader,
            optimizer, loss_fn
        )

        fold_scores.append(score)

        trial.report(np.mean(fold_scores), len(fold_scores))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(fold_scores))


# ─────────────────────────────────────────────
# OPTUNA: XGBoost
# ─────────────────────────────────────────────

def xgb_objective(trial, bundle, folds):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_jobs": -1
    }

    scores = []

    for train_idx, val_idx in folds:

        X_train, X_val, _ = scale_fold(
            bundle.X, train_idx, val_idx, bundle.desc_mask
        )

        y_train = bundle.y[train_idx]
        y_val = bundle.y[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        scores.append(rmse(y_val, preds))

    return float(np.mean(scores))


# ─────────────────────────────────────────────
# OPTUNA: Random Forest
# ─────────────────────────────────────────────

def rf_objective(trial, bundle, folds):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "n_jobs": -1
    }

    scores = []

    for train_idx, val_idx in folds:

        X_train, X_val, _ = scale_fold(
            bundle.X, train_idx, val_idx, bundle.desc_mask
        )

        y_train = bundle.y[train_idx]
        y_val = bundle.y[val_idx]

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        scores.append(rmse(y_val, preds))

    return float(np.mean(scores))


# ─────────────────────────────────────────────
# RUN OPTUNA STUDY (generic)
# ─────────────────────────────────────────────

def run_optuna(objective_fn, bundle, folds, n_trials=50):

    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective_fn(trial, bundle, folds),
        n_trials=n_trials
    )

    return study