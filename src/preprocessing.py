"""
LBVS Preprocessing Module
=========================================
Supports:
- DNN (PyTorch)
- XGBoost
- Random Forest

Design goals:
- No leakage (fold-safe scaling)
- Consistent descriptor + fingerprint handling
- Reproducible CV-ready preprocessing
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Dict, Optional


# ─────────────────────────────────────────────
# Configuration (domain knowledge layer)
# ─────────────────────────────────────────────

TARGET_COL = "pIC50"
GROUP_COL = "scaffold"

DESCRIPTOR_COLS = [
    "MolWt", "HeavyAtomCount", "MolLogP", "TPSA", "NumHDonors",
    "NumHAcceptors", "NumRotatableBonds", "RingCount",
    "NumAromaticRings", "FractionCSP3", "MaxAbsPartialCharge",
    "NumHeteroatoms"
]


# ─────────────────────────────────────────────
# Dataset container
# ─────────────────────────────────────────────

@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_cols: list

    desc_mask: np.ndarray
    fp_mask: np.ndarray

    y_mean: float
    y_std: float


# ─────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────

def load_dataset(path: str) -> DatasetBundle:
    """
    Load raw CSV and construct structured dataset bundle.
    """

    df = pd.read_csv(path)

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, GROUP_COL]]

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    groups = df[GROUP_COL].values

    # Safety cleaning (critical for descriptor generation artifacts)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature masks (descriptor vs fingerprint separation)
    desc_mask = np.array([c in DESCRIPTOR_COLS for c in feature_cols], dtype=bool)
    fp_mask = ~desc_mask

    # Target normalization (global, NOT fold-aware here)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y)) if float(np.std(y)) > 0 else 1.0
    y_norm = (y - y_mean) / y_std

    return DatasetBundle(
        X=X,
        y=y_norm.astype(np.float32),
        groups=groups,
        feature_cols=feature_cols,
        desc_mask=desc_mask,
        fp_mask=fp_mask,
        y_mean=y_mean,
        y_std=y_std
    )


# ─────────────────────────────────────────────
# Fold-safe feature scaling
# ─────────────────────────────────────────────

def scale_fold(
    X: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    desc_mask: np.ndarray,
    scaler_type: str = "standard"
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Leakage-safe scaling:
    - Fit ONLY on training descriptors
    - Apply to train + validation
    - Fingerprints remain untouched
    """

    scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()

    X_train = X[train_idx].copy()
    X_val = X[val_idx].copy()

    # Fit ONLY on descriptor columns of training set
    scaler.fit(X_train[:, desc_mask])

    # Transform descriptors only
    X_train[:, desc_mask] = scaler.transform(X_train[:, desc_mask])
    X_val[:, desc_mask] = scaler.transform(X_val[:, desc_mask])

    # Final safety cleanup
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    return X_train.astype(np.float32), X_val.astype(np.float32), scaler


# ─────────────────────────────────────────────
# Full dataset scaling (deployment only)
# ─────────────────────────────────────────────

def fit_full_scaler(X: np.ndarray, desc_mask: np.ndarray, scaler_type: str = "standard"):
    """
    Fit scaler on full dataset (ONLY for final deployed model).
    """

    scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
    scaler.fit(X[:, desc_mask])
    return scaler


def apply_scaler(X: np.ndarray, scaler, desc_mask: np.ndarray) -> np.ndarray:
    """
    Apply pretrained scaler (inference time).
    """

    X_scaled = X.copy()
    X_scaled[:, desc_mask] = scaler.transform(X[:, desc_mask])

    return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ─────────────────────────────────────────────
# Target utilities
# ─────────────────────────────────────────────

def denormalize(y_pred: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    return (y_pred * y_std) + y_mean


# ─────────────────────────────────────────────
# Validation utility (research safety check)
# ─────────────────────────────────────────────

def validate_bundle(bundle: DatasetBundle):
    """
    Ensures dataset integrity before training.
    """

    assert bundle.X.shape[0] == len(bundle.y), "X/y size mismatch"
    assert bundle.desc_mask.sum() + bundle.fp_mask.sum() == bundle.X.shape[1], "Feature mask error"

    print("\n📊 Preprocessing Validation")
    print("=" * 50)
    print(f"Samples        : {len(bundle.y)}")
    print(f"Features       : {bundle.X.shape[1]}")
    print(f"Descriptors    : {bundle.desc_mask.sum()}")
    print(f"Fingerprints   : {bundle.fp_mask.sum()}")
    print(f"Target mean    : {bundle.y_mean:.4f}")
    print(f"Target std     : {bundle.y_std:.4f}")
    print("=" * 50)

