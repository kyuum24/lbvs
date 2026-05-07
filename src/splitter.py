"""
splitter.py

Scaffold-aware splitting utilities for LBVS pipelines.

Features:
- Reproducible outer splits (GroupShuffleSplit)
- Nested CV support (GroupKFold)
- Leakage detection (critical for cheminformatics)
- Save/load splits for consistent experiments
- Model-agnostic (DNN, XGB, RF)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import joblib



# ─────────────────────────────────────────────
# SCAFFOLD SPLIT
# ─────────────────────────────────────────────

class ScaffoldSplitter:
    """
    Scaffold-aware data splitter.

    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # ── Outer Split (Train/Test) ─────────────────

    def get_outer_splits(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        groups: Union[np.ndarray, list],
        n_splits: int = 5,
        train_size: float = 0.8
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate scaffold-aware outer splits.

        Returns
        -------
        List of (train_idx, test_idx)
        """
        groups = np.asarray(groups)

        gss = GroupShuffleSplit(
            n_splits=n_splits,
            train_size=train_size,
            random_state=self.random_state
        )

        return list(gss.split(X, groups=groups))
 
    # ── Inner CV ─────────────────────────────────

    def get_inner_cv(self, n_splits: int = 3) -> GroupKFold:
        """
        Returns GroupKFold for nested CV.
        """
        return GroupKFold(n_splits=n_splits)

    # ── Split Summary + Leakage Check ────────────

    def print_split_summary(self, groups, splits):
        """
        Prints split statistics and checks scaffold leakage.
        """
        groups = np.asarray(groups)

        print("\n🧪 Scaffold Split Report")
        print("=" * 65)

        for i, (tr_idx, te_idx) in enumerate(splits):
            tr_groups = np.unique(groups[tr_idx])
            te_groups = np.unique(groups[te_idx])

            leakage = set(tr_groups).intersection(set(te_groups))

            print(
                f"Fold {i+1:02d} | "
                f"Train: {len(tr_idx):>5} ({len(tr_groups):>4} scaf) | "
                f"Test: {len(te_idx):>5} ({len(te_groups):>4} scaf) | "
                f"Leakage: {len(leakage)}"
            )

        print("=" * 65)

    
    # ── Save / Load Splits ───────────────────────
    
    def save_splits(self, splits, path: str = "outer_splits.pkl"):
        """
        Save splits to disk for reproducibility.
        """
        joblib.dump(splits, path)
        print(f"✔ Splits saved to: {path}")

    def load_splits(self, path: str = "outer_splits.pkl"):
        """
        Load previously saved splits.
        """
        splits = joblib.load(path)
        print(f"✔ Splits loaded from: {path}")
        return splits



# ─────────────────────────────────────────────
# SPLITS BUILDER
# ─────────────────────────────────────────────
def build_splits(
    X,
    groups,
    outer_splits: int = 5,
    train_size: float = 0.8,
    inner_splits: int = 3,
    random_state: int = 42,
    verbose: bool = True
):
    """
    Builder for LBVS pipelines.

    Returns
    -------
    splitter : ScaffoldSplitter
    outer    : list of splits
    inner    : GroupKFold
    """

    splitter = ScaffoldSplitter(random_state=random_state)

    outer = splitter.get_outer_splits(
        X, groups,
        n_splits=outer_splits,
        train_size=train_size
    )

    inner = splitter.get_inner_cv(n_splits=inner_splits)

    if verbose:
        splitter.print_split_summary(groups, outer)

    return splitter, outer, inner