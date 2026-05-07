import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from scipy.stats import pearsonr, spearmanr



# ─────────────────────────────────────────────
# OUT OF FOLD CONTAINER
# ─────────────────────────────────────────────


class OOFManager:
   
    
    def __init__(self, n_samples: int, target_name: str = "pIC50"):
        self.n_samples = n_samples
        self.target_name = target_name
        self.oof_preds = np.full(n_samples, np.nan, dtype=np.float32) # Use NaN to detect unfilled slots

    def update(self, preds: np.ndarray, indices: np.ndarray):
        """Update OOF array and ensure no double-filling or missing data."""
        self.oof_preds[indices] = preds.flatten()

    def finalize(self, y_true: np.ndarray) -> bool:
        """Checks if all samples were predicted exactly once."""
        missing = np.isnan(self.oof_preds).sum()
        if missing > 0:
            print(f"⚠️ Warning: {missing} samples were never predicted!")
            return False
        return True

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression and ranking metrics."""
        mask = ~np.isnan(y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        
        if len(yt) == 0: return {}

        # Error Metrics
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        mae = np.mean(np.abs(yt - yp))
        
        # R-squared
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Correlation (Ranking power is key for Virtual Screening)
        pearson, _ = pearsonr(yt, yp) if len(yt) > 1 else (0.0, 0.0)
        spearman, _ = spearmanr(yt, yp) if len(yt) > 1 else (0.0, 0.0)

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "pearson": float(pearson),
            "spearman": float(spearman)
        }

    def get_eval_df(self, y_true: np.ndarray, groups: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Constructs the evaluation DataFrame."""
        df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": self.oof_preds,
            "error": self.oof_preds - y_true,
            "abs_error": np.abs(self.oof_preds - y_true)
        })
        if groups is not None:
            df["scaffold"] = groups
        return df


# ─────────────────────────────────────────────
# SCAFFOLD'S PERFORMANCE
# ─────────────────────────────────────────────


def scaffold_summary(df: pd.DataFrame) -> pd.DataFrame:
    
    if "scaffold" not in df.columns:
        return pd.DataFrame()
        
    return df.groupby("scaffold").agg(
        count=("y_true", "count"),
        avg_error=("error", "mean"),
        rmse=("error", lambda x: np.sqrt(np.mean(x**2))),
        max_ae=("abs_error", "max")
    ).sort_values("rmse", ascending=False)

