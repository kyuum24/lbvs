import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from IPython.display import display 
from sklearn.decomposition import PCA
from umap import UMAP



# ─────────────────────────────────────────────
# COMPOUND DASHBOARD
# ─────────────────────────────────────────────


def plot_physchem_distributions(X_desc):
    """
    Plot physicochemical descriptor distributions with Ro5 annotations.
    
    Parameters
    ----------
    X_desc : pd.DataFrame
        DataFrame containing only descriptor columns.
    """
    sns.set_theme(style="whitegrid")

    RO5_RULES = {
        "MolWt": {"threshold": 500, "direction": "low"},
        "MolLogP": {"threshold": 5, "direction": "low"},
        "TPSA": {"threshold": 140, "direction": "low"},
        "NumHDonors": {"threshold": 5, "direction": "low"},
        "NumHAcceptors": {"threshold": 10, "direction": "low"},
        "NumRotatableBonds": {"threshold": 10, "direction": "low"},
    }

    descriptors = list(X_desc.columns)

    n = len(descriptors)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for i, desc in enumerate(descriptors):
        ax = axes[i]

        data = X_desc[desc].dropna()
        if data.empty:
            ax.set_visible(False)
            continue

        sns.histplot(
            data,
            bins=40,
            kde=len(data) < 5000,
            stat="density",
            ax=ax,
            color="steelblue",
            alpha=0.6,
            line_kws={"linewidth": 2, "color": "black"}
        )

        mean_val = data.mean()
        median_val = data.median()

        ax.axvline(mean_val, color='darkred', linestyle="--", linewidth=2,
                   label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color='darkorange', linestyle=":", linewidth=2,
                   label=f"Median: {median_val:.2f}")

        
        # ── Ro5 rules ────────────────────────
        
        if desc in RO5_RULES:
            thr = RO5_RULES[desc]["threshold"]
            direction = RO5_RULES[desc]["direction"]

            ax.axvline(thr, color="black", linestyle="-", linewidth=2,
                       label=f"Limit: {thr}")

            x_min, x_max = ax.get_xlim()

            if direction == "low":
                ax.axvspan(thr, x_max, color="crimson", alpha=0.1)
                pass_rate = (data <= thr).mean() * 100
            else:
                ax.axvspan(x_min, thr, color="crimson", alpha=0.1)
                pass_rate = (data >= thr).mean() * 100

            ax.text(
                0.98, 0.85,
                f"Pass: {pass_rate:.1f}%",
                transform=ax.transAxes,
                ha="right",
                fontsize=9
            )

            rule_label = "Drug-likeness rule"
        else:
            rule_label = "No rule"

        ax.set_title(
            f"{desc} (n={len(data)})\n[{rule_label}]",
            fontweight="bold",
            fontsize=12
        )

        ax.set_xlabel("")
        ax.set_ylabel("Density")
        ax.legend(fontsize="small", loc='upper right', frameon=True)

    # hide empty plots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Physicochemical Property Distributions",
                 fontweight="bold", fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# SAR RELATIONSHIP
# ─────────────────────────────────────────────


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import pandas as pd

def plot_sar_relationships(
    X_desc,
    y,
    active_threshold=6
):
    """
    SAR analysis: descriptors vs pIC50 (or any y).

    Parameters
    ----------
    X_desc : pd.DataFrame
        Descriptor matrix (ONLY descriptors)
    y : pd.Series or array-like
        Target values (e.g. pIC50)
    active_threshold : float
        Activity cutoff (default = 6 for pIC50)
    """

    sns.set_theme(style="whitegrid")

    # Ensure alignment (this is critical)
    y = pd.Series(y, index=X_desc.index)

    descriptors = X_desc.columns.tolist()

    n = len(descriptors)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for i, desc in enumerate(descriptors):
        ax = axes[i]

        # Pairwise clean data
        data = pd.DataFrame({
            desc: X_desc[desc],
            "y": y
        }).dropna()

        # Outlier trimming (robust SAR)
        data = data[
            data[desc].between(data[desc].quantile(0.01), data[desc].quantile(0.99)) &
            data["y"].between(data["y"].quantile(0.01), data["y"].quantile(0.99))
        ]

        if len(data) < 2:
            ax.set_visible(False)
            continue

        # Correlations
        p_corr, _ = pearsonr(data[desc], data["y"])
        s_corr, _ = spearmanr(data[desc], data["y"])

        # Adaptive smoothing
        use_lowess = len(data) < 3000

        sns.regplot(
            data=data,
            x=desc,
            y="y",
            ax=ax,
            lowess=use_lowess,
            scatter_kws={"alpha": 0.35, "s": 25},
            line_kws={"linewidth": 2.5}
        )

        # Activity threshold (pIC50 default = 6)
        ax.axhline(active_threshold, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axhspan(active_threshold, ax.get_ylim()[1], alpha=0.05)

        ax.set_title(
            f"{desc}\n"
            f"r = {p_corr:.2f} | ρ = {s_corr:.2f} | n = {len(data)}",
            fontsize=11,
            fontweight="bold"
        )

        ax.set_xlabel(desc)
        ax.set_ylabel("Activity")

    # Hide unused plots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Structure–Activity Relationships (SAR)",
        fontsize=16,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MORGAN PCA
# ─────────────────────────────────────────────


def plot_morgan_pca(X_fp, color_by=None, title="Chemical Space PCA (Morgan Fingerprints)"):

    sns.set_theme(style="white")

    X = X_fp.values if hasattr(X_fp, 'values') else X_fp

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_

    color_name = None
    if color_by is not None:
        color_name = getattr(color_by, "name", "Value")
        color_by = pd.Series(color_by).fillna(color_by.mean())

    plt.figure(figsize=(10, 7))

    if color_by is not None:
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=color_by,
            cmap="magma",
            s=40,
            alpha=0.6,
            edgecolors='none'
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label(color_name, fontsize=12, fontweight='bold')
    else:
        plt.scatter(
            X_pca[:, 0], X_pca[:, 1],
            color="steelblue",
            s=40,
            alpha=0.5
        )

    plt.xlabel(f"PC1 ({var_exp[0]*100:.1f}% Variance)", fontsize=11)
    plt.ylabel(f"PC2 ({var_exp[1]*100:.1f}% Variance)", fontsize=11)

    plt.title(f"{title}\n(Structural Diversity Map)", fontweight="bold", fontsize=14)

    sns.despine()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MORGAN UMAP
# ─────────────────────────────────────────────


def plot_morgan_umap(
    X_fp,
    color_by=None,
    n_neighbors=10,
    min_dist=0.05,
    metric="jaccard",
    title="UMAP: Local Chemical Topology"
):

    sns.set_theme(style="white")

    X = np.asarray(X_fp, dtype=np.float32)

    if np.isnan(X).any():   
        raise ValueError("Fingerprint matrix contains NaNs")

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )

    X_umap = umap_model.fit_transform(X)

    if color_by is not None:
        color_series = pd.Series(color_by)
        color_name = getattr(color_by, "name", "Target Value")
        color_final = color_series.fillna(color_series.mean())
    else:
        color_final = None

    plt.figure(figsize=(10, 8))

    if color_final is not None:
        scatter = plt.scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=color_final,
            cmap="magma",
            s=35,
            alpha=0.8,
            edgecolors="none"
        )
        cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04)
        cbar.set_label(color_name, fontsize=12, fontweight="bold")
    else:
        plt.scatter(
            X_umap[:, 0], X_umap[:, 1],
            s=35,
            alpha=0.7,
            color="teal"
        )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    plt.title(
        f"{title}\n(n_neighbors={n_neighbors}, metric={metric})",
        fontweight="bold",
        fontsize=14
    )

    sns.despine()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# HEAT MAP
# ─────────────────────────────────────────────


def plot_activity_cliff_heatmap(
    X_fp,
    pic50,
    max_mols=50,
    sim_threshold=0.7,
    delta_threshold=1.0,
    title="Activity Cliff Heatmap"
):
    sns.set_theme(style="white")

    # --- subsample ---
    if len(X_fp) > max_mols:
        idx = np.random.RandomState(42).choice(len(X_fp), max_mols, replace=False)
        X_fp = X_fp.iloc[idx].reset_index(drop=True)
        pic50 = pic50.iloc[idx].reset_index(drop=True)

    n = len(X_fp)

    # --- fingerprints ---
    fps = [
        DataStructs.CreateFromBitString("".join(row.astype(int).astype(str)))
        for row in X_fp.values
    ]

    tanimoto_mat = np.zeros((n, n))
    delta_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            diff = abs(pic50.iloc[i] - pic50.iloc[j])

            tanimoto_mat[i, j] = tanimoto_mat[j, i] = sim
            delta_mat[i, j] = delta_mat[j, i] = diff

    cliff_score = tanimoto_mat * delta_mat
    print("Cliff score range:", np.min(cliff_score), "→", np.max(cliff_score))
    
    # --- cliff definition (more correct than product) ---
    cliff_mask = (tanimoto_mat >= sim_threshold) & (delta_mat >= delta_threshold)
        
    plt.figure(figsize=(9, 8))

    sns.heatmap(
        cliff_mask.astype(int),
        cmap="Reds",
        cbar_kws={"label": "Activity Cliff (1 = True)"},
        linewidths=0.3
    )

    plt.title(
        f"{title}\nSim ≥ {sim_threshold}, ΔpIC50 ≥ {delta_threshold}",
        fontweight="bold"
    )

    plt.xlabel("Molecules")
    plt.ylabel("Molecules")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# FEATURES IMPORTANCE
# ─────────────────────────────────────────────


def get_general_importance(model, feature_names, X_val=None, y_val=None):
  
    
    # ── XGBoost ─────────────────────────────
    
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        scores = booster.get_score(importance_type="gain")

        if all(k.startswith('f') and k[1:].isdigit() for k in scores.keys()):
            importance_arr = np.array([
                scores.get(f"f{i}", 0.0) for i in range(len(feature_names))
            ])
        else:
            importance_arr = np.array([
                scores.get(f, 0.0) for f in feature_names
            ])

    
    # ── Random Forest ────────────────────────
    
    elif hasattr(model, "feature_importances_"):
        importance_arr = np.asarray(model.feature_importances_)

        if len(importance_arr) != len(feature_names):
            raise ValueError("Feature/importance mismatch")

    
    # ── DNN or other models ──────────────────
    
    else:
        if X_val is None or y_val is None:
            raise ValueError("Permutation importance requires X_val and y_val")

        from sklearn.inspection import permutation_importance

        results = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=10,
            random_state=42
        )

        importance_arr = np.asarray(results.importances_mean)

    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_arr
    }).sort_values("Importance", ascending=False)


# ─────────────────────────────────────────────
# TOP FEATURES INTERPRETATION
# ─────────────────────────────────────────────


def interpret_features(
    feature_names,
    mols,
    X,
    fp_prefix="FP_",
    percentiles=(25, 50, 75),
    examples_per_bit=1,
    radius=2,
    fp_size=1024
):

    descriptor_set = {name for name, _ in Descriptors._descList}

    descriptor_results = []
    fingerprint_results = []

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size
    )

    used_fp_bits = set()

    for rank, fname in enumerate(feature_names, start=1):
        top_label = f"Top {rank:02d}"

        
        # ── Fingerprints ─────────────────────
    
        if fname.startswith(fp_prefix):

            try:
                bit_number = int(fname.replace(fp_prefix, "")) - 1
            except ValueError:
                warnings.warn(f"Invalid FP name: {fname}")
                continue

            if not (0 <= bit_number < fp_size):
                warnings.warn(f"Bit {bit_number+1} out of range")
                continue

            if fname not in X.columns or bit_number in used_fp_bits:
                continue

            on_indices = X.index[X[fname] == 1][:examples_per_bit]
            if len(on_indices) == 0:
                continue

            idx = on_indices[0]
            mol = mols.loc[idx] if hasattr(mols, "loc") else mols[idx]

            if mol is None:
                continue

            info = rdFingerprintGenerator.AdditionalOutput()
            info.AllocateBitInfoMap()
            gen.GetFingerprint(mol, additionalOutput=info)
            bit_map = info.GetBitInfoMap()

            if bit_number not in bit_map:
                continue

            img = Draw.DrawMorganBit(mol, bit_number, bit_map)

            fingerprint_results.append({
                "feature": fname,
                "rank": top_label,
                "mol_idx": idx,
                "image": img
            })

            used_fp_bits.add(bit_number)

        
        # ── Descriptors ──────────────────────
        
        elif fname in descriptor_set:

            if fname not in X.columns:
                continue

            values = X[fname].dropna()
            if values.empty:
                continue

            perc = np.percentile(values, percentiles)

            descriptor_results.append({
                "feature": fname,
                "rank": top_label,
                **{f"P{p}": v for p, v in zip(percentiles, perc)}
            })

        
        # ── Other features ───────────────────
        
        else:
            warnings.warn(f" feature skipped: {fname}")

    return {
        "descriptors": pd.DataFrame(descriptor_results),
        "fingerprints": fingerprint_results
    }

    # the way to use it:

    #print(result["descriptors"])
    #for item in result["fingerprints"]:
        #print(f"Feature: {item['feature']} | Mol index: {item['mol_idx']}  | Rank: {item["rank"]}")
        #display(item["image"])


# ─────────────────────────────────────────────
# 1 MOL TOP FEATURES
# ─────────────────────────────────────────────


def mol_features(
    mol,
    feature_names,
    fp_prefix="FP_",
    radius=2,
    fp_size=1024
):

    if mol is None:
        return

    descriptor_set = {name for name, _ in Descriptors._descList}

    # ───────────── fingerprint setup ─────────────
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size
    )

    info = rdFingerprintGenerator.AdditionalOutput()
    info.AllocateBitInfoMap()
    gen.GetFingerprint(mol, additionalOutput=info)
    bit_map = info.GetBitInfoMap()

    # ───────────── descriptor precompute ─────────
    all_desc = Descriptors.CalcMolDescriptors(mol)

    # ───────────── molecule display ──────────────
    display(Draw.MolToImage(mol, size=(300, 300), legend="Query Molecule"))

    total = len(feature_names)
    print(f"\n--- Present Features (Top {total}) ---\n")

    for i, feat in enumerate(feature_names, start=1):
        prefix = f"[{i:04d}/{total}]"

        # ───────────── fingerprints ──────────────
        if feat.startswith(fp_prefix):

            try:
                bit = int(feat.replace(fp_prefix, "")) - 1
            except ValueError:
                continue

            if not (0 <= bit < fp_size):
                continue

            if bit in bit_map:
                print(f"{prefix} ✔ {feat}: Present")
                display(Draw.DrawMorganBit(mol, bit, bit_map))

        # ───────────── descriptors ───────────────
        elif feat in descriptor_set:

            val = all_desc.get(feat, None)

            if val is not None:
                print(f"{prefix} ◈ {feat}: {val:.4f}")

    print("\n--- Analysis Complete ---")


# ─────────────────────────────────────────────
# pIC50 DISTRIBUTION
# ─────────────────────────────────────────────


def plot_pic50_distribution(pic50):
    
    sns.set_theme(style="whitegrid")
    
    pic50 = pic50.dropna()
    
    plt.figure(figsize=(9, 6))
    
    sns.histplot(
        pic50,
        bins=50,
        kde=True,
        stat="density",
        color="teal",
        alpha=0.6,
        line_kws={"linewidth": 3, "color": "black"}
    )
    
    mean_val = pic50.mean()
    median_val = pic50.median()
    
    plt.axvline(mean_val, color='crimson', linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f}")
    plt.axvline(median_val, color='darkorange', linestyle=":", linewidth=2, label=f"Median = {median_val:.2f}")
    
    # Domain-specific threshold
    plt.axvline(6, color='black', linestyle='-', linewidth=2, label="Active threshold (pIC50 = 6)")
    
    plt.xlabel("pIC50 Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Distribution of pIC50 Potency (n={len(pic50)})", fontsize=14, fontweight='bold')
    
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.show()


