import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem



# ─────────────────────────────────────────────
# LOADING ACTIVITY
# ─────────────────────────────────────────────


def load_activity(target, assay_type, standard_type, standard_units):
    activity_client = new_client.activity

    results = activity_client.filter(
        target_chembl_id=target,
        assay_type=assay_type,
        standard_type=standard_type,
        standard_units=standard_units
    ).only("canonical_smiles", "standard_value")

    data = pd.DataFrame.from_records(results)

    if data.empty:
        raise ValueError("No data returned from ChEMBL query")

    # ── basic cleaning ─────────────────────
    data["standard_value"] = pd.to_numeric(data["standard_value"], errors="coerce")
    data = data.dropna(subset=["canonical_smiles", "standard_value"])

    # ── aggregation ───────────────────
    data = (
        data.groupby("canonical_smiles", as_index=False)
        .median(numeric_only=True)
    )

    # ── RDKit parsing ────────────────
    data["mol"] = data["canonical_smiles"].apply(Chem.MolFromSmiles)
    data = data.dropna(subset=["mol"]).reset_index(drop=True)

    return data




# ─────────────────────────────────────────────
# 
# ─────────────────────────────────────────────





