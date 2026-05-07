import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator




# ─────────────────────────────────────────────
# DEFAULT DESCRIPTOR LIST
# ─────────────────────────────────────────────
DESCRIPTOR_COLS = [
    "MolWt", "HeavyAtomCount", "MolLogP", "TPSA", "NumHDonors",
    "NumHAcceptors", "NumRotatableBonds", "RingCount",
    "NumAromaticRings", "FractionCSP3", "MaxAbsPartialCharge", "NumHeteroatoms"
]



# ─────────────────────────────────────────────
# Descriptors validation 
# ─────────────────────────────────────────────

def _validate_descriptors(desc_names):
    valid = set([name for name, _ in Descriptors._descList])

    invalid = set(desc_names) - valid

    if invalid:
        raise ValueError(
            f"Unknown RDKit descriptors: {invalid}\n"
            f"Check spelling or RDKit version."
        )
    

# ─────────────────────────────────────────────
# Core featurizer
# ─────────────────────────────────────────────
class MolecularFeaturizer(BaseEstimator, TransformerMixin):

    def __init__(self, desc_names="default", radius=2, fp_size=1024):
        
        if desc_names == "default":
            self.desc_names = DESCRIPTOR_COLS
        elif desc_names is None:
            self.desc_names = []
        else:
            self.desc_names = desc_names 
        
        self.radius = radius
        self.fp_size = fp_size
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.fp_size)
        self.calculator = None
        if self.desc_names:
            _validate_descriptors(self.desc_names)
            self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self.desc_names)
        

    def fit(self, X, y=None):
        return self

    # ── internal helpers ───────────────────────
    def _get_descriptors(self, mol):
        if self.calculator is None:
            return np.array([], dtype=np.float32)

        return np.array(
            self.calculator.CalcDescriptors(mol),
            dtype=np.float32
        )
    
    def _get_fp(self, mol):
        fp = self.morgan_gen.GetFingerprint(mol)
        return np.array(fp, dtype=np.uint8)

    # ── single molecule ────────────────────────
    def compute_features(self, mol):

        if mol is None:
            raise ValueError("Received None molecule in featurizer")
        
        desc = self._get_descriptors(mol)
        fp   = self._get_fp(mol)

        
        return np.concatenate([desc, fp])
    
    # ── batch transform ────────────────────────
    def transform(self, X):
        
        features_list = [self.compute_features(m) for m in X]
        
        cols = self.desc_names + [f"FP_{i+1}" for i in range(self.fp_size)]
        
        
        index = X.index if hasattr(X, 'index') else None
        return pd.DataFrame(features_list, columns=cols, index=index)
    

    


# ─────────────────────────────────────────────
# LOADING SCAFFOLDS
# ─────────────────────────────────────────────


def mol_to_scaffold(mol):
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaffold)
        return smi if smi else "acyclic"
    except Exception as e:
        raise ValueError("Scaffold extraction failed") from e

def get_scaffolds(mol_list):

    scaffolds = []

    for mol in mol_list:
        if mol is None:
            raise ValueError(f"Invalid SMILES: {mol}")

        scaffolds.append(mol_to_scaffold(mol))
    return pd.Series(scaffolds, name="scaffold")




# ─────────────────────────────────────────────
# LOADING MOLECULES
# ─────────────────────────────────────────────

def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    return mol

import pandas as pd

def get_mols(smiles_series):
    mols = smiles_series.apply(smiles_to_mol)
    return mols

# ─────────────────────────────────────────────
# LOADING pIC50
# ─────────────────────────────────────────────


def get_pic50(ic50):
   
   MAX_IC50_NM = 1e8 
   ic50_clipped = np.clip(ic50, 1e-12, MAX_IC50_NM)
   pic50 = -np.log10(ic50_clipped * 1e-9) # assumes IC50 is in nM
   
   index = ic50.index if isinstance(ic50, pd.Series) else None
   return pd.Series(pic50, name="pIC50", index=index)


# ─────────────────────────────────────────────
# 
# ─────────────────────────────────────────────