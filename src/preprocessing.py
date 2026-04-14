"""
preprocessing.py
────────────────────────────────────────────────────────────────────────────────
Data loading, validation, and preprocessing pipeline.

Design decisions:
  • MICE imputation (IterativeImputer) for clinical realism — single imputation
    is known to underestimate variance; in a full paper, multiple imputation
    (m≥20) with Rubin's rules should be used for inference. We note this as a
    limitation and provide the single-imputation version for the ML pipeline.
  • Protected attributes (race, sex, insurance) are NEVER used as model inputs
    per GUIDE framework (npj Digital Medicine, 2024); they are audit-only.
  • Robust StandardScaler used instead of MinMaxScaler — less sensitive to
    outliers common in ICU labs (e.g., creatinine 0.2–20 mg/dL).
  • Stratified split on outcome × race to ensure subgroup representation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple, Dict


# Columns used as ML features (protected attributes excluded)
FEATURE_COLS = [
    "age", "heart_rate", "map", "spo2", "temperature", "resp_rate",
    "wbc", "lactate", "creatinine", "bilirubin", "platelet",
    "sofa_score", "charlson_index", "vasopressor_use",
    "mechanical_vent", "time_to_antibiotics_h",
]

PROTECTED_ATTRS = ["race", "sex", "insurance"]
TARGET_COL      = "hospital_mortality"
ID_COL          = "subject_id"


def load_and_validate(path: str | Path) -> pd.DataFrame:
    """Load CSV and run schema / range validation."""
    df = pd.read_csv(path)

    required = set(FEATURE_COLS + PROTECTED_ATTRS + [TARGET_COL, ID_COL])
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Range assertions (clinical sanity checks)
    assert df["age"].between(18, 100).all(),          "Age out of range"
    assert df["sofa_score"].between(0, 24).all(),     "SOFA out of range"
    assert df[TARGET_COL].isin([0, 1]).all(),          "Outcome not binary"

    return df


def impute_features(df: pd.DataFrame,
                    random_state: int = 42) -> Tuple[pd.DataFrame, IterativeImputer]:
    """
    MICE imputation on feature columns.
    Returns imputed DataFrame and fitted imputer for test-set consistency.
    """
    imputer = IterativeImputer(
        max_iter=10,
        random_state=random_state,
        min_value=0,       # clinical labs are non-negative
        verbose=0,
    )
    X_imp = imputer.fit_transform(df[FEATURE_COLS])
    df_imp = df.copy()
    df_imp[FEATURE_COLS] = X_imp
    return df_imp, imputer


def encode_protected_attrs(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode protected attributes for subgroup indexing (audit only)."""
    df = df.copy()
    for col in PROTECTED_ATTRS:
        df[col] = df[col].astype("category")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size:  float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/15/15 split on outcome × race to preserve subgroup prevalence.
    Returns (train, val, test) DataFrames.
    """
    # Create stratification key = outcome + race
    strat_key = df[TARGET_COL].astype(str) + "_" + df["race"].astype(str)

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size + val_size, random_state=random_state
    )
    train_idx, temp_idx = next(sss1.split(df, strat_key))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp  = df.iloc[temp_idx].reset_index(drop=True)

    strat_temp = (df_temp[TARGET_COL].astype(str) + "_" +
                  df_temp["race"].astype(str))
    val_frac = val_size / (test_size + val_size)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - val_frac, random_state=random_state
    )
    val_idx, test_idx = next(sss2.split(df_temp, strat_temp))
    df_val  = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    return df_train, df_val, df_test


def scale_features(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    """Fit RobustScaler on train, apply to val and test."""
    scaler = RobustScaler()
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_test  = df_test.copy()

    df_train[FEATURE_COLS] = scaler.fit_transform(df_train[FEATURE_COLS])
    df_val[FEATURE_COLS]   = scaler.transform(df_val[FEATURE_COLS])
    df_test[FEATURE_COLS]  = scaler.transform(df_test[FEATURE_COLS])

    return df_train, df_val, df_test, scaler


def get_Xy(df: pd.DataFrame):
    """Return (X, y) arrays and the protected-attribute Series dict."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    A = {col: df[col] for col in PROTECTED_ATTRS}
    return X, y, A


def preprocessing_report(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
) -> Dict:
    """Print and return basic statistics about the processed splits."""
    stats = {}
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        n       = len(df)
        n_pos   = df[TARGET_COL].sum()
        prev    = n_pos / n
        stats[name] = {"n": n, "n_pos": int(n_pos), "prevalence": round(prev, 4)}

    print("\n── Cohort splits ──────────────────────────────────────────────────")
    for split, s in stats.items():
        print(f"  {split:5s}: n={s['n']:,}  deaths={s['n_pos']:,}  "
              f"prevalence={s['prevalence']:.1%}")

    print("\n── Race × mortality (test set) ────────────────────────────────────")
    race_mort = df_test.groupby("race")[TARGET_COL].agg(["sum", "count", "mean"])
    race_mort.columns = ["deaths", "n", "mortality_rate"]
    race_mort["mortality_rate"] = race_mort["mortality_rate"].round(3)
    print(race_mort.to_string())
    return stats
