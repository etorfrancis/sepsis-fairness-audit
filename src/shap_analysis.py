"""
shap_analysis.py
────────────────────────────────────────────────────────────────────────────────
SHAP-based feature importance with fairness attribution extension.

Two analyses:
  1. Standard SHAP summary — global feature importance for each model.

  2. Fairness-SHAP attribution (novel contribution) —
     For each feature f, compute the difference in mean |SHAP| between the
     most-disparate group (Black) and the reference group (White). Features
     with large positive gaps drive higher model attention to Black patients
     (which may encode proxy discrimination). Features with negative gaps
     are used less for Black patients (under-service).

     This operationalises the method proposed in:
     "Explainable AI for Fair Sepsis Mortality Predictive Model" (AIME'24)
     but extends it to continuous SHAP magnitudes across all features.

  3. Subgroup SHAP waterfall — per-group mean SHAP decomposition, showing
     which features push predictions up/down differentially by race.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, Tuple


FEATURE_COLS = [
    "age", "heart_rate", "map", "spo2", "temperature", "resp_rate",
    "wbc", "lactate", "creatinine", "bilirubin", "platelet",
    "sofa_score", "charlson_index", "vasopressor_use",
    "mechanical_vent", "time_to_antibiotics_h",
]


def compute_shap_values(
    model:     Any,
    X_test:    np.ndarray,
    model_name: str = "XGBoost",
    n_background: int = 200,
    max_evals: int = 500,
    rng_seed:  int = 42,
) -> shap.Explanation:
    """
    Compute SHAP values using the appropriate explainer for each model type.
    - TreeExplainer for XGBoost (exact, fast)
    - KernelExplainer for LR and MLP (sampled approximation)
    """
    rng = np.random.default_rng(rng_seed)
    print(f"  Computing SHAP for {model_name} ...")

    if model_name == "XGBoost":
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer(X_test[:500])   # cap for speed
    else:
        background = shap.sample(X_test, n_background,
                                 random_state=rng_seed)
        explainer  = shap.KernelExplainer(
            model.predict_proba, background
        )
        # KernelExplainer returns list [class_0, class_1]; take class_1
        raw = explainer.shap_values(X_test[:max_evals], silent=True)
        if isinstance(raw, list):
            raw = raw[1]
        shap_vals = shap.Explanation(
            values          = raw,
            base_values     = explainer.expected_value[1] if hasattr(
                explainer.expected_value, "__len__") else explainer.expected_value,
            data            = X_test[:max_evals],
            feature_names   = FEATURE_COLS,
        )

    return shap_vals


def global_feature_importance(
    shap_vals: shap.Explanation,
    model_name: str,
) -> pd.DataFrame:
    """Mean absolute SHAP per feature (global importance)."""
    vals   = shap_vals.values if hasattr(shap_vals, "values") else shap_vals
    if vals.ndim == 3:
        vals = vals[:, :, 1]  # binary classification: take positive class
    mean_abs = np.abs(vals).mean(axis=0)
    df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "mean_abs_shap": mean_abs,
        "model":      model_name,
    }).sort_values("mean_abs_shap", ascending=False)
    return df


def fairness_shap_attribution(
    shap_vals: shap.Explanation,
    race:      pd.Series,
    n_test:    int,
    ref_group: str = "White",
    cmp_group: str = "Black",
) -> pd.DataFrame:
    """
    Compute per-feature SHAP gap between comparison and reference group.

    A positive gap for feature f means the model assigns MORE weight to f
    for the comparison group — a potential source of proxy discrimination if
    f is correlated with race (e.g., time_to_antibiotics_h).
    """
    vals = shap_vals.values if hasattr(shap_vals, "values") else shap_vals
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    # Align race Series to the subset used for SHAP (first n_test rows)
    race_sub = race.iloc[:vals.shape[0]].reset_index(drop=True)

    rows = []
    for feat_idx, feat_name in enumerate(FEATURE_COLS):
        feat_shap = vals[:, feat_idx]
        ref_mask  = (race_sub == ref_group).values
        cmp_mask  = (race_sub == cmp_group).values

        if ref_mask.sum() < 10 or cmp_mask.sum() < 10:
            continue

        mean_abs_ref = np.abs(feat_shap[ref_mask]).mean()
        mean_abs_cmp = np.abs(feat_shap[cmp_mask]).mean()
        gap          = mean_abs_cmp - mean_abs_ref

        rows.append({
            "feature":        feat_name,
            "mean_abs_shap_ref": round(mean_abs_ref, 5),
            "mean_abs_shap_cmp": round(mean_abs_cmp, 5),
            "shap_gap":          round(gap, 5),
            "ref_group":      ref_group,
            "cmp_group":      cmp_group,
        })

    df = pd.DataFrame(rows).sort_values("shap_gap", ascending=False)
    return df


def subgroup_mean_shap(
    shap_vals: shap.Explanation,
    race:      pd.Series,
) -> pd.DataFrame:
    """
    Compute signed mean SHAP per feature × race group.
    Positive = pushes mortality prediction up for that group.
    """
    vals     = shap_vals.values if hasattr(shap_vals, "values") else shap_vals
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    race_sub = race.iloc[:vals.shape[0]].reset_index(drop=True)
    rows     = []

    for grp in race_sub.unique():
        mask = (race_sub == grp).values
        if mask.sum() < 10:
            continue
        mean_shap = vals[mask].mean(axis=0)
        for i, feat in enumerate(FEATURE_COLS):
            rows.append({
                "race":         grp,
                "feature":      feat,
                "mean_shap":    round(mean_shap[i], 5),
            })

    return pd.DataFrame(rows)


def run_shap_analysis(
    suite,
    X_test:   np.ndarray,
    A_test:   Dict[str, pd.Series],
) -> Dict[str, Any]:
    """Run full SHAP analysis for XGBoost (primary model) and LR (baseline)."""
    results = {}
    print("\n── SHAP Analysis ──────────────────────────────────────────────────")

    for model_name, model in [("XGBoost", suite.xgb), ("LR", suite.lr)]:
        shap_vals = compute_shap_values(model, X_test, model_name)
        importance_df = global_feature_importance(shap_vals, model_name)
        fairness_df   = fairness_shap_attribution(
            shap_vals, A_test["race"], X_test.shape[0]
        )
        subgrp_df = subgroup_mean_shap(shap_vals, A_test["race"])

        results[model_name] = {
            "shap_vals":   shap_vals,
            "importance":  importance_df,
            "fairness":    fairness_df,
            "subgroup":    subgrp_df,
        }

        print(f"\n  {model_name} — Top 5 features by global SHAP:")
        print(importance_df.head(5)[["feature", "mean_abs_shap"]].to_string(index=False))

        print(f"\n  {model_name} — Top 5 features by Black–White SHAP gap:")
        print(fairness_df.head(5)[
            ["feature", "mean_abs_shap_ref", "mean_abs_shap_cmp", "shap_gap"]
        ].to_string(index=False))

    return results
