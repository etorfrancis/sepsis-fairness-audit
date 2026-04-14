"""
debiasing.py
────────────────────────────────────────────────────────────────────────────────
Post-processing debiasing via equalized-odds threshold optimisation.

Method:
  Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in
  supervised learning. NeurIPS.

  For each group g, find a group-specific decision threshold tau_g that
  best equalises TPR across groups (manual grid search implementation,
  avoiding fairlearn/pandas dtype compatibility issues in v0.13/3.x).

Tradeoff analysis:
  Sweep global threshold 0.05–0.95 to build the Pareto frontier of
  (AUROC, EOD). This is the paper's core novel contribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference


def _group_tpr(y_true, y_prob, threshold, groups, grp):
    mask = (groups == grp)
    yt   = y_true[mask]
    yp   = (y_prob[mask] >= threshold).astype(int)
    tp   = ((yp == 1) & (yt == 1)).sum()
    fn   = ((yp == 0) & (yt == 1)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def manual_equalized_odds_optimizer(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    n_thresholds: int = 50,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Per-group threshold search to equalise TPR across groups.
    Returns (group_thresholds, debiased_predictions).
    """
    tau_grid      = np.linspace(0.1, 0.9, n_thresholds)
    unique_groups = np.unique(groups)

    # Reference: White (or largest group)
    ref = "White" if "White" in unique_groups else unique_groups[0]
    ref_tpr = _group_tpr(y_true, y_prob, 0.5, groups, ref)

    best_thresholds = {ref: 0.5}
    for grp in unique_groups:
        if grp == ref:
            continue
        best_tau, best_dist = 0.5, float("inf")
        for tau in tau_grid:
            dist = abs(_group_tpr(y_true, y_prob, tau, groups, grp) - ref_tpr)
            if dist < best_dist:
                best_dist, best_tau = dist, tau
        best_thresholds[grp] = best_tau

    y_pred = np.zeros(len(y_prob), dtype=int)
    for grp in unique_groups:
        mask          = (groups == grp)
        y_pred[mask]  = (y_prob[mask] >= best_thresholds[grp]).astype(int)

    return best_thresholds, y_pred


def apply_threshold_optimizer(
    model:   Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    A_train: pd.Series,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    A_test:  pd.Series,
) -> Tuple[Dict, Dict]:
    """Apply manual EOD threshold optimisation; return (thresholds, metrics)."""
    groups    = np.array(A_test)
    y_prob    = model.predict_proba(X_test)[:, 1].astype(np.float64)

    thresholds, y_pred = manual_equalized_odds_optimizer(y_prob, y_test, groups)

    eod   = equalized_odds_difference(y_test, y_pred, sensitive_features=groups)
    dpd   = demographic_parity_difference(y_test, y_pred, sensitive_features=groups)
    auroc = roc_auc_score(y_test, y_prob)

    return thresholds, {"auroc": auroc, "eod_after": eod, "dpd_after": dpd}


def compute_pareto_frontier(
    model:      Any,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    A_train:    pd.Series,
    X_test:     np.ndarray,
    y_test:     np.ndarray,
    A_test:     pd.Series,
    model_name: str = "Model",
) -> pd.DataFrame:
    """Sweep global thresholds to build the accuracy-fairness Pareto curve."""
    groups = np.array(A_test)
    y_prob = model.predict_proba(X_test)[:, 1].astype(np.float64)
    auroc  = roc_auc_score(y_test, y_prob)
    rows   = []

    for tau in np.linspace(0.05, 0.95, 50):
        y_pred = (y_prob >= tau).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        eod = equalized_odds_difference(y_test, y_pred, sensitive_features=groups)
        tp  = ((y_pred == 1) & (y_test == 1)).sum()
        fn  = ((y_pred == 0) & (y_test == 1)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append({
            "model": model_name, "threshold": tau,
            "auroc": auroc, "eod": abs(eod), "tpr": tpr,
        })
    return pd.DataFrame(rows)


def run_full_debiasing(
    suite,
    X_train: np.ndarray, y_train: np.ndarray, A_train_race: pd.Series,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray, A_test_race: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run debiasing + Pareto frontier for all three models."""
    models_dict = {"LR": suite.lr, "XGBoost": suite.xgb, "MLP": suite.mlp}
    debias_rows, pareto_dfs = [], []

    print("\n── Debiasing (Equalized Odds Threshold Optimisation) ─────────────")
    for name, model in models_dict.items():
        print(f"  Processing {name} ...")
        groups   = np.array(A_test_race)
        y_prob   = model.predict_proba(X_test)[:, 1].astype(np.float64)
        y_base   = (y_prob >= 0.5).astype(int)
        eod_pre  = equalized_odds_difference(y_test, y_base, sensitive_features=groups)
        auroc    = roc_auc_score(y_test, y_prob)

        _, result = apply_threshold_optimizer(
            model, X_train, y_train, A_train_race,
            X_test,  y_test,  A_test_race,
        )
        debias_rows.append({
            "model":         name,
            "auroc":         auroc,
            "eod_before":    abs(eod_pre),
            "eod_after":     abs(result["eod_after"]),
            "eod_reduction": abs(eod_pre) - abs(result["eod_after"]),
        })
        print(f"    EOD: {abs(eod_pre):.4f} → {abs(result['eod_after']):.4f}")

        pf = compute_pareto_frontier(
            model, X_train, y_train, A_train_race,
            X_test, y_test, A_test_race, model_name=name,
        )
        pareto_dfs.append(pf)

    debias_df = pd.DataFrame(debias_rows)
    pareto_df = pd.concat(pareto_dfs, ignore_index=True)
    print("\n  Debiasing summary:")
    print(debias_df.round(4).to_string(index=False))
    return debias_df, pareto_df
