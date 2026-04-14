"""
fairness_audit.py
────────────────────────────────────────────────────────────────────────────────
Comprehensive fairness audit of sepsis prediction models.

Four fairness metrics computed for every (model × protected attribute × subgroup):

  1. Demographic Parity Difference (DPD)
     — Difference in positive prediction rates between groups.
     — Concerns: should high-risk groups be predicted differently?

  2. Equalized Odds Difference (EOD)
     — Max of |TPR difference| and |FPR difference| across groups.
     — The key clinical metric: are we missing more sepsis deaths in one group?

  3. Predictive Parity (PPV gap)
     — Difference in Precision (PPV) across groups.
     — Concerns: if the model fires, is it equally reliable for all groups?

  4. Calibration-within-groups (subgroup ECE)
     — Do model probabilities reflect true risk equally across groups?
     — Obermeyer et al. (2019) showed racial bias lives in miscalibration.

  5. Subgroup AUROC
     — Overall discrimination per subgroup (not a fairness metric per se,
       but essential to report alongside the fairness metrics).

Reference:
  Verma & Rubin (2018) "Fairness Definitions Explained" — FAT* Workshop
  Chouldechova (2017) "Fair Prediction with Disparate Impact"
  Obermeyer et al. (2019) Science 366(6464):447-453
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, precision_score,
)
from models import expected_calibration_error


# ── Metric computation ────────────────────────────────────────────────────────

def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all 5 metrics for a single subgroup."""
    y_pred = (y_prob >= threshold).astype(int)
    n      = len(y_true)
    if n < 20:
        return {k: float("nan") for k in
                ["n", "prevalence", "auroc", "tpr", "fpr", "ppv", "ece",
                 "pred_pos_rate"]}

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return {
        "n":             n,
        "prevalence":    y_true.mean(),
        "pred_pos_rate": y_pred.mean(),
        "auroc":         _safe_auroc(y_true, y_prob),
        "tpr":           tpr,
        "fpr":           fpr,
        "ppv":           ppv if not np.isnan(ppv) else float("nan"),
        "ece":           expected_calibration_error(y_true, y_prob),
    }


def audit_protected_attribute(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    groups:    pd.Series,
    attr_name: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute per-subgroup metrics for a single protected attribute.
    Returns a tidy DataFrame with one row per group.
    """
    rows = []
    for grp in groups.unique():
        mask = (groups == grp).values
        metrics = compute_subgroup_metrics(y_true[mask], y_prob[mask], threshold)
        metrics["group"]     = grp
        metrics["attribute"] = attr_name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("group")
    df.index.name = "subgroup"

    # Reference group = largest group (or White for race)
    if attr_name == "race":
        ref = "White"
    else:
        ref = df["n"].idxmax()

    ref_row = df.loc[ref]

    # Gap metrics (absolute difference from reference group)
    df["dpd"]        = df["pred_pos_rate"] - ref_row["pred_pos_rate"]
    df["tpr_gap"]    = df["tpr"]           - ref_row["tpr"]
    df["fpr_gap"]    = df["fpr"]           - ref_row["fpr"]
    df["ppv_gap"]    = df["ppv"]           - ref_row["ppv"]
    df["auroc_gap"]  = df["auroc"]         - ref_row["auroc"]
    df["eod"]        = df[["tpr_gap", "fpr_gap"]].abs().max(axis=1)

    df["reference_group"] = ref
    df["attribute"]       = attr_name
    return df.reset_index()


def full_fairness_audit(
    models:    Dict[str, Any],
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    A_test:    Dict[str, pd.Series],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run full fairness audit across all models × all protected attributes.
    Returns a long-form DataFrame suitable for plotting and tabular reporting.
    """
    all_rows = []
    print("\n── Fairness Audit ─────────────────────────────────────────────────")

    for model_name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]

        for attr_name, groups in A_test.items():
            df_attr = audit_protected_attribute(
                y_test, y_prob, groups, attr_name, threshold
            )
            df_attr["model"] = model_name
            all_rows.append(df_attr)

    audit_df = pd.concat(all_rows, ignore_index=True)

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n  Equalized Odds Difference (EOD) by model × race:\n")
    pivot = audit_df[audit_df["attribute"] == "race"].pivot_table(
        index="subgroup", columns="model", values="eod"
    ).round(4)
    print(pivot.to_string())

    print("\n  AUROC by race:\n")
    pivot_auroc = audit_df[audit_df["attribute"] == "race"].pivot_table(
        index="subgroup", columns="model", values="auroc"
    ).round(4)
    print(pivot_auroc.to_string())

    return audit_df


def compute_fairness_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse audit results to a per-(model, attribute) summary:
    max EOD, max |AUROC gap|, max |PPV gap|.
    These are the headline numbers that go in the paper's Table 2.
    """
    summary_rows = []
    for (model, attr), grp in audit_df.groupby(["model", "attribute"]):
        summary_rows.append({
            "model":         model,
            "attribute":     attr,
            "max_eod":       grp["eod"].abs().max(),
            "max_auroc_gap": grp["auroc_gap"].abs().max(),
            "max_tpr_gap":   grp["tpr_gap"].abs().max(),
            "max_fpr_gap":   grp["fpr_gap"].abs().max(),
            "max_ppv_gap":   grp["ppv_gap"].abs().max(),
            "max_ece_gap":   (grp["ece"] - grp["ece"].min()).max(),
        })
    return pd.DataFrame(summary_rows).sort_values(["attribute", "model"])
