"""
visualisation.py
────────────────────────────────────────────────────────────────────────────────
Publication-quality figures for all study components.

Figure 1: ROC curves (all 3 models)
Figure 2: Calibration plots (all 3 models)
Figure 3: Fairness heatmap — EOD × model × race
Figure 4: Accuracy–Fairness Pareto frontier
Figure 5: Subgroup AUROC comparison (grouped bar)
Figure 6: SHAP global feature importance (XGBoost)
Figure 7: Fairness-SHAP attribution (Black vs White SHAP gap)
Figure 8: Debiasing summary — EOD before/after

All figures saved as 300 DPI PDFs suitable for journal submission.
Style: clean academic aesthetic, colorblind-safe palette (Wong 2011).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Colorblind-safe palette (Wong 2011, Nature Methods) ──────────────────────
PALETTE = {
    "LR":      "#0072B2",  # blue
    "XGBoost": "#D55E00",  # vermillion
    "MLP":     "#009E73",  # green
    "White":   "#56B4E9",  # sky blue
    "Black":   "#E69F00",  # orange
    "Hispanic":"#CC79A7",  # mauve
    "Asian":   "#009E73",  # green
    "Other":   "#999999",  # grey
}

RACE_ORDER = ["White", "Black", "Hispanic", "Asian", "Other"]
FIG_DIR    = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def _save(fig, name: str, dpi: int = 300):
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    path_png = FIG_DIR / f"{name}.png"
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_curves(suite, X_test, y_test):
    """Figure 1: ROC curves for all models."""
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(6, 5.5))

    for name, model in [("LR", suite.lr), ("XGBoost", suite.xgb), ("MLP", suite.mlp)]:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=PALETTE[name], lw=2,
                label=f"{name} (AUROC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Figure 1: ROC Curves — Sepsis Mortality Prediction", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    _save(fig, "fig1_roc_curves")


def plot_calibration(suite, X_test, y_test):
    """Figure 2: Calibration curves."""
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect calibration")

    for name, model in [("LR", suite.lr), ("XGBoost", suite.xgb), ("MLP", suite.mlp)]:
        y_prob = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, "o-", color=PALETTE[name], lw=2,
                markersize=5, label=name)

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Figure 2: Calibration Curves", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, "fig2_calibration")


def plot_fairness_heatmap(audit_df: pd.DataFrame):
    """Figure 3: EOD heatmap — race × model."""
    pivot = (
        audit_df[audit_df["attribute"] == "race"]
        .pivot_table(index="subgroup", columns="model", values="eod")
        .reindex(RACE_ORDER)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=0.25)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=12)
    ax.set_title("Figure 3: Equalized Odds Difference by Race × Model", fontsize=12)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=11,
                        color="white" if val > 0.15 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label("EOD (higher = more unfair)", fontsize=10)
    _save(fig, "fig3_fairness_heatmap")


def plot_pareto_frontier(pareto_df: pd.DataFrame):
    """Figure 4: Accuracy–Fairness Pareto frontier."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name in ["LR", "XGBoost", "MLP"]:
        sub = pareto_df[pareto_df["model"] == model_name].sort_values("threshold")
        if sub.empty:
            continue
        ax.plot(sub["eod"], sub["auroc"], "o-", color=PALETTE[model_name],
                lw=2, markersize=4, alpha=0.8, label=model_name)

        # Mark the default (τ=0.5) point
        mid = sub.iloc[(sub["threshold"] - 0.5).abs().argsort()[:1]]
        ax.scatter(mid["eod"], mid["auroc"], color=PALETTE[model_name],
                   s=120, zorder=5, marker="*")

    ax.set_xlabel("Equalized Odds Difference (↓ fairer)", fontsize=12)
    ax.set_ylabel("AUROC (↑ more accurate)", fontsize=12)
    ax.set_title("Figure 4: Accuracy–Fairness Pareto Frontier\n"
                 "(★ = default threshold τ=0.5)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save(fig, "fig4_pareto_frontier")


def plot_subgroup_auroc(audit_df: pd.DataFrame):
    """Figure 5: Subgroup AUROC grouped bar chart."""
    sub = (
        audit_df[audit_df["attribute"] == "race"]
        .pivot_table(index="subgroup", columns="model", values="auroc")
        .reindex(RACE_ORDER)
        .dropna(how="all")
    )

    x    = np.arange(len(sub))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model_name in enumerate(["LR", "XGBoost", "MLP"]):
        if model_name not in sub.columns:
            continue
        ax.bar(x + (i - 1) * w, sub[model_name], w,
               color=PALETTE[model_name], label=model_name, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(sub.index, fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Figure 5: Subgroup AUROC by Race", fontsize=12)
    ax.legend(fontsize=11)
    ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.4)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig5_subgroup_auroc")


def plot_shap_importance(shap_results: dict):
    """Figure 6: Global SHAP feature importance for XGBoost."""
    df = shap_results["XGBoost"]["importance"].head(16)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(df)))
    ax.barh(df["feature"][::-1], df["mean_abs_shap"][::-1],
            color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Figure 6: Global Feature Importance (XGBoost SHAP)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "fig6_shap_importance")


def plot_fairness_shap(shap_results: dict):
    """Figure 7: Fairness-SHAP — Black vs White SHAP gap per feature."""
    df = shap_results["XGBoost"]["fairness"]
    df = df.sort_values("shap_gap")

    colors = ["#D55E00" if v > 0 else "#0072B2" for v in df["shap_gap"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"], df["shap_gap"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("SHAP gap: Black − White (mean |SHAP|)", fontsize=12)
    ax.set_title("Figure 7: Fairness-SHAP Attribution\n"
                 "(+ve = model attends more to feature for Black patients)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    pos_patch = mpatches.Patch(color="#D55E00", label="Higher attention → Black")
    neg_patch = mpatches.Patch(color="#0072B2", label="Higher attention → White")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=10)
    _save(fig, "fig7_fairness_shap")


def plot_debiasing_summary(debias_df: pd.DataFrame):
    """Figure 8: EOD before/after debiasing."""
    x   = np.arange(len(debias_df))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(x - w/2, debias_df["eod_before"], w, label="Before debiasing",
           color="#D55E00", alpha=0.9)
    ax.bar(x + w/2, debias_df["eod_after"],  w, label="After debiasing",
           color="#009E73", alpha=0.9)

    # Arrows showing reduction
    for i, row in debias_df.iterrows():
        ax.annotate(
            f"↓{row['eod_reduction']:.3f}",
            xy=(i + w/2, row["eod_after"] + 0.003),
            fontsize=9, ha="center", color="#009E73",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(debias_df["model"], fontsize=12)
    ax.set_ylabel("Equalized Odds Difference", fontsize=12)
    ax.set_title("Figure 8: EOD Before and After\nEqualized Odds Threshold Optimisation",
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig8_debiasing")


def generate_all_figures(
    suite, X_test, y_test,
    audit_df, pareto_df, debias_df, shap_results,
):
    """Generate all 8 publication figures."""
    print("\n── Generating Figures ─────────────────────────────────────────────")
    plot_roc_curves(suite, X_test, y_test)
    plot_calibration(suite, X_test, y_test)
    plot_fairness_heatmap(audit_df)
    plot_pareto_frontier(pareto_df)
    plot_subgroup_auroc(audit_df)
    plot_shap_importance(shap_results)
    plot_fairness_shap(shap_results)
    plot_debiasing_summary(debias_df)
    print("  All 8 figures generated.")
