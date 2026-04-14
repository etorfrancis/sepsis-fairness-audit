"""
main.py
────────────────────────────────────────────────────────────────────────────────
Full pipeline orchestrator for:
  "Fairness Audit & Debiasing of Sepsis Mortality Prediction Models
   Across Demographic Subgroups"

Usage
-----
  # Step 1: generate the dataset (once)
  python src/generate_dataset.py

  # Step 2: run the full study
  python src/main.py

Output
------
  results/
    performance_summary.csv      — AUROC/AUPRC/Brier/ECE for all models × splits
    fairness_audit_full.csv      — Per-subgroup metrics for all models × attributes
    fairness_summary.csv         — Headline fairness numbers (Table 2 of paper)
    debiasing_results.csv        — EOD before/after debiasing
    pareto_frontier.csv          — Full Pareto frontier data
    shap_importance_xgb.csv      — XGBoost global SHAP importance
    shap_fairness_xgb.csv        — XGBoost Fairness-SHAP attribution
    shap_subgroup_xgb.csv        — Subgroup mean SHAP values

  figures/
    fig1_roc_curves.pdf
    fig2_calibration.pdf
    fig3_fairness_heatmap.pdf
    fig4_pareto_frontier.pdf
    fig5_subgroup_auroc.pdf
    fig6_shap_importance.pdf
    fig7_fairness_shap.pdf
    fig8_debiasing.pdf
"""

import sys
import os
import time
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Ensure src/ is on path when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from generate_dataset import generate_cohort
from preprocessing    import (
    load_and_validate, impute_features, encode_protected_attrs,
    split_data, scale_features, get_Xy, preprocessing_report,
    FEATURE_COLS, TARGET_COL,
)
from models           import train_all_models
from fairness_audit   import full_fairness_audit, compute_fairness_summary
from debiasing        import run_full_debiasing
from shap_analysis    import run_shap_analysis
from visualisation    import generate_all_figures

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("figures", exist_ok=True)


def save_results(name: str, df: pd.DataFrame):
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def main():
    t0 = time.time()
    print("=" * 70)
    print("  SEPSIS FAIRNESS AUDIT — Full Pipeline")
    print("=" * 70)

    # ── Step 0: Data ──────────────────────────────────────────────────────────
    print("\n── Step 0: Dataset ────────────────────────────────────────────────")
    DATA_PATH = "sepsis_cohort.csv"
    if not os.path.exists(DATA_PATH):
        print("  Generating synthetic cohort (N=10,000) ...")
        df_raw = generate_cohort(10_000)
        df_raw.to_csv(DATA_PATH, index=False)
        print(f"  Dataset saved: {df_raw.shape}")
    else:
        df_raw = pd.read_csv(DATA_PATH)
        print(f"  Loaded existing dataset: {df_raw.shape}")

    df_raw = load_and_validate(DATA_PATH)
    print(f"  Validation passed. Mortality rate: {df_raw[TARGET_COL].mean():.1%}")

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    print("\n── Step 1: Preprocessing ──────────────────────────────────────────")
    df_imp, imputer = impute_features(df_raw)
    df_enc          = encode_protected_attrs(df_imp)
    df_train, df_val, df_test = split_data(df_enc)
    df_train, df_val, df_test, scaler = scale_features(df_train, df_val, df_test)
    preprocessing_report(df_train, df_val, df_test)

    X_train, y_train, A_train = get_Xy(df_train)
    X_val,   y_val,   A_val   = get_Xy(df_val)
    X_test,  y_test,  A_test  = get_Xy(df_test)

    # ── Step 2: Model Training ────────────────────────────────────────────────
    suite, perf_df = train_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    save_results("performance_summary", perf_df)

    # ── Step 3: Fairness Audit ────────────────────────────────────────────────
    models_dict = {"LR": suite.lr, "XGBoost": suite.xgb, "MLP": suite.mlp}
    audit_df = full_fairness_audit(
        models_dict, X_test, y_test, A_test
    )
    summary_df = compute_fairness_summary(audit_df)
    save_results("fairness_audit_full",  audit_df)
    save_results("fairness_summary",     summary_df)

    print("\n  Fairness summary (max EOD across race groups):")
    print(summary_df[["model", "attribute", "max_eod", "max_tpr_gap",
                       "max_auroc_gap"]].round(4).to_string(index=False))

    # ── Step 4: Debiasing ─────────────────────────────────────────────────────
    debias_df, pareto_df = run_full_debiasing(
        suite,
        X_train, y_train, A_train["race"],
        X_val,   y_val,
        X_test,  y_test,  A_test["race"],
    )
    save_results("debiasing_results",  debias_df)
    save_results("pareto_frontier",    pareto_df)

    # ── Step 5: SHAP Analysis ─────────────────────────────────────────────────
    shap_results = run_shap_analysis(suite, X_test, A_test)

    for model_key in ["XGBoost", "LR"]:
        save_results(
            f"shap_importance_{model_key.lower()}",
            shap_results[model_key]["importance"]
        )
        save_results(
            f"shap_fairness_{model_key.lower()}",
            shap_results[model_key]["fairness"]
        )
        save_results(
            f"shap_subgroup_{model_key.lower()}",
            shap_results[model_key]["subgroup"]
        )

    # ── Step 6: Figures ───────────────────────────────────────────────────────
    generate_all_figures(
        suite, X_test, y_test,
        audit_df, pareto_df, debias_df, shap_results,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Results: ./{RESULTS_DIR}/")
    print(f"  Figures: ./figures/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
