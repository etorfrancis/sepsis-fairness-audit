"""
models.py
────────────────────────────────────────────────────────────────────────────────
Train and evaluate three sepsis mortality prediction models:
  1. Logistic Regression with L2 regularisation (clinical interpretable baseline)
  2. XGBoost gradient boosting (state-of-the-art tabular)
  3. Multi-Layer Perceptron (shallow neural network)

Performance metrics reported:
  - AUROC, AUPRC  (discrimination)
  - Brier score   (probabilistic accuracy)
  - ECE           (Expected Calibration Error — how reliable are the probabilities?)

All hyperparameters chosen by 5-fold stratified cross-validation on the training
set. Best parameters are logged. Models are stored as attributes of ModelSuite.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from dataclasses import dataclass, field

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import xgboost as xgb


# ── Expected Calibration Error ────────────────────────────────────────────────

def expected_calibration_error(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    Compute ECE: weighted mean absolute difference between predicted
    probability and observed frequency across probability bins.
    Lower is better. Perfect calibration = 0.
    """
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    ece        = 0.0
    n          = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        avg_conf  = y_prob[mask].mean()
        avg_acc   = y_true[mask].mean()
        ece      += mask.sum() * abs(avg_conf - avg_acc)
    return ece / n


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "",
) -> Dict[str, float]:
    """Full discrimination + calibration evaluation."""
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "model":        model_name,
        "auroc":        roc_auc_score(y, y_prob),
        "auprc":        average_precision_score(y, y_prob),
        "brier":        brier_score_loss(y, y_prob),
        "ece":          expected_calibration_error(y, y_prob),
    }


def get_roc_data(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    return fpr, tpr, thresholds


def get_pr_data(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_prob)
    return precision, recall


def get_calibration_data(model, X, y, n_bins=10):
    y_prob = model.predict_proba(X)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y, y_prob, n_bins=n_bins)
    return fraction_pos, mean_pred


# ── Model training ────────────────────────────────────────────────────────────

@dataclass
class ModelSuite:
    """Container for all trained models and their predictions."""
    lr:  Any = None
    xgb: Any = None
    mlp: Any = None
    results_train: Dict = field(default_factory=dict)
    results_val:   Dict = field(default_factory=dict)
    results_test:  Dict = field(default_factory=dict)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> LogisticRegression:
    """
    Logistic Regression with L2 penalty.
    C selected by cross-validation over log-spaced grid.
    """
    print("  Training Logistic Regression ...")
    best_C, best_score = 1.0, 0.0
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=skf, scoring="roc_auc", n_jobs=-1)
        mean_auc = scores.mean()
        if mean_auc > best_score:
            best_score, best_C = mean_auc, C

    print(f"    Best C={best_C}, CV-AUROC={best_score:.4f}")
    model = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> xgb.XGBClassifier:
    """
    XGBoost with early stopping on validation AUROC.
    Key hyperparameters chosen to balance performance and overfitting:
      - max_depth=5   prevents over-deep trees common in small ICU data
      - subsample=0.8 row subsampling for variance reduction
      - colsample=0.8 feature subsampling
      - scale_pos_weight handles class imbalance (~18% positive)
    """
    print("  Training XGBoost ...")
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators        = 800,
        max_depth           = 5,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        scale_pos_weight    = pos_weight,
        eval_metric         = "auc",
        early_stopping_rounds = 30,
        random_state        = 42,
        verbosity           = 0,
        use_label_encoder   = False,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iter = model.best_iteration
    print(f"    Best iteration: {best_iter}")
    return model


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> MLPClassifier:
    """
    Shallow MLP (2 hidden layers).
    Architecture and alpha chosen by cross-validation.
    Shallow depth is appropriate for tabular clinical data of this size.
    """
    print("  Training MLP ...")
    best_arch, best_alpha, best_score = (128, 64), 0.001, 0.0
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for arch in [(64,), (128, 64), (256, 128)]:
        for alpha in [1e-4, 1e-3, 1e-2]:
            model = MLPClassifier(
                hidden_layer_sizes=arch,
                alpha=alpha,
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )
            scores = cross_val_score(model, X_train, y_train,
                                     cv=skf, scoring="roc_auc", n_jobs=-1)
            mean_auc = scores.mean()
            if mean_auc > best_score:
                best_score = mean_auc
                best_arch, best_alpha = arch, alpha

    print(f"    Best arch={best_arch}, alpha={best_alpha}, CV-AUROC={best_score:.4f}")
    model = MLPClassifier(
        hidden_layer_sizes=best_arch,
        alpha=best_alpha,
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
) -> Tuple[ModelSuite, pd.DataFrame]:
    """
    Train all three models and return ModelSuite + performance summary DataFrame.
    """
    suite = ModelSuite()
    print("\n── Model Training ─────────────────────────────────────────────────")

    suite.lr  = train_logistic_regression(X_train, y_train)
    suite.xgb = train_xgboost(X_train, y_train, X_val, y_val)
    suite.mlp = train_mlp(X_train, y_train)

    print("\n── Performance Summary ────────────────────────────────────────────")
    rows = []
    for name, model in [("LR", suite.lr), ("XGBoost", suite.xgb), ("MLP", suite.mlp)]:
        for split, X, y in [("Train", X_train, y_train),
                             ("Val",   X_val,   y_val),
                             ("Test",  X_test,  y_test)]:
            r = evaluate_model(model, X, y, name)
            r["split"] = split
            rows.append(r)

    perf_df = pd.DataFrame(rows)
    test_rows = perf_df[perf_df["split"] == "Test"]

    print(f"\n{'Model':<10} {'AUROC':>8} {'AUPRC':>8} {'Brier':>8} {'ECE':>8}")
    print("─" * 50)
    for _, row in test_rows.iterrows():
        print(f"{row['model']:<10} {row['auroc']:>8.4f} {row['auprc']:>8.4f} "
              f"{row['brier']:>8.4f} {row['ece']:>8.4f}")

    suite.results_test = {
        r["model"]: r for _, r in test_rows.iterrows()
    }
    return suite, perf_df
