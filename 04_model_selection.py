"""
04_model_selection.py
======================
Phase 4 — Train, cross-validate, and compare multiple candidate models.
Produces a ranked comparison and saves the best model.

Models evaluated
----------------
1. Logistic Regression          (fast baseline, interpretable)
2. Random Forest                (strong ensemble baseline)
3. LightGBM                     (gradient boosting — usually top performer)
4. XGBoost                      (gradient boosting alternative)
5. CatBoost                     (handles categoricals natively)
6. Hist Gradient Boosting       (sklearn native, no install needed)
7. Extra Trees                  (variance-reducer)

Metrics tracked
---------------
- ROC-AUC        (primary — standard for imbalanced credit scoring)
- PR-AUC         (area under precision-recall curve — better for minorities)
- F1 @ best threshold (threshold tuned on val)
- KS statistic   (common in credit risk)
- Log loss
- CV mean ± std AUC (5-fold stratified)

Outputs
-------
    reports/model_comparison.xlsx
    models/{model_name}_best.pkl
    models/feature_importance.csv

Run
---
    python 04_model_selection.py
"""

import warnings, time, json, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, log_loss, roc_curve,
)

# Optional imports — skip gracefully if not installed
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

from config import (
    CV_FOLDS, RANDOM_STATE, OUTPUT_DIR, MODELS_DIR, REPORTS_DIR, logger
)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_splits():
    X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")
    X_val   = pd.read_parquet(OUTPUT_DIR / "X_val.parquet")
    y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet").squeeze()
    y_val   = pd.read_parquet(OUTPUT_DIR / "y_val.parquet").squeeze()
    feature_names = (OUTPUT_DIR / "feature_names.txt").read_text().splitlines()
    logger.info(
        f"Loaded: X_train {X_train.shape}  X_val {X_val.shape}  "
        f"features: {len(feature_names)}"
    )
    return X_train, X_val, y_train, y_val, feature_names


# ── Metrics helpers ───────────────────────────────────────────────────────────

def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic — max separation between TPR and FPR curves."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Find threshold that maximises F1 on validation set."""
    thresholds = np.linspace(0.01, 0.99, 200)
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_f1, best_thresh


def evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Train model, evaluate all metrics, return result dict."""
    logger.info(f"  Training {name} …")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 1)

    # Probabilities
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_val   = model.predict_proba(X_val)[:, 1]

    # Metrics
    train_auc = roc_auc_score(y_train, y_prob_train)
    val_auc   = roc_auc_score(y_val,   y_prob_val)
    pr_auc    = average_precision_score(y_val, y_prob_val)
    logloss   = log_loss(y_val, y_prob_val)
    ks        = ks_statistic(y_val.values, y_prob_val)
    f1, thresh = best_f1_threshold(y_val.values, y_prob_val)

    # Overfit gap
    overfit_gap = round(train_auc - val_auc, 4)

    result = {
        "model":         name,
        "val_roc_auc":   round(val_auc,   4),
        "train_roc_auc": round(train_auc, 4),
        "overfit_gap":   overfit_gap,
        "pr_auc":        round(pr_auc,    4),
        "ks_stat":       round(ks,        4),
        "best_f1":       round(f1,        4),
        "best_threshold":round(thresh,    3),
        "log_loss":      round(logloss,   4),
        "train_time_s":  train_time,
    }
    logger.info(
        f"    val AUC={val_auc:.4f}  PR-AUC={pr_auc:.4f}  "
        f"KS={ks:.4f}  gap={overfit_gap:+.4f}  [{train_time}s]"
    )
    return result, model


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_model(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = CV_FOLDS,
) -> dict:
    """Return mean and std AUC from stratified k-fold CV."""
    logger.info(f"  Cross-validating {name} ({n_folds}-fold) …")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return {
        "model":       name,
        "cv_auc_mean": round(float(scores.mean()), 4),
        "cv_auc_std":  round(float(scores.std()),  4),
        "cv_folds":    n_folds,
    }


# ── Feature importance ────────────────────────────────────────────────────────

def extract_feature_importance(
    models: dict,
    feature_names: list[str],
) -> pd.DataFrame:
    """Collect feature importances from tree-based models."""
    records = []
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
        else:
            continue

        for feat, score in zip(feature_names, imp):
            records.append({"model": name, "feature": feat, "importance": score})

    df = pd.DataFrame(records)
    # Average importance across models (normalised)
    pivot = df.pivot_table(index="feature", columns="model", values="importance")
    pivot = pivot.div(pivot.sum())               # normalise each model
    pivot["avg_importance"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg_importance", ascending=False).reset_index()
    return pivot


# ── Model definitions ─────────────────────────────────────────────────────────

def build_candidates(class_weight: dict) -> dict:
    """Return dict of {name: model} for all available frameworks."""

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            solver="saga",
            C=0.1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=50,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=50,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if HAS_LGB:
        pos_weight = (class_weight[0] / class_weight[1]) if isinstance(class_weight, dict) else 1.0
        candidates["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    if HAS_XGB:
        pos_weight = (class_weight[0] / class_weight[1]) if isinstance(class_weight, dict) else 1.0
        candidates["XGBoost"] = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    if HAS_CAT:
        candidates["CatBoost"] = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=0,
        )

    return candidates


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Phase 4 — Model Selection")
    logger.info("=" * 60)

    X_train, X_val, y_train, y_val, feature_names = load_splits()

    # Class weight for imbalanced target
    pos = y_train.sum()
    neg = len(y_train) - pos
    class_weight = {0: 1.0, 1: round(neg / pos, 2)}
    logger.info(f"Class weights: {class_weight}  (pos={pos:,}  neg={neg:,})")

    candidates = build_candidates(class_weight)
    logger.info(f"Candidates: {list(candidates.keys())}")

    val_results  = []
    cv_results   = []
    fitted_models = {}

    for name, model in candidates.items():
        logger.info(f"\n{'─'*50}")
        logger.info(f"Model: {name}")

        result, fitted = evaluate_model(
            name, model, X_train, y_train, X_val, y_val
        )
        val_results.append(result)
        fitted_models[name] = fitted

        cv_res = cross_validate_model(name, model, X_train, y_train)
        cv_results.append(cv_res)

    # ── Rankings ──────────────────────────────────────────────────────────
    val_df = pd.DataFrame(val_results).sort_values("val_roc_auc", ascending=False)
    cv_df  = pd.DataFrame(cv_results).sort_values("cv_auc_mean", ascending=False)

    # Merge CV into val table
    comparison = val_df.merge(cv_df[["model","cv_auc_mean","cv_auc_std"]], on="model")
    comparison["rank"] = range(1, len(comparison) + 1)
    comparison = comparison[["rank","model","val_roc_auc","cv_auc_mean","cv_auc_std",
                               "pr_auc","ks_stat","best_f1","overfit_gap",
                               "log_loss","train_time_s","best_threshold"]]

    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON (sorted by val ROC-AUC)")
    logger.info("=" * 60)
    logger.info(f"\n{comparison[['rank','model','val_roc_auc','cv_auc_mean','pr_auc','ks_stat']].to_string(index=False)}")

    # ── Best model ────────────────────────────────────────────────────────
    best_name  = comparison.iloc[0]["model"]
    best_model = fitted_models[best_name]
    best_path  = MODELS_DIR / f"{best_name}_best.pkl"

    with open(best_path, "wb") as f:
        pickle.dump({"model": best_model, "feature_names": feature_names}, f)

    logger.info(f"\n✓ Best model: {best_name}  AUC={comparison.iloc[0]['val_roc_auc']}")
    logger.info(f"  Saved → {best_path}")

    # ── Feature importance ────────────────────────────────────────────────
    feat_imp = extract_feature_importance(fitted_models, feature_names)
    feat_imp_path = MODELS_DIR / "feature_importance.csv"
    feat_imp.to_csv(feat_imp_path, index=False)
    logger.info(f"\nTop 15 features (avg across models):")
    logger.info(f"\n{feat_imp[['feature','avg_importance']].head(15).to_string(index=False)}")

    # ── Save comparison report ────────────────────────────────────────────
    report_path = REPORTS_DIR / "model_comparison.xlsx"
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="01_model_ranking",      index=False)
        cv_df.to_excel(writer,      sheet_name="02_cross_validation",   index=False)
        feat_imp.to_excel(writer,   sheet_name="03_feature_importance", index=False)

    logger.info(f"\nReport saved → {report_path}")
    logger.info("Model selection complete.")

    return comparison, fitted_models


if __name__ == "__main__":
    main()
