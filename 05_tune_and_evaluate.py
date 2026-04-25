"""
05_tune_and_evaluate.py
========================
Phase 5 — Hyperparameter optimisation for the best model, final evaluation
with business metrics, SHAP explainability, and Kaggle submission file.

Steps
-----
1. Load best model name from model_comparison report
2. Run Optuna Bayesian optimisation (LightGBM / XGBoost / RandomForest)
3. Retrain best params on full train+val data
4. Final evaluation:
   - ROC / PR curves
   - Calibration (reliability diagram)
   - Confusion matrix at optimal threshold
   - Score distribution (good vs bad)
5. SHAP feature importance (waterfall + beeswarm)
6. Generate Kaggle submission CSV

Outputs
-------
    models/tuned_model.pkl
    reports/final_evaluation.xlsx
    reports/submission.csv

Run
---
    python 05_tune_and_evaluate.py

Dependencies (optional — script degrades gracefully)
-----------------------------------------------------
    pip install optuna shap
"""

import warnings, pickle, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, log_loss, roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold

from config import (
    CV_FOLDS, RANDOM_STATE, OUTPUT_DIR, MODELS_DIR,
    REPORTS_DIR, ID_COL, TARGET, FILES, logger,
)

# Optional imports
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("optuna not installed — skipping hyperparameter tuning.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("shap not installed — skipping SHAP explainability.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ── Load data ─────────────────────────────────────────────────────────────────

def load_all():
    X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")
    X_val   = pd.read_parquet(OUTPUT_DIR / "X_val.parquet")
    X_test  = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")
    y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet").squeeze()
    y_val   = pd.read_parquet(OUTPUT_DIR / "y_val.parquet").squeeze()

    # Full train = train + val (for final model)
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)

    test_ids = pd.read_csv(FILES["test"])[ID_COL]

    logger.info(
        f"X_full {X_full.shape}  X_test {X_test.shape}  "
        f"test_ids {len(test_ids):,}"
    )
    return X_train, X_val, X_test, y_train, y_val, X_full, y_full, test_ids


# ── Optuna objective for LightGBM ─────────────────────────────────────────────

def lgb_objective(trial, X_train, y_train, n_folds=CV_FOLDS):
    """Optuna objective: maximise mean CV AUC."""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 300, 2000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "verbose":           -1,
    }

    # Compute class ratio for scale_pos_weight
    pos = y_train.sum()
    neg = len(y_train) - pos
    params["scale_pos_weight"] = neg / pos

    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        Xtr, Xvl = X_train.iloc[train_idx], X_train.iloc[val_idx]
        ytr, yvl = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xvl)[:, 1]
        scores.append(roc_auc_score(yvl, prob))

    return float(np.mean(scores))


def run_optuna(X_train, y_train, n_trials: int = 50) -> dict:
    """Run Optuna search and return best params."""
    if not HAS_OPTUNA or not HAS_LGB:
        logger.warning("Skipping Optuna (requires optuna + lightgbm)")
        return {}

    logger.info(f"Running Optuna ({n_trials} trials) …")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: lgb_objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    logger.info(
        f"  Best trial: AUC={study.best_value:.4f}  "
        f"params={study.best_params}"
    )
    return study.best_params


# ── Build final tuned model ───────────────────────────────────────────────────

def build_tuned_model(best_params: dict, pos_weight: float):
    """Build LightGBM model with tuned params (fallback to defaults)."""
    if not HAS_LGB:
        from sklearn.ensemble import HistGradientBoostingClassifier
        logger.info("LightGBM not available — using HistGradientBoosting as tuned model")
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)

    params = {
        "n_estimators":      best_params.get("n_estimators", 1000),
        "learning_rate":     best_params.get("learning_rate", 0.05),
        "num_leaves":        best_params.get("num_leaves", 63),
        "max_depth":         best_params.get("max_depth", -1),
        "min_child_samples": best_params.get("min_child_samples", 50),
        "subsample":         best_params.get("subsample", 0.8),
        "colsample_bytree":  best_params.get("colsample_bytree", 0.8),
        "reg_alpha":         best_params.get("reg_alpha", 0.1),
        "reg_lambda":        best_params.get("reg_lambda", 0.1),
        "scale_pos_weight":  pos_weight,
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "verbose":           -1,
    }
    return lgb.LGBMClassifier(**params)


# ── Final evaluation metrics ──────────────────────────────────────────────────

def full_evaluation(model, X_val, y_val) -> dict:
    """Compute comprehensive evaluation metrics on validation set."""
    y_prob = model.predict_proba(X_val)[:, 1]

    # Optimal threshold (maximise F1)
    thresholds = np.linspace(0.01, 0.99, 300)
    f1s = [f1_score(y_val, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    opt_thresh = float(thresholds[np.argmax(f1s)])
    y_pred = (y_prob >= opt_thresh).astype(int)

    # ROC curve
    fpr, tpr, roc_thresh = roc_curve(y_val, y_prob)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresh})

    # PR curve
    prec, rec, pr_thresh = precision_recall_curve(y_val, y_prob)
    pr_df = pd.DataFrame({"precision": prec, "recall": rec,
                           "threshold": np.append(pr_thresh, np.nan)})

    # Calibration
    prob_true, prob_pred = calibration_curve(y_val, y_prob, n_bins=10)
    cal_df = pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Score distribution
    score_df = pd.DataFrame({"probability": y_prob, "actual": y_val})

    metrics = {
        "roc_auc":        round(roc_auc_score(y_val, y_prob), 4),
        "pr_auc":         round(average_precision_score(y_val, y_prob), 4),
        "f1_opt":         round(float(np.max(f1s)), 4),
        "precision_opt":  round(precision_score(y_val, y_pred, zero_division=0), 4),
        "recall_opt":     round(recall_score(y_val, y_pred, zero_division=0), 4),
        "opt_threshold":  round(opt_thresh, 3),
        "log_loss":       round(log_loss(y_val, y_prob), 4),
        "ks_stat":        round(float(np.max(tpr - fpr)), 4),
        "true_pos":       int(tp),
        "false_pos":      int(fp),
        "true_neg":       int(tn),
        "false_neg":      int(fn),
        "accuracy":       round((tp + tn) / len(y_val), 4),
        "lift_at_10pct":  _lift(y_val, y_prob, top_pct=0.10),
        "lift_at_20pct":  _lift(y_val, y_prob, top_pct=0.20),
    }

    return metrics, roc_df, pr_df, cal_df, score_df


def _lift(y_true, y_prob, top_pct=0.10) -> float:
    """Lift at top X% of predicted scores (standard credit risk metric)."""
    n = int(len(y_true) * top_pct)
    idx = np.argsort(y_prob)[::-1][:n]
    top_rate = y_true.values[idx].mean()
    base_rate = y_true.mean()
    return round(float(top_rate / base_rate), 3) if base_rate > 0 else 0.0


# ── SHAP explainability ───────────────────────────────────────────────────────

def compute_shap(model, X_val: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    """Compute SHAP values and return mean absolute values per feature."""
    if not HAS_SHAP:
        return pd.DataFrame()

    logger.info("Computing SHAP values …")
    sample = X_val.sample(min(max_rows, len(X_val)), random_state=RANDOM_STATE)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        # For binary classification, use class-1 SHAP values
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        mean_abs = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature":    sample.columns.tolist(),
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_df["rank"] = range(1, len(shap_df) + 1)
        logger.info(f"  SHAP computed for {len(sample):,} samples")
        return shap_df

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return pd.DataFrame()


# ── Submission file ───────────────────────────────────────────────────────────

def make_submission(model, X_test: pd.DataFrame, test_ids: pd.Series) -> pd.DataFrame:
    """Generate Kaggle-format submission."""
    y_prob = model.predict_proba(X_test)[:, 1]
    sub = pd.DataFrame({ID_COL: test_ids.values, TARGET: y_prob})
    sub_path = REPORTS_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    logger.info(f"Submission saved → {sub_path}  ({len(sub):,} rows)")
    return sub


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Phase 5 — Tune & Final Evaluate")
    logger.info("=" * 60)

    (X_train, X_val, X_test,
     y_train, y_val,
     X_full, y_full, test_ids) = load_all()

    # ── 1. Optuna tuning ────────────────────────────────────────────────────
    best_params = run_optuna(X_train, y_train, n_trials=50)

    # ── 2. Build and train tuned model ──────────────────────────────────────
    pos_weight = (len(y_full) - y_full.sum()) / y_full.sum()
    tuned_model = build_tuned_model(best_params, float(pos_weight))

    logger.info("Training tuned model on full train+val data …")
    t0 = time.time()
    tuned_model.fit(X_full, y_full)
    logger.info(f"  Done in {time.time()-t0:.1f}s")

    # Quick val-set check (fitted on full data, so this is slightly optimistic)
    val_auc = roc_auc_score(y_val, tuned_model.predict_proba(X_val)[:, 1])
    logger.info(f"  Val AUC (post full-data fit): {val_auc:.4f}")

    # ── 3. Comprehensive evaluation on held-out val ──────────────────────────
    logger.info("Running final evaluation …")
    # Refit on train only for unbiased val evaluation
    eval_model = build_tuned_model(best_params, float(pos_weight))
    eval_model.fit(X_train, y_train)

    metrics, roc_df, pr_df, cal_df, score_df = full_evaluation(eval_model, X_val, y_val)

    logger.info("\n── Final Metrics ─────────────────────────────────────────")
    for k, v in metrics.items():
        logger.info(f"  {k:<25} {v}")

    # ── 4. SHAP explainability ───────────────────────────────────────────────
    shap_df = compute_shap(eval_model, X_val)

    # ── 5. Submission ────────────────────────────────────────────────────────
    submission = make_submission(tuned_model, X_test, test_ids)

    # ── 6. Save tuned model ──────────────────────────────────────────────────
    tuned_path = MODELS_DIR / "tuned_model.pkl"
    with open(tuned_path, "wb") as f:
        pickle.dump({
            "model":        tuned_model,
            "best_params":  best_params,
            "val_auc":      val_auc,
            "threshold":    metrics["opt_threshold"],
            "feature_names": X_full.columns.tolist(),
        }, f)
    logger.info(f"Tuned model saved → {tuned_path}")

    # ── 7. Evaluation report ─────────────────────────────────────────────────
    report_path = REPORTS_DIR / "final_evaluation.xlsx"
    metrics_df  = pd.DataFrame([metrics])
    params_df   = pd.DataFrame([best_params]) if best_params else pd.DataFrame()

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer,  sheet_name="01_final_metrics",    index=False)
        roc_df.to_excel(writer,      sheet_name="02_roc_curve",        index=False)
        pr_df.to_excel(writer,       sheet_name="03_pr_curve",         index=False)
        cal_df.to_excel(writer,      sheet_name="04_calibration",      index=False)
        score_df.to_excel(writer,    sheet_name="05_score_distribution",index=False)
        if not shap_df.empty:
            shap_df.to_excel(writer, sheet_name="06_shap_importance",  index=False)
        if not params_df.empty:
            params_df.to_excel(writer, sheet_name="07_best_params",    index=False)
        submission.head(100).to_excel(writer, sheet_name="08_submission_preview", index=False)

    logger.info(f"Final evaluation report → {report_path}")
    logger.info("\nPipeline complete ✓")
    logger.info("──────────────────────────────────────────────────────────")
    logger.info(f"  Best model   : {type(tuned_model).__name__}")
    logger.info(f"  ROC-AUC      : {metrics['roc_auc']}")
    logger.info(f"  PR-AUC       : {metrics['pr_auc']}")
    logger.info(f"  KS statistic : {metrics['ks_stat']}")
    logger.info(f"  Lift @ 10%   : {metrics['lift_at_10pct']}x")
    logger.info(f"  Threshold    : {metrics['opt_threshold']}")
    logger.info("──────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
