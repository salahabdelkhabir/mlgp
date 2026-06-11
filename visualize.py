"""
visualize.py
=============
Standalone visualisation suite for the Home Credit pipeline.  Reads saved
artefacts and produces publication-quality figures saved to reports/figures/.

Usage
-----
    # After running the full pipeline:
    python visualize.py

    # Regenerate a specific figure:
    python visualize.py --figures roc feature_importance
"""

import argparse, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve

from config import OUTPUT_DIR, MODELS_DIR, REPORTS_DIR, TARGET, logger

warnings.filterwarnings("ignore")

FIGS_DIR = REPORTS_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
C_BLUE  = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN = "#2ca02c"
C_RED   = "#d62728"
C_GREY  = "#7f7f7f"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ── Data loading ───────────────────────────────────────────────────────────────

def load_val_data():
    """Return (y_val, y_prob) from saved parquet and pickled model."""
    tune_path = MODELS_DIR / "tuned_model.pkl"
    if not tune_path.exists():
        logger.warning(f"tuned_model.pkl not found at {tune_path}")
        return None, None

    with open(tune_path, "rb") as f:
        obj = pickle.load(f)
    model = obj["model"] if isinstance(obj, dict) else obj

    y_val = pd.read_parquet(OUTPUT_DIR / "y_val.parquet").squeeze()
    X_val = pd.read_parquet(OUTPUT_DIR / "X_val.parquet")
    y_prob = model.predict_proba(X_val)[:, 1]
    return y_val.values, y_prob


def load_feature_importance():
    """Read feature_importance.csv from the models directory."""
    path = MODELS_DIR / "feature_importance.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.sort_values("avg_importance", ascending=False)


def load_metrics() -> dict | None:
    """Read final metrics from evaluation report."""
    path = REPORTS_DIR / "final_evaluation.xlsx"
    if not path.exists():
        return None
    df = pd.read_excel(path, sheet_name="01_final_metrics")
    return df.iloc[0].to_dict()


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, ax=None):
    """ROC curve with AUC annotation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = np.trapz(tpr, fpr)
    ax.plot(fpr, tpr, color=C_BLUE, lw=2,
            label=f"LightGBM (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    return ax.figure if ax is None else ax


def plot_pr_curve(y_true, y_prob, ax=None):
    """Precision-Recall curve with baseline."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = np.trapz(prec[::-1], rec[::-1])
    baseline = y_true.mean()
    ax.plot(rec, prec, color=C_ORANGE, lw=2,
            label=f"LightGBM (PR-AUC = {pr_auc:.4f})")
    ax.axhline(baseline, color=C_GREY, ls="--", lw=1,
               label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return ax.figure if ax is None else ax


def plot_confusion_matrix(y_true, y_prob, threshold=0.5, ax=None):
    """Confusion matrix heatmap at a given threshold."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.2)
    ax.figure.colorbar(im, ax=ax, shrink=0.75)
    labels = [["TN\n" + str(tn), "FP\n" + str(fp)],
              ["FN\n" + str(fn), "TP\n" + str(tp)]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)
    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title(f"Confusion Matrix  (threshold = {threshold:.2f})",
                 fontweight="bold")
    return ax.figure if ax is None else ax


def plot_feature_importance(top_n: int = 20):
    """Horizontal bar chart of top-N features by mean importance."""
    df = load_feature_importance()
    if df is None or df.empty:
        logger.warning("feature_importance.csv not found — skipping")
        return None

    top = df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top)))
    ax.barh(range(len(top)), top["avg_importance"].values, color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Normalised Importance")
    ax.set_title(f"Top {top_n} Features  (averaged across models)",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def plot_score_distribution(y_true, y_prob, ax=None):
    """Histogram of predicted probabilities split by actual class."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 51)
    ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.6, label="Actual 0 (No difficulty)",
            color=C_BLUE, density=True)
    ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.6, label="Actual 1 (Difficulty)",
            color=C_RED, density=True)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Actual Class", fontweight="bold")
    ax.legend()
    return ax.figure if ax is None else ax


def plot_calibration_curve(y_true, y_prob, ax=None):
    """Reliability diagram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(prob_pred, prob_true, "o-", color=C_GREEN, lw=2,
            label="LightGBM")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    return ax.figure if ax is None else ax


def plot_lift_chart(y_true, y_prob, ax=None):
    """Cumulative lift / gains chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    n = len(y_true)
    idx = np.argsort(y_prob)[::-1]
    sorted_true = y_true[idx]
    cum_positives = np.cumsum(sorted_true)
    total_positives = cum_positives[-1]
    pct_population = np.arange(1, n + 1) / n * 100
    pct_positives = cum_positives / total_positives * 100

    ax.plot(pct_population, pct_positives, color=C_BLUE, lw=2,
            label="Model")
    ax.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.5,
            label="Random")
    ax.set_xlabel("% Population")
    ax.set_ylabel("% Positives Captured")
    ax.set_title("Cumulative Lift / Gains Chart", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    return ax.figure if ax is None else ax


# ── Composite figure (summary dashboard) ──────────────────────────────────────

def plot_summary_dashboard(y_true, y_prob):
    """4-panel dashboard for a quick resume screenshot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plot_roc_curve(y_true, y_prob, ax=axes[0, 0])
    axes[0, 0].set_title("ROC Curve", fontweight="bold")

    plot_pr_curve(y_true, y_prob, ax=axes[0, 1])
    axes[0, 1].set_title("Precision-Recall", fontweight="bold")

    # Threshold from metrics if available
    metrics = load_metrics()
    threshold = metrics.get("opt_threshold", 0.5) if metrics else 0.5
    plot_confusion_matrix(y_true, y_prob, threshold=threshold, ax=axes[1, 0])
    axes[1, 0].set_title("Confusion Matrix", fontweight="bold")

    plot_score_distribution(y_true, y_prob, ax=axes[1, 1])
    axes[1, 1].set_title("Score Distribution", fontweight="bold")

    fig.tight_layout(pad=3)
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

AVAILABLE_FIGURES = {
    "roc":           ("ROC curve", lambda y, p: plot_roc_curve(y, p)),
    "pr":            ("PR curve", lambda y, p: plot_pr_curve(y, p)),
    "confusion":     ("Confusion matrix", lambda y, p: plot_confusion_matrix(y, p)),
    "importance":    ("Feature importance", lambda y, p: plot_feature_importance()),
    "score_dist":    ("Score distribution", lambda y, p: plot_score_distribution(y, p)),
    "calibration":   ("Calibration curve", lambda y, p: plot_calibration_curve(y, p)),
    "lift":          ("Lift chart", lambda y, p: plot_lift_chart(y, p)),
    "dashboard":     ("Summary dashboard", lambda y, p: plot_summary_dashboard(y, p)),
}


def main():
    parser = argparse.ArgumentParser(description="Generate pipeline figures")
    parser.add_argument(
        "--figures", nargs="+",
        choices=list(AVAILABLE_FIGURES) + ["all"],
        default=["all"],
        help="Figures to generate (default: all)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Visualisation Suite")
    logger.info("=" * 60)

    y_true, y_prob = load_val_data()
    figs_to_plot = list(AVAILABLE_FIGURES) if "all" in args.figures else args.figures

    for key in figs_to_plot:
        name, plot_fn = AVAILABLE_FIGURES[key]
        logger.info(f"  Plotting {name} …")

        # Some figures need y_true/y_prob, some (importance) don't
        try:
            fig = plot_fn(y_true, y_prob)
        except TypeError:
            # Feature importance doesn't use y data
            fig = plot_fn(None, None)

        if fig is None:
            continue

        save_path = FIGS_DIR / f"{key}.png"
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"    → saved {save_path}")

    # Print metrics summary
    metrics = load_metrics()
    if metrics:
        logger.info("\n── Final Metrics ──────────────────────────────────")
        for k in ["roc_auc", "pr_auc", "f1_opt", "ks_stat", "opt_threshold",
                    "lift_at_10pct", "lift_at_20pct", "precision_opt", "recall_opt"]:
            if k in metrics:
                logger.info(f"  {k:<25} {metrics[k]}")

    logger.info(f"\nAll figures saved to {FIGS_DIR}")
    logger.info("Visualisation complete.")


if __name__ == "__main__":
    main()
