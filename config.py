"""
config.py
=========
Central configuration for the Home Credit Default Risk pipeline.
Edit DATA_DIR to point to your local dataset folder before running anything.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# DATA_DIR points to the folder containing all raw CSVs.
# Your files are in D:\ML GP\DATA\ — the pipeline script lives in D:\ML GP\
# so we use a path relative to the working directory, or an absolute path.
DATA_DIR    = Path("DATA")             # relative to D:\ML GP  (where you run the script)
OUTPUT_DIR  = Path("DATA/processed")  # engineered features land here
REPORTS_DIR = Path("reports")         # EDA reports, plots
MODELS_DIR  = Path("models")          # serialised model artefacts

for _d in [DATA_DIR, OUTPUT_DIR, REPORTS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── File names ─────────────────────────────────────────────────────────────────
FILES = {
    "train":          DATA_DIR / "application_train.csv",
    "test":           DATA_DIR / "application_test.csv",
    "bureau":         DATA_DIR / "bureau.csv",
    "bureau_balance": DATA_DIR / "bureau_balance.csv",
    "prev_app":       DATA_DIR / "previous_application.csv",
    "pos_cash":       DATA_DIR / "POS_CASH_balance.csv",
    "installments":   DATA_DIR / "installments_payments.csv",
    "cc_balance":     DATA_DIR / "credit_card_balance.csv",
}

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET       = "TARGET"
ID_COL       = "SK_ID_CURR"

# ── Known sentinel / anomaly values ───────────────────────────────────────────
DAYS_EMPLOYED_SENTINEL = 365243   # means "not employed" — replace with NaN

# ── Categorical encoding ───────────────────────────────────────────────────────
# Columns with ≤ this many unique values → label-encoded; rest → target-encoded
MAX_ONEHOT_CARDINALITY = 10

# ── Train / validation split ───────────────────────────────────────────────────
VALIDATION_FRAC = 0.20
CV_FOLDS        = 5
RANDOM_STATE    = 42

# ── Logging ────────────────────────────────────────────────────────────────────
import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("homecredit")