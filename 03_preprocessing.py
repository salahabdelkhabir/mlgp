"""
03_preprocessing.py
====================
Phase 3 — Clean, encode, impute, and scale the feature matrix produced
by 02_feature_engineering.py.  Produces model-ready arrays and saves:

    data/processed/X_train.parquet
    data/processed/X_val.parquet
    data/processed/y_train.parquet
    data/processed/y_val.parquet
    data/processed/X_test.parquet
    data/processed/feature_names.txt
    data/processed/preprocessing_report.xlsx

Steps
-----
1. Train / validation split (stratified)
2. Drop near-zero-variance and near-duplicate columns
3. Impute numerics  (median) and categoricals (mode / "Missing")
4. Encode categoricals
   - Low cardinality  (≤ MAX_ONEHOT_CARDINALITY) → Label encoding
   - High cardinality               → Target encoding (fit on train only)
5. Cap extreme outliers  (Winsorise at 1st / 99th percentile)
6. Optional log-transform for heavily skewed AMT_* columns
7. Align train and test columns

All transformations are FIT on train, APPLIED to val and test to prevent
data leakage.

Run
---
    python 03_preprocessing.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import (
    TARGET, ID_COL, MAX_ONEHOT_CARDINALITY,
    VALIDATION_FRAC, RANDOM_STATE, OUTPUT_DIR, logger,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(OUTPUT_DIR / "train_features.parquet")
    test  = pd.read_parquet(OUTPUT_DIR / "test_features.parquet")
    logger.info(f"Loaded train {train.shape}  test {test.shape}")
    return train, test


def split_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


# ── Step 1: Remove near-zero-variance columns ─────────────────────────────────

def drop_low_variance(X_train: pd.DataFrame, threshold: float = 0.01):
    """
    Drop columns where > (1-threshold)*100 % of values are the same.
    Only numeric columns are checked; categoricals are handled separately.
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = []
    for col in num_cols:
        top_freq = X_train[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq >= (1 - threshold):
            drop_cols.append(col)
    logger.info(f"  Low-variance columns dropped: {len(drop_cols)}")
    return drop_cols


# ── Step 2: Imputation ────────────────────────────────────────────────────────

class SimpleImputer:
    """
    Fit medians (numeric) and modes (categorical) on train,
    apply to any split.
    """
    def __init__(self):
        self.num_fill: dict = {}
        self.cat_fill: dict = {}

    def fit(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns

        for col in num_cols:
            self.num_fill[col] = X[col].median()
        for col in cat_cols:
            mode = X[col].mode()
            self.cat_fill[col] = mode.iloc[0] if len(mode) else "Missing"

        logger.info(
            f"  Imputer fit: {len(self.num_fill)} numeric, "
            f"{len(self.cat_fill)} categorical columns"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, val in self.num_fill.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        for col, val in self.cat_fill.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X


# ── Step 3: Target encoding  (high cardinality categoricals) ─────────────────

class TargetEncoder:
    """
    Replace each category value with the smoothed mean of TARGET.
    Smoothing prevents overfitting on rare categories.
    Fit ONLY on training data.
    """
    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.global_mean: float = 0.0
        self.encoding_map: dict[str, dict] = {}   # col → {value: encoded_float}

    def fit(self, X: pd.DataFrame, y: pd.Series, cols: list[str]):
        self.global_mean = float(y.mean())
        for col in cols:
            stats = (
                pd.DataFrame({"cat": X[col], "target": y})
                .groupby("cat")["target"]
                .agg(["mean", "count"])
            )
            smoother = self.smoothing / (stats["count"] + self.smoothing)
            encoded = (1 - smoother) * stats["mean"] + smoother * self.global_mean
            self.encoding_map[col] = encoded.to_dict()
        logger.info(f"  Target encoder fit on {len(cols)} high-cardinality columns")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.encoding_map.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(self.global_mean)
        return X


# ── Step 4: Label encoding  (low cardinality categoricals) ───────────────────

class MultiLabelEncoder:
    """Label-encode multiple columns; fit on train."""
    def __init__(self):
        self.encoders: dict[str, LabelEncoder] = {}

    def fit(self, X: pd.DataFrame, cols: list[str]):
        for col in cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        logger.info(f"  Label encoder fit on {len(cols)} low-cardinality columns")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, le in self.encoders.items():
            if col in X.columns:
                # Handle unseen values gracefully
                known = set(le.classes_)
                X[col] = X[col].astype(str).apply(
                    lambda v: v if v in known else le.classes_[0]
                )
                X[col] = le.transform(X[col])
        return X


# ── Step 5: Winsorisation  (outlier capping) ──────────────────────────────────

class Winsoriser:
    """Clip numeric features to [p01, p99] computed from train."""
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
        self.bounds: dict[str, tuple] = {}

    def fit(self, X: pd.DataFrame, cols: list[str]):
        for col in cols:
            lo = X[col].quantile(self.lower)
            hi = X[col].quantile(self.upper)
            self.bounds[col] = (lo, hi)
        logger.info(f"  Winsoriser fit on {len(cols)} numeric columns")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lo, upper=hi)
        return X


# ── Step 6: Log-transform skewed AMT_* columns ───────────────────────────────

def log_transform_skewed(X: pd.DataFrame, skew_threshold: float = 2.0) -> tuple[pd.DataFrame, list]:
    """Apply log1p to numeric columns with skewness > threshold."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    skewed = [
        col for col in num_cols
        if abs(X[col].skew()) > skew_threshold and X[col].min() >= 0
    ]
    X = X.copy()
    for col in skewed:
        X[col] = np.log1p(X[col])
    logger.info(f"  Log-transformed {len(skewed)} skewed columns")
    return X, skewed


# ── Step 7: Column alignment ──────────────────────────────────────────────────

def align_columns(X_train: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Ensure test/val has exactly the same columns as train (same order)."""
    missing = set(X_train.columns) - set(X_other.columns)
    for col in missing:
        X_other[col] = 0
    extra = set(X_other.columns) - set(X_train.columns)
    X_other = X_other.drop(columns=list(extra))
    return X_other[X_train.columns]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Phase 3 — Preprocessing")
    logger.info("=" * 60)

    train_df, test_df = load_features()

    # Separate target
    X_all, y_all = split_target(train_df)
    X_test_raw = test_df.drop(columns=[TARGET], errors="ignore")

    # ── 1. Train / val split ────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=VALIDATION_FRAC,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )
    logger.info(
        f"Split: train {X_train.shape[0]:,}  val {X_val.shape[0]:,}  "
        f"test {X_test_raw.shape[0]:,}"
    )

    # Drop ID column from features
    for df in [X_train, X_val, X_test_raw]:
        if ID_COL in df.columns:
            df.drop(columns=[ID_COL], inplace=True)

    # ── 2. Drop low-variance columns ────────────────────────────────────────
    low_var_cols = drop_low_variance(X_train)
    X_train = X_train.drop(columns=low_var_cols)
    X_val   = X_val.drop(columns=low_var_cols, errors="ignore")
    X_test_raw = X_test_raw.drop(columns=low_var_cols, errors="ignore")

    # ── 3. Identify column types ─────────────────────────────────────────────
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card  = [c for c in cat_cols if X_train[c].nunique() <= MAX_ONEHOT_CARDINALITY]
    high_card = [c for c in cat_cols if X_train[c].nunique() >  MAX_ONEHOT_CARDINALITY]
    num_cols  = X_train.select_dtypes(include=[np.number]).columns.tolist()

    logger.info(
        f"  Column types: {len(num_cols)} numeric  "
        f"{len(low_card)} low-card cat  {len(high_card)} high-card cat"
    )

    # ── 4. Impute ────────────────────────────────────────────────────────────
    imputer = SimpleImputer().fit(X_train)
    X_train     = imputer.transform(X_train)
    X_val       = imputer.transform(X_val)
    X_test_raw  = imputer.transform(X_test_raw)

    # ── 5. Encode categoricals ───────────────────────────────────────────────
    # Label encode low cardinality
    le_enc = MultiLabelEncoder().fit(X_train, low_card)
    X_train    = le_enc.transform(X_train)
    X_val      = le_enc.transform(X_val)
    X_test_raw = le_enc.transform(X_test_raw)

    # Target encode high cardinality
    te_enc = TargetEncoder().fit(X_train, y_train, high_card)
    X_train    = te_enc.transform(X_train)
    X_val      = te_enc.transform(X_val)
    X_test_raw = te_enc.transform(X_test_raw)

    # ── 6. Winsorise ─────────────────────────────────────────────────────────
    num_cols_now = X_train.select_dtypes(include=[np.number]).columns.tolist()
    wins = Winsoriser().fit(X_train, num_cols_now)
    X_train    = wins.transform(X_train)
    X_val      = wins.transform(X_val)
    X_test_raw = wins.transform(X_test_raw)

    # ── 7. Log-transform skewed columns ──────────────────────────────────────
    X_train, log_cols = log_transform_skewed(X_train)
    for col in log_cols:
        for df in [X_val, X_test_raw]:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

    # ── 8. Align test columns ────────────────────────────────────────────────
    X_val      = align_columns(X_train, X_val)
    X_test_raw = align_columns(X_train, X_test_raw)

    # ── Save ─────────────────────────────────────────────────────────────────
    logger.info("Saving processed splits …")
    X_train.to_parquet(OUTPUT_DIR / "X_train.parquet",  index=False)
    X_val.to_parquet(  OUTPUT_DIR / "X_val.parquet",    index=False)
    X_test_raw.to_parquet(OUTPUT_DIR / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(OUTPUT_DIR / "y_train.parquet", index=False)
    y_val.to_frame().to_parquet(  OUTPUT_DIR / "y_val.parquet",   index=False)

    # Save feature names
    feature_names = X_train.columns.tolist()
    (OUTPUT_DIR / "feature_names.txt").write_text("\n".join(feature_names))

    # ── Preprocessing report ─────────────────────────────────────────────────
    report = {
        "summary": pd.DataFrame([{
            "train_rows": X_train.shape[0],
            "val_rows":   X_val.shape[0],
            "test_rows":  X_test_raw.shape[0],
            "n_features": X_train.shape[1],
            "low_var_dropped": len(low_var_cols),
            "low_card_encoded": len(low_card),
            "high_card_target_encoded": len(high_card),
            "log_transformed": len(log_cols),
            "target_rate_train": round(float(y_train.mean()), 4),
            "target_rate_val":   round(float(y_val.mean()),   4),
        }]),
        "low_var_dropped":  pd.DataFrame({"column": low_var_cols}),
        "high_card_cols":   pd.DataFrame({"column": high_card}),
        "log_cols":         pd.DataFrame({"column": log_cols}),
        "imputer_numeric":  pd.DataFrame(
            [{"col": k, "fill_value": v} for k, v in imputer.num_fill.items()]
        ),
    }

    rpt_path = OUTPUT_DIR / "preprocessing_report.xlsx"
    with pd.ExcelWriter(rpt_path, engine="openpyxl") as writer:
        for sheet, df in report.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)

    logger.info("")
    logger.info(f"Final feature count : {X_train.shape[1]}")
    logger.info(f"Train rows          : {X_train.shape[0]:,}")
    logger.info(f"Val rows            : {X_val.shape[0]:,}")
    logger.info(f"Test rows           : {X_test_raw.shape[0]:,}")
    logger.info(f"Preprocessing report → {rpt_path}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
