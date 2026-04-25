"""
01_eda.py
=========
Phase 1 — Load all tables, profile each one, and produce a structured EDA
report saved to reports/eda_summary.xlsx.

What this script does
---------------------
1. Loads every raw CSV and snapshots its shape, dtypes, missing-value rates.
2. Profiles the application train table in depth:
   - Target distribution & class imbalance
   - Numeric column statistics (mean, std, skew, % missing)
   - Categorical column statistics (cardinality, top value, target rate per value)
   - Correlation of every numeric feature with TARGET
3. Checks known anomalies (DAYS_EMPLOYED sentinel, negative DAYS_* columns).
4. Writes a multi-sheet Excel report: one sheet per section.

Run
---
    python 01_eda.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from config import FILES, TARGET, ID_COL, DAYS_EMPLOYED_SENTINEL, REPORTS_DIR, logger


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_table(key: str) -> pd.DataFrame:
    path = FILES[key]
    logger.info(f"Loading {key}  ({path.name})")
    df = pd.read_csv(path)
    logger.info(f"  → {df.shape[0]:,} rows  {df.shape[1]} cols")
    return df


def missing_profile(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a dataframe summarising dtype and missingness for each column."""
    total = len(df)
    rec = []
    for col in df.columns:
        n_miss = df[col].isna().sum()
        rec.append({
            "table":   label,
            "column":  col,
            "dtype":   str(df[col].dtype),
            "n_miss":  n_miss,
            "pct_miss": round(100 * n_miss / total, 2),
            "n_unique": df[col].nunique(),
        })
    return pd.DataFrame(rec)


def numeric_stats(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Descriptive statistics + correlation with TARGET for numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [ID_COL, target_col]]

    records = []
    for col in num_cols:
        series = df[col].dropna()
        corr, pval = pointbiserialr(
            df.loc[df[col].notna(), target_col],
            series
        ) if len(series) > 10 else (np.nan, np.nan)

        records.append({
            "column":      col,
            "count":       int(series.count()),
            "pct_miss":    round(100 * df[col].isna().mean(), 2),
            "mean":        round(series.mean(), 4),
            "std":         round(series.std(), 4),
            "min":         round(series.min(), 4),
            "p25":         round(series.quantile(0.25), 4),
            "median":      round(series.median(), 4),
            "p75":         round(series.quantile(0.75), 4),
            "max":         round(series.max(), 4),
            "skew":        round(series.skew(), 4),
            "corr_target": round(corr, 4) if not np.isnan(corr) else None,
            "pval":        round(pval, 6) if not np.isnan(pval) else None,
        })

    return pd.DataFrame(records).sort_values("corr_target", key=abs, ascending=False)


def categorical_stats(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Cardinality, top value, missing rate, and default rate per category."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    records = []
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        top_val = vc.index[0] if len(vc) else None
        top_freq = vc.iloc[0] if len(vc) else 0

        # target rate per category — identify which values have elevated risk
        cat_target = (
            df.groupby(col, dropna=False)[target_col]
            .agg(["mean", "count"])
            .rename(columns={"mean": "default_rate", "count": "n"})
            .sort_values("default_rate", ascending=False)
            .head(3)
            .to_dict("index")
        )

        records.append({
            "column":         col,
            "n_unique":       df[col].nunique(dropna=True),
            "pct_miss":       round(100 * df[col].isna().mean(), 2),
            "top_value":      str(top_val),
            "top_freq":       int(top_freq),
            "top_pct":        round(100 * top_freq / len(df), 2),
            "high_risk_cats": str(list(cat_target.keys())[:3]),
        })

    return pd.DataFrame(records).sort_values("n_unique", ascending=False)


def anomaly_check(df: pd.DataFrame) -> pd.DataFrame:
    """Flag known data quality issues in the application table."""
    issues = []

    # DAYS_EMPLOYED sentinel
    if "DAYS_EMPLOYED" in df.columns:
        n = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).sum()
        issues.append({
            "check": "DAYS_EMPLOYED sentinel (365243)",
            "n_affected": int(n),
            "pct": round(100 * n / len(df), 2),
            "action": "Replace with NaN — represents retired/unemployed clients",
        })

    # Positive DAYS_BIRTH (should always be negative)
    if "DAYS_BIRTH" in df.columns:
        n = (df["DAYS_BIRTH"] > 0).sum()
        issues.append({
            "check": "DAYS_BIRTH > 0 (should be negative)",
            "n_affected": int(n),
            "pct": round(100 * n / len(df), 2),
            "action": "Flag rows — likely data entry errors",
        })

    # AMT_INCOME_TOTAL extreme outliers (> 99.9th percentile)
    if "AMT_INCOME_TOTAL" in df.columns:
        threshold = df["AMT_INCOME_TOTAL"].quantile(0.999)
        n = (df["AMT_INCOME_TOTAL"] > threshold).sum()
        issues.append({
            "check": f"AMT_INCOME_TOTAL > 99.9th pctile ({threshold:,.0f})",
            "n_affected": int(n),
            "pct": round(100 * n / len(df), 2),
            "action": "Cap or log-transform — extreme outliers distort means",
        })

    # CODE_GENDER == 'XNA'
    if "CODE_GENDER" in df.columns:
        n = (df["CODE_GENDER"] == "XNA").sum()
        issues.append({
            "check": "CODE_GENDER = 'XNA'",
            "n_affected": int(n),
            "pct": round(100 * n / len(df), 2),
            "action": "Treat as missing / separate category",
        })

    return pd.DataFrame(issues)


def target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Class counts and rates."""
    vc = df[TARGET].value_counts().reset_index()
    vc.columns = ["target_value", "count"]
    vc["pct"] = round(100 * vc["count"] / len(df), 2)
    vc["label"] = vc["target_value"].map({0: "No difficulty", 1: "Payment difficulty"})
    return vc


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Phase 1 — EDA")
    logger.info("=" * 60)

    # 1. Load all tables
    tables = {key: load_table(key) for key in FILES}

    # 2. Missing-value profile for every table
    logger.info("Building missing-value profiles …")
    all_profiles = pd.concat(
        [missing_profile(df, key) for key, df in tables.items()],
        ignore_index=True,
    )

    # 3. Deep profile of application train
    train = tables["train"]
    logger.info("Profiling application_train …")

    target_dist   = target_distribution(train)
    num_stats     = numeric_stats(train, TARGET)
    cat_stats     = categorical_stats(train, TARGET)
    anomalies     = anomaly_check(train)

    # High-missing columns summary (> 40 % missing)
    high_miss = (
        all_profiles[all_profiles["pct_miss"] > 40]
        .sort_values("pct_miss", ascending=False)
        .reset_index(drop=True)
    )

    # Top correlations
    top_corr = num_stats.dropna(subset=["corr_target"]).head(30)

    # 4. Write Excel report
    report_path = REPORTS_DIR / "eda_summary.xlsx"
    logger.info(f"Writing EDA report → {report_path}")

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        target_dist.to_excel(writer,  sheet_name="01_target_distribution", index=False)
        all_profiles.to_excel(writer, sheet_name="02_all_tables_missing",  index=False)
        high_miss.to_excel(writer,    sheet_name="03_high_missing_cols",    index=False)
        anomalies.to_excel(writer,    sheet_name="04_anomaly_checks",       index=False)
        num_stats.to_excel(writer,    sheet_name="05_numeric_stats",        index=False)
        cat_stats.to_excel(writer,    sheet_name="06_categorical_stats",    index=False)
        top_corr.to_excel(writer,     sheet_name="07_top_correlations",     index=False)

    # 5. Console summary
    logger.info("")
    logger.info("── Target distribution ──────────────────────────────")
    logger.info(f"\n{target_dist.to_string(index=False)}")

    logger.info("")
    logger.info("── Top 10 correlated numeric features ──────────────")
    logger.info(f"\n{top_corr[['column','corr_target','pct_miss']].head(10).to_string(index=False)}")

    logger.info("")
    logger.info("── Anomaly checks ───────────────────────────────────")
    logger.info(f"\n{anomalies[['check','n_affected','pct']].to_string(index=False)}")

    logger.info("")
    logger.info(f"EDA complete. Report saved → {report_path}")
    logger.info(f"Total tables loaded: {len(tables)}")
    logger.info(f"Total columns described: {all_profiles.shape[0]}")

    return tables   # pass forward if running in notebook


if __name__ == "__main__":
    main()
