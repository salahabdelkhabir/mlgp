"""
02_feature_engineering.py
==========================
Phase 2 — Aggregate all support tables and engineer features from the
application table.  Produces:

    data/processed/train_features.parquet
    data/processed/test_features.parquet

Feature groups built
---------------------
A. Application table
   - Fix sentinel / clean anomalies
   - Ratio features  (credit_to_income, annuity_to_income, …)
   - Age / employment tenure in years
   - Document count  (sum of FLAG_DOCUMENT_*)
   - Address mismatch score
   - Days features converted to positive years

B. Bureau + bureau_balance aggregates  (prefix: bur_)
   - Counts, active credits, overdue amounts, DPD history

C. Previous application aggregates  (prefix: prev_)
   - Approval / rejection counts, recency, interest rate stats

D. POS CASH balance aggregates  (prefix: pos_)
   - Max DPD, contract status breakdown, instalment completion

E. Installments payment aggregates  (prefix: ins_)
   - Payment ratio, late-payment count, max days late

F. Credit card balance aggregates  (prefix: cc_)
   - Utilisation ratio, drawing frequency, payment regularity

Run
---
    python 02_feature_engineering.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from config import (
    FILES, TARGET, ID_COL, DAYS_EMPLOYED_SENTINEL,
    OUTPUT_DIR, logger,
)


# ══════════════════════════════════════════════════════════════════════════════
# A. Application table cleaning & direct features
# ══════════════════════════════════════════════════════════════════════════════

def engineer_application(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and derive features directly from the application table."""
    df = df.copy()

    # ── Fix anomalies ──────────────────────────────────────────────────────
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
        DAYS_EMPLOYED_SENTINEL, np.nan
    )
    df.loc[df["CODE_GENDER"] == "XNA", "CODE_GENDER"] = np.nan

    # ── Convert negative DAYS_* to positive years ──────────────────────────
    # All DAYS_* fields are stored as negative integers (days before application)
    df["app_age_years"]        = (-df["DAYS_BIRTH"])    / 365.25
    df["app_employed_years"]   = (-df["DAYS_EMPLOYED"]) / 365.25
    df["app_reg_years"]        = (-df["DAYS_REGISTRATION"]) / 365.25
    df["app_id_publish_years"] = (-df["DAYS_ID_PUBLISH"])   / 365.25
    df["app_phone_change_years"] = (-df["DAYS_LAST_PHONE_CHANGE"]) / 365.25

    # ── Financial ratios ───────────────────────────────────────────────────
    df["app_credit_to_income"]  = df["AMT_CREDIT"]  / (df["AMT_INCOME_TOTAL"] + 1)
    df["app_annuity_to_income"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["app_credit_to_goods"]   = df["AMT_CREDIT"]  / (df["AMT_GOODS_PRICE"]  + 1)
    df["app_annuity_to_credit"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"]       + 1)
    df["app_goods_to_income"]   = df["AMT_GOODS_PRICE"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["app_income_per_person"] = (
        df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"].clip(lower=1))
    )

    # ── Employment ratio (how long employed relative to age) ───────────────
    df["app_employed_to_age_ratio"] = (
        df["app_employed_years"] / (df["app_age_years"] + 1)
    )

    # ── Document provision score  (total docs provided) ───────────────────
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["app_doc_count"] = df[doc_cols].sum(axis=1)

    # ── Address mismatch score  (sum of REG/LIVE/WORK flags) ──────────────
    mismatch_cols = [
        "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",      "LIVE_CITY_NOT_WORK_CITY",
    ]
    exist = [c for c in mismatch_cols if c in df.columns]
    df["app_address_mismatch_score"] = df[exist].sum(axis=1)

    # ── Social circle default rates ────────────────────────────────────────
    for dpd in [30, 60]:
        obs = f"OBS_{dpd}_CNT_SOCIAL_CIRCLE"
        dft = f"DEF_{dpd}_CNT_SOCIAL_CIRCLE"
        if obs in df.columns and dft in df.columns:
            df[f"app_social_default_rate_{dpd}"] = (
                df[dft] / (df[obs] + 1)
            )

    # ── Credit bureau enquiry recency ─────────────────────────────────────
    enquiry_cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",  "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    exist_enq = [c for c in enquiry_cols if c in df.columns]
    df["app_total_bureau_enquiries"] = df[exist_enq].sum(axis=1)

    # ── Building info average (compress AVG/MODE/MEDI into one score) ─────
    avg_cols = [c for c in df.columns if c.endswith("_AVG")]
    if avg_cols:
        df["app_building_avg_score"] = df[avg_cols].mean(axis=1)

    logger.info(
        f"  Application features: {df.shape[1]} cols "
        f"(started with {len(df.columns)} cols)"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# B. Bureau aggregates
# ══════════════════════════════════════════════════════════════════════════════

def agg_bureau(bureau: pd.DataFrame, bureau_bal: pd.DataFrame) -> pd.DataFrame:
    """Aggregate external credit bureau records per SK_ID_CURR."""
    logger.info("  Aggregating bureau …")

    # ── Bureau balance: status-level features ────────────────────────────
    # STATUS: C=closed, X=unknown, 0=no DPD, 1-5=DPD buckets
    bb = bureau_bal.copy()
    bb["dpd_flag"] = bb["STATUS"].isin(["1","2","3","4","5"]).astype(int)
    bb["closed_flag"] = (bb["STATUS"] == "C").astype(int)

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        bur_bal_n_months       = ("MONTHS_BALANCE", "count"),
        bur_bal_dpd_count      = ("dpd_flag",       "sum"),
        bur_bal_dpd_rate       = ("dpd_flag",       "mean"),
        bur_bal_months_closed  = ("closed_flag",    "sum"),
    ).reset_index()

    # Join bureau balance back to bureau
    b = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    # ── Active credit flags ───────────────────────────────────────────────
    b["active_flag"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["closed_flag_b"] = (b["CREDIT_ACTIVE"] == "Closed").astype(int)

    # ── Overdue flag ──────────────────────────────────────────────────────
    b["overdue_flag"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype(int)

    agg = b.groupby(ID_COL).agg(
        bur_n_credits              = ("SK_ID_BUREAU",          "count"),
        bur_n_active               = ("active_flag",           "sum"),
        bur_n_closed               = ("closed_flag_b",         "sum"),
        bur_active_rate            = ("active_flag",           "mean"),
        bur_max_overdue_days       = ("CREDIT_DAY_OVERDUE",    "max"),
        bur_sum_overdue_days       = ("CREDIT_DAY_OVERDUE",    "sum"),
        bur_n_overdue              = ("overdue_flag",          "sum"),
        bur_max_overdue_amt        = ("AMT_CREDIT_MAX_OVERDUE","max"),
        bur_sum_credit             = ("AMT_CREDIT_SUM",        "sum"),
        bur_sum_debt               = ("AMT_CREDIT_SUM_DEBT",   "sum"),
        bur_sum_overdue_amt        = ("AMT_CREDIT_SUM_OVERDUE","sum"),
        bur_mean_credit            = ("AMT_CREDIT_SUM",        "mean"),
        bur_max_prolonged          = ("CNT_CREDIT_PROLONG",    "max"),
        bur_days_since_last_update = ("DAYS_CREDIT_UPDATE",    "max"),  # least negative = most recent
        bur_days_first_credit      = ("DAYS_CREDIT",           "min"),  # most negative = oldest
        bur_days_last_credit       = ("DAYS_CREDIT",           "max"),  # least negative = most recent
        # bureau_balance roll-up
        bur_total_dpd_months       = ("bur_bal_dpd_count",     "sum"),
        bur_max_dpd_rate           = ("bur_bal_dpd_rate",      "max"),
        bur_mean_dpd_rate          = ("bur_bal_dpd_rate",      "mean"),
    ).reset_index()

    # Derived
    agg["bur_debt_to_credit_ratio"] = (
        agg["bur_sum_debt"] / (agg["bur_sum_credit"] + 1)
    )

    logger.info(f"    → {agg.shape[1]-1} bureau features for {agg.shape[0]:,} clients")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# C. Previous application aggregates
# ══════════════════════════════════════════════════════════════════════════════

def agg_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prior Home Credit loan applications per SK_ID_CURR."""
    logger.info("  Aggregating previous_application …")

    p = prev.copy()
    p["approved_flag"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["refused_flag"]  = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    p["insured_flag"]  = p["NFLAG_INSURED_ON_APPROVAL"].fillna(0) if "NFLAG_INSURED_ON_APPROVAL" in p.columns else 0
    p["credit_vs_application"] = p["AMT_CREDIT"] / (p["AMT_APPLICATION"] + 1)

    # Build agg spec only for columns that actually exist
    agg_spec = {
        "prev_n_applications":     ("SK_ID_PREV",            "count"),
        "prev_n_approved":         ("approved_flag",         "sum"),
        "prev_n_refused":          ("refused_flag",          "sum"),
        "prev_approval_rate":      ("approved_flag",         "mean"),
        "prev_mean_annuity":       ("AMT_ANNUITY",           "mean"),
        "prev_max_annuity":        ("AMT_ANNUITY",           "max"),
        "prev_mean_credit":        ("AMT_CREDIT",            "mean"),
        "prev_mean_down_payment":  ("AMT_DOWN_PAYMENT",      "mean"),
        "prev_mean_down_rate":     ("RATE_DOWN_PAYMENT",     "mean"),
        "prev_mean_credit_vs_app": ("credit_vs_application", "mean"),
        "prev_days_last_decision": ("DAYS_DECISION",         "max"),
        "prev_days_first_decision":("DAYS_DECISION",         "min"),
        "prev_mean_cnt_payment":   ("CNT_PAYMENT",           "mean"),
        "prev_n_insured":          ("insured_flag",          "sum"),
    }
    # Optional columns — include only if present
    if "RATE_INTEREST_PRIMARY" in p.columns:
        agg_spec["prev_mean_interest_rate"] = ("RATE_INTEREST_PRIMARY", "mean")
        agg_spec["prev_max_interest_rate"]  = ("RATE_INTEREST_PRIMARY", "max")

    agg = p.groupby(ID_COL).agg(**agg_spec).reset_index()

    agg["prev_refusal_rate"] = (
        agg["prev_n_refused"] / (agg["prev_n_applications"] + 1)
    )

    logger.info(f"    → {agg.shape[1]-1} prev_app features for {agg.shape[0]:,} clients")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# D. POS CASH balance aggregates
# ══════════════════════════════════════════════════════════════════════════════

def agg_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS/CASH monthly balance snapshots per SK_ID_CURR."""
    logger.info("  Aggregating POS_CASH_balance …")

    p = pos.copy()
    p["active_flag"] = (p["NAME_CONTRACT_STATUS"] == "Active").astype(int)
    p["completed_flag"] = (p["NAME_CONTRACT_STATUS"] == "Completed").astype(int)

    agg = p.groupby(ID_COL).agg(
        pos_n_records         = ("SK_DPD",              "count"),
        pos_max_dpd           = ("SK_DPD",              "max"),
        pos_mean_dpd          = ("SK_DPD",              "mean"),
        pos_sum_dpd           = ("SK_DPD",              "sum"),
        pos_max_dpd_def       = ("SK_DPD_DEF",          "max"),
        pos_n_active          = ("active_flag",         "sum"),
        pos_n_completed       = ("completed_flag",      "sum"),
        pos_mean_instalment   = ("CNT_INSTALMENT",      "mean"),
        pos_mean_inst_future  = ("CNT_INSTALMENT_FUTURE","mean"),
    ).reset_index()

    agg["pos_active_rate"] = agg["pos_n_active"] / (agg["pos_n_records"] + 1)

    logger.info(f"    → {agg.shape[1]-1} pos_cash features for {agg.shape[0]:,} clients")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# E. Installments payment aggregates
# ══════════════════════════════════════════════════════════════════════════════

def agg_installments(ins: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installment payment records per SK_ID_CURR."""
    logger.info("  Aggregating installments_payments …")

    i = ins.copy()

    # Key derived signals
    i["payment_ratio"]    = i["AMT_PAYMENT"] / (i["AMT_INSTALMENT"] + 1)
    i["days_late"]        = i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]
    i["late_flag"]        = (i["days_late"] > 0).astype(int)
    i["paid_in_full"]     = (i["AMT_PAYMENT"] >= i["AMT_INSTALMENT"]).astype(int)
    i["underpaid_amount"] = (i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]).clip(lower=0)

    agg = i.groupby(ID_COL).agg(
        ins_n_payments             = ("AMT_PAYMENT",    "count"),
        ins_mean_payment_ratio     = ("payment_ratio",  "mean"),
        ins_min_payment_ratio      = ("payment_ratio",  "min"),
        ins_max_days_late          = ("days_late",      "max"),
        ins_mean_days_late         = ("days_late",      "mean"),
        ins_n_late_payments        = ("late_flag",      "sum"),
        ins_late_payment_rate      = ("late_flag",      "mean"),
        ins_n_paid_in_full         = ("paid_in_full",   "sum"),
        ins_total_underpaid        = ("underpaid_amount","sum"),
        ins_mean_underpaid         = ("underpaid_amount","mean"),
        ins_max_underpaid          = ("underpaid_amount","max"),
    ).reset_index()

    agg["ins_paid_in_full_rate"] = (
        agg["ins_n_paid_in_full"] / (agg["ins_n_payments"] + 1)
    )

    logger.info(f"    → {agg.shape[1]-1} installment features for {agg.shape[0]:,} clients")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# F. Credit card balance aggregates
# ══════════════════════════════════════════════════════════════════════════════

def agg_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly credit card balance snapshots per SK_ID_CURR."""
    logger.info("  Aggregating credit_card_balance …")

    c = cc.copy()

    # Utilisation: balance / credit limit
    c["utilisation"] = c["AMT_BALANCE"] / (c["AMT_CREDIT_LIMIT_ACTUAL"] + 1)

    # Payment ratio: actual payment / balance
    c["cc_payment_ratio"] = (
        c["AMT_PAYMENT_TOTAL_CURRENT"] / (c["AMT_BALANCE"] + 1)
    )

    c["active_flag"] = (c["NAME_CONTRACT_STATUS"] == "Active").astype(int)

    agg = c.groupby(ID_COL).agg(
        cc_n_records                 = ("AMT_BALANCE",              "count"),
        cc_mean_balance              = ("AMT_BALANCE",              "mean"),
        cc_max_balance               = ("AMT_BALANCE",              "max"),
        cc_mean_limit                = ("AMT_CREDIT_LIMIT_ACTUAL",  "mean"),
        cc_mean_utilisation          = ("utilisation",              "mean"),
        cc_max_utilisation           = ("utilisation",              "max"),
        cc_mean_payment_ratio        = ("cc_payment_ratio",         "mean"),
        cc_min_payment_ratio         = ("cc_payment_ratio",         "min"),
        cc_mean_drawings_atm         = ("AMT_DRAWINGS_ATM_CURRENT", "mean"),
        cc_mean_drawings_total       = ("AMT_DRAWINGS_CURRENT",     "mean"),
        cc_max_dpd                   = ("SK_DPD",                   "max"),
        cc_mean_dpd                  = ("SK_DPD",                   "mean"),
        cc_max_dpd_def               = ("SK_DPD_DEF",               "max"),
        cc_mean_min_payment_required = ("AMT_INST_MIN_REGULARITY",  "mean"),
        cc_n_active_months           = ("active_flag",              "sum"),
        cc_total_instalment_mature   = ("CNT_INSTALMENT_MATURE_CUM","max"),
    ).reset_index()

    logger.info(f"    → {agg.shape[1]-1} credit card features for {agg.shape[0]:,} clients")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# Master join
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    app: pd.DataFrame,
    bur_agg: pd.DataFrame,
    prev_agg: pd.DataFrame,
    pos_agg: pd.DataFrame,
    ins_agg: pd.DataFrame,
    cc_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join all aggregated feature tables onto application."""
    logger.info("  Joining all feature tables onto application …")

    df = app
    for agg, name in [
        (bur_agg,  "bureau"),
        (prev_agg, "previous_application"),
        (pos_agg,  "pos_cash"),
        (ins_agg,  "installments"),
        (cc_agg,   "credit_card"),
    ]:
        before = df.shape[1]
        df = df.merge(agg, on=ID_COL, how="left")
        added = df.shape[1] - before
        logger.info(f"    + {name}: +{added} cols  → {df.shape[1]} total")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("Phase 2 — Feature Engineering")
    logger.info("=" * 60)

    # Load raw tables
    logger.info("Loading raw tables …")
    train_raw      = pd.read_csv(FILES["train"])
    test_raw       = pd.read_csv(FILES["test"])
    bureau         = pd.read_csv(FILES["bureau"])
    bureau_balance = pd.read_csv(FILES["bureau_balance"])
    prev_app       = pd.read_csv(FILES["prev_app"])
    pos_cash       = pd.read_csv(FILES["pos_cash"])
    installments   = pd.read_csv(FILES["installments"])
    cc_balance     = pd.read_csv(FILES["cc_balance"])

    # Build aggregates (shared between train and test)
    bur_agg  = agg_bureau(bureau, bureau_balance)
    prev_agg = agg_previous_application(prev_app)
    pos_agg  = agg_pos_cash(pos_cash)
    ins_agg  = agg_installments(installments)
    cc_agg   = agg_credit_card(cc_balance)

    # Engineer application-level features
    logger.info("Engineering application features …")
    train_app = engineer_application(train_raw)
    test_app  = engineer_application(test_raw)

    # Build master feature matrices
    logger.info("Building train feature matrix …")
    train_feat = build_feature_matrix(
        train_app, bur_agg, prev_agg, pos_agg, ins_agg, cc_agg
    )

    logger.info("Building test feature matrix …")
    test_feat = build_feature_matrix(
        test_app, bur_agg, prev_agg, pos_agg, ins_agg, cc_agg
    )

    # Save
    train_out = OUTPUT_DIR / "train_features.parquet"
    test_out  = OUTPUT_DIR / "test_features.parquet"

    train_feat.to_parquet(train_out, index=False)
    test_feat.to_parquet(test_out,   index=False)

    logger.info("")
    logger.info(f"Train features shape : {train_feat.shape}")
    logger.info(f"Test  features shape : {test_feat.shape}")
    logger.info(f"Saved → {train_out}")
    logger.info(f"Saved → {test_out}")
    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    main()