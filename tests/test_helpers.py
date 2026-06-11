"""Unit tests for pipeline helper functions."""
import importlib
import numpy as np
import pandas as pd
import pytest

# Numbered modules — import via importlib
eda    = importlib.import_module("01_eda")
fe     = importlib.import_module("02_feature_engineering")
pre    = importlib.import_module("03_preprocessing")
select = importlib.import_module("04_model_selection")

from config import TARGET, ID_COL, DAYS_EMPLOYED_SENTINEL


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        ID_COL: range(n),
        TARGET: rng.binomial(1, 0.08, n),
        "AMT_INCOME_TOTAL": rng.lognormal(12, 0.5, n),
        "AMT_CREDIT": rng.lognormal(13, 0.6, n),
        "DAYS_BIRTH": rng.integers(-20000, -5000, n),
        "DAYS_REGISTRATION": rng.integers(-15000, -1, n),
        "DAYS_ID_PUBLISH": rng.integers(-10000, -1, n),
        "DAYS_LAST_PHONE_CHANGE": rng.integers(-5000, -1, n),
        "DAYS_EMPLOYED": np.where(
            rng.random(n) < 0.1,
            DAYS_EMPLOYED_SENTINEL,
            rng.integers(-15000, -1, n),
        ),
        "CODE_GENDER": rng.choice(["M", "F", "XNA"], n, p=[0.4, 0.59, 0.01]),
        "NAME_EDUCATION_TYPE": rng.choice(
            ["Secondary", "Higher", "Incomplete higher", "Academic degree"],
            n, p=[0.5, 0.3, 0.15, 0.05],
        ),
        "FLAG_DOCUMENT_2": rng.integers(0, 2, n),
        "FLAG_DOCUMENT_3": rng.integers(0, 2, n),
        "REG_REGION_NOT_LIVE_REGION": rng.integers(0, 2, n),
        "REG_REGION_NOT_WORK_REGION": rng.integers(0, 2, n),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, n),
        "AMT_GOODS_PRICE": rng.lognormal(13, 0.5, n),
        "AMT_ANNUITY": rng.lognormal(11, 0.4, n),
        "OBS_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 20, n),
        "DEF_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 5, n),
    })


# ══════════════════════════════════════════════════════════════════════════════
# 01_eda.py
# ══════════════════════════════════════════════════════════════════════════════

class TestEDAHelpers:

    def test_missing_profile_shape(self, sample_df):
        prof = eda.missing_profile(sample_df, "train")
        assert isinstance(prof, pd.DataFrame)
        assert list(prof.columns) == ["table", "column", "dtype", "n_miss", "pct_miss", "n_unique"]
        assert len(prof) == sample_df.shape[1]
        assert (prof["table"] == "train").all()

    def test_missing_profile_with_nan(self, sample_df):
        sample_df.loc[0, "AMT_INCOME_TOTAL"] = np.nan
        prof = eda.missing_profile(sample_df, "train")
        row = prof[prof["column"] == "AMT_INCOME_TOTAL"].iloc[0]
        assert row["n_miss"] == 1
        assert row["pct_miss"] > 0

    def test_target_distribution(self, sample_df):
        td = eda.target_distribution(sample_df)
        assert list(td.columns) == ["target_value", "count", "pct", "label"]
        assert td["count"].sum() == len(sample_df)

    def test_numeric_stats_excludes_id_and_target(self, sample_df):
        ns = eda.numeric_stats(sample_df, TARGET)
        assert ID_COL not in ns["column"].values
        assert TARGET not in ns["column"].values

    def test_numeric_stats_sorted_by_abs_corr(self, sample_df):
        ns = eda.numeric_stats(sample_df, TARGET)
        abs_corr = ns["corr_target"].dropna().abs().values
        for i in range(len(abs_corr) - 1):
            assert abs_corr[i] >= abs_corr[i + 1]

    def test_categorical_stats_has_required_columns(self, sample_df):
        cs = eda.categorical_stats(sample_df, TARGET)
        for col in ["column", "n_unique", "high_risk_cats", "top_value"]:
            assert col in cs.columns

    def test_anomaly_detects_sentinel(self, sample_df):
        issues = eda.anomaly_check(sample_df)
        assert any("DAYS_EMPLOYED sentinel" in c for c in issues["check"])
        n_sentinel = (sample_df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).sum()
        row = issues[issues["check"].str.contains("sentinel")].iloc[0]
        assert row["n_affected"] == n_sentinel

    def test_anomaly_detects_xna(self, sample_df):
        issues = eda.anomaly_check(sample_df)
        assert any("XNA" in c for c in issues["check"])


# ══════════════════════════════════════════════════════════════════════════════
# 02_feature_engineering.py
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    def test_engineer_application_adds_expected_features(self, sample_df):
        result = fe.engineer_application(sample_df)
        expected = [
            "app_age_years", "app_employed_years", "app_credit_to_income",
            "app_annuity_to_income", "app_doc_count", "app_address_mismatch_score",
            "app_social_default_rate_30",
            "app_income_per_person",
        ]
        for col in expected:
            assert col in result.columns

    def test_sentinel_replaced_with_nan(self, sample_df):
        result = fe.engineer_application(sample_df)
        n_sentinel_orig = (sample_df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).sum()
        n_nan_new = result["DAYS_EMPLOYED"].isna().sum()
        assert n_nan_new == n_sentinel_orig

    def test_xna_replaced_with_nan(self, sample_df):
        result = fe.engineer_application(sample_df)
        n_xna_orig = (sample_df["CODE_GENDER"] == "XNA").sum()
        n_nan_new = result["CODE_GENDER"].isna().sum()
        assert n_nan_new == n_xna_orig

    def test_app_age_positive(self, sample_df):
        result = fe.engineer_application(sample_df)
        assert (result["app_age_years"] > 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# 03_preprocessing.py
# ══════════════════════════════════════════════════════════════════════════════

class TestSimpleImputer:

    def test_fit_stores_medians(self, sample_df):
        X = sample_df.drop(columns=[TARGET, ID_COL])
        imputer = pre.SimpleImputer().fit(X)
        assert "AMT_INCOME_TOTAL" in imputer.num_fill

    def test_transform_fills_nan(self, sample_df):
        X = sample_df.drop(columns=[TARGET, ID_COL]).copy()
        X.iloc[:5, 0] = np.nan
        imputer = pre.SimpleImputer().fit(X)
        result = imputer.transform(X)
        assert result.iloc[:5, 0].isna().sum() == 0

    def test_transform_preserves_shape(self, sample_df):
        X = sample_df.drop(columns=[TARGET, ID_COL])
        imputer = pre.SimpleImputer().fit(X)
        result = imputer.transform(X)
        assert result.shape == X.shape


class TestTargetEncoder:

    def test_encoded_values_between_0_and_1(self, sample_df):
        X = sample_df[["NAME_EDUCATION_TYPE"]]
        y = sample_df[TARGET]
        te = pre.TargetEncoder(smoothing=5.0).fit(X, y, ["NAME_EDUCATION_TYPE"])
        encoded = te.transform(X)
        assert encoded["NAME_EDUCATION_TYPE"].between(0, 1).all()

    def test_unknown_category_gets_global_mean(self, sample_df):
        X_train = sample_df[["NAME_EDUCATION_TYPE"]]
        y_train = sample_df[TARGET]
        te = pre.TargetEncoder().fit(X_train, y_train, ["NAME_EDUCATION_TYPE"])
        X_test = pd.DataFrame({"NAME_EDUCATION_TYPE": ["Unknown_Degree"]})
        encoded = te.transform(X_test)
        assert encoded["NAME_EDUCATION_TYPE"].iloc[0] == pytest.approx(y_train.mean(), abs=0.05)


class TestMultiLabelEncoder:

    def test_output_is_integer(self, sample_df):
        X = sample_df[["NAME_EDUCATION_TYPE", "CODE_GENDER"]]
        mle = pre.MultiLabelEncoder().fit(X, ["NAME_EDUCATION_TYPE", "CODE_GENDER"])
        encoded = mle.transform(X)
        for col in ["NAME_EDUCATION_TYPE", "CODE_GENDER"]:
            assert encoded[col].dtype in (np.int64, np.int32)

    def test_unseen_value_defaults_to_first_class(self, sample_df):
        X_train = sample_df[["CODE_GENDER"]]
        mle = pre.MultiLabelEncoder().fit(X_train, ["CODE_GENDER"])
        X_test = pd.DataFrame({"CODE_GENDER": ["Z"]})
        encoded = mle.transform(X_test)
        first = mle.encoders["CODE_GENDER"].classes_[0]
        expected = mle.encoders["CODE_GENDER"].transform([first])[0]
        assert encoded["CODE_GENDER"].iloc[0] == expected


class TestWinsoriser:

    def test_clips_extremes(self, sample_df):
        X = sample_df[["AMT_INCOME_TOTAL", "AMT_CREDIT"]]
        wins = pre.Winsoriser(0.05, 0.95).fit(X, ["AMT_INCOME_TOTAL", "AMT_CREDIT"])
        clipped = wins.transform(X)
        lo, hi = wins.bounds["AMT_INCOME_TOTAL"]
        assert clipped["AMT_INCOME_TOTAL"].min() >= lo
        assert clipped["AMT_INCOME_TOTAL"].max() <= hi


class TestLogTransformSkewed:

    def test_constant_column_not_transformed(self, sample_df):
        X = sample_df[["AMT_INCOME_TOTAL", "AMT_CREDIT"]].copy()
        X["const"] = 1.0
        _, skewed = pre.log_transform_skewed(X, skew_threshold=2.0)
        assert "const" not in skewed

    def test_reduces_skewness(self, sample_df):
        X = sample_df[["AMT_INCOME_TOTAL", "AMT_CREDIT"]].copy()
        orig_skew = X["AMT_INCOME_TOTAL"].skew()
        result, skewed = pre.log_transform_skewed(X, skew_threshold=0.0)
        if "AMT_INCOME_TOTAL" in skewed:
            assert abs(result["AMT_INCOME_TOTAL"].skew()) < abs(orig_skew)


# ══════════════════════════════════════════════════════════════════════════════
# 04_model_selection.py
# ══════════════════════════════════════════════════════════════════════════════

class TestModelSelectionHelpers:

    def test_ks_statistic_range(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        ks = select.ks_statistic(y_true, y_prob)
        assert 0 <= ks <= 1

    def test_best_f1_threshold_perfect_separation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        f1, thresh = select.best_f1_threshold(y_true, y_prob)
        assert f1 == pytest.approx(1.0, abs=0.01)
        assert 0.3 < thresh < 0.7

    def test_best_f1_threshold_random(self):
        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.2, 500)
        y_prob = rng.random(500)
        f1, thresh = select.best_f1_threshold(y_true, y_prob)
        assert 0 <= f1 <= 1
        assert 0 <= thresh <= 1
