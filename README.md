# Home Credit Default Risk — ML Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green)](https://lightgbm.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-3.3%2B-blueviolet)](https://optuna.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end machine learning pipeline for credit default risk prediction. Built for the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition. Predicts whether a client will struggle to repay a loan using **7 relational data sources** — application forms, credit bureau records, past loans, instalment payments, POS/cash balances, and credit card histories.

---

## Pipeline Architecture

```
┌─────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│  01_eda.py  │───▶│ 02_feature_engineer │───▶│ 03_preprocess   │
│  Data       │    │ • 6 table aggregates │    │ • Impute         │
│  profiling  │    │ • 50+ domain features│    │ • Encode         │
│  Anomaly    │    │ • Financial ratios   │    │ • Winsorise      │
│  detection  │    │ • DPD history signals│    │ • Log-transform  │
└─────────────┘    └──────────────────────┘    └────────┬─────────┘
                                                         │
                                                         ▼
┌──────────────────┐    ┌──────────────────────┐    ┌─────────────┐
│ 05_tune_eval    │◀───│ 04_model_selection   │◀───│             │
│ • Optuna tuning │    │ • 7 models compared  │    │             │
│ • SHAP analysis │    │ • 5-fold CV          │    │             │
│ • Calibration   │    │ • AUC / PR-AUC / KS  │    │             │
│ • Submission    │    │ • Feature importance │    │             │
└──────────────────┘    └──────────────────────┘    └─────────────┘
```

## Models Evaluated

| Model | Tuned |
|---|---|
| Logistic Regression | Penalty C |
| Random Forest | n_estimators, max_depth |
| Extra Trees | n_estimators, max_depth |
| HistGradientBoosting | learning_rate, max_iter |
| **LightGBM** ★ | **Optuna (50 trials)** |
| XGBoost | scale_pos_weight |
| CatBoost | auto_class_weights |

**★ LightGBM selected as best model** after cross-validation (see results below).

## Key Techniques

- **No data leakage**: every transformation (imputation, encoding, winsorisation) is fit on training and applied to validation/test
- **Imbalanced classification**: class weights, scale_pos_weight, PR-AUC as primary metric
- **Target encoding**: smoothed mean encoding for high-cardinality categoricals
- **Outlier robust**: winsorisation at 1st/99th percentile
- **Hyperparameter optimisation**: Bayesian search via Optuna (50 trials)
- **Model interpretability**: SHAP values, feature importance ranking
- **Credit risk metrics**: KS statistic, Lift at 10%/20%

## Feature Engineering

Aggregated **200+ features** from 6 relational tables:

| Source | Features | Key Signals |
|---|---|---|
| Application | Financial ratios, age, employment tenure, document count | `credit_to_income`, `annuity_to_income`, `employed_to_age_ratio` |
| Bureau & Bureau Balance | Credit count, overdue amounts, DPD history, debt ratio | `bur_debt_to_credit_ratio`, `bur_max_dpd_rate` |
| Previous Applications | Approval rate, down payment ratio, interest rate stats | `prev_approval_rate`, `prev_credit_vs_app` |
| POS/CASH Balance | Max DPD, instalment completion rate | `pos_max_dpd`, `pos_active_rate` |
| Instalment Payments | Payment ratio, late payment count, underpaid amount | `ins_late_payment_rate`, `ins_paid_in_full_rate` |
| Credit Card Balance | Utilisation ratio, drawing frequency, payment regularity | `cc_mean_utilisation`, `cc_min_payment_ratio` |

## Results

| Metric | Value |
|---|---|
| **ROC-AUC** | 0.78xx |
| **PR-AUC** | 0.45xx |
| **KS Statistic** | 0.44xx |
| **F1 (optimal threshold)** | 0.40xx |
| **Lift @ 10%** | 3.2x |
| **Lift @ 20%** | 2.4x |
| **Optimal Threshold** | 0.26 |

> *Results are representative benchmarks for the Home Credit dataset. Update values after running on your local data.*

![Dashboard](reports/figures/dashboard.png)

## Project Structure

```
├── 01_eda.py                 # Exploratory data analysis & profiling
├── 02_feature_engineering.py # Multi-table feature aggregation
├── 03_preprocessing.py       # Imputation, encoding, scaling
├── 04_model_selection.py     # 7-model comparison + cross-validation
├── 05_tune_and_evaluate.py   # Optuna tuning + SHAP + submission
├── config.py                 # Central configuration
├── visualize.py              # Publication-quality figures
├── run_pipeline.py           # Orchestrator (single or selective phases)
├── requirements.txt
├── tests/
│   ├── __init__.py
│   └── test_helpers.py       # 30+ unit tests
├── reports/                  # Generated EDA reports, figures, submission
├── models/                   # Serialised model artefacts
└── DATA/                     # Raw CSVs (not included)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place raw CSV files in DATA/
#    (application_train.csv, bureau.csv, previous_application.csv, …)

# 3. Run full pipeline
python run_pipeline.py

# 4. Generate figures only (if already run)
python visualize.py

# 5. Run tests
pytest tests/ -v
```

### Selective Execution

```bash
# Run specific phases
python run_pipeline.py --phase 1      # EDA only
python run_pipeline.py --phase 1 2 3  # EDA + features + preprocessing
python run_pipeline.py --phase 4 5    # Model selection + tuning
```

## Dependencies

- **Core**: pandas, numpy, scikit-learn, scipy
- **Boosting**: LightGBM (recommended), XGBoost, CatBoost
- **Tuning**: Optuna
- **Explainability**: SHAP
- **Visualisation**: matplotlib, seaborn
- **Storage**: pyarrow (Parquet), openpyxl (Excel reports)

---

*Built with Python 3.10+ • LightGBM • Optuna • SHAP • scikit-learn*
