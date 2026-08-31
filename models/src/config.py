"""Configuration parameters for the financial machine learning pipeline."""

from pathlib import Path
from typing import Dict, List

# ==========================================
# Paths
# ==========================================
MODELS_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = MODELS_DIR / "data"
METRICS_DIR = MODELS_DIR / "metrics"

# ==========================================
# Time Series & Prediction Horizons
# ==========================================
DEFAULT_HORIZON: int = 36  # Target prediction horizon in months
FECHA_INICIO: str = "1900-01-31"
FECHA_OBJETIVO: str = "2035-01-31"

# Minimum fraction of valid (non-null) history required for a feature
MIN_HISTORY_RATIO: float = 0.6

# Walk-forward validation window sizes (in months)
MIN_TRAIN_SIZE: int = 240  # Minimum 20 years of initial training history
TEST_SIZE: int = 12  # 1-year test folds
VAL_SIZE_RATIO: float = 0.2  # Fraction of train fold used for internal temporal validation
SCORE_FRAC: float = 0.5  # Split between early stopping and threshold score validation

# Final roll-out evaluation window (last 10 years)
FINAL_ROLLOUT_MONTHS: int = 120
FINAL_GAP_MONTHS: int = 36

# ==========================================
# Strategy Simulation Parameters
# ==========================================
MONTHLY_AMOUNT: float = 1.0
SIGNAL_MULTIPLIER: float = 2.0

# ==========================================
# Model Hyperparameters (XGBoost)
# ==========================================
DO_RANDOM_SEARCH: bool = False
TUNE_EACH_FOLD: bool = False
RANDOM_SEARCH_N_ITER: int = 180
RANDOM_SEARCH_SEED: int = 42

PARAM_DIST: Dict[str, List] = {
    "learning_rate": [0.03, 0.05, 0.07],
    "max_depth": [4, 5],
    "min_child_weight": [5, 6],
    "gamma": [0.5, 1.0],
    "reg_lambda": [8, 10, 12],
    "reg_alpha": [0.05, 0.1, 0.2],
}

FIXED_PARAMS_BASE: Dict = {
    "objective": "binary:logistic",
    "n_estimators": 5000,
    "random_state": 42,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "early_stopping_rounds": 100,
}

MANUAL_PARAMS_BASE: Dict = {
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 5,
    "gamma": 1.0,
    "reg_lambda": 9,
    "reg_alpha": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
}

FINAL_FIXED_PARAMS: Dict = {
    "objective": "binary:logistic",
    "n_estimators": 5000,
    "random_state": 42,
    "tree_method": "hist",
    "eval_metric": "auc",
    "subsample": 0.9,
    "colsample_bytree": 0.8,
}

# ==========================================
# Feature Definitions
# ==========================================
DEFAULT_FEATURES: List[str] = [
    "equity_risk_premium",
    "credit_spread",
    "unemp_change_12m",
    "m2_yoy",
    "permit_yoy",
]

ALL_CANDIDATE_FEATURES: List[str] = [
    "balance_yoy",
    "unemp_change_12m",
    "fund_rate_change_3m",
    "BAMLC0A0CM",
    "BAMLH0A0HYM2",
    "vix_level",
    "vix_3m_change",
    "m2_yoy",
    "NFCI",
    "permit_yoy",
    "DFII10",
    "T10Y3M",
    "curve_slope_3m_change",
    "T10YIE",
    "inflation_expectations_3m_change",
    "sp500_earnings_yield",
    "cape_earnings_yield",
    "liquidity_impulse",
    "curve_change_12m",
    "CORESTICKM159SFRBATL",
    "equity_risk_premium",
    "NFCI_3m_change",
    "real_rate_change_6m",
    "dxy_12m",
    "vix_z_score",
    "earnings_growth_12m",
    "hy_spread_change_3m",
    "credit_impulse",
    "real_rate",
    "HOUST",
    "TOTALSA",
    "T10Y2Y",
    "curve_slope",
    "USSLIND",
    "credit_spread",
    "vol_regime",
    "credit_stress",
    "liquidity_trend",
    "high_inflation",
]

TOP_SHAP_FEATURES: List[str] = [
    "m2_yoy",
    "permit_yoy",
    "equity_risk_premium",
    "sp500_earnings_yield",
    "HOUST",
    "unemp_change_12m",
]
