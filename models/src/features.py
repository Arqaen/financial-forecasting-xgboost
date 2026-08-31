"""Feature engineering and target creation for financial machine learning."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from .config import DEFAULT_HORIZON, FECHA_INICIO, FECHA_OBJETIVO, MIN_HISTORY_RATIO


def add_technical_indicators(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Compute horizon-adapted Exponential Moving Averages, RSI, and Rate of Change (ROC).

    Args:
        df: DataFrame containing a 'Close' price column.
        horizon: Target forecast horizon in months (used to scale EMA windows).

    Returns:
        DataFrame enriched with technical features.
    """
    df = df.copy()
    close_col = "Close"
    h = int(horizon)

    short = max(3, h // 2)
    mid = h
    long_w = h * 2

    df[f"ema_{short}"] = df[close_col].ewm(span=short, adjust=False).mean()
    df[f"ema_{mid}"] = df[close_col].ewm(span=mid, adjust=False).mean()
    df[f"ema_{long_w}"] = df[close_col].ewm(span=long_w, adjust=False).mean()
    df[f"ema_{short}_dist"] = df[close_col] / df[f"ema_{short}"] - 1.0
    df[f"ema_{mid}_dist"] = df[close_col] / df[f"ema_{mid}"] - 1.0
    df[f"ema_{long_w}_dist"] = df[close_col] / df[f"ema_{long_w}"] - 1.0
    df["ema_spread"] = df[f"ema_{short}"] / df[f"ema_{long_w}"] - 1.0

    # RSI (14-month window)
    rsi_window = 14
    delta = df[close_col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{rsi_window}"] = 100.0 - (100.0 / (1.0 + rs))

    # Rate of Change (ROC) aligned with horizon
    df[f"roc_{h}"] = df[close_col].pct_change(h)

    return df


def apply_macro_release_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Apply realistic publication release lags to macroeconomic indicators to prevent lookahead bias.

    Args:
        df: DataFrame containing raw macroeconomic series.

    Returns:
        DataFrame with lagged macroeconomic series.
    """
    df = df.copy()

    if "GDPC1" in df.columns:
        df["GDPC1"] = df["GDPC1"].shift(3)  # Quarterly GDP release lag
    if "UNRATE" in df.columns:
        df["UNRATE"] = df["UNRATE"].shift(1)
    if "PERMIT" in df.columns:
        df["PERMIT"] = df["PERMIT"].shift(1)
    if "M2SL" in df.columns:
        df["M2SL"] = df["M2SL"].shift(1)
    if "TOTALSA" in df.columns:
        df["TOTALSA"] = df["TOTALSA"].shift(1)
    if "HOUST" in df.columns:
        df["HOUST"] = df["HOUST"].shift(1)
    if "CORESTICKM159SFRBATL" in df.columns:
        df["CORESTICKM159SFRBATL"] = df["CORESTICKM159SFRBATL"].shift(1)
    if "WALCL" in df.columns:
        df["WALCL"] = df["WALCL"].shift(1)

    return df


def compute_macro_and_valuation_features(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Calculate derived macroeconomic, monetary, liquidity, credit, and valuation features.

    Args:
        df: DataFrame containing lagged macro data and market series.
        horizon: Prediction horizon in months.

    Returns:
        DataFrame enriched with financial and macroeconomic features.
    """
    df = df.copy()
    h = int(horizon)

    if "WALCL" in df.columns:
        df["balance_yoy"] = df["WALCL"].pct_change(12)
        df["liquidity_trend"] = df["WALCL"].pct_change(6) - df["WALCL"].pct_change(12)

    if "Close" in df.columns:
        df["sp500_12m"] = df["Close"].pct_change(12)
        df["sp500_horizon"] = df["Close"].pct_change(h)
        df["drawdown_12m"] = df["Close"] / df["Close"].rolling(12).max() - 1.0
        df["momentum_12m"] = df["Close"].pct_change(12)
        df["ret_6m"] = df["Close"].pct_change(6)
        df["ret_12m"] = df["Close"].pct_change(12)
        df["momentum_change"] = df["momentum_12m"] - df["momentum_12m"].shift(6)

    if "GDPC1" in df.columns:
        df["gdp_yoy"] = df["GDPC1"].pct_change(12)
        df["gdp_yoy_lag6"] = df["gdp_yoy"].shift(6)
        df["gdp_yoy_ma6"] = df["gdp_yoy"].rolling(6).mean()
        df["gdp_yoy_diff6"] = df["gdp_yoy"] - df["gdp_yoy"].shift(6)

    if "UNRATE" in df.columns:
        df["unemp_change_12m"] = df["UNRATE"].diff(12)
        df["recession"] = (df["UNRATE"] > df["UNRATE"].rolling(24).mean()).astype(int)

    if "FEDFUNDS" in df.columns:
        df["fund_rate_change_3m"] = df["FEDFUNDS"].diff(3)

    if "VIX_Close" in df.columns:
        df["vix_level"] = df["VIX_Close"]
        df["vix_3m_change"] = df["VIX_Close"].pct_change(3)
        vix_mean_12 = df["VIX_Close"].rolling(12).mean()
        vix_std_12 = df["VIX_Close"].rolling(12).std()
        df["vix_z_score"] = (df["VIX_Close"] - vix_mean_12) / vix_std_12
        df["vol_regime"] = df["VIX_Close"] / vix_mean_12

    if "M2SL" in df.columns:
        df["m2_yoy"] = df["M2SL"].pct_change(12)
        if "gdp_yoy" in df.columns:
            df["liquidity_impulse"] = df["m2_yoy"] - df["gdp_yoy"]
            df["liquidity_impulse_lag6"] = df["liquidity_impulse"].shift(6)

    if "PERMIT" in df.columns:
        df["permit_yoy"] = df["PERMIT"].pct_change(12)

    if "T10Y3M" in df.columns:
        df["curve_slope_3m_change"] = df["T10Y3M"].diff(3)
        df["curve_change_12m"] = df["T10Y3M"].diff(12)
        if "DGS10" in df.columns:
            df["curve_slope"] = df["DGS10"] - df["T10Y3M"]

    if "T10YIE" in df.columns:
        df["inflation_expectations_3m_change"] = df["T10YIE"].diff(3)
        df["high_inflation"] = (df["T10YIE"] > 2.5).astype(int)

    if "sp500_pe_ratio" in df.columns:
        df["sp500_earnings_yield"] = 1.0 / df["sp500_pe_ratio"]
        df["earnings_growth_12m"] = df["sp500_pe_ratio"].diff(12) / df["sp500_pe_ratio"].shift(12)
        if f"roc_{h}" in df.columns:
            df["value_momentum"] = df["sp500_earnings_yield"] * df[f"roc_{h}"]
        if "DGS10" in df.columns:
            df["equity_risk_premium"] = df["sp500_earnings_yield"] - df["DGS10"]

    if "cape_data" in df.columns:
        df["cape_earnings_yield"] = 1.0 / df["cape_data"]

    if "NFCI" in df.columns:
        df["NFCI_3m_change"] = df["NFCI"].diff(3)

    if "DFII10" in df.columns:
        df["real_rate_change_6m"] = df["DFII10"].diff(6)
        if "CORESTICKM159SFRBATL" in df.columns:
            df["real_rate"] = df["DFII10"] - df["CORESTICKM159SFRBATL"]

    if "DXY_Close" in df.columns:
        df["dxy_12m"] = df["DXY_Close"].pct_change(12)
        df["dxy_3m_change"] = df["DXY_Close"].pct_change(3)

    if "BAMLH0A0HYM2" in df.columns:
        df["hy_spread_change_3m"] = df["BAMLH0A0HYM2"].diff(3)
        df["credit_impulse"] = -df["BAMLH0A0HYM2"].diff(12)
        df["credit_stress"] = df["BAMLH0A0HYM2"].diff(6)

    if "BAA" in df.columns and "AAA" in df.columns:
        df["credit_spread"] = df["BAA"] - df["AAA"]

    return df


def create_targets(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Construct forward return variables, log-return regression target, and binary directional label.

    Target definitions:
        - close_fwd: Price at t + horizon
        - future_return: Simple forward return (close_{t+h} / close_t - 1)
        - target_reg: Log-return regression target ln(close_{t+h} / close_t)
        - target: Binary classification label (1 if close_{t+h} > close_t else 0)

    Args:
        df: DataFrame with 'Close' price series.
        horizon: Forecast horizon in months.

    Returns:
        DataFrame enriched with target variables.
    """
    df = df.copy()
    h = int(horizon)

    close_fwd = df["Close"].shift(-h)
    df["close_fwd"] = close_fwd
    df["future_return"] = close_fwd / df["Close"] - 1.0
    df["target_reg"] = np.where(
        close_fwd.notna(),
        np.log(close_fwd / df["Close"]),
        np.nan,
    )
    df["target"] = np.where(
        close_fwd.notna() & df["Close"].notna(),
        (close_fwd > df["Close"]).astype(float),
        np.nan,
    )

    return df


def filter_features_by_history(
    df: pd.DataFrame,
    feature_names: List[str],
    min_history: float = MIN_HISTORY_RATIO,
) -> Tuple[List[str], List[str]]:
    """Filter out features that do not meet the minimum non-null historical coverage threshold.

    Args:
        df: DataFrame sliced to the target historical period.
        feature_names: List of candidate feature column names.
        min_history: Minimum required fraction of non-null observations.

    Returns:
        Tuple containing (valid_features, dropped_features).
    """
    valid_features = [
        f
        for f in feature_names
        if f in df.columns and float(df[f].notna().mean()) > float(min_history)
    ]
    dropped_features = [f for f in feature_names if f not in valid_features]
    return valid_features, dropped_features


def prepare_modeling_dataset(
    raw_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    horizon: int = DEFAULT_HORIZON,
    start_date: str = FECHA_INICIO,
    end_date: str = FECHA_OBJETIVO,
    min_history: float = MIN_HISTORY_RATIO,
) -> Tuple[pd.DataFrame, List[str]]:
    """End-to-end dataset preparation pipeline: features, lags, targets, date filtering, and NaN cleaning.

    Args:
        raw_df: Raw joined monthly DataFrame.
        features: Optional subset of features to retain (defaults to config.DEFAULT_FEATURES).
        horizon: Forecast horizon in months.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        min_history: Minimum valid historical ratio for features.

    Returns:
        Tuple containing (cleaned_modeling_df, validated_features_list).
    """
    from .config import DEFAULT_FEATURES

    feature_list = list(features or DEFAULT_FEATURES)

    # 1. Feature Engineering
    df = add_technical_indicators(raw_df, horizon=horizon)
    df = apply_macro_release_lags(df)
    df = compute_macro_and_valuation_features(df, horizon=horizon)
    df = create_targets(df, horizon=horizon)

    # 2. Date slicing
    df = df.loc[start_date:end_date].copy()

    # 3. Filter features by valid history
    valid_features, dropped_features = filter_features_by_history(
        df, feature_list, min_history=min_history
    )
    if dropped_features:
        print(
            f"[Features] Dropped features with insufficient history (<{min_history*100:.0f}%): {dropped_features}"
        )

    # 4. Clean NaNs in target and selected features
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    df = df.dropna(subset=valid_features)

    return df, valid_features
