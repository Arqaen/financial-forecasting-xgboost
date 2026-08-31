"""Unit tests for feature engineering, publication release lags, and target construction."""

import numpy as np
import pandas as pd
import pytest

from models.src.features import (
    add_technical_indicators,
    apply_macro_release_lags,
    compute_macro_and_valuation_features,
    create_targets,
    filter_features_by_history,
    prepare_modeling_dataset,
)


@pytest.fixture
def sample_monthly_dates() -> pd.DatetimeIndex:
    """Generate 40 consecutive month-end dates for testing."""
    return pd.date_range(start="2020-01-31", periods=40, freq="ME")


def test_apply_macro_release_lags_shifts_gdp_and_monthly(
    sample_monthly_dates: pd.DatetimeIndex,
) -> None:
    """Verify macro release lags: GDP +3m, monthly indicators +1m to avoid lookahead bias."""
    n = len(sample_monthly_dates)
    raw_df = pd.DataFrame(
        {
            "GDPC1": np.arange(100.0, 100.0 + n),
            "UNRATE": np.linspace(3.5, 7.5, n),
            "PERMIT": np.linspace(1000.0, 1500.0, n),
            "M2SL": np.linspace(15000.0, 21000.0, n),
            "TOTALSA": np.linspace(12.0, 18.0, n),
            "HOUST": np.linspace(1100.0, 1600.0, n),
            "CORESTICKM159SFRBATL": np.linspace(2.0, 5.0, n),
            "WALCL": np.linspace(4000.0, 8000.0, n),
        },
        index=sample_monthly_dates,
    )

    lagged_df = apply_macro_release_lags(raw_df)

    # GDP must be shifted by 3 months (first 3 rows are NaN)
    assert lagged_df["GDPC1"].iloc[:3].isna().all()
    assert lagged_df["GDPC1"].iloc[3] == raw_df["GDPC1"].iloc[0]
    assert lagged_df["GDPC1"].iloc[10] == raw_df["GDPC1"].iloc[7]

    # Monthly indicators must be shifted by 1 month (first 1 row is NaN)
    monthly_cols = [
        "UNRATE",
        "PERMIT",
        "M2SL",
        "TOTALSA",
        "HOUST",
        "CORESTICKM159SFRBATL",
        "WALCL",
    ]
    for col in monthly_cols:
        assert pd.isna(lagged_df[col].iloc[0]), f"{col} first row should be NaN"
        assert lagged_df[col].iloc[1] == raw_df[col].iloc[0]
        assert lagged_df[col].iloc[15] == raw_df[col].iloc[14]


def test_add_technical_indicators(sample_monthly_dates: pd.DatetimeIndex) -> None:
    """Verify EMA, RSI, and ROC calculations."""
    n = len(sample_monthly_dates)
    prices = pd.Series(np.linspace(100.0, 200.0, n), index=sample_monthly_dates)
    df = pd.DataFrame({"Close": prices})

    res = add_technical_indicators(df, horizon=12)

    # Horizon 12 -> short=6, mid=12, long=24
    assert "ema_6" in res.columns
    assert "ema_12" in res.columns
    assert "ema_24" in res.columns
    assert "ema_6_dist" in res.columns
    assert "ema_spread" in res.columns
    assert "rsi_14" in res.columns
    assert "roc_12" in res.columns

    # For strictly increasing prices, ROC over 12 months must be positive
    valid_roc = res["roc_12"].dropna()
    assert (valid_roc > 0).all()

    # For strictly increasing prices, RSI(14) should be 100.0 (no negative diffs)
    valid_rsi = res["rsi_14"].dropna()
    assert (valid_rsi == 100.0).all()


def test_compute_macro_and_valuation_features(sample_monthly_dates: pd.DatetimeIndex) -> None:
    """Verify calculation of derived financial and macroeconomic ratios."""
    n = len(sample_monthly_dates)
    df = pd.DataFrame(
        {
            "Close": np.linspace(100.0, 150.0, n),
            "WALCL": np.linspace(5000.0, 6000.0, n),
            "GDPC1": np.linspace(20000.0, 22000.0, n),
            "UNRATE": np.linspace(4.0, 6.0, n),
            "FEDFUNDS": np.linspace(1.0, 5.0, n),
            "VIX_Close": np.full(n, 20.0),
            "M2SL": np.linspace(15000.0, 18000.0, n),
            "PERMIT": np.linspace(1200.0, 1400.0, n),
            "T10Y3M": np.linspace(0.5, 1.5, n),
            "DGS10": np.linspace(2.5, 4.0, n),
            "T10YIE": np.full(n, 2.8),
            "sp500_pe_ratio": np.full(n, 25.0),
            "cape_data": np.full(n, 30.0),
            "NFCI": np.linspace(-0.5, 0.5, n),
            "DFII10": np.linspace(0.5, 2.0, n),
            "CORESTICKM159SFRBATL": np.full(n, 3.0),
            "DXY_Close": np.linspace(95.0, 105.0, n),
            "BAMLH0A0HYM2": np.linspace(3.0, 5.0, n),
            "BAA": np.linspace(4.5, 5.5, n),
            "AAA": np.linspace(3.5, 4.2, n),
        },
        index=sample_monthly_dates,
    )

    feat_df = compute_macro_and_valuation_features(df, horizon=12)

    # Check valuation
    assert np.isclose(feat_df["sp500_earnings_yield"].iloc[0], 1.0 / 25.0)
    assert np.isclose(feat_df["cape_earnings_yield"].iloc[0], 1.0 / 30.0)

    # Check credit spread = BAA - AAA
    expected_spread = df["BAA"] - df["AAA"]
    assert np.allclose(feat_df["credit_spread"], expected_spread)

    # Check equity risk premium = earnings_yield - DGS10
    expected_erp = (1.0 / 25.0) - df["DGS10"]
    assert np.allclose(feat_df["equity_risk_premium"], expected_erp)

    # Check high_inflation flag when T10YIE > 2.5
    assert (feat_df["high_inflation"] == 1).all()

    # Check curve slope = DGS10 - T10Y3M
    assert np.allclose(feat_df["curve_slope"], df["DGS10"] - df["T10Y3M"])


def test_create_targets_forward_returns_and_lookahead_gap(
    sample_monthly_dates: pd.DatetimeIndex,
) -> None:
    """Verify target construction: forward shift, binary classification, and end NaNs."""
    n = len(sample_monthly_dates)
    h = 6
    # Prices: 100, 110, 120, ...
    prices = pd.Series(np.arange(100.0, 100.0 + n * 10, 10), index=sample_monthly_dates)
    df = pd.DataFrame({"Close": prices})

    target_df = create_targets(df, horizon=h)

    # Check forward price shift
    assert target_df["close_fwd"].iloc[0] == prices.iloc[h]
    assert target_df["close_fwd"].iloc[10] == prices.iloc[10 + h]

    # Last h rows must have NaN in close_fwd, future_return, target_reg, and target
    assert target_df["close_fwd"].iloc[-h:].isna().all()
    assert target_df["future_return"].iloc[-h:].isna().all()
    assert target_df["target_reg"].iloc[-h:].isna().all()
    assert target_df["target"].iloc[-h:].isna().all()

    # All non-NaN targets for increasing series should be 1.0 (close_fwd > close)
    valid_targets = target_df["target"].dropna()
    assert (valid_targets == 1.0).all()

    # Check log-return regression target
    expected_log_ret = np.log(prices.iloc[h] / prices.iloc[0])
    assert np.isclose(target_df["target_reg"].iloc[0], expected_log_ret)


def test_filter_features_by_history() -> None:
    """Verify filtering of candidate features based on non-null threshold."""
    dates = pd.date_range("2020-01-31", periods=10, freq="ME")
    df = pd.DataFrame(
        {
            "good_feature_1": [1.0] * 10,
            "good_feature_2": [1.0] * 8 + [np.nan] * 2,  # 80% non-null
            "bad_feature": [1.0] * 3 + [np.nan] * 7,  # 30% non-null
        },
        index=dates,
    )

    valid, dropped = filter_features_by_history(
        df,
        ["good_feature_1", "good_feature_2", "bad_feature", "non_existent"],
        min_history=0.6,
    )

    assert set(valid) == {"good_feature_1", "good_feature_2"}
    assert set(dropped) == {"bad_feature", "non_existent"}


def test_prepare_modeling_dataset_end_to_end() -> None:
    """Verify full dataset preparation with date slicing, NaN filtering, and target validation."""
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    raw_df = pd.DataFrame(
        {
            "Close": np.linspace(100.0, 200.0, 60),
            "GDPC1": np.linspace(18000.0, 21000.0, 60),
            "UNRATE": np.linspace(4.0, 6.0, 60),
            "M2SL": np.linspace(12000.0, 16000.0, 60),
            "PERMIT": np.linspace(1000.0, 1500.0, 60),
            "DGS10": np.linspace(2.0, 4.0, 60),
            "sp500_pe_ratio": np.full(60, 20.0),
            "BAA": np.linspace(4.0, 5.0, 60),
            "AAA": np.linspace(3.0, 3.8, 60),
        },
        index=dates,
    )

    features = ["equity_risk_premium", "credit_spread", "unemp_change_12m", "m2_yoy", "permit_yoy"]

    df_model, valid_feats = prepare_modeling_dataset(
        raw_df,
        features=features,
        horizon=6,
        start_date="2015-01-31",
        end_date="2019-12-31",
        min_history=0.5,
    )

    assert not df_model.empty
    assert df_model["target"].isin([0, 1]).all()
    assert df_model["target"].dtype in (int, np.int32, np.int64)
    assert not df_model[valid_feats].isna().any().any()
    assert set(valid_feats).issubset(set(features))
