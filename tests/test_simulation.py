"""Unit tests for investment strategy simulation (DCA, Value Averaging, and ML Signal Allocation)."""

import numpy as np
import pandas as pd

from models.src.simulation import (
    simulate_monthly_dca_roi,
    simulate_signal_allocation_roi,
    simulate_value_averaging_modified_roi,
)


def test_simulate_monthly_dca_roi_controlled_values() -> None:
    """Verify DCA calculation for shares, cumulative invested, portfolio value, and ROI %."""
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.Series([10.0, 20.0, 20.0], index=dates)
    contributions = pd.Series([100.0, 100.0, 100.0], index=dates)

    dca = simulate_monthly_dca_roi(prices, contributions)

    # Step 1: price=10, contrib=100 -> shares_bought = 10 -> total shares = 10 -> invested = 100 -> value = 100 -> ROI = 0%
    assert dca["shares"].iloc[0] == 10.0
    assert dca["invested"].iloc[0] == 100.0
    assert dca["value"].iloc[0] == 100.0
    assert dca["roi_pct"].iloc[0] == 0.0

    # Step 2: price=20, contrib=100 -> shares_bought = 5 -> total shares = 15 -> invested = 200 -> value = 300 -> ROI = 50%
    assert dca["shares"].iloc[1] == 15.0
    assert dca["invested"].iloc[1] == 200.0
    assert dca["value"].iloc[1] == 300.0
    assert dca["roi_pct"].iloc[1] == 50.0

    # Step 3: price=20, contrib=100 -> shares_bought = 5 -> total shares = 20 -> invested = 300 -> value = 400 -> ROI = 33.333%
    assert dca["shares"].iloc[2] == 20.0
    assert dca["invested"].iloc[2] == 300.0
    assert dca["value"].iloc[2] == 400.0
    assert np.isclose(dca["roi_pct"].iloc[2], 100.0 / 3.0)


def test_simulate_value_averaging_modified_roi() -> None:
    """Verify modified Value Averaging with bounded contributions [base, max_multiplier * base]."""
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    # Prices: 10, 5, 20
    prices = pd.Series([10.0, 5.0, 20.0], index=dates)

    va = simulate_value_averaging_modified_roi(prices, monthly_amount=100.0, max_multiplier=3.0)

    assert not va.empty
    assert "target_value" in va.columns
    assert "contribution" in va.columns
    assert "portfolio_value_before" in va.columns
    assert "roi_pct" in va.columns

    # Target values: t=1 -> 100, t=2 -> 200, t=3 -> 300
    assert va["target_value"].iloc[0] == 100.0
    assert va["target_value"].iloc[1] == 200.0
    assert va["target_value"].iloc[2] == 300.0

    # Contributions bounded between 100 and 300
    assert (va["contribution"] >= 100.0).all()
    assert (va["contribution"] <= 300.0).all()


def test_simulate_signal_allocation_roi() -> None:
    """Verify signal allocation allocating only on positive signal (signal > 0.5)."""
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    prices = pd.Series([10.0, 10.0, 10.0, 10.0], index=dates)
    signal = pd.Series([1, 0, 1, 0], index=dates)

    res = simulate_signal_allocation_roi(prices, signal, monthly_amount=50.0, multiplier=2.0)

    # Multiplier=2.0 * 50.0 = 100 when signal=1, 0 when signal=0
    assert res["contribution"].iloc[0] == 100.0
    assert res["contribution"].iloc[1] == 0.0
    assert res["contribution"].iloc[2] == 100.0
    assert res["contribution"].iloc[3] == 0.0
