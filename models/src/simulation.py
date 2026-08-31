"""Investment strategy backtesting and simulation module (DCA, Value Averaging, ML dynamic allocation)."""

import numpy as np
import pandas as pd


def simulate_monthly_dca_roi(
    prices: pd.Series,
    contributions: pd.Series,
) -> pd.DataFrame:
    """Simulate a periodic contribution strategy (e.g. Dollar-Cost Averaging) and track ROI.

    Args:
        prices: Asset price series (indexed by Datetime).
        contributions: Dollar amount contributed at each period.

    Returns:
        DataFrame containing price, periodic contribution, cumulative invested, cumulative shares,
        portfolio value, and percentage ROI.
    """
    prices_clean = prices.astype(float)
    contrib_clean = contributions.astype(float).reindex(prices_clean.index).fillna(0.0)

    shares_bought = contrib_clean.div(prices_clean).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    shares = shares_bought.cumsum()
    invested = contrib_clean.cumsum()
    value = shares.mul(prices_clean)
    roi_pct = np.where(
        invested.values > 0,
        (value.values - invested.values) / invested.values * 100.0,
        np.nan,
    )

    return pd.DataFrame(
        {
            "price": prices_clean,
            "contribution": contrib_clean,
            "invested": invested,
            "shares": shares,
            "value": value,
            "roi_pct": roi_pct,
        },
        index=prices_clean.index,
    )


def simulate_value_averaging_modified_roi(
    prices: pd.Series,
    monthly_amount: float = 1.0,
    max_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Simulate a modified Value Averaging (VA) investment strategy with bounded contributions.

    Target portfolio value increases linearly: V_target(t) = base_amount * t.
    Contribution is bounded between [base_amount, max_multiplier * base_amount].

    Args:
        prices: Asset price series.
        monthly_amount: Base target growth per period.
        max_multiplier: Maximum allowable contribution multiple (defaults to 3.0x).

    Returns:
        DataFrame containing detailed portfolio progression, target values, and ROI.
    """
    prices_clean = prices.astype(float)
    base_amount = float(monthly_amount)
    max_amount = float(max_multiplier * base_amount)

    rows = []
    shares = 0.0
    invested = 0.0

    for t, (date, price) in enumerate(prices_clean.items(), start=1):
        price_val = float(price)
        portfolio_value_before = shares * price_val
        target_value = base_amount * float(t)
        contribution = min(max(target_value - portfolio_value_before, base_amount), max_amount)

        shares_bought = contribution / price_val if price_val > 0 else 0.0
        shares += shares_bought
        invested += contribution
        value = shares * price_val
        roi_pct = (value - invested) / invested * 100.0 if invested > 0 else np.nan

        rows.append(
            {
                "price": price_val,
                "target_value": target_value,
                "portfolio_value_before": portfolio_value_before,
                "contribution": contribution,
                "invested": invested,
                "shares": shares,
                "value": value,
                "roi_pct": roi_pct,
            }
        )

    return pd.DataFrame(rows, index=prices_clean.index)


def simulate_signal_allocation_roi(
    prices: pd.Series,
    signal: pd.Series,
    monthly_amount: float = 1.0,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """Simulate an ML signal-directed investment strategy.

    Contributes (multiplier * monthly_amount) when signal == 1, and 0 when signal == 0.

    Args:
        prices: Asset price series.
        signal: Binary or continuous signal series (1 = bullish, 0 = neutral/bearish).
        monthly_amount: Base contribution unit.
        multiplier: Multiplier factor for positive signal periods.

    Returns:
        DataFrame of strategy performance.
    """
    aligned_signal = signal.reindex(prices.index).fillna(0.0)
    contributions = pd.Series(
        float(monthly_amount)
        * float(multiplier)
        * (aligned_signal.astype(float) > 0.5).astype(float)
    )
    return simulate_monthly_dca_roi(prices, contributions)
