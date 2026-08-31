"""Tests for dataset files and data_loader functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.src.config import DATA_DIR
from models.src.data_loader import load_series_csv, to_monthly_last

EXPECTED_FILES = [
    "sp500.csv",
    "vix.csv",
    "balance_fed.csv",
    "corporate_profit.csv",
    "corporate_spread.csv",
    "fund_rate.csv",
    "gdp.csv",
    "high_yield_spread.csv",
    "unemployment.csv",
    "DFII10.csv",
    "DGS10.csv",
    "M2SL.csv",
    "NFCI.csv",
    "PERMIT.csv",
    "T10Y3M.csv",
    "T10YIE.csv",
    "sp-500-pe-ratio-price-to-earnings-chart.csv",
    "cape_data.csv",
    "CORESTICKM159SFRBATL.csv",
    "dxy.csv",
    "TOTALSA.csv",
    "HOUST.csv",
    "TB3MS.csv",
    "DGS3MO.csv",
    "T10Y2Y.csv",
    "USSLIND.csv",
    "BAA.csv",
    "AAA.csv",
    "ie_data.xls",
]


def test_all_raw_data_files_exist():
    """Verify that all expected historical dataset files exist in models/data."""
    missing = []
    for fname in EXPECTED_FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            missing.append(fname)
    assert not missing, f"Missing dataset files in {DATA_DIR}: {missing}"


def test_data_files_are_non_empty():
    """Verify that data files have content (> 0 bytes)."""
    for fname in EXPECTED_FILES:
        fpath = DATA_DIR / fname
        assert fpath.stat().st_size > 0, f"File {fname} is empty"


def test_load_series_csv_synthetic(tmp_path: Path):
    """Verify CSV loading, date parsing, column dropping, and numeric conversion."""
    csv_file = tmp_path / "test_series.csv"
    csv_file.write_text(
        "Date, Close , High , Volume \n"
        "2020-01-01, 100.5 , 105.0 , 1000\n"
        "2020-01-02, invalid , 106.0 , 1200\n"
        "2020-01-03, 102.3 , 107.0 , 1100\n"
    )

    df = load_series_csv(
        "test_series.csv",
        date_col="Date",
        drop_columns=["High", "Volume"],
        data_dir=tmp_path,
    )

    assert "Close" in df.columns
    assert "High" not in df.columns
    assert "Volume" not in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df["Close"].iloc[0] == 100.5
    assert np.isnan(df["Close"].iloc[1])
    assert df["Close"].iloc[2] == 102.3

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_series_csv("non_existent.csv", date_col="Date", data_dir=tmp_path)


def test_to_monthly_last():
    """Verify resampling of daily series to month-end frequency with forward-fill."""
    daily_dates = pd.date_range("2020-01-01", "2020-03-15", freq="D")
    df_daily = pd.DataFrame(
        {"val": np.arange(len(daily_dates), dtype=float)},
        index=daily_dates,
    )

    monthly_df = to_monthly_last(df_daily)

    assert len(monthly_df) == 3
    # Check month-end dates (Jan 31, Feb 29 2020, Mar 31)
    assert monthly_df.index[0] == pd.Timestamp("2020-01-31")
    assert monthly_df.index[1] == pd.Timestamp("2020-02-29")
    assert monthly_df.index[2] == pd.Timestamp("2020-03-31")
    # Jan 31 was day 30 (0-indexed)
    assert monthly_df["val"].iloc[0] == 30.0
