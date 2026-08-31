"""Data loading and preprocessing module for market and macroeconomic time series."""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from .config import DATA_DIR


def load_series_csv(
    filename: str,
    *,
    date_col: str,
    index_name: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load a time series CSV file, parse dates, strip column names, and coerce numeric types.

    Args:
        filename: Name of the CSV file.
        date_col: Column name containing dates.
        index_name: Optional index name to set.
        drop_columns: Optional list of columns to drop.
        data_dir: Path to data directory (defaults to config.DATA_DIR).

    Returns:
        DataFrame indexed by DateTime with cleaned numeric columns.
    """
    directory = data_dir or DATA_DIR
    file_path = directory / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    df = pd.read_csv(file_path, parse_dates=[date_col], index_col=date_col)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.columns = df.columns.str.strip()

    if index_name:
        df.index.name = index_name

    if drop_columns:
        existing = [col for col in drop_columns if col in df.columns]
        if existing:
            df = df.drop(columns=existing)

    return df


def to_monthly_last(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any series (daily/monthly/quarterly) to monthly frequency.

    Uses the last observed value of each calendar month and forward-fills missing observations.

    Args:
        df: Input DataFrame with DatetimeIndex.

    Returns:
        Monthly resampled DataFrame with forward-filled values.
    """
    df = df.sort_index()
    df = df.resample("M").last()
    df = df.ffill()
    return df


def load_raw_dataset(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all 28 market and macroeconomic time series and align them to a monthly calendar.

    Args:
        data_dir: Path to the data directory containing CSV files.

    Returns:
        Consolidated monthly DataFrame indexed by month-end dates.
    """
    target_dir = data_dir or DATA_DIR

    sp500 = load_series_csv(
        "sp500.csv",
        date_col="Date",
        drop_columns=["Price", "High", "Low", "Open", "Volume"],
        data_dir=target_dir,
    )

    series_map: Dict[str, pd.DataFrame] = {
        "vix": load_series_csv(
            "vix.csv",
            date_col="Date",
            drop_columns=["Price", "High", "Low", "Open", "Volume"],
            data_dir=target_dir,
        ).rename(columns={"Close": "VIX_Close"}),
        "balance": load_series_csv("balance_fed.csv", date_col="observation_date", data_dir=target_dir),
        "corp_profit": load_series_csv("corporate_profit.csv", date_col="observation_date", data_dir=target_dir),
        "corp_spread": load_series_csv("corporate_spread.csv", date_col="observation_date", data_dir=target_dir),
        "fund_rate": load_series_csv("fund_rate.csv", date_col="observation_date", data_dir=target_dir),
        "gdp": load_series_csv("gdp.csv", date_col="observation_date", data_dir=target_dir),
        "hy_spread": load_series_csv("high_yield_spread.csv", date_col="observation_date", data_dir=target_dir),
        "unemp": load_series_csv("unemployment.csv", date_col="observation_date", data_dir=target_dir),
        "dfii10": load_series_csv("DFII10.csv", date_col="observation_date", data_dir=target_dir),
        "dgs10": load_series_csv("DGS10.csv", date_col="observation_date", data_dir=target_dir),
        "m2sl": load_series_csv("M2SL.csv", date_col="observation_date", data_dir=target_dir),
        "nfci": load_series_csv("NFCI.csv", date_col="observation_date", data_dir=target_dir),
        "permit": load_series_csv("PERMIT.csv", date_col="observation_date", data_dir=target_dir),
        "t10y3m": load_series_csv("T10Y3M.csv", date_col="observation_date", data_dir=target_dir),
        "t10yie": load_series_csv("T10YIE.csv", date_col="observation_date", data_dir=target_dir),
        "sp500_pe_ratio": load_series_csv(
            "sp-500-pe-ratio-price-to-earnings-chart.csv",
            date_col="date",
            data_dir=target_dir,
        ).rename(columns={"value": "sp500_pe_ratio"}),
        "cape_data": load_series_csv("cape_data.csv", date_col="Date", data_dir=target_dir).rename(
            columns={"CAPE": "cape_data"}
        ),
        "core_cpi": load_series_csv("CORESTICKM159SFRBATL.csv", date_col="observation_date", data_dir=target_dir),
        "dxy": load_series_csv(
            "dxy.csv",
            date_col="Date",
            drop_columns=["High", "Low", "Open", "Volume"],
            data_dir=target_dir,
        ).rename(columns={"Close": "DXY_Close"}),
        "TOTALSA": load_series_csv("TOTALSA.csv", date_col="observation_date", data_dir=target_dir),
        "HOUST": load_series_csv("HOUST.csv", date_col="observation_date", data_dir=target_dir),
        "TB3MS": load_series_csv("TB3MS.csv", date_col="observation_date", data_dir=target_dir),
        "DGS3MO": load_series_csv("DGS3MO.csv", date_col="observation_date", data_dir=target_dir),
        "T10Y2Y": load_series_csv("T10Y2Y.csv", date_col="observation_date", data_dir=target_dir),
        "USSLIND": load_series_csv("USSLIND.csv", date_col="observation_date", data_dir=target_dir),
        "BAA": load_series_csv("BAA.csv", date_col="observation_date", data_dir=target_dir),
        "AAA": load_series_csv("AAA.csv", date_col="observation_date", data_dir=target_dir),
    }

    # Convert all datasets to monthly frequency (last day of month)
    sp500 = to_monthly_last(sp500)
    for name, dataset in list(series_map.items()):
        series_map[name] = to_monthly_last(dataset)

    # Consolidate via left join onto SP500 index
    joined_df = sp500.join(
        [
            series_map["vix"],
            series_map["balance"],
            series_map["corp_profit"],
            series_map["corp_spread"],
            series_map["fund_rate"],
            series_map["gdp"],
            series_map["hy_spread"],
            series_map["unemp"],
            series_map["dfii10"],
            series_map["dgs10"],
            series_map["m2sl"],
            series_map["nfci"],
            series_map["permit"],
            series_map["t10y3m"],
            series_map["t10yie"],
            series_map["sp500_pe_ratio"],
            series_map["cape_data"],
            series_map["core_cpi"],
            series_map["dxy"],
            series_map["TOTALSA"],
            series_map["HOUST"],
            series_map["TB3MS"],
            series_map["DGS3MO"],
            series_map["T10Y2Y"],
            series_map["USSLIND"],
            series_map["BAA"],
            series_map["AAA"],
        ],
        how="left",
    )

    return joined_df
