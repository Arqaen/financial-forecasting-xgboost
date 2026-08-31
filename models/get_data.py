"""Data acquisition and preprocessing utilities for market and macroeconomic time series."""

import argparse
import logging
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

# Centralized Path Resolution
MODELS_DIR = Path(__file__).resolve().parent
DATA_DIR = MODELS_DIR / "data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FRED_COMMON_SERIES: Dict[str, str] = {
    "M2SL": "M2SL.csv",
    "UNRATE": "unemployment.csv",
    "PERMIT": "PERMIT.csv",
    "HOUST": "HOUST.csv",
    "TOTALSA": "TOTALSA.csv",
    "USSLIND": "USSLIND.csv",
    "BAA": "BAA.csv",
    "AAA": "AAA.csv",
    "DGS10": "DGS10.csv",
    "DGS3MO": "DGS3MO.csv",
    "TB3MS": "TB3MS.csv",
    "T10Y2Y": "T10Y2Y.csv",
    "T10Y3M": "T10Y3M.csv",
    "T10YIE": "T10YIE.csv",
    "DFII10": "DFII10.csv",
    "NFCI": "NFCI.csv",
    "CORESTICKM159SFRBATL": "CORESTICKM159SFRBATL.csv",
    "WALCL": "balance_fed.csv",
    "CP": "corporate_profit.csv",
    "BAA10Y": "corporate_spread.csv",
    "GDPC1": "gdp.csv",
    "BAMLH0A0HYM2": "high_yield_spread.csv",
    "FEDFUNDS": "fund_rate.csv",
}


def ensure_data_dir(data_dir: Path = DATA_DIR) -> Path:
    """Ensure the target data directory exists."""
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_shiller_ie_data(data_dir: Path = DATA_DIR) -> Path:
    """Download Robert Shiller's historical S&P / CAPE Excel file (ie_data.xls)."""
    ensure_data_dir(data_dir)
    target_path = data_dir / "ie_data.xls"
    url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    logger.info("Downloading Shiller ie_data.xls from %s...", url)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as f:
        f.write(response.read())
    logger.info("Saved Shiller data to %s", target_path)
    return target_path


def get_cape_data(data_dir: Path = DATA_DIR) -> Path:
    """Extract and format CAPE series from ie_data.xls into cape_data.csv."""
    ensure_data_dir(data_dir)
    xls_path = data_dir / "ie_data.xls"
    if not xls_path.exists():
        logger.info("ie_data.xls not found locally, downloading...")
        download_shiller_ie_data(data_dir)

    logger.info("Reading %s...", xls_path)
    df = pd.read_excel(
        xls_path,
        sheet_name="Data",
        usecols="A,M",
        skiprows=128,
        engine="xlrd",
    )
    df.columns = ["Date", "CAPE"]
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y.%m") + pd.offsets.MonthEnd(0)
    out_path = data_dir / "cape_data.csv"
    df.to_csv(out_path, index=False)
    logger.info("Successfully saved CAPE data to %s (%d rows)", out_path, len(df))
    return out_path


def get_spy_data(data_dir: Path = DATA_DIR) -> Path:
    """Download historical S&P 500 (SPY) daily price series from Yahoo Finance."""
    ensure_data_dir(data_dir)
    logger.info("Downloading SPY historical data from Yahoo Finance...")
    df = yf.download("SPY", period="max", auto_adjust=False)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if col[1] == "" or col[1] == "SPY" else f"{col[0]}_{col[1]}"
            for col in df.columns
        ]
    out_path = data_dir / "sp500.csv"
    df.to_csv(out_path, index=False)
    logger.info("Successfully saved S&P 500 data to %s (%d rows)", out_path, len(df))
    return out_path


def get_dxy_data(data_dir: Path = DATA_DIR) -> Path:
    """Download historical US Dollar Index (DX-Y.NYB) from Yahoo Finance."""
    ensure_data_dir(data_dir)
    logger.info("Downloading DXY historical data from Yahoo Finance...")
    df = yf.download("DX-Y.NYB", period="max", auto_adjust=False)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if col[1] == "" or col[1] == "DX-Y.NYB" else f"{col[0]}_{col[1]}"
            for col in df.columns
        ]
    out_path = data_dir / "dxy.csv"
    df.to_csv(out_path, index=False)
    logger.info("Successfully saved DXY data to %s (%d rows)", out_path, len(df))
    return out_path


def get_vix_data(data_dir: Path = DATA_DIR) -> Path:
    """Download historical CBOE Volatility Index (^VIX) from Yahoo Finance."""
    ensure_data_dir(data_dir)
    logger.info("Downloading ^VIX historical data from Yahoo Finance...")
    df = yf.download("^VIX", period="max", auto_adjust=False)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if col[1] == "" or col[1] == "^VIX" else f"{col[0]}_{col[1]}"
            for col in df.columns
        ]
    out_path = data_dir / "vix.csv"
    df.to_csv(out_path, index=False)
    logger.info("Successfully saved VIX data to %s (%d rows)", out_path, len(df))
    return out_path


def download_fred_series(
    series_id: str, filename: Optional[str] = None, data_dir: Path = DATA_DIR
) -> Path:
    """Download a macroeconomic time series from St. Louis Fed FRED API/CSV endpoint."""
    ensure_data_dir(data_dir)
    target_file = filename or f"{series_id}.csv"
    out_path = data_dir / target_file
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    logger.info("Downloading FRED series %s -> %s...", series_id, target_file)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response, open(out_path, "wb") as f:
        f.write(response.read())
    logger.info("Saved FRED series %s to %s", series_id, out_path)
    return out_path


def get_pmi_data(data_dir: Path = DATA_DIR) -> Optional[Path]:
    """Download ISM Manufacturing PMI data from investing.com with cloudscraper."""
    ensure_data_dir(data_dir)
    out_path = data_dir / "pmi.csv"
    try:
        import cloudscraper

        scraper = cloudscraper.create_scraper()
        url = (
            "https://endpoints.investing.com/pd-instruments/v1/calendars/economic/events/173/occurrences"
            "?domain_id=1&limit=1000"
        )
        logger.info("Fetching PMI data from investing.com endpoint...")
        response = scraper.get(url, timeout=30)
        if response.status_code != 200:
            logger.warning("Investing.com returned status %s for PMI request", response.status_code)
            return None

        data = response.json()
        rows: List[Dict] = []
        for item in data.get("occurrences", []):
            if "occurrence_time" in item and "actual" in item:
                date = datetime.fromisoformat(item["occurrence_time"].replace("Z", "+00:00"))
                rows.append({"observation_date": date.strftime("%Y-%m-%d"), "PMI": item["actual"]})

        if not rows:
            logger.warning("No occurrence rows returned from PMI endpoint")
            return None

        df = pd.DataFrame(rows)
        df = df.sort_values("observation_date")
        df.to_csv(out_path, index=False)
        logger.info("Successfully saved PMI data to %s (%d rows)", out_path, len(df))
        return out_path
    except Exception as e:
        logger.warning(
            "PMI scraping failed: %s. Existing pmi.csv will be preserved if available.", e
        )
        return out_path if out_path.exists() else None


def download_all(data_dir: Path = DATA_DIR) -> None:
    """Download/refresh all major downloadable datasets."""
    logger.info("Beginning full dataset acquisition into %s...", data_dir)
    get_spy_data(data_dir)
    get_dxy_data(data_dir)
    get_vix_data(data_dir)
    get_cape_data(data_dir)
    get_pmi_data(data_dir)
    for series_id, fname in FRED_COMMON_SERIES.items():
        try:
            download_fred_series(series_id, filename=fname, data_dir=data_dir)
        except Exception as e:
            logger.warning("Could not download FRED series %s: %s", series_id, e)
    logger.info("Data acquisition complete.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for data acquisition."""
    parser = argparse.ArgumentParser(
        description="Download and prepare macroeconomic and financial datasets for ML models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["all", "spy", "dxy", "vix", "cape", "pmi", "fred"],
        default="all",
        help="Data source to download or refresh",
    )
    parser.add_argument(
        "--fred-id",
        type=str,
        default=None,
        help="Specific FRED series ID to download (when --source fred)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR,
        help="Target data directory (defaults to models/data)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    target_dir: Path = args.out_dir.resolve()

    if args.source == "all":
        download_all(target_dir)
    elif args.source == "spy":
        get_spy_data(target_dir)
    elif args.source == "dxy":
        get_dxy_data(target_dir)
    elif args.source == "vix":
        get_vix_data(target_dir)
    elif args.source == "cape":
        get_cape_data(target_dir)
    elif args.source == "pmi":
        get_pmi_data(target_dir)
    elif args.source == "fred":
        if args.fred_id:
            download_fred_series(args.fred_id, data_dir=target_dir)
        else:
            for s_id, fn in FRED_COMMON_SERIES.items():
                download_fred_series(s_id, filename=fn, data_dir=target_dir)


if __name__ == "__main__":
    main()
