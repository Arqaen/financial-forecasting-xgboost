"""Tests for dataset files and data_loader functions."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.src.config import DATA_DIR

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
