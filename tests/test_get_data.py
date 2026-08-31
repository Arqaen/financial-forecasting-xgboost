"""Tests for get_data acquisition and path resolution."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("yfinance")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.get_data import DATA_DIR, FRED_COMMON_SERIES, MODELS_DIR


def test_get_data_paths():
    """Verify get_data paths point to models/data directory."""
    assert MODELS_DIR.exists()
    assert DATA_DIR == MODELS_DIR / "data"
    assert DATA_DIR.exists()


def test_fred_common_series_mappings():
    """Verify core FRED series mappings exist."""
    assert "M2SL" in FRED_COMMON_SERIES
    assert "UNRATE" in FRED_COMMON_SERIES
    assert "PERMIT" in FRED_COMMON_SERIES
    assert "GDPC1" in FRED_COMMON_SERIES
    assert "BAMLH0A0HYM2" in FRED_COMMON_SERIES
