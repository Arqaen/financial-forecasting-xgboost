"""Tests for pipeline configuration and path resolution."""

from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.src.config import (
    ALL_CANDIDATE_FEATURES,
    DATA_DIR,
    DEFAULT_FEATURES,
    DEFAULT_HORIZON,
    METRICS_DIR,
    MIN_HISTORY_RATIO,
    MIN_TRAIN_SIZE,
    MODELS_DIR,
    TEST_SIZE,
)


def test_paths_exist_and_resolve_correctly():
    """Verify that MODELS_DIR, DATA_DIR, and METRICS_DIR resolve to models directory."""
    assert MODELS_DIR.exists(), f"MODELS_DIR does not exist: {MODELS_DIR}"
    assert MODELS_DIR.is_dir()
    assert MODELS_DIR.name == "models"

    assert DATA_DIR == MODELS_DIR / "data"
    assert DATA_DIR.exists(), f"DATA_DIR does not exist: {DATA_DIR}"
    assert DATA_DIR.is_dir()

    assert METRICS_DIR == MODELS_DIR / "metrics"


def test_config_constants():
    """Verify core modeling parameters and feature definitions."""
    assert DEFAULT_HORIZON == 36
    assert MIN_TRAIN_SIZE == 240
    assert TEST_SIZE == 12
    assert 0.0 < MIN_HISTORY_RATIO <= 1.0

    assert isinstance(DEFAULT_FEATURES, list)
    assert len(DEFAULT_FEATURES) > 0
    assert set(DEFAULT_FEATURES).issubset(set(ALL_CANDIDATE_FEATURES))
