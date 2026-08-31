"""Financial Market Directional Prediction Pipeline.

This script is the main entry point for the quantitative research workflow.
It executes the modular pipeline including Data ETL, Feature Engineering,
Exploratory Data Analysis, Purged Walk-Forward Temporal Cross-Validation,
10-Year Out-of-Sample Final Roll-Out, and SHAP Explainability.

For advanced stage-by-stage CLI options, refer to `run_pipeline.py`.
"""

import sys
from pathlib import Path

# Ensure package modules can be imported when executing directly
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from run_pipeline import run_pipeline
from src.config import (
    DEFAULT_FEATURES,
    DEFAULT_HORIZON,
    DO_RANDOM_SEARCH,
    FECHA_INICIO,
    FECHA_OBJETIVO,
    MIN_HISTORY_RATIO,
)


def main() -> None:
    """Execute the complete end-to-end quantitative financial research pipeline."""
    run_pipeline(
        stage="all",
        horizon=DEFAULT_HORIZON,
        start_date=FECHA_INICIO,
        end_date=FECHA_OBJETIVO,
        min_history=MIN_HISTORY_RATIO,
        features=DEFAULT_FEATURES,
        do_random_search=DO_RANDOM_SEARCH,
        clean_metrics=True,
    )


if __name__ == "__main__":
    main()
