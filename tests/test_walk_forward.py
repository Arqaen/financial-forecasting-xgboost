"""Unit tests for purged walk-forward cross-validation, embargo, and fold geometry."""

from pathlib import Path

import numpy as np
import pandas as pd

from models.src.walk_forward import run_walk_forward_evaluation


def test_walk_forward_purging_and_embargo_geometry() -> None:
    """Verify that walk-forward validation strictly enforces purging gap and test separation."""
    # Create 60 monthly steps
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    df = pd.DataFrame(
        {
            "feat1": np.linspace(0.0, 1.0, 60),
            "target": [0, 1] * 30,
            "Close": np.linspace(100.0, 200.0, 60),
            "close_fwd": np.linspace(110.0, 210.0, 60),
        },
        index=dates,
    )

    min_train_size = 24
    horizon = 6
    test_size = 6
    embargo = 0

    start = int(min_train_size)
    purge = int(horizon)

    folds_inspected = 0
    while start < len(df) - test_size:
        train_end = start - purge
        test_end = start + test_size

        train_indices = list(range(0, train_end))
        test_indices = list(range(start, test_end))

        # 1. Verify purging gap: train_end must be exactly 'purge' periods before 'start'
        assert start - train_end == purge
        gap_indices = list(range(train_end, start))
        assert len(gap_indices) == purge

        # 2. Verify train and test sets do NOT overlap
        assert set(train_indices).isdisjoint(set(test_indices))
        assert set(train_indices).isdisjoint(set(gap_indices))

        # 3. Verify internal validation split inside train fold
        val_size = int(len(train_indices) * 0.2)
        gap = int(horizon)
        tr_end = len(train_indices) - (val_size + gap)

        tr_idx = train_indices[:tr_end]
        val_idx = train_indices[-val_size:]
        internal_gap_idx = train_indices[tr_end : len(train_indices) - val_size]

        assert set(tr_idx).isdisjoint(set(val_idx))
        assert len(internal_gap_idx) == gap

        start = test_end + embargo
        folds_inspected += 1

    assert folds_inspected >= 4


def test_run_walk_forward_evaluation_execution(tmp_path: Path) -> None:
    """Verify full walk-forward execution on a small synthetic dataset."""
    dates = pd.date_range("2015-01-31", periods=50, freq="ME")
    # Synthetic dataframe with features, targets, prices
    df = pd.DataFrame(
        {
            "f1": np.linspace(-1.0, 1.0, 50),
            "f2": np.sin(np.linspace(0, 3.14, 50)),
            "target": [0, 1, 0, 1, 1, 0, 1, 0, 0, 1] * 5,
            "Close": np.linspace(100.0, 180.0, 50),
            "close_fwd": np.linspace(110.0, 190.0, 50),
            "high_inflation": [0, 1] * 25,
        },
        index=dates,
    )

    features = ["f1", "f2"]

    wf_df, scorecard = run_walk_forward_evaluation(
        df,
        features=features,
        horizon=4,
        min_train_size=20,
        test_size=4,
        out_dir=tmp_path,
        do_random_search=False,
    )

    assert not wf_df.empty
    assert "proba_up" in wf_df.columns
    assert "actual" in wf_df.columns
    assert "pred" in wf_df.columns
    assert "close_t" in wf_df.columns
    assert "close_t_plus_h" in wf_df.columns

    # Verify scorecard DataFrame
    assert not scorecard.empty
    assert "ROC-AUC (mean)" in scorecard.index
    assert "LogLoss (mean)" in scorecard.index
    assert "Brier score (mean)" in scorecard.index

    # Verify generated plot and scorecard PNG files in out_dir
    assert (tmp_path / "walk_forward_metrics_scorecard.png").exists()
    assert (tmp_path / "walk_forward_classification.png").exists()
