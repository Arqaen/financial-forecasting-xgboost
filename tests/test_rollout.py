"""Unit tests for final out-of-sample rollout validation and regime evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd

from models.src.rollout import run_final_rollout


def test_run_final_rollout_insufficient_data(tmp_path: Path) -> None:
    """Verify that rollout safely returns None when historical length is insufficient."""
    dates = pd.date_range("2020-01-31", periods=10, freq="ME")
    df = pd.DataFrame(
        {
            "f1": np.zeros(10),
            "target": [0, 1] * 5,
            "Close": np.ones(10),
            "close_fwd": np.ones(10),
        },
        index=dates,
    )

    result = run_final_rollout(
        df,
        features=["f1"],
        horizon=6,
        rollout_months=6,
        gap_months=3,
        min_train_size=100,
        out_dir=tmp_path,
    )
    assert result is None


def test_run_final_rollout_execution(tmp_path: Path) -> None:
    """Verify final rollout execution on a synthetic dataset with train-rollout gap."""
    dates = pd.date_range("2010-01-31", periods=100, freq="ME")
    df = pd.DataFrame(
        {
            "f1": np.linspace(-1.0, 1.0, 100),
            "f2": np.cos(np.linspace(0, 6.28, 100)),
            "target": [0, 1] * 50,
            "Close": np.linspace(100.0, 200.0, 100),
            "close_fwd": np.linspace(105.0, 205.0, 100),
            "high_inflation": [0, 1] * 50,
        },
        index=dates,
    )

    roll_df = run_final_rollout(
        df,
        features=["f1", "f2"],
        horizon=4,
        rollout_months=20,
        gap_months=4,
        min_train_size=20,
        out_dir=tmp_path,
    )

    assert roll_df is not None
    assert not roll_df.empty
    assert "proba_up" in roll_df.columns
    assert "actual" in roll_df.columns
    assert "pred" in roll_df.columns
    assert "close_t" in roll_df.columns
    assert "close_t_plus_h" in roll_df.columns

    # Verify diagnostic figures created in out_dir
    assert (tmp_path / "final_rollout_ranking_metrics.png").exists()
    assert (tmp_path / "final_rollout_classification.png").exists()
