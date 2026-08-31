"""Unit and Property-Based tests for purged walk-forward cross-validation, embargo, and fold geometry."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from models.src.walk_forward import (
    generate_folds,
    run_walk_forward_evaluation,
    split_train_val_internal,
)

# ==============================================================================
# Hypothesis Strategies
# ==============================================================================


@st.composite
def valid_wf_params(draw) -> Tuple[int, int, int, int, int]:
    """Generate structurally valid parameter combinations for walk-forward validation."""
    purge = draw(st.integers(min_value=0, max_value=24))
    min_train = draw(st.integers(min_value=purge + 1, max_value=120))
    test_size = draw(st.integers(min_value=1, max_value=24))
    embargo = draw(st.integers(min_value=0, max_value=12))
    n = draw(st.integers(min_value=min_train + test_size + 1, max_value=600))
    return n, min_train, test_size, purge, embargo


# ==============================================================================
# Property-Based Invariant Tests (Hypothesis)
# ==============================================================================


@given(params=valid_wf_params())
@settings(max_examples=250)
def test_property_zero_lookahead_and_purge_gap(params: Tuple[int, int, int, int, int]) -> None:
    """Invariant: Train and Test indices are strictly disjoint, separated by >= purge gap."""
    n, min_train, test_size, purge, embargo = params
    folds = generate_folds(
        n=n, min_train=min_train, test_size=test_size, purge=purge, embargo=embargo
    )
    assert len(folds) >= 1

    for train_slice, test_slice in folds:
        train_indices = set(range(train_slice.start, train_slice.stop))
        test_indices = set(range(test_slice.start, test_slice.stop))

        # 1. Zero lookahead: No test observation is in train
        assert train_indices.isdisjoint(
            test_indices
        ), f"Train {train_slice} and Test {test_slice} overlap!"

        # 2. Train strictly precedes test
        assert train_slice.stop <= test_slice.start

        # 3. Purging gap is strictly enforced: test_start - train_end == purge
        purging_gap = test_slice.start - train_slice.stop
        assert (
            purging_gap == purge
        ), f"Expected purge gap of {purge}, got {purging_gap} for train {train_slice}, test {test_slice}"

        # 4. Gap indices are disjoint from both train and test
        gap_indices = set(range(train_slice.stop, test_slice.start))
        assert len(gap_indices) == purge
        assert gap_indices.isdisjoint(train_indices)
        assert gap_indices.isdisjoint(test_indices)


@given(params=valid_wf_params())
@settings(max_examples=250)
def test_property_fold_bounds_and_test_sizes(params: Tuple[int, int, int, int, int]) -> None:
    """Invariant: All fold slices remain within [0, n] and test window length matches test_size."""
    n, min_train, test_size, purge, embargo = params
    folds = generate_folds(
        n=n, min_train=min_train, test_size=test_size, purge=purge, embargo=embargo
    )

    for train_slice, test_slice in folds:
        # Bounds check
        assert 0 <= train_slice.start < train_slice.stop <= n
        assert 0 < test_slice.start < test_slice.stop <= n

        # Test window size exactness
        assert (test_slice.stop - test_slice.start) == test_size


@given(params=valid_wf_params())
@settings(max_examples=250)
def test_property_temporal_monotonicity_and_expansion(
    params: Tuple[int, int, int, int, int]
) -> None:
    """Invariant: Test windows advance forward in time, and train windows expand monotonically."""
    n, min_train, test_size, purge, embargo = params
    folds = generate_folds(
        n=n, min_train=min_train, test_size=test_size, purge=purge, embargo=embargo
    )

    for k in range(len(folds) - 1):
        curr_train, curr_test = folds[k]
        next_train, next_test = folds[k + 1]

        # Monotonic expansion of training set (anchored at 0)
        assert curr_train.start == next_train.start == 0
        assert next_train.stop > curr_train.stop

        # Test advancement with embargo spacing
        assert next_test.start == curr_test.stop + embargo
        assert next_test.start > curr_test.start
        assert next_test.stop > curr_test.stop


@given(
    n=st.integers(min_value=-50, max_value=300),
    min_train=st.integers(min_value=-50, max_value=300),
    test_size=st.integers(min_value=-50, max_value=300),
    purge=st.integers(min_value=-50, max_value=300),
    embargo=st.integers(min_value=-50, max_value=300),
)
@settings(max_examples=300)
def test_property_robustness_on_arbitrary_inputs(
    n: int, min_train: int, test_size: int, purge: int, embargo: int
) -> None:
    """Invariant: generate_folds never throws exceptions on arbitrary/negative inputs."""
    folds = generate_folds(
        n=n, min_train=min_train, test_size=test_size, purge=purge, embargo=embargo
    )
    assert isinstance(folds, list)

    if min_train <= purge or n <= 0 or min_train <= 0 or test_size <= 0 or purge < 0 or embargo < 0:
        assert len(folds) == 0
    else:
        for tr, te in folds:
            assert isinstance(tr, slice)
            assert isinstance(te, slice)
            assert 0 <= tr.start <= tr.stop <= n
            assert 0 <= te.start <= te.stop <= n


@given(
    train_len=st.integers(min_value=1, max_value=500),
    horizon=st.integers(min_value=0, max_value=50),
    val_ratio=st.floats(min_value=0.05, max_value=0.5),
    score_frac=st.floats(min_value=0.1, max_value=0.9),
)
@settings(max_examples=250)
def test_property_internal_validation_split_invariants(
    train_len: int, horizon: int, val_ratio: float, score_frac: float
) -> None:
    """Invariant: Internal validation splitting guarantees zero lookahead between tr and val."""
    tr_slice, es_slice, score_slice = split_train_val_internal(
        train_len=train_len,
        horizon=horizon,
        val_ratio=val_ratio,
        score_frac=score_frac,
    )

    # All slices inside [0, train_len]
    for s in (tr_slice, es_slice, score_slice):
        assert 0 <= s.start <= s.stop <= train_len

    val_size = int(train_len * val_ratio)
    tr_end = train_len - (val_size + int(horizon))

    if tr_end > 0 and val_size > 0:
        # Internal train and validation sets are disjoint
        tr_set = set(range(tr_slice.start, tr_slice.stop))
        val_start = train_len - val_size
        val_set = set(range(val_start, train_len))
        assert tr_set.isdisjoint(val_set)

        # Internal gap is >= horizon
        internal_gap = val_start - tr_slice.stop
        assert internal_gap == int(horizon)

        # Early stopping and scoring slices stay within validation range
        if es_slice.stop > es_slice.start:
            assert set(range(es_slice.start, es_slice.stop)).issubset(val_set)
        if score_slice.stop > score_slice.start:
            assert set(range(score_slice.start, score_slice.stop)).issubset(val_set)


# ==============================================================================
# End-to-End Integration Tests
# ==============================================================================


def test_run_walk_forward_evaluation_execution(tmp_path: Path) -> None:
    """Verify full walk-forward execution on a small synthetic dataset."""
    dates = pd.date_range("2015-01-31", periods=50, freq="ME")
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
