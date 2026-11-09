"""Unit tests for time-aware XGBoost hyperparameter tuning."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from models.src.config import PARAM_DIST
from models.src.tuning import sample_param_combo, tune_xgb_random_search_timeval


def test_sample_param_combo() -> None:
    """Verify that parameter sampling selects valid keys and values from distribution."""
    rng = np.random.RandomState(42)
    sample = sample_param_combo(PARAM_DIST, rng)

    for param, values in PARAM_DIST.items():
        assert param in sample
        assert sample[param] in values


def test_tune_xgb_random_search_timeval() -> None:
    """Verify time-aware random search on synthetic temporal splits."""
    np.random.seed(42)
    # Generate train, early-stopping, and score datasets
    X_tr = pd.DataFrame(np.random.randn(30, 3), columns=["f1", "f2", "f3"])
    y_tr = pd.Series(np.random.binomial(1, 0.5, 30))

    X_es = pd.DataFrame(np.random.randn(10, 3), columns=["f1", "f2", "f3"])
    y_es = pd.Series([0, 1] * 5)

    X_score = pd.DataFrame(np.random.randn(10, 3), columns=["f1", "f2", "f3"])
    y_score = pd.Series([0, 1] * 5)

    fixed_params = {
        "objective": "binary:logistic",
        "n_estimators": 10,
        "random_state": 42,
        "tree_method": "hist",
        "eval_metric": "logloss",
    }

    param_dist = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    }

    best_params, best_ll = tune_xgb_random_search_timeval(
        X_tr,
        y_tr,
        X_es,
        y_es,
        X_score,
        y_score,
        fixed_params=fixed_params,
        param_dist=param_dist,
        n_iter=3,
        random_state=42,
    )

    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params
    assert "max_depth" in best_params
    assert np.isfinite(best_ll)
    assert best_ll > 0.0


@pytest.mark.parametrize(
    ("labels", "objective", "selected"),
    [
        ([0, 1, 0, 1], "AP_0", 2),
        ([0, 1, 0, 1], "logloss", 1),
        ([1, 1, 1, 1], "AP_0", 1),
    ],
)
def test_selection_objective_and_training_weights(
    labels: list[int], objective: str, selected: int
) -> None:
    # The second candidate ranks both negatives first but has poorly calibrated probabilities.
    X = pd.DataFrame({"x": [0, 1, 2, 3]})
    y_train = pd.Series([0, 1, 0, 1])
    first, second = Mock(), Mock()
    for model, probs in ((first, [0.2, 0.3, 0.4, 0.7]), (second, [0.01, 0.03, 0.02, 0.04])):
        model.predict_proba.return_value = np.column_stack([1 - np.array(probs), probs])
    with (
        patch("models.src.tuning.XGBClassifier", side_effect=[first, second]),
        patch(
            "models.src.tuning.sample_param_combo", side_effect=[{"max_depth": 1}, {"max_depth": 2}]
        ),
    ):
        params, score = tune_xgb_random_search_timeval(
            X,
            y_train,
            X,
            y_train,
            X,
            pd.Series(labels),
            fixed_params={},
            param_dist={},
            n_iter=2,
            class_zero_weight=3,
            tuning_objective=objective,
        )
    assert params == {"max_depth": selected}
    if selected == 2:
        assert score == 1.0
    else:
        expected = np.where(
            np.array(labels) == 1,
            first.predict_proba.return_value[:, 1],
            first.predict_proba.return_value[:, 0],
        )
        assert score == pytest.approx(-np.log(expected).mean())
    for model in (first, second):
        np.testing.assert_array_equal(model.fit.call_args.kwargs["sample_weight"], [3, 1, 3, 1])
        assert "sample_weight_eval_set" not in model.fit.call_args.kwargs
