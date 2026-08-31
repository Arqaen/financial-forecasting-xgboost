"""Unit tests for time-aware XGBoost hyperparameter tuning."""

import numpy as np
import pandas as pd

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
