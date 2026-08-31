"""Time-aware hyperparameter tuning using random search and temporal validation."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .metrics import binary_logloss


def sample_param_combo(param_dist: Dict[str, List], rng: np.random.RandomState) -> Dict:
    """Sample a single random hyperparameter configuration from distributions."""
    return {key: rng.choice(values) for key, values in param_dist.items()}


def tune_xgb_random_search_timeval(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_es: pd.DataFrame,
    y_es: pd.Series,
    X_score: pd.DataFrame,
    y_score: pd.Series,
    *,
    fixed_params: Dict,
    param_dist: Dict[str, List],
    n_iter: int = 30,
    random_state: int = 42,
) -> Tuple[Dict, float]:
    """Perform temporal random search optimizing LogLoss on an out-of-time scoring block.

    Prevents lookahead leakage by strictly ordering time blocks:
        Train (past) -> Early Stopping Val (intermediate) -> Scoring Val (most recent).

    Args:
        X_tr: Training features.
        y_tr: Training targets.
        X_es: Early-stopping evaluation features.
        y_es: Early-stopping evaluation targets.
        X_score: Out-of-time scoring features for metric evaluation.
        y_score: Out-of-time scoring targets.
        fixed_params: Base parameters dictionary.
        param_dist: Hyperparameter sampling distributions.
        n_iter: Number of random iterations.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple containing (best_sampled_parameters_dict, best_validation_logloss).
    """
    rng = np.random.RandomState(random_state)

    best_params: Dict = {}
    best_logloss = np.inf
    fixed_n_estimators = int(fixed_params.get("n_estimators", 5000))

    for _ in range(int(n_iter)):
        params = sample_param_combo(param_dist, rng)
        model_params = dict(fixed_params)
        model_params.update(params)

        model = XGBClassifier(**model_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_es, y_es)],
            verbose=False,
            early_stopping_rounds=model_params.get("early_stopping_rounds", 100),
        )

        score_proba = model.predict_proba(X_score)[:, 1]
        score_proba = np.clip(score_proba, 1e-6, 1.0 - 1e-6)
        score_ll = binary_logloss(y_score.values, score_proba)

        if score_ll < best_logloss:
            best_logloss = float(score_ll)
            best_params = params

    print(
        f"[RandomSearch] best logloss={best_logloss:.5f} "
        f"n_estimators(fijo)={fixed_n_estimators} params={best_params}"
    )
    return best_params, float(best_logloss)
