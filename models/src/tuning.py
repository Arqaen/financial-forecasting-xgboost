"""Time-aware hyperparameter tuning using random search and temporal validation."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from .config import CLASS_ZERO_WEIGHT, TUNING_OBJECTIVE
from .metrics import binary_logloss


def training_sample_weights(y: pd.Series, class_zero_weight: float) -> Optional[np.ndarray]:
    """Apply the configured weight to class 0 in training only."""
    if not np.isfinite(class_zero_weight) or class_zero_weight <= 0:
        raise ValueError("class_zero_weight must be positive and finite")
    if class_zero_weight == 1.0:
        return None
    return np.where(y.to_numpy() == 0, class_zero_weight, 1.0)


def fit_temporal_classifier(
    model: XGBClassifier,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_es: pd.DataFrame,
    y_es: pd.Series,
    *,
    class_zero_weight: float = CLASS_ZERO_WEIGHT,
) -> None:
    """Fit on weighted training rows, using the observed distribution for early stopping."""
    eval_set = [(X_es, y_es)] if y_es.nunique() >= 2 else None
    if eval_set is None:
        model.set_params(early_stopping_rounds=None)
    model.fit(
        X_tr,
        y_tr,
        sample_weight=training_sample_weights(y_tr, class_zero_weight),
        eval_set=eval_set,
        verbose=False,
    )


def sample_param_combo(param_dist: Dict[str, Any], rng: np.random.RandomState) -> Dict[str, Any]:
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
    fixed_params: Dict[str, Any],
    param_dist: Dict[str, Any],
    n_iter: int = 30,
    random_state: int = 42,
    class_zero_weight: float = CLASS_ZERO_WEIGHT,
    tuning_objective: str = TUNING_OBJECTIVE,
) -> Tuple[Dict[str, Any], float]:
    """Select candidates by log loss or class-0 average precision on a later scoring block.

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
        class_zero_weight: Training weight for class 0. Class 1 keeps unit weight.
        tuning_objective: Either logloss (minimize) or AP_0 (maximize).

    Returns:
        Selected parameters and validation score. Single-class blocks use log loss.
    """
    if tuning_objective not in {"logloss", "AP_0"}:
        raise ValueError("tuning_objective must be logloss or AP_0")
    if not len(y_score) or n_iter < 1:
        raise ValueError("random search requires a scoring block and at least one candidate")
    metric = tuning_objective if y_score.nunique() >= 2 else "logloss"
    rng = np.random.RandomState(random_state)

    best_params: Dict = {}
    best_score = -np.inf if metric == "AP_0" else np.inf
    fixed_n_estimators = int(fixed_params.get("n_estimators", 5000))

    for _ in range(int(n_iter)):
        params = sample_param_combo(param_dist, rng)
        model_params = dict(fixed_params)
        model_params.update(params)

        model = XGBClassifier(**model_params)
        fit_temporal_classifier(model, X_tr, y_tr, X_es, y_es, class_zero_weight=class_zero_weight)

        score_proba = model.predict_proba(X_score)[:, 1]
        if metric == "AP_0":
            score = float(average_precision_score(y_score, 1.0 - score_proba, pos_label=0))
            improved = score > best_score
        else:
            score = binary_logloss(y_score, np.clip(score_proba, 1e-6, 1.0 - 1e-6))
            improved = score < best_score
        if improved:
            best_score = score
            best_params = params

    print(
        f"[RandomSearch] best {metric}={best_score:.5f} "
        f"n_estimators(fijo)={fixed_n_estimators} params={best_params}"
    )
    return best_params, float(best_score)
