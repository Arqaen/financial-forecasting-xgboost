"""Unit tests for model training and explainability diagnostics."""

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from models.src.explainability import train_final_model_and_explain


def test_train_final_model_and_explain(tmp_path: Path) -> None:
    """Verify final model training, output predictions, and diagnostics dictionary."""
    dates = pd.date_range("2020-01-31", periods=30, freq="M")
    df = pd.DataFrame(
        {
            "f1": np.linspace(0.0, 1.0, 30),
            "f2": np.linspace(1.0, 0.0, 30),
            "target": [0, 1] * 15,
        },
        index=dates,
    )

    model, proba, diagnostics = train_final_model_and_explain(
        df,
        features=["f1", "f2"],
        horizon=12,
        top_shap_features=["f1"],
        out_dir=tmp_path,
    )

    assert isinstance(model, XGBClassifier)
    assert 0.0 <= proba <= 1.0
    assert "last_date" in diagnostics
    assert "final_proba" in diagnostics
    assert "in_sample_auc" in diagnostics
    assert "in_sample_logloss" in diagnostics
    assert "in_sample_brier" in diagnostics
    assert "historical_percentile" in diagnostics
