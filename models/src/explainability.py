from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    shap = None
    _HAS_SHAP = False

from .config import (
    DEFAULT_HORIZON,
    DO_RANDOM_SEARCH,
    FINAL_FIXED_PARAMS,
    MANUAL_PARAMS_BASE,
    METRICS_DIR,
    TOP_SHAP_FEATURES,
)
from .metrics import binary_logloss, precision_at_k


def train_final_model_and_explain(
    df: pd.DataFrame,
    features: List[str],
    *,
    horizon: int = DEFAULT_HORIZON,
    top_shap_features: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[XGBClassifier, float, Dict]:
    """Train the final XGBoost model on the complete historical dataset and generate SHAP explainability plots.

    Args:
        df: Prepared modeling DataFrame.
        features: Predictive feature columns.
        horizon: Forecast horizon in months.
        top_shap_features: Subset of features for which to produce SHAP dependence interaction plots.
        out_dir: Output directory for saving SHAP plots.

    Returns:
        Tuple containing (trained_xgb_model, latest_prediction_probability, diagnostic_metrics_dict).
    """
    output_dir = out_dir or METRICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    features_to_plot = top_shap_features or TOP_SHAP_FEATURES

    X = df[features]
    y = df["target"].astype(int)

    final_model_params = dict(FINAL_FIXED_PARAMS)
    final_model_params.update(MANUAL_PARAMS_BASE)

    final_model = XGBClassifier(**final_model_params)
    final_model.fit(X, y)

    last_X = X.iloc[[-1]]
    final_proba = float(final_model.predict_proba(last_X)[:, 1][0])
    last_date = X.index[-1]

    # In-sample Diagnostics
    final_proba_all = final_model.predict_proba(X)[:, 1]
    final_logloss = float(binary_logloss(y.values, final_proba_all))
    final_brier = float(brier_score_loss(y, final_proba_all))
    final_precision_top20 = precision_at_k(y, final_proba_all, top_frac=0.2)
    final_auc = float(roc_auc_score(y, final_proba_all)) if len(np.unique(y)) > 1 else float("nan")

    hist_mean = float(np.nanmean(final_proba_all))
    hist_std = float(np.nanstd(final_proba_all, ddof=0))
    final_proba_percentile = float(np.mean(final_proba_all <= final_proba) * 100.0)
    final_proba_z = float((final_proba - hist_mean) / hist_std) if hist_std > 0 else float("nan")

    diagnostics = {
        "last_date": last_date,
        "final_proba": final_proba,
        "in_sample_auc": final_auc,
        "in_sample_logloss": final_logloss,
        "in_sample_brier": final_brier,
        "precision_top20": final_precision_top20,
        "historical_percentile": final_proba_percentile,
        "historical_z_score": final_proba_z,
    }

    print("\n" + "=" * 60)
    print("FINAL MODEL & IN-SAMPLE DIAGNOSTICS")
    print("=" * 60)
    print(f"Última fecha: {last_date.date() if hasattr(last_date, 'date') else last_date}")
    print(f"P(sube) en {horizon} meses: {final_proba:.4f}")
    print(f"Percentil histórico de P(sube): {final_proba_percentile:.1f}% (z-score: {final_proba_z:.2f})")
    print(f"In-sample ROC-AUC: {final_auc:.4f} | LogLoss: {final_logloss:.4f} | Brier: {final_brier:.4f}")

    # ==========================================
    # SHAP Tree Explanations
    # ==========================================
    if not _HAS_SHAP:
        print("[SHAP] Warning: shap library is not installed in the active environment. Skipping SHAP plots.")
        return final_model, final_proba, diagnostics

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_pos = shap_values[1]
        expected_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
    else:
        shap_values_pos = shap_values
        expected_value = explainer.expected_value

    # 1. SHAP Beeswarm Summary Plot
    plt.figure()
    shap.summary_plot(shap_values_pos, X, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_cls.png", dpi=160)
    plt.close()

    # 2. SHAP Global Feature Importance Bar Chart
    plt.figure()
    shap.summary_plot(shap_values_pos, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance_bar_cls.png", dpi=160)
    plt.close()

    # 3. SHAP Dependence Interaction Plots for Top Features
    for feature in features_to_plot:
        if feature not in X.columns:
            continue
        plt.figure(figsize=(6, 4))
        shap.dependence_plot(
            feature,
            shap_values_pos,
            X,
            interaction_index="auto",
            show=False,
        )
        fname = f"shap_dependence_cls_{feature}.png"
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=120)
        plt.close()

    # 4. SHAP Waterfall Decomposition for Latest Observation
    shap_values_last = explainer.shap_values(last_X)
    if isinstance(shap_values_last, list) and len(shap_values_last) == 2:
        shap_values_last_pos = shap_values_last[1][0]
    else:
        shap_values_last_pos = shap_values_last[0]

    plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_last_pos,
            base_values=expected_value,
            data=last_X.iloc[0],
            feature_names=X.columns,
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_last_prediction_cls.png", dpi=160)
    plt.close()

    print(f"SHAP explanation figures successfully saved in: {output_dir}\n")

    return final_model, final_proba, diagnostics
