"""Unit tests for calibration, classification, financial risk, and ranking metrics."""

import numpy as np
import pandas as pd

from models.src.metrics import (
    best_threshold_by_f1,
    binary_logloss,
    brier_decomposition,
    compute_calibration_deciles_table,
    compute_exposure_turnover,
    compute_return_risk_metrics,
    compute_signal_stability_metrics,
    compute_spearman_rank_corr,
    confusion_matrix_by_thresholds,
    correlation_report,
    expected_calibration_error,
    lift_at_k,
    max_drawdown_from_equity,
    precision_at_k,
)


def test_binary_logloss_known_values() -> None:
    """Verify binary log-loss on controlled probability outputs."""
    y_true = np.array([1, 0, 1, 0])

    # 1. Prediction with constant 0.5 -> logloss = -ln(0.5) = ln(2) ~= 0.693147
    p_half = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.isclose(binary_logloss(y_true, p_half), np.log(2.0), atol=1e-5)

    # 2. Perfect prediction -> clamped to 1 - 1e-9 -> logloss near 0
    p_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    assert np.isclose(binary_logloss(y_true, p_perfect), 0.0, atol=1e-6)

    # 3. Completely wrong prediction -> clamped to 1e-9 -> logloss ~ -ln(1e-9) ~ 20.723
    p_wrong = np.array([0.0, 1.0, 0.0, 1.0])
    assert np.isclose(binary_logloss(y_true, p_wrong), -np.log(1e-9), atol=1e-3)


def test_best_threshold_by_f1() -> None:
    """Verify threshold optimization finding maximum F1 score."""
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    # Probabilities that cleanly separate classes around 0.6
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    best_t, best_f1 = best_threshold_by_f1(y_true, y_proba)
    assert 0.3 < best_t <= 0.7
    assert np.isclose(best_f1, 1.0)

    # Single-class edge case returns default 0.5, nan
    single_class_t, single_class_f1 = best_threshold_by_f1(pd.Series([1, 1, 1]), y_proba[:3])
    assert single_class_t == 0.5
    assert np.isnan(single_class_f1)


def test_precision_and_lift_at_k() -> None:
    """Verify precision and lift calculation at top K%."""
    # 10 samples, 4 positives (base_rate = 0.4)
    y_true = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    # Probabilities highest for positive samples
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

    # Top 20% (top 2 samples): both are class 1 -> precision = 1.0
    p20 = precision_at_k(y_true, y_proba, top_frac=0.2)
    assert np.isclose(p20, 1.0)

    # Lift = precision_at_top20 / base_rate = 1.0 / 0.4 = 2.5
    l20 = lift_at_k(y_true, y_proba, top_frac=0.2)
    assert np.isclose(l20, 2.5)

    # Edge cases
    assert np.isnan(precision_at_k(pd.Series([]), np.array([])))
    assert np.isnan(lift_at_k(pd.Series([0, 0]), np.array([0.1, 0.2])))


def test_confusion_matrix_by_thresholds() -> None:
    """Verify confusion matrix counts and metrics across multiple thresholds."""
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.6, 0.6, 0.9])

    cm_df = confusion_matrix_by_thresholds(y_true, y_proba, thresholds=(0.5, 0.8))

    assert cm_df.loc[0.5, "tp"] == 2
    assert cm_df.loc[0.5, "fp"] == 1
    assert cm_df.loc[0.5, "tn"] == 1
    assert cm_df.loc[0.5, "fn"] == 0

    assert cm_df.loc[0.8, "tp"] == 1
    assert cm_df.loc[0.8, "fp"] == 0
    assert cm_df.loc[0.8, "tn"] == 2
    assert cm_df.loc[0.8, "fn"] == 1


def test_expected_calibration_error() -> None:
    """Verify ECE calculation for calibrated vs uncalibrated predictions."""
    # Perfectly calibrated probabilities
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    ece_perfect = expected_calibration_error(y_true, y_proba, n_bins=5, strategy="uniform")
    assert np.isclose(ece_perfect, 0.0, atol=1e-5)

    # Completely uncalibrated
    y_uncal = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    ece_bad = expected_calibration_error(y_uncal, y_proba, n_bins=5, strategy="uniform")
    assert np.isclose(ece_bad, 1.0, atol=1e-5)


def test_brier_decomposition_identity() -> None:
    """Verify Brier Score decomposition identity: Brier = Reliability - Resolution + Uncertainty."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.6, size=100)
    y_proba = np.clip(y_true * 0.5 + np.random.uniform(0.1, 0.4, size=100), 0.01, 0.99)

    decomp = brier_decomposition(y_true, y_proba, n_bins=5, strategy="quantile")

    assert "brier" in decomp
    assert "reliability" in decomp
    assert "resolution" in decomp
    assert "uncertainty" in decomp
    assert "brier_decomp" in decomp

    # Verify Brier = Reliability - Resolution + Uncertainty
    expected_decomp = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
    assert np.isclose(decomp["brier_decomp"], expected_decomp, atol=1e-7)


def test_compute_calibration_deciles_table() -> None:
    """Verify decile calibration breakdown table generation."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame(
        {
            "actual": np.random.binomial(1, 0.5, size=n),
            "proba_up": np.linspace(0.05, 0.95, n),
        }
    )

    table = compute_calibration_deciles_table(df, n_bins=5)
    assert not table.empty
    assert "n" in table.columns
    assert "mean_proba" in table.columns
    assert "empirical_rate" in table.columns
    assert "logloss" in table.columns
    assert "brier" in table.columns


def test_max_drawdown_from_equity() -> None:
    """Verify Maximum Drawdown computation on known equity curves."""
    # Curve: 100 -> 150 (peak) -> 120 (trough) -> 180
    # Peak = 150, Min = 120 -> Drawdown = (120 - 150) / 150 = -0.20 (-20%)
    equity = np.array([100.0, 150.0, 120.0, 180.0])
    mdd = max_drawdown_from_equity(equity)
    assert np.isclose(mdd, -0.20)

    # Strictly increasing equity has 0% drawdown
    assert np.isclose(max_drawdown_from_equity(np.array([100.0, 110.0, 120.0])), 0.0)

    # Edge case: single value
    assert np.isnan(max_drawdown_from_equity(np.array([100.0])))


def test_compute_return_risk_metrics() -> None:
    """Verify CAGR, Annual Volatility, Sharpe Ratio, and Drawdown calculations."""
    # 24 months of constant 1% monthly return
    r = np.full(24, 0.01)
    metrics = compute_return_risk_metrics(r, periods_per_year=12.0)

    # Total return = (1.01)^24 - 1
    expected_total = (1.01**24) - 1.0
    assert np.isclose(metrics["total_return"], expected_total, atol=1e-5)

    # CAGR = (1.01)^12 - 1
    expected_cagr = (1.01**12) - 1.0
    assert np.isclose(metrics["cagr"], expected_cagr, atol=1e-5)

    # Volatility with constant return is 0
    assert np.isclose(metrics["vol"], 0.0)
    assert np.isclose(metrics["ann_vol"], 0.0)

    # Drawdown with strictly positive returns is 0
    assert np.isclose(metrics["max_drawdown"], 0.0)


def test_compute_signal_stability_metrics() -> None:
    """Verify switching frequency and holding duration for binary signals."""
    # Signal: 1, 1, 0, 0, 1 (4 switches out of 4 transitions: 1->1 (no), 1->0 (yes), 0->0 (no), 0->1 (yes))
    # Changes: 2 / 4 = 50%
    signal = pd.Series([1, 1, 0, 0, 1])
    stab = compute_signal_stability_metrics(signal)

    assert stab["n"] == 5.0
    assert np.isclose(stab["pct_long"], 3.0 / 5.0)
    assert np.isclose(stab["change_pct"], 2.0 / 4.0)
    # Long runs: [1, 1] (len 2) and [1] (len 1) -> mean = 1.5
    assert np.isclose(stab["avg_hold_long"], 1.5)
    # Flat runs: [0, 0] (len 2) -> mean = 2.0
    assert np.isclose(stab["avg_hold_flat"], 2.0)


def test_compute_exposure_turnover() -> None:
    """Verify mean and median exposure turnover."""
    # Exposure: 0.2, 0.5, 0.5, 0.8
    # Absolute diffs: |0.5-0.2|=0.3, |0.5-0.5|=0.0, |0.8-0.5|=0.3 -> mean = 0.2, median = 0.3
    exposure = pd.Series([0.2, 0.5, 0.5, 0.8])
    turnover = compute_exposure_turnover(exposure)

    assert turnover["n"] == 4.0
    assert np.isclose(turnover["mean_abs_change"], 0.2)
    assert np.isclose(turnover["median_abs_change"], 0.3)


def test_correlation_and_spearman_rank() -> None:
    """Verify Pearson and Spearman rank correlation helpers."""
    df = pd.DataFrame(
        {
            "feat_pos": [1, 2, 3, 4, 5],
            "feat_neg": [5, 4, 3, 2, 1],
            "target": [2, 4, 6, 8, 10],
        }
    )

    p_corr = correlation_report(df, ["feat_pos", "target"])
    assert np.isclose(p_corr.loc["feat_pos", "target"], 1.0)

    s_corr = compute_spearman_rank_corr(df, ["feat_pos", "feat_neg"], target_col="target")
    assert np.isclose(s_corr["feat_pos"], 1.0)
    assert np.isclose(s_corr["feat_neg"], -1.0)
