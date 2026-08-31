"""Statistical, probabilistic calibration, and financial risk metrics module."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def binary_logloss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculate binary cross-entropy (log-loss) with numerical probability clamping.

    Args:
        y_true: Array of binary truth labels (0 or 1).
        y_proba: Array of predicted probabilities for class 1.

    Returns:
        Scalar binary log-loss value.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_proba, dtype=float)
    y_p = np.clip(y_p, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p)))


def best_threshold_by_f1(
    y_true: pd.Series,
    y_proba: np.ndarray,
    *,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Find the optimal classification decision threshold that maximizes F1 score.

    Args:
        y_true: Ground truth series or array.
        y_proba: Model predicted probabilities.
        thresholds: Array of candidate thresholds to test (defaults to 50 values from 0.1 to 0.9).

    Returns:
        Tuple containing (best_threshold, best_f1_score).
    """
    y_arr = np.asarray(y_true, dtype=int)
    if len(np.unique(y_arr)) < 2:
        return 0.5, float("nan")

    proba = np.asarray(y_proba, dtype=float)
    proba = np.clip(proba, 0.0, 1.0)

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 50)

    best_t = 0.5
    best_f1 = -np.inf
    for t in thresholds:
        preds = (proba >= float(t)).astype(int)
        score = float(f1_score(y_arr, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return float(best_t), float(best_f1)


def precision_at_k(y_true: pd.Series, y_proba: np.ndarray, *, top_frac: float = 0.2) -> float:
    """Compute empirical precision among the top K% highest probability predictions.

    Args:
        y_true: Ground truth series or array.
        y_proba: Model predicted probabilities.
        top_frac: Top quantile fraction to evaluate (e.g. 0.2 for top 20%).

    Returns:
        Mean empirical rate of class 1 in the top fraction.
    """
    y_arr = np.asarray(y_true, dtype=int)
    proba = np.asarray(y_proba, dtype=float)
    if len(y_arr) == 0:
        return float("nan")

    k = max(1, int(float(top_frac) * len(y_arr)))
    idx = np.argsort(proba)[-k:]
    return float(y_arr[idx].mean())


def lift_at_k(y_true: pd.Series, y_proba: np.ndarray, *, top_frac: float = 0.2) -> float:
    """Compute model lift (precision at top K% / baseline positive rate).

    Args:
        y_true: Ground truth series or array.
        y_proba: Model predicted probabilities.
        top_frac: Top quantile fraction.

    Returns:
        Lift factor over random selection.
    """
    y_arr = np.asarray(y_true, dtype=int)
    if len(y_arr) == 0:
        return float("nan")

    base_rate = float(y_arr.mean())
    if base_rate <= 0.0:
        return float("nan")

    return float(precision_at_k(y_arr, y_proba, top_frac=top_frac) / base_rate)


def confusion_matrix_by_thresholds(
    y_true: pd.Series,
    y_proba: np.ndarray,
    *,
    thresholds: Tuple[float, ...] = (0.5, 0.7, 0.8),
) -> pd.DataFrame:
    """Compute detailed confusion matrix statistics across multiple classification thresholds.

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities.
        thresholds: Tuple of cut-off values.

    Returns:
        DataFrame with counts (TN, FP, FN, TP) and class recalls/precisions per threshold.
    """
    y_arr = np.asarray(y_true, dtype=int)
    proba = np.asarray(y_proba, dtype=float)
    rows = []

    for threshold in thresholds:
        y_pred = (proba >= float(threshold)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": float(precision_score(y_arr, y_pred, zero_division=0)),
                "recall_1": float(recall_score(y_arr, y_pred, pos_label=1, zero_division=0)),
                "recall_0": float(recall_score(y_arr, y_pred, pos_label=0, zero_division=0)),
            }
        )

    return pd.DataFrame(rows).set_index("threshold")


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """Calculate Expected Calibration Error (ECE) via quantile or uniform binning.

    Args:
        y_true: Array of actual binary labels.
        y_proba: Array of predicted probabilities.
        n_bins: Number of bins.
        strategy: 'quantile' (equal frequency) or 'uniform' (equal width).

    Returns:
        Scalar ECE metric.
    """
    dfp = pd.DataFrame(
        {"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_proba, dtype=float)}
    )
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return float("nan")

    dfp["p"] = np.clip(dfp["p"].astype(float), 1e-9, 1.0 - 1e-9)
    dfp["y"] = dfp["y"].astype(int)

    if str(strategy).lower() == "quantile":
        dfp["bin"] = pd.qcut(dfp["p"], int(n_bins), labels=False, duplicates="drop")
    else:
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        dfp["bin"] = pd.cut(dfp["p"], bins=edges, labels=False, include_lowest=True)

    n_total = float(len(dfp))
    if n_total <= 0:
        return float("nan")

    ece = 0.0
    for _, g in dfp.groupby("bin"):
        n_k = float(len(g))
        if n_k <= 0:
            continue
        p_k = float(g["p"].mean())
        o_k = float(g["y"].mean())
        ece += (n_k / n_total) * abs(o_k - p_k)

    return float(ece)


def brier_decomposition(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Dict[str, float]:
    """Decompose Brier Score into Reliability (calibration), Resolution, and Uncertainty.

    Formula: Brier = Reliability - Resolution + Uncertainty

    Args:
        y_true: Actual binary labels.
        y_proba: Predicted probabilities.
        n_bins: Number of bins.
        strategy: Binning strategy ('quantile' or 'uniform').

    Returns:
        Dictionary with decomposed components and effective bin counts.
    """
    dfp = pd.DataFrame(
        {"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_proba, dtype=float)}
    )
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return {
            "brier": float("nan"),
            "reliability": float("nan"),
            "resolution": float("nan"),
            "uncertainty": float("nan"),
            "brier_decomp": float("nan"),
            "n_bins_eff": float("nan"),
        }

    dfp["p"] = np.clip(dfp["p"].astype(float), 1e-9, 1.0 - 1e-9)
    dfp["y"] = dfp["y"].astype(int)

    if str(strategy).lower() == "quantile":
        dfp["bin"] = pd.qcut(dfp["p"], int(n_bins), labels=False, duplicates="drop")
    else:
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        dfp["bin"] = pd.cut(dfp["p"], bins=edges, labels=False, include_lowest=True)

    y_bar = float(dfp["y"].mean())
    uncertainty = float(y_bar * (1.0 - y_bar))

    n_total = float(len(dfp))
    reliability = 0.0
    resolution = 0.0
    n_bins_eff = 0
    for _, g in dfp.groupby("bin"):
        n_k = float(len(g))
        if n_k <= 0:
            continue
        n_bins_eff += 1
        w_k = n_k / n_total
        p_k = float(g["p"].mean())
        o_k = float(g["y"].mean())
        reliability += w_k * (p_k - o_k) ** 2
        resolution += w_k * (o_k - y_bar) ** 2

    brier = float(np.mean((dfp["y"].astype(float) - dfp["p"].astype(float)) ** 2))
    brier_decomp = float(reliability - resolution + uncertainty)
    return {
        "brier": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_decomp": brier_decomp,
        "n_bins_eff": float(n_bins_eff),
    }


def compute_calibration_deciles_table(
    df: pd.DataFrame,
    *,
    n_bins: int = 10,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> pd.DataFrame:
    """Generate a comprehensive decile calibration breakdown table.

    Args:
        df: DataFrame containing ground truth and predicted probability columns.
        n_bins: Number of quantiles/deciles.
        actual_col: Column name of actual binary labels.
        proba_col: Column name of predicted probabilities.

    Returns:
        DataFrame indexed by decile with empirical rates, mean probabilities, lift, logloss, and Brier.
    """
    dfp = df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return pd.DataFrame()

    y_true_all = dfp[actual_col].astype(int).to_numpy()
    y_proba_all = np.clip(dfp[proba_col].astype(float).to_numpy(), 1e-9, 1.0 - 1e-9)
    base_rate = float(np.mean(y_true_all)) if len(y_true_all) else np.nan

    dfp["bin"] = pd.qcut(
        dfp[proba_col].astype(float),
        int(n_bins),
        labels=False,
        duplicates="drop",
    )

    rows = []
    for b, g in dfp.groupby("bin"):
        y_true = g[actual_col].astype(int).to_numpy()
        y_proba = np.clip(g[proba_col].astype(float).to_numpy(), 1e-9, 1.0 - 1e-9)
        n = int(len(g))

        mean_proba = float(np.mean(y_proba)) if n else np.nan
        emp_rate = float(np.mean(y_true)) if n else np.nan
        ll = float(binary_logloss(y_true, y_proba)) if n else np.nan
        br = float(brier_score_loss(y_true, y_proba)) if n else np.nan
        lift = float(emp_rate / base_rate) if (n and base_rate and base_rate > 0) else np.nan

        rows.append(
            {
                "decil": int(b) + 1,
                "n": n,
                "p_min": float(np.min(y_proba)) if n else np.nan,
                "p_max": float(np.max(y_proba)) if n else np.nan,
                "mean_proba": mean_proba,
                "empirical_rate": emp_rate,
                "lift_vs_base": lift,
                "logloss": ll,
                "brier": br,
            }
        )

    out = pd.DataFrame(rows).sort_values("decil")
    if out.empty:
        return out

    return out.set_index("decil")


def correlation_report(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Calculate Pearson correlation matrix for selected columns."""
    return df[cols].corr()


def compute_spearman_rank_corr(
    df: pd.DataFrame,
    features: List[str],
    *,
    target_col: str = "target",
) -> pd.Series:
    """Compute monotonic Spearman rank correlation between individual features and the target."""
    values: Dict[str, float] = {}
    for col in features:
        if col not in df.columns or target_col not in df.columns:
            continue
        pair = df[[col, target_col]].dropna()
        if len(pair) < 3:
            values[col] = np.nan
            continue
        values[col] = float(spearmanr(pair[col], pair[target_col]).correlation)
    return pd.Series(values, dtype=float)


def max_drawdown_from_equity(equity: np.ndarray) -> float:
    """Calculate Maximum Drawdown from an equity curve array."""
    eq = np.asarray(equity, dtype=float)
    if eq.size < 2:
        return float("nan")
    eq = np.where(np.isfinite(eq), eq, np.nan)
    if np.all(np.isnan(eq)):
        return float("nan")
    eq = pd.Series(eq).ffill().bfill().to_numpy(dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak == 0, np.nan, peak) - 1.0
    return float(np.nanmin(dd))


def compute_return_risk_metrics(
    returns: np.ndarray,
    *,
    periods_per_year: float = 12.0,
    risk_free_rate_annual: float = 0.0,
) -> Dict[str, float]:
    """Compute financial risk and performance metrics: CAGR, Annualized Volatility, Sharpe Ratio, and Max Drawdown.

    Args:
        returns: Array of periodic returns.
        periods_per_year: Frequency factor (12.0 for monthly series).
        risk_free_rate_annual: Annualized benchmark risk-free rate.

    Returns:
        Dictionary containing CAGR, annual volatility, Sharpe ratio, Max Drawdown, etc.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 2:
        return {
            "n": float(n),
            "total_return": float("nan"),
            "cagr": float("nan"),
            "vol": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "mean_ret": float("nan"),
            "std_ret": float("nan"),
        }

    equity = np.cumprod(1.0 + np.nan_to_num(r, nan=0.0))
    total_return = float(equity[-1] - 1.0)

    years = float(n) / float(periods_per_year) if periods_per_year else float("nan")
    cagr = (
        float(equity[-1] ** (1.0 / years) - 1.0)
        if years and years > 0 and equity[-1] > 0
        else float("nan")
    )

    vol = float(np.std(r, ddof=1))
    ann_vol = float(vol * np.sqrt(float(periods_per_year))) if periods_per_year else float("nan")

    excess_ann = float(cagr - float(risk_free_rate_annual)) if np.isfinite(cagr) else float("nan")
    sharpe = (
        float(excess_ann / ann_vol)
        if ann_vol and ann_vol > 0 and np.isfinite(excess_ann)
        else float("nan")
    )

    mdd = max_drawdown_from_equity(equity)
    return {
        "n": float(n),
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "mean_ret": float(np.mean(r)),
        "std_ret": float(np.std(r, ddof=1)),
    }


def compute_signal_stability_metrics(signal: pd.Series) -> Dict[str, float]:
    """Analyze the switching frequency, active long percentage, and average holding duration of a signal."""
    s = signal.copy()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {
            "n": float(0),
            "change_pct": float("nan"),
            "pct_long": float("nan"),
            "avg_hold_long": float("nan"),
            "avg_hold_flat": float("nan"),
        }

    s_bin = (s.astype(float) > 0.5).astype(int)
    n = int(len(s_bin))
    if n < 2:
        return {
            "n": float(n),
            "change_pct": float("nan"),
            "pct_long": float(s_bin.mean()),
            "avg_hold_long": float("nan"),
            "avg_hold_flat": float("nan"),
        }

    change_pct = float((s_bin != s_bin.shift(1)).iloc[1:].mean())
    pct_long = float(s_bin.mean())

    run_id = (s_bin != s_bin.shift(1)).cumsum()
    run_len = s_bin.groupby(run_id).size().astype(float)
    run_val = s_bin.groupby(run_id).first().astype(int)

    avg_hold_long = float(run_len[run_val == 1].mean()) if (run_val == 1).any() else float("nan")
    avg_hold_flat = float(run_len[run_val == 0].mean()) if (run_val == 0).any() else float("nan")

    return {
        "n": float(n),
        "change_pct": change_pct,
        "pct_long": pct_long,
        "avg_hold_long": avg_hold_long,
        "avg_hold_flat": avg_hold_flat,
    }


def compute_exposure_turnover(exposure: pd.Series) -> Dict[str, float]:
    """Compute mean and median absolute monthly turnover in portfolio exposure."""
    x = exposure.copy()
    x = x.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(x) < 2:
        return {
            "n": float(len(x)),
            "mean_abs_change": float("nan"),
            "median_abs_change": float("nan"),
        }

    dx = x.diff().abs().iloc[1:]
    return {
        "n": float(len(x)),
        "mean_abs_change": float(dx.mean()),
        "median_abs_change": float(dx.median()),
    }
