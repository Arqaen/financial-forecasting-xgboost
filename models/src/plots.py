"""Visualization module for financial ML diagnostics, calibration curves, scorecards, and strategy comparisons."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .metrics import binary_logloss


def save_table_figure(
    table_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    footer: Optional[str] = None,
    float_fmt: str = "{:.4f}",
) -> None:
    """Render a DataFrame as a clean, publication-ready table figure saved to disk."""
    if table_df is None or table_df.empty:
        return

    disp = table_df.copy()
    for col in disp.columns:
        if pd.api.types.is_numeric_dtype(disp[col]):
            disp[col] = disp[col].map(lambda x: "" if pd.isna(x) else float_fmt.format(float(x)))
        else:
            disp[col] = disp[col].astype(str)

    n_rows, n_cols = disp.shape
    fig_w = max(7.5, 1.25 * (n_cols + 1))
    fig_h = max(2.2, 0.45 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=disp.values,
        colLabels=[str(c) for c in disp.columns],
        rowLabels=[str(i) for i in disp.index],
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.3)

    fig.suptitle(title, y=0.97)
    if footer:
        fig.text(0.01, 0.02, footer, ha="left", va="bottom", fontsize=9)

    fig.tight_layout(rect=(0.0, 0.04 if footer else 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix_heatmap(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    out_path: Path,
    title: str,
) -> None:
    """Render a 2x2 confusion matrix heatmap with cell labels and count annotations."""
    y_arr = np.asarray(y_true, dtype=int)
    y_hat = np.asarray(y_pred, dtype=int)
    if y_arr.size == 0 or y_hat.size == 0 or y_arr.size != y_hat.size:
        return

    cm = confusion_matrix(y_arr, y_hat, labels=[0, 1])
    if cm.shape != (2, 2):
        return
    tn, fp, fn, tp = cm.ravel()

    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    vmax = float(np.max(mat)) if np.isfinite(mat).any() else 1.0

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=max(1.0, vmax))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Real 0", "Real 1"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)

    labels = np.array([["TN", "FP"], ["FN", "TP"]], dtype=object)
    for i in range(2):
        for j in range(2):
            val = int(mat[i, j])
            txt = f"{labels[i, j]}\n{val}"
            color = "white" if (vmax > 0 and mat[i, j] / vmax > 0.55) else "black"
            ax.text(
                j, i, txt, ha="center", va="center", fontsize=14, color=color, fontweight="bold"
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Conteo")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    target_col: str = "target",
    max_vars: int = 25,
) -> None:
    """Plot a Pearson correlation heatmap for selected feature columns."""
    if corr is None or corr.empty:
        return

    corr_plot = corr.copy()
    if corr_plot.shape[0] > max_vars:
        corr_plot = corr_plot.iloc[:max_vars, :max_vars]
        title = f"{title} (primeras {max_vars})"

    labels = [str(c) for c in corr_plot.columns]
    fig_w = max(8.0, 0.35 * len(labels))
    fig_h = max(7.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = corr_plot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlación")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_spearman_rank_corr_bar(
    spearman_corr: pd.Series,
    *,
    out_path: Path,
    title: str,
    top_n: Optional[int] = None,
) -> None:
    """Plot a horizontal bar chart of Spearman rank correlation coefficients."""
    if spearman_corr is None or spearman_corr.empty:
        return

    s = spearman_corr.dropna().copy()
    if s.empty:
        return

    abs_s = pd.Series(np.abs(s))
    s = pd.Series(s.reindex(abs_s.sort_values(ascending=False).index))
    if top_n is not None:
        s = pd.Series(s.head(int(top_n)))

    s = pd.Series(s.sort_values())

    fig_h = max(6.0, 0.22 * len(s) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.barh(s.index.astype(str), s.values, color="tab:blue", alpha=0.85)
    ax.axvline(0, color="grey", lw=1, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Spearman rho")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_classification_timeline(
    plot_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    year_locator: int = 2,
    date_col: str = "date",
    proba_col: str = "proba_up",
    actual_col: str = "actual",
    pred_col: str = "pred",
    price_col: str = "close_t",
    price_fwd_col: str = "close_t_plus_h",
) -> None:
    """Plot timeline showing asset price, predicted class hits/misses, and forecast probabilities."""
    dfp = plot_df.copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col).reset_index(drop=True)

    if dfp.empty:
        return

    y_true = dfp[actual_col].astype(int).to_numpy()
    y_proba = (
        np.clip(dfp[proba_col].astype(float).to_numpy(), 0.0, 1.0)
        if proba_col in dfp.columns
        else None
    )
    if pred_col in dfp.columns:
        y_pred = dfp[pred_col].astype(int).to_numpy()
    elif y_proba is not None:
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_pred = np.zeros_like(y_true)
    hit_mask = y_pred == y_true

    auc = float("nan")
    if y_proba is not None and len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_proba))
    ll = float("nan")
    br = float("nan")
    if y_proba is not None:
        ll = binary_logloss(y_true, y_proba)
        br = float(brier_score_loss(y_true, y_proba))

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.4]},
    )

    ax0.plot(
        dfp[date_col],
        y_pred,
        label="Predicción de clase (0/1)",
        color="tab:gray",
        alpha=0.35,
        lw=1.1,
        drawstyle="steps-post",
        zorder=2,
    )
    ax0.scatter(
        dfp[date_col],
        y_pred,
        c=np.where(hit_mask, "green", "red"),
        s=18,
        alpha=0.85,
        zorder=4,
    )
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_ylabel("Clase")

    ax0b = ax0.twinx()
    if price_col in dfp.columns:
        ax0b.plot(
            dfp[date_col],
            dfp[price_col],
            label="Precio (Close t)",
            color="tab:blue",
            alpha=0.35,
            lw=1.2,
        )
    ax0b.set_ylabel("Precio (Close)")
    ax0b.set_yscale("log")

    lines0, labels0 = ax0.get_legend_handles_labels()
    lines0b, labels0b = ax0b.get_legend_handles_labels()
    ax0.legend(
        lines0 + lines0b,
        labels0 + labels0b,
        loc="upper left",
        ncol=2,
        fontsize=9,
    )

    if y_proba is not None:
        ax1.plot(
            dfp[date_col],
            y_proba,
            color="purple",
            alpha=0.50,
            lw=1.3,
            label="P(sube)",
        )
        ax1.set_ylim(0.0, 1.0)
        ax1.set_ylabel("Prob.")
        ax1.legend(loc="upper left", ncol=2, fontsize=9)

    ax1.set_xlabel("Fecha")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator(year_locator))

    ax0.grid(True, alpha=0.25)
    ax1.grid(True, alpha=0.25)

    fig.suptitle(f"{title}\nAUC={auc:.3f} | LogLoss={ll:.3f} | Brier={br:.3f}", y=0.98)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_calibration_curve_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    n_bins: int = 10,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    """Plot quantile calibration curve comparing predicted probability against empirical positive frequency."""
    dfp = wf_df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    y_true = dfp[actual_col].astype(int).to_numpy()
    y_proba = np.clip(dfp[proba_col].astype(float).to_numpy(), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    if len(np.unique(y_true)) < 2:
        ax.text(
            0.5,
            0.5,
            "Calibration curve no definida\n(solo 1 clase en datos)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        frac_pos, mean_pred = calibration_curve(
            y_true,
            y_proba,
            n_bins=int(n_bins),
            strategy="quantile",
        )
        ax.plot(mean_pred, frac_pos, marker="o", lw=1.8, label="Modelo")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1.2, label="Ideal")

    ax.set_title(title)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia real de clase=1")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_proba_hist_by_class(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
    bins: int = 25,
    kde: bool = True,
) -> None:
    """Plot probability distribution histograms and KDE overlays separated by true binary class."""
    dfp = wf_df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    y_true = dfp[actual_col].astype(int)
    y_proba = np.clip(dfp[proba_col].astype(float).to_numpy(), 0.0, 1.0)

    p0 = y_proba[(y_true == 0).to_numpy()]
    p1 = y_proba[(y_true == 1).to_numpy()]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(p0, bins=bins, density=True, alpha=0.45, color="tab:blue", label="Real=0")
    ax.hist(p1, bins=bins, density=True, alpha=0.45, color="tab:orange", label="Real=1")

    if kde:
        grid = np.linspace(0.0, 1.0, 300)
        if len(p0) > 3 and np.std(p0) > 1e-12:
            ax.plot(grid, gaussian_kde(p0)(grid), color="tab:blue", lw=1.6)
        if len(p1) > 3 and np.std(p1) > 1e-12:
            ax.plot(grid, gaussian_kde(p1)(grid), color="tab:orange", lw=1.6)

    ax.set_title(title)
    ax.set_xlabel("P(sube)")
    ax.set_ylabel("Densidad")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_roc_pr_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    """Plot dual ROC and Precision-Recall curves with benchmark reference baselines."""
    dfp = wf_df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    y_true = dfp[actual_col].astype(int).to_numpy()
    y_proba = np.clip(dfp[proba_col].astype(float).to_numpy(), 0.0, 1.0)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.suptitle(title, y=0.98)

    if len(np.unique(y_true)) < 2:
        ax_roc.text(
            0.5,
            0.5,
            "ROC no definida\n(solo 1 clase)",
            ha="center",
            va="center",
            transform=ax_roc.transAxes,
        )
        ax_pr.text(
            0.5,
            0.5,
            "PR no definida\n(solo 1 clase)",
            ha="center",
            va="center",
            transform=ax_pr.transAxes,
        )
    else:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = float(roc_auc_score(y_true, y_proba))
        ax_roc.plot(fpr, tpr, lw=1.8, label=f"ROC-AUC={roc_auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1.0)

        ax_roc.set_title("ROC")
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_xlim(0.0, 1.0)
        ax_roc.set_ylim(0.0, 1.0)
        ax_roc.grid(True, alpha=0.25)
        ax_roc.legend(loc="lower right")

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = float(average_precision_score(y_true, y_proba))
        base_rate = float(np.mean(y_true))
        ax_pr.plot(recall, precision, lw=1.8, label=f"AP={ap:.3f}")
        ax_pr.axhline(
            base_rate, linestyle="--", color="grey", lw=1.0, label=f"Base-rate={base_rate:.3f}"
        )

        ax_pr.set_title("Precision-Recall")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_xlim(0.0, 1.0)
        ax_pr.set_ylim(0.0, 1.05)
        ax_pr.grid(True, alpha=0.25)
        ax_pr.legend(loc="lower left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_metrics_by_proba_bin(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    n_bins: int = 10,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    """Plot calibration decile progression comparing mean predicted probability against empirical hit rate."""
    dfp = wf_df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    dfp["bin"] = pd.qcut(
        dfp[proba_col].astype(float),
        int(n_bins),
        labels=False,
        duplicates="drop",
    )

    rows: List[Dict[str, Any]] = []
    for b, g in dfp.groupby("bin"):
        y_true = g[actual_col].astype(int).to_numpy()
        n = int(len(g))
        mean_proba = float(np.mean(g[proba_col].astype(float))) if n > 0 else np.nan
        emp_rate = float(np.mean(y_true)) if n > 0 else np.nan
        rows.append({"bin": int(b), "n": n, "mean_proba": mean_proba, "empirical_rate": emp_rate})

    if not rows:
        return
    mdf = pd.DataFrame(rows).sort_values("bin")
    if mdf.empty or "bin" not in mdf.columns:
        return

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(mdf["bin"], mdf["empirical_rate"], marker="o", lw=1.8, label="P(real=1) por bin")
    ax.plot(mdf["bin"], mdf["mean_proba"], marker="o", lw=1.8, label="Mean P(sube) por bin")

    ax.set_title(title)
    ax.set_xlabel("Decil de P(sube) (bajo -> alto)")
    ax.set_ylabel("Probabilidad")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cumulative_gains_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    """Plot cumulative gains curve (fraction of captured positive events vs fraction of sample selected)."""
    dfp = wf_df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    dfp = dfp.sort_values(proba_col, ascending=False).reset_index(drop=True)
    y_true = dfp[actual_col].astype(int).to_numpy()
    total_pos = int(np.sum(y_true))
    n = int(len(dfp))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    if total_pos == 0:
        ax.text(
            0.5,
            0.5,
            "Cumulative gains no definido\n(no hay positivos)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        cum_pos = np.cumsum(y_true)
        x = np.arange(1, n + 1) / n
        gain = cum_pos / total_pos

        ax.plot(x, gain, lw=2.0, label="Modelo")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1.2, label="Random")

    ax.set_title(title)
    ax.set_xlabel("Fracción seleccionada (ordenado por P(sube) desc)")
    ax.set_ylabel("Fracción de positivos capturados")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rolling_logloss_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    window: int = 36,
    date_col: str = "date",
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    """Plot rolling log-loss over time to inspect model temporal stability."""
    dfp = wf_df[[date_col, actual_col, proba_col]].copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col).replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    y_true = dfp[actual_col].astype(int)
    y_proba = np.clip(dfp[proba_col].astype(float), 1e-9, 1.0 - 1e-9)
    point_ll = -(y_true * np.log(y_proba) + (1.0 - y_true) * np.log(1.0 - y_proba))
    roll = point_ll.rolling(int(window), min_periods=max(5, int(window // 3))).mean()

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(
        dfp[date_col], roll, color="tab:blue", lw=1.8, label=f"Rolling LogLoss ({int(window)}m)"
    )
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("LogLoss")
    ax.set_ylim(
        0.0, max(0.05, float(np.nanquantile(roll.dropna(), 0.98))) if roll.notna().any() else 1.0
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rolling_brier_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    window: int = 36,
    date_col: str = "date",
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    """Plot rolling Brier score over time."""
    dfp = wf_df[[date_col, actual_col, proba_col]].copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col).replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    y_true = dfp[actual_col].astype(float)
    y_proba = np.clip(dfp[proba_col].astype(float), 0.0, 1.0)
    point_bs = (y_true - y_proba) ** 2
    roll = point_bs.rolling(int(window), min_periods=max(5, int(window // 3))).mean()

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(
        dfp[date_col], roll, color="tab:orange", lw=1.8, label=f"Rolling Brier ({int(window)}m)"
    )
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Brier")
    ax.set_ylim(
        0.0, max(0.05, float(np.nanquantile(roll.dropna(), 0.98))) if roll.notna().any() else 1.0
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_regime_performance_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    regime_series: pd.Series,
    date_col: str = "date",
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> pd.DataFrame:
    """Plot comparative model performance (AUC, Brier, LogLoss) across distinct macroeconomic regimes."""
    dfp = wf_df[[date_col, actual_col, proba_col]].copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col)

    reg = regime_series.copy()
    if not isinstance(reg.index, pd.DatetimeIndex):
        reg.index = pd.to_datetime(reg.index)
    reg = reg.sort_index()

    dfp = dfp.merge(reg.rename("regime"), left_on=date_col, right_index=True, how="left")
    dfp = dfp.dropna(subset=["regime"])
    if dfp.empty:
        return pd.DataFrame()

    rows = []
    for regime_name, g in dfp.groupby("regime"):
        y_true = g[actual_col].astype(int).to_numpy()
        n = int(len(g))
        y_proba = np.clip(g[proba_col].astype(float).to_numpy(), 0.0, 1.0)
        base_rate = float(np.mean(y_true)) if n > 0 else np.nan

        ll = binary_logloss(y_true, y_proba) if n > 0 else np.nan
        br = float(brier_score_loss(y_true, y_proba)) if n > 0 else np.nan

        auc = float("nan")
        ap = float("nan")
        if n > 2 and len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_proba))
            ap = float(average_precision_score(y_true, y_proba))
        rows.append(
            {
                "regime": str(regime_name),
                "n": n,
                "base_rate": base_rate,
                "mean_proba": float(np.mean(y_proba)) if n > 0 else np.nan,
                "logloss": float(ll),
                "brier": float(br),
                "auc": float(auc) if not pd.isna(auc) else np.nan,
                "ap": float(ap) if not pd.isna(ap) else np.nan,
            }
        )

    rdf = pd.DataFrame(rows).sort_values("regime")
    if rdf.empty:
        return rdf

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(rdf))
    w = 0.22
    ax.bar(x - w, rdf["auc"], width=w, label="AUC")
    ax.bar(x, rdf["brier"], width=w, label="Brier")
    ax.bar(x + w, rdf["logloss"], width=w, label="LogLoss")
    ax.set_xticks(x)
    ax.set_xticklabels(rdf["regime"].tolist())
    ax.set_ylim(
        0.0, max(1.05, float(np.nanmax(rdf[["auc", "brier", "logloss"]].to_numpy())) * 1.05)
    )
    ax.set_title(title)
    ax.set_ylabel("Métrica")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right", ncol=3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return rdf


def plot_equity_curve_directional_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    date_col: str = "date",
    proba_col: str = "proba_up",
    close_col: str = "close_t",
    close_fwd_col: str = "close_t_plus_h",
) -> None:
    """Plot theoretical equity curve comparison: Buy&Hold vs Continuous Probability Exposure Strategy."""
    dfp = wf_df[[date_col, proba_col, close_col, close_fwd_col]].copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col).replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    fwd_ret = (
        dfp[close_fwd_col].astype(float).to_numpy() / dfp[close_col].astype(float).to_numpy() - 1.0
    )
    exposure = np.clip(dfp[proba_col].astype(float).to_numpy(), 0.0, 1.0)
    strat_ret = exposure * fwd_ret

    equity_strat = np.cumprod(1.0 + np.nan_to_num(strat_ret, nan=0.0))
    equity_bh = np.cumprod(1.0 + np.nan_to_num(fwd_ret, nan=0.0))

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.plot(
        dfp[date_col], equity_bh, lw=1.8, color="tab:blue", alpha=0.75, label="Buy&Hold (horizon)"
    )
    ax.plot(
        dfp[date_col], equity_strat, lw=2.0, color="purple", label="Estrategia (exposure=P(sube))"
    )
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Equity (multiplicador)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_roi_strategies_comparison(
    bh_curve: pd.DataFrame,
    sig_curve: pd.DataFrame,
    va_curve: Optional[pd.DataFrame] = None,
    *,
    out_path: Path,
    title: str,
    year_locator: int = 2,
    monthly_amount: float = 1.0,
    signal_multiplier: float = 2.0,
) -> None:
    """Plot cumulative percentage ROI comparison across DCA, Value Averaging, and ML Signal Allocation."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        bh_curve.index,
        bh_curve["roi_pct"],
        label=f"Buy&Hold DCA (x={monthly_amount:g}/mes)",
        color="tab:blue",
    )
    ax.plot(
        sig_curve.index,
        sig_curve["roi_pct"],
        label=f"Señal (clase 1: {signal_multiplier:g}x, clase 0: 0x)",
        color="purple",
    )
    if va_curve is not None:
        ax.plot(
            va_curve.index,
            va_curve["roi_pct"],
            label=f"Value Averaging Modified (x={monthly_amount:g}/mes, min=x, max=3x)",
            color="tab:green",
        )

    ax.axhline(0, linestyle="--", color="grey", alpha=0.6)
    ax.set_title(title)
    ax.set_ylabel("ROI (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(year_locator))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_target_distribution(
    target_series: pd.Series,
    *,
    out_path: Path,
    title: str = "Distribución del retorno futuro (target)",
) -> None:
    """Plot histogram distribution of future forward returns."""
    plt.figure(figsize=(8, 4))
    target_series.dropna().hist(bins=50)
    plt.title(title)
    plt.axvline(0, linestyle="--", color="grey")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_decile_accuracy(
    deciles: pd.Series,
    *,
    out_path: Path,
    title: str = "Tasa de acierto (clase=1) por decil de P(sube)",
) -> None:
    """Plot bar chart of empirical positive rates grouped by forecast probability decile."""
    fig, ax = plt.subplots(figsize=(8, 4))
    deciles.plot(kind="bar", ax=ax, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Decil P(sube) (bajo -> alto)")
    ax.set_ylabel("P(real=1)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
