"""Final out-of-sample rollout evaluation module (last 10 years) with subperiod and regime stability tests."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from .config import (
    DEFAULT_HORIZON,
    DO_RANDOM_SEARCH,
    FINAL_GAP_MONTHS,
    FINAL_ROLLOUT_MONTHS,
    FIXED_PARAMS_BASE,
    MANUAL_PARAMS_BASE,
    METRICS_DIR,
    MIN_TRAIN_SIZE,
    MONTHLY_AMOUNT,
    SCORE_FRAC,
    SIGNAL_MULTIPLIER,
)
from .metrics import (
    best_threshold_by_f1,
    binary_logloss,
    brier_decomposition,
    compute_calibration_deciles_table,
    compute_exposure_turnover,
    compute_return_risk_metrics,
    compute_signal_stability_metrics,
    confusion_matrix_by_thresholds,
    expected_calibration_error,
    lift_at_k,
    precision_at_k,
)
from .plots import (
    plot_calibration_curve_wf,
    plot_classification_timeline,
    plot_metrics_by_proba_bin,
    plot_regime_performance_wf,
    plot_roi_strategies_comparison,
    plot_rolling_brier_wf,
    plot_rolling_logloss_wf,
    save_table_figure,
)
from .simulation import simulate_monthly_dca_roi


def run_final_rollout(
    df: pd.DataFrame,
    features: List[str],
    *,
    horizon: int = DEFAULT_HORIZON,
    rollout_months: int = FINAL_ROLLOUT_MONTHS,
    gap_months: int = FINAL_GAP_MONTHS,
    min_train_size: int = MIN_TRAIN_SIZE,
    out_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Evaluate model performance on an untouched out-of-sample period (e.g. final 10 years).

    Trains only on historical data prior to the rollout period (respecting the forward return gap)
    and tests stability across early/late subperiods and macroeconomic regimes.

    Args:
        df: Prepared modeling DataFrame.
        features: Predictive feature columns.
        horizon: Forecast horizon in months.
        rollout_months: Length of final out-of-sample testing window in months.
        gap_months: Temporal gap between training end and rollout start.
        min_train_size: Minimum required historical observations for training.
        out_dir: Output directory for saving figures and tables.

    Returns:
        DataFrame of rollout predictions or None if insufficient history.
    """
    output_dir = out_dir or METRICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout_end = pd.to_datetime(df.index.max())
    rollout_start = rollout_end - pd.DateOffset(months=rollout_months - 1)
    rollout_df = df.loc[rollout_start:rollout_end].copy()

    train_end_date = rollout_start - pd.DateOffset(months=gap_months)
    train_df = df.loc[:train_end_date].copy()

    if len(rollout_df) < 5 or len(train_df) < (min_train_size // 2):
        print(
            f"[FinalRollout] Insufficient historical data to train/validate "
            f"(train={len(train_df)} rollout={len(rollout_df)}). Skipping."
        )
        return None

    X_train_full = train_df[features]
    y_train_full = train_df["target"].astype(int)
    X_roll = rollout_df[features]
    y_roll = rollout_df["target"].astype(int)

    # Internal temporal validation split
    val_size = int(len(X_train_full) * 0.2)
    gap = int(horizon)
    tr_end = -(val_size + gap)

    X_tr = X_train_full.iloc[:tr_end]
    y_tr = y_train_full.iloc[:tr_end]
    X_val = X_train_full.iloc[-val_size:]
    y_val = y_train_full.iloc[-val_size:]

    if len(X_val) < 3:
        X_es = X_val
        y_es = y_val
        X_score = X_val
        y_score = y_val
    else:
        score_size = max(1, int(len(X_val) * SCORE_FRAC))
        es_size = len(X_val) - score_size
        if es_size < 1:
            es_size = 1
            score_size = len(X_val) - 1

        X_es = X_val.iloc[:es_size]
        y_es = y_val.iloc[:es_size]
        X_score = X_val.iloc[es_size:]
        y_score = y_val.iloc[es_size:]

    if len(np.unique(y_tr)) < 2:
        best_t_roll, _ = 0.5, float("nan")
        p_const = float(np.clip(y_tr.mean(), 1e-6, 1.0 - 1e-6))
        roll_proba = np.full(len(X_roll), p_const, dtype=float)
        roll_pred = (roll_proba >= best_t_roll).astype(int)
    else:
        fixed_params = dict(FIXED_PARAMS_BASE)
        manual_params = dict(MANUAL_PARAMS_BASE)

        model_params = dict(fixed_params)
        model_params.update(manual_params)

        model_roll = XGBClassifier(**model_params)
        if len(np.unique(y_es)) < 2:
            model_roll.set_params(early_stopping_rounds=None)
            model_roll.fit(X_tr, y_tr, verbose=False)
        else:
            model_roll.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)

        if len(X_score) < 1:
            best_t_roll, _ = 0.5, float("nan")
        else:
            score_proba = model_roll.predict_proba(X_score)[:, 1]
            best_t_roll, _ = best_threshold_by_f1(y_score, score_proba)

        roll_proba = model_roll.predict_proba(X_roll)[:, 1]
        roll_pred = (roll_proba >= best_t_roll).astype(int)

    # Rollout Metrics
    roll_logloss = float(binary_logloss(y_roll.values, roll_proba))
    roll_brier = float(brier_score_loss(y_roll, roll_proba))
    roll_base_rate = float(y_roll.mean())
    roll_precision_top20 = precision_at_k(y_roll, roll_proba, top_frac=0.2)
    roll_lift_top20 = lift_at_k(y_roll, roll_proba, top_frac=0.2)
    roll_recall_0 = float(recall_score(y_roll, roll_pred, pos_label=0, zero_division=0))
    roll_threshold_cm = confusion_matrix_by_thresholds(y_roll, roll_proba)

    if len(np.unique(y_roll)) > 1:
        roll_auc = float(roc_auc_score(y_roll, roll_proba))
        roll_ap = float(average_precision_score(y_roll, roll_proba))
    else:
        roll_auc = float("nan")
        roll_ap = float("nan")

    p0 = float(np.clip(y_train_full.mean(), 1e-6, 1.0 - 1e-6))
    baseline_ll = float(binary_logloss(y_roll.values, np.full_like(roll_proba, p0)))
    baseline_br = float(brier_score_loss(y_roll, np.full_like(roll_proba, p0)))

    roll_ranking_metrics = pd.DataFrame(
        {
            "Valor": [
                roll_base_rate,
                roll_precision_top20,
                roll_lift_top20,
                roll_recall_0,
            ]
        },
        index=[
            "Base rate clase 1",
            "Precision@top20%",
            "Lift@top20%",
            "Recall clase 0",
        ],
    )
    save_table_figure(
        roll_ranking_metrics,
        out_path=output_dir / "final_rollout_ranking_metrics.png",
        title=f"Final Roll-out — Métricas para señal top 20% ({horizon}m)",
    )
    save_table_figure(
        roll_threshold_cm,
        out_path=output_dir / "final_rollout_confusion_matrix_thresholds.png",
        title=f"Final Roll-out — Confusion matrix por threshold ({horizon}m)",
    )

    roll_plot_df = pd.DataFrame(
        {
            "date": pd.to_datetime(rollout_df.index),
            "proba_up": roll_proba,
            "actual": y_roll.values,
            "pred": roll_pred,
            "close_t": rollout_df["Close"].values,
            "close_t_plus_h": rollout_df["close_fwd"].values,
        }
    ).sort_values("date")

    # Risk, Turnover, and Hard Calibration
    roll_ret_df = roll_plot_df[["date", "close_t", "proba_up", "pred", "actual"]].copy()
    roll_ret_df["date"] = pd.to_datetime(roll_ret_df["date"])
    roll_ret_df = (
        roll_ret_df.sort_values("date")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["close_t", "proba_up"])
    )

    if len(roll_ret_df) >= 3:
        roll_ret_df["ret_1m"] = (
            roll_ret_df["close_t"].astype(float).shift(-1) / roll_ret_df["close_t"].astype(float)
            - 1.0
        )
        roll_ret_df = roll_ret_df.iloc[:-1].copy()

        roll_exposure_proba = np.clip(roll_ret_df["proba_up"].astype(float).to_numpy(), 0.0, 1.0)
        roll_exposure_pred = (roll_ret_df["pred"].astype(float).to_numpy() > 0.5).astype(float)
        roll_ret_1m = roll_ret_df["ret_1m"].astype(float).to_numpy()

        roll_risk_bh = compute_return_risk_metrics(roll_ret_1m, periods_per_year=12.0)
        roll_risk_proba = compute_return_risk_metrics(
            roll_exposure_proba * roll_ret_1m, periods_per_year=12.0
        )
        roll_risk_pred = compute_return_risk_metrics(
            roll_exposure_pred * roll_ret_1m, periods_per_year=12.0
        )

        roll_risk_table = pd.DataFrame(
            {
                "Buy&Hold": roll_risk_bh,
                "Estrategia (exposure=P)": roll_risk_proba,
                "Estrategia (pred 0/1)": roll_risk_pred,
            }
        ).reindex(
            [
                "n",
                "total_return",
                "cagr",
                "ann_vol",
                "sharpe",
                "max_drawdown",
                "mean_ret",
                "std_ret",
            ]
        )
        save_table_figure(
            roll_risk_table,
            out_path=output_dir / "final_rollout_risk_metrics_1m.png",
            title=f"Final Roll-out — Riesgo / Sharpe (retornos 1M, target {horizon}m)",
        )

        roll_stab = compute_signal_stability_metrics(
            roll_plot_df.set_index(pd.to_datetime(roll_plot_df["date"]))["pred"]
        )
        roll_turn = compute_exposure_turnover(
            roll_plot_df.set_index(pd.to_datetime(roll_plot_df["date"]))["proba_up"]
        )
        roll_turnover_table = pd.DataFrame(
            {
                "Valor": [
                    roll_stab.get("n", np.nan),
                    roll_stab.get("pct_long", np.nan),
                    roll_stab.get("change_pct", np.nan),
                    roll_stab.get("avg_hold_long", np.nan),
                    roll_stab.get("avg_hold_flat", np.nan),
                    roll_turn.get("mean_abs_change", np.nan),
                    roll_turn.get("median_abs_change", np.nan),
                ]
            },
            index=[
                "n (meses)",
                "% tiempo long (pred=1)",
                "% cambios de señal (pred)",
                "Duración media long (meses)",
                "Duración media flat (meses)",
                "Turnover exposure (mean |ΔP|)",
                "Turnover exposure (median |ΔP|)",
            ],
        )
        save_table_figure(
            roll_turnover_table,
            out_path=output_dir / "final_rollout_turnover_signal_stability.png",
            title=f"Final Roll-out — Turnover / Estabilidad de señal ({horizon}m)",
        )

        roll_y_true = roll_plot_df["actual"].astype(int).to_numpy()
        roll_y_proba = np.clip(roll_plot_df["proba_up"].astype(float).to_numpy(), 1e-9, 1.0 - 1e-9)
        roll_base = float(np.mean(roll_y_true)) if len(roll_y_true) else float("nan")
        roll_ece = expected_calibration_error(
            roll_y_true, roll_y_proba, n_bins=10, strategy="quantile"
        )
        roll_bd = brier_decomposition(roll_y_true, roll_y_proba, n_bins=10, strategy="quantile")

        roll_y_proba_base = np.full_like(roll_y_proba, float(np.clip(roll_base, 1e-9, 1.0 - 1e-9)))
        roll_ece_base = expected_calibration_error(
            roll_y_true, roll_y_proba_base, n_bins=10, strategy="quantile"
        )
        roll_bd_base = brier_decomposition(
            roll_y_true, roll_y_proba_base, n_bins=10, strategy="quantile"
        )

        roll_calib_hard_table = pd.DataFrame(
            {
                "Modelo (rollout)": [
                    roll_ece,
                    float(brier_score_loss(roll_y_true, roll_y_proba)),
                    roll_bd.get("reliability", np.nan),
                    roll_bd.get("resolution", np.nan),
                    roll_bd.get("uncertainty", np.nan),
                    roll_bd.get("brier_decomp", np.nan),
                    roll_bd.get("n_bins_eff", np.nan),
                ],
                "Baseline (p const)": [
                    roll_ece_base,
                    float(brier_score_loss(roll_y_true, roll_y_proba_base)),
                    roll_bd_base.get("reliability", np.nan),
                    roll_bd_base.get("resolution", np.nan),
                    roll_bd_base.get("uncertainty", np.nan),
                    roll_bd_base.get("brier_decomp", np.nan),
                    roll_bd_base.get("n_bins_eff", np.nan),
                ],
            },
            index=[
                "ECE (quantile bins)",
                "Brier",
                "Brier: Reliability",
                "Brier: Resolution",
                "Brier: Uncertainty",
                "Brier: Decomposition total",
                "Bins efectivos",
            ],
        )
        save_table_figure(
            roll_calib_hard_table,
            out_path=output_dir / "final_rollout_calibration_hard_metrics.png",
            title=f"Final Roll-out — Calibration dura (ECE + Brier decomposition, {horizon}m)",
        )

    plot_classification_timeline(
        roll_plot_df,
        out_path=output_dir / "final_rollout_classification.png",
        title=f"Final Roll-out — Probabilidades vs clase real ({horizon}m)",
        year_locator=1,
    )
    plot_rolling_logloss_wf(
        roll_plot_df,
        out_path=output_dir / "final_rollout_rolling_logloss_36m.png",
        title=f"Final Roll-out — Rolling LogLoss (36m, horizonte {horizon}m)",
        window=36,
    )
    plot_rolling_brier_wf(
        roll_plot_df,
        out_path=output_dir / "final_rollout_rolling_brier_36m.png",
        title=f"Final Roll-out — Rolling Brier (36m, horizonte {horizon}m)",
        window=36,
    )

    # Subperiod Split (Early vs Late)
    if len(roll_plot_df) >= 10:
        split_idx = int(len(roll_plot_df) // 2)
        split_date = pd.to_datetime(roll_plot_df["date"].iloc[split_idx])

        def _block_metrics(g: pd.DataFrame) -> Dict[str, float]:
            y_t = g["actual"].astype(int).to_numpy()
            y_p = np.clip(g["proba_up"].astype(float).to_numpy(), 1e-9, 1.0 - 1e-9)
            y_pr = (y_p >= float(best_t_roll)).astype(int)
            n_obs = int(len(g))

            m_dict: Dict[str, float] = {
                "n": float(n_obs),
                "base_rate": float(np.mean(y_t)) if n_obs else np.nan,
                "logloss": float(binary_logloss(y_t, y_p)) if n_obs else np.nan,
                "brier": float(brier_score_loss(y_t, y_p)) if n_obs else np.nan,
                "precision_top20": (
                    float(precision_at_k(pd.Series(y_t), y_p, top_frac=0.2)) if n_obs else np.nan
                ),
                "lift_top20": (
                    float(lift_at_k(pd.Series(y_t), y_p, top_frac=0.2)) if n_obs else np.nan
                ),
                "recall_0": (
                    float(recall_score(y_t, y_pr, pos_label=0, zero_division=0))
                    if n_obs
                    else np.nan
                ),
                "accuracy": float(accuracy_score(y_t, y_pr)) if n_obs else np.nan,
                "f1": float(f1_score(y_t, y_pr, zero_division=0)) if n_obs else np.nan,
            }
            if n_obs > 2 and len(np.unique(y_t)) > 1:
                m_dict["auc"] = float(roc_auc_score(y_t, y_p))
                m_dict["ap"] = float(average_precision_score(y_t, y_p))
            else:
                m_dict["auc"] = np.nan
                m_dict["ap"] = np.nan
            return m_dict

        g_early = roll_plot_df.loc[roll_plot_df["date"] <= split_date]
        g_late = roll_plot_df.loc[roll_plot_df["date"] > split_date]

        roll_split_metrics = pd.DataFrame(
            {
                "All": _block_metrics(roll_plot_df),
                f"Early (<= {split_date.date()})": _block_metrics(g_early),
                f"Late (> {split_date.date()})": _block_metrics(g_late),
            }
        ).reindex(
            [
                "n",
                "base_rate",
                "auc",
                "ap",
                "logloss",
                "brier",
                "precision_top20",
                "lift_top20",
                "recall_0",
                "accuracy",
                "f1",
            ]
        )
        save_table_figure(
            roll_split_metrics,
            out_path=output_dir / "final_rollout_subperiod_metrics.png",
            title=f"Final Roll-out — Métricas por subperiodo (threshold={best_t_roll:.3f})",
        )

    # Calibration Deciles
    plot_calibration_curve_wf(
        roll_plot_df,
        out_path=output_dir / "final_rollout_calibration.png",
        title=f"Final Roll-out — Calibration curve ({horizon}m)",
        n_bins=10,
    )
    plot_metrics_by_proba_bin(
        roll_plot_df,
        out_path=output_dir / "final_rollout_metrics_by_decile.png",
        title=f"Final Roll-out — Calibration por decil (P(sube) vs P(real=1), {horizon}m)",
        n_bins=10,
    )
    roll_calib_deciles = compute_calibration_deciles_table(roll_plot_df, n_bins=10)
    if roll_calib_deciles is not None and not roll_calib_deciles.empty:
        save_table_figure(
            roll_calib_deciles,
            out_path=output_dir / "final_rollout_calibration_deciles_table.png",
            title=f"Final Roll-out — Tabla de calibración por decil ({horizon}m)",
        )

    # Regime analysis
    if "high_inflation" in df.columns:
        regime_series_roll = (
            df["high_inflation"].astype(float).map({1.0: "high_inflation", 0.0: "low_inflation"})
        )
        roll_regime_df = plot_regime_performance_wf(
            roll_plot_df,
            out_path=output_dir / "final_rollout_regime_performance.png",
            title="Final Roll-out — Performance por régimen (high/low inflation)",
            regime_series=regime_series_roll,
        )
        if roll_regime_df is not None and not roll_regime_df.empty:
            save_table_figure(
                roll_regime_df.set_index("regime"),
                out_path=output_dir / "final_rollout_regime_performance_table.png",
                title="Final Roll-out — Tabla performance por régimen",
            )

    # Strategy ROI Comparison
    roll_signal = roll_plot_df[["date", "pred"]].copy()
    roll_signal["date"] = pd.to_datetime(roll_signal["date"])
    roll_signal = roll_signal.set_index("date").sort_index()

    prices_eval_roll = pd.Series(
        roll_plot_df["close_t"].values,
        index=pd.to_datetime(roll_plot_df["date"]),
        name="Close",
    ).sort_index()
    pred_aligned_roll = roll_signal["pred"].reindex(prices_eval_roll.index)

    has_pred_roll = pred_aligned_roll.notna()
    prices_eval_roll = prices_eval_roll.loc[has_pred_roll]
    pred_aligned_roll = pred_aligned_roll.loc[has_pred_roll]

    contrib_bh_roll = pd.Series(MONTHLY_AMOUNT, index=prices_eval_roll.index)
    contrib_signal_roll = (
        MONTHLY_AMOUNT
        * float(SIGNAL_MULTIPLIER)
        * (pred_aligned_roll.astype(int) == 1).astype(float)
    )

    bh_curve_roll = simulate_monthly_dca_roi(prices_eval_roll, contrib_bh_roll)
    sig_curve_roll = simulate_monthly_dca_roi(prices_eval_roll, contrib_signal_roll)

    plot_roi_strategies_comparison(
        bh_curve_roll,
        sig_curve_roll,
        out_path=output_dir / "roi_strategies_final_rollout.png",
        title=f"ROI acumulado (%) — DCA mensual vs Señal (Final Roll-out, horizonte {horizon}m)",
        year_locator=1,
        monthly_amount=MONTHLY_AMOUNT,
        signal_multiplier=SIGNAL_MULTIPLIER,
    )

    print("\n" + "=" * 60)
    print("FINAL ROLL-OUT EVALUATION (LAST 10 YEARS)")
    print("=" * 60)
    print(f"Ventana: {rollout_df.index.min().date()} -> {rollout_df.index.max().date()}")
    print(f"Train hasta: {train_df.index.max().date()} (gap={gap_months}m)")
    print(f"ROC-AUC: {roll_auc:.4f} | PR-AUC: {roll_ap:.4f}")
    print(f"LogLoss: {roll_logloss:.4f} (Baseline: {baseline_ll:.4f})")
    print(f"Brier: {roll_brier:.4f} (Baseline: {baseline_br:.4f})")
    print(f"ROI Final DCA: {float(bh_curve_roll['roi_pct'].dropna().iloc[-1]):.2f}%")
    print(f"ROI Final Señal ML: {float(sig_curve_roll['roi_pct'].dropna().iloc[-1]):.2f}%\n")

    return roll_plot_df
