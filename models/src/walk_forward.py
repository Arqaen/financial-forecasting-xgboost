"""Purged Walk-Forward validation engine with temporal embargo and internal validation threshold tuning."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from .config import (
    DEFAULT_HORIZON,
    DO_RANDOM_SEARCH,
    FIXED_PARAMS_BASE,
    MANUAL_PARAMS_BASE,
    METRICS_DIR,
    MIN_TRAIN_SIZE,
    MONTHLY_AMOUNT,
    PARAM_DIST,
    RANDOM_SEARCH_N_ITER,
    RANDOM_SEARCH_SEED,
    SCORE_FRAC,
    SIGNAL_MULTIPLIER,
    TEST_SIZE,
    TUNE_EACH_FOLD,
)
from .metrics import (
    best_threshold_by_f1,
    binary_logloss,
    brier_decomposition,
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
    plot_confusion_matrix_heatmap,
    plot_cumulative_gains_wf,
    plot_decile_accuracy,
    plot_equity_curve_directional_wf,
    plot_metrics_by_proba_bin,
    plot_proba_hist_by_class,
    plot_regime_performance_wf,
    plot_roc_pr_wf,
    plot_roi_strategies_comparison,
    plot_rolling_logloss_wf,
    save_table_figure,
)
from .simulation import (
    simulate_monthly_dca_roi,
    simulate_value_averaging_modified_roi,
)
from .tuning import tune_xgb_random_search_timeval


def run_walk_forward_evaluation(
    df: pd.DataFrame,
    features: List[str],
    *,
    horizon: int = DEFAULT_HORIZON,
    min_train_size: int = MIN_TRAIN_SIZE,
    test_size: int = TEST_SIZE,
    out_dir: Optional[Path] = None,
    do_random_search: bool = DO_RANDOM_SEARCH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute purged walk-forward temporal cross-validation with out-of-sample evaluation.

    Args:
        df: Prepared modeling DataFrame.
        features: List of predictive feature columns.
        horizon: Forecast horizon in months (used for purging overlap).
        min_train_size: Initial training window in months.
        test_size: Out-of-sample test window in months.
        out_dir: Directory to save generated diagnostic plots and scorecards.
        do_random_search: Whether to run random search on validation folds.

    Returns:
        Tuple containing (wf_df_predictions, wf_metrics_scorecard).
    """
    output_dir = out_dir or METRICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    aucs: List[float] = []
    loglosses: List[float] = []
    ap_scores: List[float] = []
    briers: List[float] = []
    balanced_accs: List[float] = []
    mccs: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    recalls_0: List[float] = []
    f1s: List[float] = []
    accuracies: List[float] = []
    precision_at_top20s: List[float] = []
    lift_top20s: List[float] = []

    baseline_loglosses: List[float] = []
    baseline_briers: List[float] = []

    all_proba: List[float] = []
    all_actuals: List[int] = []
    all_dates: List[pd.Timestamp] = []
    all_close: List[float] = []
    all_close_fwd: List[float] = []
    all_pred: List[int] = []
    all_thresholds: List[float] = []

    best_params_global: Optional[Dict] = None

    start = int(min_train_size)
    purge = int(horizon)
    embargo = 0

    while start < len(df) - test_size:
        train_end = start - purge
        test_end = start + test_size

        train_df = df.iloc[:train_end]
        test_df = df.iloc[start:test_end]

        X_train = train_df[features]
        y_train = train_df["target"].astype(int)

        X_test = test_df[features]
        y_test = test_df["target"].astype(int)

        # Internal temporal validation split
        val_size = int(len(X_train) * 0.2)
        gap = int(horizon)
        tr_end = -(val_size + gap)

        X_tr = X_train.iloc[:tr_end]
        y_tr = y_train.iloc[:tr_end]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]

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
            best_t, _ = 0.5, float("nan")
            p_const = float(np.clip(y_tr.mean(), 1e-6, 1.0 - 1e-6))
            proba = np.full(len(X_test), p_const, dtype=float)
            y_pred = (proba >= best_t).astype(int)
        else:
            fixed_params = dict(FIXED_PARAMS_BASE)
            manual_params = dict(MANUAL_PARAMS_BASE)

            if not do_random_search:
                best_params = manual_params
            elif do_random_search and (TUNE_EACH_FOLD or best_params_global is None):
                best_params, _ = tune_xgb_random_search_timeval(
                    X_tr,
                    y_tr,
                    X_es,
                    y_es,
                    X_score,
                    y_score,
                    fixed_params=fixed_params,
                    param_dist=PARAM_DIST,
                    n_iter=RANDOM_SEARCH_N_ITER,
                    random_state=RANDOM_SEARCH_SEED,
                )
                if not TUNE_EACH_FOLD:
                    best_params_global = best_params
            else:
                best_params = best_params_global or {}

            model_params = dict(fixed_params)
            model_params.update(best_params)

            model = XGBClassifier(**model_params)
            if len(np.unique(y_es)) < 2:
                model.set_params(early_stopping_rounds=None)
                model.fit(X_tr, y_tr, verbose=False)
            else:
                model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)

            if len(X_score) < 1:
                best_t, _ = 0.5, float("nan")
            else:
                score_proba = model.predict_proba(X_score)[:, 1]
                best_t, _ = best_threshold_by_f1(y_score, score_proba)

            proba = model.predict_proba(X_test)[:, 1]
            y_pred = (proba >= best_t).astype(int)

        all_thresholds.append(float(best_t))
        all_proba.extend(proba.tolist())
        all_actuals.extend(y_test.values.tolist())
        all_dates.extend(y_test.index.tolist())
        all_close.extend(test_df["Close"].values.tolist())
        all_close_fwd.extend(test_df["close_fwd"].values.tolist())
        all_pred.extend(y_pred.tolist())

        # Baseline evaluation (constant train base-rate)
        p0 = float(np.clip(y_train.mean(), 1e-6, 1.0 - 1e-6))
        baseline_loglosses.append(binary_logloss(y_test.values, np.full_like(proba, p0)))
        baseline_briers.append(brier_score_loss(y_test, np.full_like(proba, p0)))

        # Test fold metrics
        balanced_accs.append(float(balanced_accuracy_score(y_test, y_pred)))
        mccs.append(float(matthews_corrcoef(y_test, y_pred)))
        precisions.append(float(precision_score(y_test, y_pred, zero_division=0)))
        recalls.append(float(recall_score(y_test, y_pred, zero_division=0)))
        recalls_0.append(float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)))
        f1s.append(float(f1_score(y_test, y_pred, zero_division=0)))
        accuracies.append(float(accuracy_score(y_test, y_pred)))
        precision_at_top20s.append(precision_at_k(y_test, proba, top_frac=0.2))
        lift_top20s.append(lift_at_k(y_test, proba, top_frac=0.2))

        if len(np.unique(y_test)) > 1:
            aucs.append(float(roc_auc_score(y_test, proba)))
            ap_scores.append(float(average_precision_score(y_test, proba)))
        else:
            aucs.append(np.nan)
            ap_scores.append(np.nan)

        loglosses.append(binary_logloss(y_test.values, proba))
        briers.append(float(brier_score_loss(y_test, proba)))

        start = test_end + embargo

    # ==========================================
    # Scorecards & Consolidated WF DataFrame
    # ==========================================
    wf_metrics_scorecard = pd.DataFrame(
        {
            "Valor": [
                float(np.nanmean(aucs)),
                float(np.nanmean(ap_scores)),
                float(np.nanmean(loglosses)),
                float(np.nanmean(briers)),
                float(np.nanmean(balanced_accs)),
                float(np.nanmean(mccs)),
                float(np.nanmean(precisions)),
                float(np.nanmean(recalls)),
                float(np.nanmean(recalls_0)),
                float(np.nanmean(f1s)),
                float(np.nanmean(accuracies)),
                float(np.nanmean(precision_at_top20s)),
                float(np.nanmean(lift_top20s)),
            ]
        },
        index=[
            "ROC-AUC (mean)",
            "PR-AUC / AvgPrecision (mean)",
            "LogLoss (mean)",
            "Brier score (mean)",
            "Balanced Accuracy (mean)",
            "MCC (mean)",
            "Precision (mean)",
            "Recall (mean)",
            "Recall clase 0 (mean)",
            "F1 (mean)",
            "Accuracy (mean)",
            "Precision@top20% (mean)",
            "Lift@top20% (mean)",
        ],
    )
    save_table_figure(
        wf_metrics_scorecard,
        out_path=output_dir / "walk_forward_metrics_scorecard.png",
        title=f"Walk-Forward — Métricas promedio (horizonte {horizon}m)",
    )

    wf_baselines_table = pd.DataFrame(
        {
            "LogLoss": [float(np.nanmean(loglosses)), float(np.nanmean(baseline_loglosses))],
            "Brier": [float(np.nanmean(briers)), float(np.nanmean(baseline_briers))],
        },
        index=["Modelo (WF)", "Baseline"],
    )
    save_table_figure(
        wf_baselines_table,
        out_path=output_dir / "walk_forward_baselines_scorecard.png",
        title="Walk-Forward — Modelo vs Baselines",
    )

    wf_df = pd.DataFrame(
        {
            "date": pd.to_datetime(all_dates),
            "proba_up": all_proba,
            "actual": all_actuals,
            "pred": all_pred,
            "close_t": all_close,
            "close_t_plus_h": all_close_fwd,
        }
    )
    wf_df = (
        wf_df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    )

    # ==========================================
    # 1M Realized Risk Metrics & Turnover
    # ==========================================
    wf_ret_df = wf_df[["date", "close_t", "proba_up", "pred", "actual"]].copy()
    wf_ret_df["date"] = pd.to_datetime(wf_ret_df["date"])
    wf_ret_df = (
        wf_ret_df.sort_values("date")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["close_t", "proba_up"])
    )

    if len(wf_ret_df) >= 3:
        wf_ret_df["ret_1m"] = (
            wf_ret_df["close_t"].astype(float).shift(-1) / wf_ret_df["close_t"].astype(float) - 1.0
        )
        wf_ret_df = wf_ret_df.iloc[:-1].copy()

        exposure_proba = np.clip(wf_ret_df["proba_up"].astype(float).to_numpy(), 0.0, 1.0)
        exposure_pred = (wf_ret_df["pred"].astype(float).to_numpy() > 0.5).astype(float)
        ret_1m = wf_ret_df["ret_1m"].astype(float).to_numpy()

        wf_risk_bh = compute_return_risk_metrics(ret_1m, periods_per_year=12.0)
        wf_risk_proba = compute_return_risk_metrics(exposure_proba * ret_1m, periods_per_year=12.0)
        wf_risk_pred = compute_return_risk_metrics(exposure_pred * ret_1m, periods_per_year=12.0)

        wf_risk_table = pd.DataFrame(
            {
                "Buy&Hold": wf_risk_bh,
                "Estrategia (exposure=P)": wf_risk_proba,
                "Estrategia (pred 0/1)": wf_risk_pred,
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
            wf_risk_table,
            out_path=output_dir / "walk_forward_risk_metrics_1m.png",
            title=f"Walk-Forward — Riesgo / Sharpe (retornos 1M, horizonte target {horizon}m)",
        )

        wf_signal_series = wf_df.set_index(pd.to_datetime(wf_df["date"]))["pred"]
        wf_stab = compute_signal_stability_metrics(wf_signal_series)
        wf_turn = compute_exposure_turnover(
            wf_df.set_index(pd.to_datetime(wf_df["date"]))["proba_up"]
        )

        wf_turnover_table = pd.DataFrame(
            {
                "Valor": [
                    wf_stab.get("n", np.nan),
                    wf_stab.get("pct_long", np.nan),
                    wf_stab.get("change_pct", np.nan),
                    wf_stab.get("avg_hold_long", np.nan),
                    wf_stab.get("avg_hold_flat", np.nan),
                    wf_turn.get("mean_abs_change", np.nan),
                    wf_turn.get("median_abs_change", np.nan),
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
            wf_turnover_table,
            out_path=output_dir / "walk_forward_turnover_signal_stability.png",
            title=f"Walk-Forward — Turnover / Estabilidad de señal ({horizon}m)",
        )

        wf_y_true = wf_df["actual"].astype(int).to_numpy()
        wf_y_proba = np.clip(wf_df["proba_up"].astype(float).to_numpy(), 1e-9, 1.0 - 1e-9)
        wf_base = float(np.mean(wf_y_true)) if len(wf_y_true) else float("nan")

        wf_ece = expected_calibration_error(wf_y_true, wf_y_proba, n_bins=10, strategy="quantile")
        wf_bd = brier_decomposition(wf_y_true, wf_y_proba, n_bins=10, strategy="quantile")

        wf_y_proba_base = np.full_like(wf_y_proba, float(np.clip(wf_base, 1e-9, 1.0 - 1e-9)))
        wf_ece_base = expected_calibration_error(
            wf_y_true, wf_y_proba_base, n_bins=10, strategy="quantile"
        )
        wf_bd_base = brier_decomposition(wf_y_true, wf_y_proba_base, n_bins=10, strategy="quantile")

        wf_calib_hard_table = pd.DataFrame(
            {
                "Modelo (WF)": [
                    wf_ece,
                    float(brier_score_loss(wf_y_true, wf_y_proba)),
                    wf_bd.get("reliability", np.nan),
                    wf_bd.get("resolution", np.nan),
                    wf_bd.get("uncertainty", np.nan),
                    wf_bd.get("brier_decomp", np.nan),
                    wf_bd.get("n_bins_eff", np.nan),
                ],
                "Baseline (p const)": [
                    wf_ece_base,
                    float(brier_score_loss(wf_y_true, wf_y_proba_base)),
                    wf_bd_base.get("reliability", np.nan),
                    wf_bd_base.get("resolution", np.nan),
                    wf_bd_base.get("uncertainty", np.nan),
                    wf_bd_base.get("brier_decomp", np.nan),
                    wf_bd_base.get("n_bins_eff", np.nan),
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
            wf_calib_hard_table,
            out_path=output_dir / "walk_forward_calibration_hard_metrics.png",
            title=f"Walk-Forward — Calibration dura (ECE + Brier decomposition, {horizon}m)",
        )

    # ==========================================
    # Ranking Metrics & Confusion Matrices
    # ==========================================
    wf_top20_precision = precision_at_k(wf_df["actual"], wf_df["proba_up"].to_numpy(), top_frac=0.2)
    wf_top20_lift = lift_at_k(wf_df["actual"], wf_df["proba_up"].to_numpy(), top_frac=0.2)
    wf_recall_0 = float(recall_score(wf_df["actual"], wf_df["pred"], pos_label=0, zero_division=0))
    wf_base_rate = float(wf_df["actual"].mean())
    wf_threshold_cm = confusion_matrix_by_thresholds(wf_df["actual"], wf_df["proba_up"].to_numpy())

    wf_ranking_metrics = pd.DataFrame(
        {
            "Valor": [
                wf_base_rate,
                wf_top20_precision,
                wf_top20_lift,
                wf_recall_0,
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
        wf_ranking_metrics,
        out_path=output_dir / "walk_forward_ranking_metrics.png",
        title=f"Walk-Forward — Métricas para señal top 20% ({horizon}m)",
    )
    save_table_figure(
        wf_threshold_cm,
        out_path=output_dir / "walk_forward_confusion_matrix_thresholds.png",
        title=f"Walk-Forward — Confusion matrix por threshold ({horizon}m)",
    )
    plot_confusion_matrix_heatmap(
        wf_df["actual"],
        wf_df["pred"].to_numpy(),
        out_path=output_dir / "walk_forward_confusion_matrix_heatmap.png",
        title=f"Walk-Forward — Matriz de confusión (pred optimizado, {horizon}m)",
    )

    # ==========================================
    # Diagnostic Plots
    # ==========================================
    plot_classification_timeline(
        wf_df,
        out_path=output_dir / "walk_forward_classification.png",
        title=f"Walk-Forward — Probabilidades vs clase real ({horizon}m)",
        year_locator=2,
    )
    plot_calibration_curve_wf(
        wf_df,
        out_path=output_dir / "walk_forward_calibration.png",
        title=f"Walk-Forward — Calibration curve ({horizon}m)",
        n_bins=10,
    )
    plot_proba_hist_by_class(
        wf_df,
        out_path=output_dir / "walk_forward_proba_hist_by_class.png",
        title=f"Walk-Forward — Distribución de P(sube) por clase ({horizon}m)",
        bins=25,
        kde=True,
    )
    plot_roc_pr_wf(
        wf_df,
        out_path=output_dir / "walk_forward_roc_pr.png",
        title=f"Walk-Forward — ROC & PR ({horizon}m)",
    )
    plot_metrics_by_proba_bin(
        wf_df,
        out_path=output_dir / "walk_forward_metrics_by_decile.png",
        title=f"Walk-Forward — Calibration por decil (P(sube) vs P(real=1), {horizon}m)",
        n_bins=10,
    )
    plot_cumulative_gains_wf(
        wf_df,
        out_path=output_dir / "walk_forward_cumulative_gains.png",
        title=f"Walk-Forward — Cumulative gains / lift ({horizon}m)",
    )
    plot_rolling_logloss_wf(
        wf_df,
        out_path=output_dir / "walk_forward_rolling_accuracy_36m.png",
        title=f"Walk-Forward — Rolling LogLoss (36m, horizonte {horizon}m)",
        window=36,
    )

    if "high_inflation" in df.columns:
        regime_series = (
            df["high_inflation"].astype(float).map({1.0: "high_inflation", 0.0: "low_inflation"})
        )
        plot_regime_performance_wf(
            wf_df,
            out_path=output_dir / "walk_forward_regime_performance.png",
            title="Walk-Forward — Performance por régimen (high/low inflation)",
            regime_series=regime_series,
        )

    plot_equity_curve_directional_wf(
        wf_df,
        out_path=output_dir / "walk_forward_equity_curve_directional.png",
        title=f"Walk-Forward — Equity curve direccional ({horizon}m)",
    )

    dec_df = wf_df.copy()
    dec_df["decile"] = pd.qcut(dec_df["proba_up"], 10, labels=False, duplicates="drop")
    deciles = dec_df.groupby("decile")["actual"].mean()
    plot_decile_accuracy(
        deciles,
        out_path=output_dir / "decile_plot_classification.png",
        title="Tasa de acierto (clase=1) por decil de P(sube)",
    )

    # ==========================================
    # Investment Strategy ROI Simulation
    # ==========================================
    wf_signal = wf_df[["date", "pred"]].copy()
    wf_signal["date"] = pd.to_datetime(wf_signal["date"])
    wf_signal = wf_signal.set_index("date").sort_index()

    eval_start = pd.to_datetime(wf_df["date"].min())
    eval_end = pd.to_datetime(wf_df["date"].max())
    prices_eval = df.loc[eval_start:eval_end, "Close"].copy()
    pred_aligned = wf_signal["pred"].reindex(prices_eval.index)

    has_pred = pred_aligned.notna()
    prices_eval = prices_eval.loc[has_pred]
    pred_aligned = pred_aligned.loc[has_pred]

    contrib_bh = pd.Series(MONTHLY_AMOUNT, index=prices_eval.index)
    contrib_signal = (
        MONTHLY_AMOUNT * float(SIGNAL_MULTIPLIER) * (pred_aligned.astype(int) == 1).astype(float)
    )

    bh_curve = simulate_monthly_dca_roi(prices_eval, contrib_bh)
    sig_curve = simulate_monthly_dca_roi(prices_eval, contrib_signal)
    va_curve = simulate_value_averaging_modified_roi(prices_eval, MONTHLY_AMOUNT)

    plot_roi_strategies_comparison(
        bh_curve,
        sig_curve,
        va_curve,
        out_path=output_dir / "roi_strategies_walk_forward.png",
        title=f"ROI acumulado (%) — DCA mensual vs Señal vs Value Averaging Modified (Walk-Forward, horizonte {horizon}m)",
        year_locator=2,
        monthly_amount=MONTHLY_AMOUNT,
        signal_multiplier=SIGNAL_MULTIPLIER,
    )

    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 60)
    print(wf_metrics_scorecard)
    print(f"\nROI final Buy&Hold DCA (%): {float(bh_curve['roi_pct'].dropna().iloc[-1]):.2f}%")
    print(f"ROI final Señal ML (%): {float(sig_curve['roi_pct'].dropna().iloc[-1]):.2f}%")
    print(
        f"ROI final Value Averaging Modified (%): {float(va_curve['roi_pct'].dropna().iloc[-1]):.2f}%\n"
    )

    return wf_df, wf_metrics_scorecard
