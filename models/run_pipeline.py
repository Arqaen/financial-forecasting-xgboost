import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

# Ensure models/ is in sys.path when executed directly
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from src.config import (
    CLASS_ZERO_WEIGHT,
    DEFAULT_FEATURES,
    DEFAULT_HORIZON,
    DO_RANDOM_SEARCH,
    FECHA_INICIO,
    FECHA_OBJETIVO,
    METRICS_DIR,
    MIN_HISTORY_RATIO,
    THRESHOLD_CLASS,
    TUNING_OBJECTIVE,
)
from src.data_loader import load_raw_dataset
from src.explainability import train_final_model_and_explain
from src.features import prepare_modeling_dataset
from src.metrics import compute_spearman_rank_corr, correlation_report
from src.plots import (
    plot_correlation_heatmap,
    plot_spearman_rank_corr_bar,
    plot_target_distribution,
)
from src.rollout import run_final_rollout
from src.tracking import ExperimentTracker
from src.walk_forward import run_walk_forward_evaluation


def run_eda_stage(
    df: pd.DataFrame,
    features: List[str],
    out_dir: Path,
) -> None:
    """Execute Exploratory Data Analysis (EDA) and generate correlation and target distribution plots."""
    print("\n" + "=" * 60)
    print("STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # 1. Pearson Correlation
    corr = correlation_report(df, features + ["target"])
    plot_correlation_heatmap(
        corr,
        out_path=out_dir / "correlation_heatmap.png",
        title="Matriz de correlación (Pearson)",
        target_col="target",
        max_vars=25,
    )

    # 2. Spearman Monotonic Rank Correlation
    spearman_corr = compute_spearman_rank_corr(df, features, target_col="target")
    plot_spearman_rank_corr_bar(
        spearman_corr,
        out_path=out_dir / "spearman_rank_corr.png",
        title="Spearman rank correlation vs target",
    )

    # 3. Target Forward Return Distribution
    if "target_reg" in df.columns:
        plot_target_distribution(
            df["target_reg"],
            out_path=out_dir / "return_dist.png",
            title="Distribución del retorno futuro (target)",
        )

    print(f"EDA diagnostics successfully generated in: {out_dir}")


def run_pipeline(
    stage: str = "all",
    horizon: int = DEFAULT_HORIZON,
    start_date: str = FECHA_INICIO,
    end_date: str = FECHA_OBJETIVO,
    min_history: float = MIN_HISTORY_RATIO,
    features: Optional[List[str]] = None,
    do_random_search: bool = DO_RANDOM_SEARCH,
    clean_metrics: bool = True,
    out_dir: Optional[Path] = None,
    use_mlflow: Optional[bool] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    registered_model_name: Optional[str] = None,
    threshold_class: int = THRESHOLD_CLASS,
    class_zero_weight: float = CLASS_ZERO_WEIGHT,
    tuning_objective: str = TUNING_OBJECTIVE,
) -> None:
    """Run specified or all stages of the financial machine learning pipeline.

    Args:
        stage: Pipeline stage ('all', 'eda', 'walk-forward', 'rollout', 'explain').
        horizon: Forecast horizon in months.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        min_history: Minimum valid historical ratio for features.
        features: Optional list of predictive features (defaults to config.DEFAULT_FEATURES).
        do_random_search: Whether to execute random search hyperparameter tuning.
        clean_metrics: Whether to clear existing output artifacts directory before execution.
        out_dir: Custom directory for metrics output.
        use_mlflow: Enable tracking (defaults to MLFLOW_ENABLED, or True).
        tracking_uri: MLflow store/server URI (defaults to repository-local SQLite).
        experiment_name: MLflow experiment (CLI overrides environment variables).
        run_name: Optional display name for this pipeline execution.
        registered_model_name: Optional Model Registry name for the final model.
        threshold_class: Class whose F1 is maximized in temporal threshold selection.
        class_zero_weight: Training sample weight for observations labeled 0.
        tuning_objective: Scoring metric when random search is enabled.
    """
    load_dotenv(_CURRENT_DIR.parent / ".env", override=False)
    if stage not in ("all", "eda", "walk-forward", "rollout", "explain"):
        raise ValueError(f"Unknown pipeline stage: {stage}")
    tracker = ExperimentTracker(
        enabled=use_mlflow,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        registered_model_name=registered_model_name,
    )
    with tracker.run(
        {
            "stage": stage,
            "horizon": horizon,
            "start_date": start_date,
            "end_date": end_date,
            "min_history": min_history,
            "do_random_search": do_random_search,
            "threshold_class": threshold_class,
            "class_zero_weight": class_zero_weight,
            "tuning_objective": tuning_objective,
            "requested_features": ",".join(features or DEFAULT_FEATURES),
        }
    ):
        _execute_pipeline(
            stage,
            horizon,
            start_date,
            end_date,
            min_history,
            features,
            do_random_search,
            clean_metrics,
            out_dir,
            tracker,
            threshold_class,
            class_zero_weight,
            tuning_objective,
        )


def _execute_pipeline(
    stage: str,
    horizon: int,
    start_date: str,
    end_date: str,
    min_history: float,
    features: Optional[List[str]],
    do_random_search: bool,
    clean_metrics: bool,
    out_dir: Optional[Path],
    tracker: ExperimentTracker,
    threshold_class: int,
    class_zero_weight: float,
    tuning_objective: str,
) -> None:
    """Execute stages within the pipeline's tracking context."""
    output_dir = out_dir or METRICS_DIR
    if clean_metrics and stage in ("all", "eda"):
        if output_dir.exists():
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and assembling 28 market & macroeconomic series...")
    raw_df = load_raw_dataset()

    print(
        f"Applying feature engineering, publication lags, and target labels (horizon={horizon}m)..."
    )
    selected_features = features or DEFAULT_FEATURES
    df, valid_features = prepare_modeling_dataset(
        raw_df,
        features=selected_features,
        horizon=horizon,
        start_date=start_date,
        end_date=end_date,
        min_history=min_history,
    )
    print(
        f"Active modeling dataset: {len(df)} monthly observations | Features ({len(valid_features)}): {valid_features}"
    )
    tracker.log_dataset(df, valid_features)

    # Stage Dispatcher
    if stage in ("all", "eda"):
        with tracker.stage("eda", output_dir):
            run_eda_stage(df, valid_features, output_dir)

    if stage in ("all", "walk-forward"):
        print("\n" + "=" * 60)
        print("STAGE 2: PURGED WALK-FORWARD TEMPORAL CROSS-VALIDATION")
        print("=" * 60)
        with tracker.stage("walk-forward", output_dir):
            predictions, scorecard = run_walk_forward_evaluation(
                df,
                valid_features,
                horizon=horizon,
                out_dir=output_dir,
                do_random_search=do_random_search,
                threshold_class=threshold_class,
                class_zero_weight=class_zero_weight,
                tuning_objective=tuning_objective,
                tracker=tracker,
            )
            tracker.log_frame(predictions, "predictions.csv")
            tracker.log_frame(scorecard, "scorecard.csv")

    if stage in ("all", "rollout"):
        print("\n" + "=" * 60)
        print("STAGE 3: FINAL 10-YEAR OUT-OF-SAMPLE ROLL-OUT EVALUATION")
        print("=" * 60)
        with tracker.stage("rollout", output_dir):
            rollout = run_final_rollout(
                df,
                valid_features,
                horizon=horizon,
                out_dir=output_dir,
                tracker=tracker,
                threshold_class=threshold_class,
                class_zero_weight=class_zero_weight,
            )
            if rollout is not None:
                tracker.log_frame(rollout, "predictions.csv")
            else:
                tracker.log_params({"skipped": True, "skip_reason": "insufficient_history"})

    if stage in ("all", "explain"):
        print("\n" + "=" * 60)
        print("STAGE 4: FULL DATASET MODEL TRAINING & SHAP EXPLAINABILITY")
        print("=" * 60)
        with tracker.stage("explain", output_dir):
            model, _, diagnostics = train_final_model_and_explain(
                df,
                valid_features,
                horizon=horizon,
                out_dir=output_dir,
                class_zero_weight=class_zero_weight,
            )
            tracker.log_metrics(diagnostics)
            tracker.log_dict(
                {
                    key: str(value) if key == "last_date" else value
                    for key, value in diagnostics.items()
                },
                "diagnostics.json",
            )
            tracker.log_model(model, df[valid_features].head(5))

    print("\n" + "=" * 60)
    print(f"PIPELINE EXECUTION COMPLETE! All artifacts saved in: {output_dir.resolve()}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Financial ML Research & Quantitative Prediction Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "eda", "walk-forward", "rollout", "explain"],
        default="all",
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Target prediction horizon in months",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=FECHA_INICIO,
        help="Start date for analysis (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=FECHA_OBJETIVO,
        help="End date for analysis (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--min-history",
        type=float,
        default=MIN_HISTORY_RATIO,
        help="Minimum non-null ratio for features",
    )
    parser.add_argument(
        "--random-search",
        action="store_true",
        default=DO_RANDOM_SEARCH,
        help="Enable random search hyperparameter tuning",
    )
    parser.add_argument(
        "--threshold-class",
        type=int,
        choices=[0, 1],
        default=THRESHOLD_CLASS,
        help="Class whose F1 selects the walk-forward and rollout decision threshold",
    )
    parser.add_argument(
        "--class-zero-weight",
        type=float,
        default=CLASS_ZERO_WEIGHT,
        help="Weight of class 0 training samples (class 1 has weight 1)",
    )
    parser.add_argument(
        "--tuning-objective",
        choices=["logloss", "AP_0"],
        default=TUNING_OBJECTIVE,
        help="Walk-forward random search selection metric",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not wipe the metrics directory prior to running",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        default=None,
        help="Disable MLflow tracking",
    )
    parser.add_argument("--tracking-uri", help="MLflow tracking URI; overrides MLFLOW_TRACKING_URI")
    parser.add_argument("--experiment-name", help="MLflow experiment name")
    parser.add_argument("--run-name", help="MLflow run display name")
    parser.add_argument(
        "--registered-model-name", help="Register the final XGBoost model under this name"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        stage=args.stage,
        horizon=args.horizon,
        start_date=args.start_date,
        end_date=args.end_date,
        min_history=args.min_history,
        do_random_search=args.random_search,
        clean_metrics=not args.no_clean,
        use_mlflow=args.use_mlflow,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        registered_model_name=args.registered_model_name,
        threshold_class=args.threshold_class,
        class_zero_weight=args.class_zero_weight,
        tuning_objective=args.tuning_objective,
    )
