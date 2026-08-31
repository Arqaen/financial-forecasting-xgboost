import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Ensure models/ is in sys.path when executed directly
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from src.config import (
    DEFAULT_FEATURES,
    DEFAULT_HORIZON,
    DO_RANDOM_SEARCH,
    FECHA_INICIO,
    FECHA_OBJETIVO,
    METRICS_DIR,
    MIN_HISTORY_RATIO,
)
from src.data_loader import load_raw_dataset
from src.explainability import train_final_model_and_explain
from src.features import prepare_modeling_dataset
from src.metrics import (
    compute_spearman_rank_corr,
    correlation_report,
)
from src.plots import (
    plot_correlation_heatmap,
    plot_spearman_rank_corr_bar,
    plot_target_distribution,
)
from src.rollout import run_final_rollout
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
    """
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

    # Stage Dispatcher
    if stage in ("all", "eda"):
        run_eda_stage(df, valid_features, output_dir)

    if stage in ("all", "walk-forward"):
        print("\n" + "=" * 60)
        print("STAGE 2: PURGED WALK-FORWARD TEMPORAL CROSS-VALIDATION")
        print("=" * 60)
        run_walk_forward_evaluation(
            df,
            valid_features,
            horizon=horizon,
            out_dir=output_dir,
            do_random_search=do_random_search,
        )

    if stage in ("all", "rollout"):
        print("\n" + "=" * 60)
        print("STAGE 3: FINAL 10-YEAR OUT-OF-SAMPLE ROLL-OUT EVALUATION")
        print("=" * 60)
        run_final_rollout(
            df,
            valid_features,
            horizon=horizon,
            out_dir=output_dir,
        )

    if stage in ("all", "explain"):
        print("\n" + "=" * 60)
        print("STAGE 4: FULL DATASET MODEL TRAINING & SHAP EXPLAINABILITY")
        print("=" * 60)
        train_final_model_and_explain(
            df,
            valid_features,
            horizon=horizon,
            out_dir=output_dir,
        )

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
        "--no-clean",
        action="store_true",
        help="Do not wipe the metrics directory prior to running",
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
    )
