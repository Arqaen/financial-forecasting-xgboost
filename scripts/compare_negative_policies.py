"""Compare class weights and threshold objectives on the 36-month forecast.

Run from the repository root: python -m scripts.compare_negative_policies
"""

from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd
from sklearn.metrics import log_loss

from models.src.data_loader import load_raw_dataset
from models.src.features import prepare_modeling_dataset
from models.src.metrics import class_zero_metrics
from models.src.tracking import ExperimentTracker
from models.src.walk_forward import run_walk_forward_evaluation

# Freeze the reference configurations independently of future pipeline defaults.
SHALLOW = {
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_child_weight": 5,
    "gamma": 1.0,
    "reg_lambda": 9,
    "reg_alpha": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_jobs": 2,
}
FLEXIBLE = {
    **SHALLOW,
    "max_depth": 6,
    "min_child_weight": 2,
    "gamma": 0.1,
    "reg_lambda": 3,
    "reg_alpha": 0.0,
}
EXPERIMENT = "class-zero-policies-h36"
HORIZON = 36
DEVELOPMENT_END = pd.Timestamp("2001-12-31")
EVALUATION_START = pd.Timestamp("2005-01-01")


def summarize(predictions):
    actual = predictions["actual"]
    pred = predictions["pred"]
    proba = predictions["proba_up"]
    return {
        "months": len(actual),
        "actual_0": int((actual == 0).sum()),
        **class_zero_metrics(actual, proba, pred),
        "logloss": log_loss(actual, proba, labels=[0, 1]),
    }


def main():
    dataset, features = prepare_modeling_dataset(load_raw_dataset(), horizon=HORIZON)
    runs = []
    reference = None
    tracker = ExperimentTracker(enabled=True, experiment_name=EXPERIMENT)
    for (name, params), weight, target in product(
        [("shallow", SHALLOW), ("flexible", FLEXIBLE)], [1, 2, 4], [1, 0]
    ):
        run_name = f"{name}-weight{weight}-f1_{target}"
        print(f"Evaluating {run_name}", flush=True)
        tracker.run_name = run_name
        with (
            tracker.run(
                {
                    "stage": "walk-forward",
                    "setup": name,
                    "horizon": HORIZON,
                    "development_end": str(DEVELOPMENT_END.date()),
                    "evaluation_start": str(EVALUATION_START.date()),
                    "class_zero_weight": weight,
                    "threshold_class": target,
                    "selection_metric": "development.F1_0",
                }
            ),
            TemporaryDirectory() as tmp,
        ):
            run = mlflow.active_run()
            tracker.log_dataset(dataset, features)
            tracker.log_dict(params, "manual_params.json")
            predictions, _ = run_walk_forward_evaluation(
                dataset,
                features,
                horizon=HORIZON,
                min_train_size=240,
                test_size=12,
                manual_params=params,
                threshold_class=target,
                class_zero_weight=weight,
                do_random_search=False,
                tracker=tracker,
                out_dir=Path(tmp),
            )
            factual = predictions[["date", "actual", "close_t", "close_t_plus_h"]]
            if reference is None:
                reference = factual.copy()
            else:
                pd.testing.assert_frame_equal(reference, factual)
            tracker.log_frame(predictions, "predictions.csv")
            development = summarize(predictions.loc[predictions.date <= DEVELOPMENT_END])
            tracker.log_metrics({f"development.{k}": v for k, v in development.items()})
            tracker.log_metrics(summarize(predictions))
            runs.append((run_name, run.info.run_id, development, predictions))
            client = mlflow.MlflowClient(tracking_uri=mlflow.get_tracking_uri())
            experiment_id = run.info.experiment_id

    # Freeze selection before computing any evaluation-period comparisons.
    winner = max(runs, key=lambda r: (r[2]["F1_0"], -r[2]["false_alarms_0"]))[0]
    print(f"Development-period winner: {winner}", flush=True)
    for name, run_id, _, predictions in runs:
        evaluation = summarize(predictions.loc[predictions.date >= EVALUATION_START])
        client.set_tag(run_id, "development_winner", str(name == winner).lower())
        for key, value in evaluation.items():
            client.log_metric(run_id, f"evaluation.{key}", float(value))
        if name in (winner, "shallow-weight1-f1_1"):
            print(name, evaluation, flush=True)
    print(f"MLflow experiment ID: {experiment_id}")


if __name__ == "__main__":
    main()
