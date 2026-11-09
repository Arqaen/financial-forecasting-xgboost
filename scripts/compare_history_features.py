"""Compare complete cases, longer history and training-selected extra features.

Run from the repository root with python -m scripts.compare_history_features.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import mlflow
import pandas as pd

from models.src.config import ALL_CANDIDATE_FEATURES
from models.src.data_loader import load_raw_dataset
from models.src.features import prepare_modeling_dataset
from models.src.tracking import ExperimentTracker
from models.src.walk_forward import run_walk_forward_evaluation
from scripts.compare_negative_policies import SHALLOW, summarize


def main(evaluation_scope: str = "both") -> None:
    matplotlib.use("Agg")
    raw = load_raw_dataset()
    baseline, base_features = prepare_modeling_dataset(raw, horizon=36)
    extended, _ = prepare_modeling_dataset(raw, horizon=36, keep_missing_features=True)
    rich, rich_features = prepare_modeling_dataset(
        raw,
        features=ALL_CANDIDATE_FEATURES,
        horizon=36,
        keep_missing_features=True,
    )
    start = baseline.index[240]
    reference_start = str(baseline.index.min().date())
    extras = [f for f in rich_features if f not in base_features]
    output = Path(".mlflow") / (
        "history-features-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True)
    tracker = ExperimentTracker(enabled=True, experiment_name="history-and-features-h36")
    references: Dict[str, pd.DataFrame] = {}
    reference_windows: Dict[str, Any] = {}
    records = []
    manifest: Dict[str, Any] = {"experiment_name": tracker.experiment_name, "runs": []}
    earliest = extended.index[240]
    full_start = pd.Timestamp(earliest.year + int(earliest.month > 1), 1, 31)
    variants: List[Tuple[str, pd.DataFrame, List[str], pd.Timestamp, Optional[str], str]] = []
    if evaluation_scope in ("common", "both"):
        variants.extend(
            [
                ("01-reference-complete", baseline, [], start, reference_start, "common"),
                ("02-long-history-missing", extended, [], start, reference_start, "common"),
                ("03-long-history-selected-extras", rich, extras, start, reference_start, "common"),
            ]
        )
    if evaluation_scope in ("full", "both"):
        variants.extend(
            [
                ("02-long-history-missing", extended, [], full_start, None, "full"),
                ("03-long-history-selected-extras", rich, extras, full_start, None, "full"),
            ]
        )
    for variant, frame, candidates, test_start, validation_anchor, scope in variants:
        name = f"{variant}-{scope}-{test_start.year}-2022"
        initial_rows = int(frame.index.get_loc(test_start))
        assert frame.index.equals(pd.date_range(frame.index.min(), frame.index.max(), freq="ME"))
        tracker.run_name = name
        run_output = output / name
        run_output.mkdir()
        print("START", name, len(frame), "rows", flush=True)
        with tracker.run(
            {
                "stage": "walk-forward",
                "horizon": 36,
                "variant": variant,
                "evaluation_scope": scope,
                "threshold_class": 1,
                "class_zero_weight": 1.0,
                "evaluation_start": str(test_start.date()),
                "selection_policy": "Training-only absolute Spearman; at least 60 observations; top 5 extras",
            }
        ):
            active = mlflow.active_run()
            assert active is not None
            run_id = active.info.run_id
            client = mlflow.MlflowClient(tracking_uri=mlflow.get_tracking_uri())
            tracker.log_dataset(frame, base_features + candidates)
            tracker.log_dict(SHALLOW, "manual_params.json")
            mlflow.log_artifact(__file__, artifact_path="reproducibility")
            predictions, scorecard = run_walk_forward_evaluation(
                frame,
                base_features,
                horizon=36,
                min_train_size=initial_rows,
                test_size=12,
                out_dir=run_output,
                do_random_search=False,
                manual_params=SHALLOW,
                threshold_class=1,
                class_zero_weight=1.0,
                tracker=tracker,
                validation_reference_start=validation_anchor,
                additional_features=candidates,
                min_feature_observations=60,
                max_additional_features=5,
            )
            factual = predictions[["date", "actual", "close_t", "close_t_plus_h"]]
            if scope not in references:
                references[scope] = factual.copy()
            else:
                pd.testing.assert_frame_equal(references[scope], factual)
            windows = []
            for artifact in sorted(client.list_artifacts(run_id, "folds"), key=lambda a: a.path):
                path = client.download_artifacts(run_id, artifact.path, str(run_output))
                fold = json.loads(Path(path).read_text(encoding="utf-8"))
                assert pd.Timestamp(fold["fit_end"]) < pd.Timestamp(fold["early_stopping_start"])
                assert pd.Timestamp(fold["threshold_validation_end"]) < pd.Timestamp(
                    fold["test_start"]
                )
                windows.append(
                    {
                        key: fold[key]
                        for key in (
                            "test_start",
                            "test_end",
                            "fit_end",
                            "early_stopping_start",
                            "early_stopping_end",
                            "threshold_validation_start",
                            "threshold_validation_end",
                        )
                    }
                )
            if scope not in reference_windows:
                reference_windows[scope] = windows
            else:
                assert windows == reference_windows[scope]
            expected_folds = 41 if scope == "common" else 75
            assert len(windows) == expected_folds and len(predictions) == expected_folds * 12
            assert predictions.date.min() == test_start
            assert predictions.date.max() == pd.Timestamp("2022-12-31")
            tracker.log_frame(predictions, "predictions.csv")
            tracker.log_frame(scorecard, "scorecard.csv")
            tracker.log_metrics(summarize(predictions))
            shared = summarize(predictions.loc[predictions.date >= start])
            tracker.log_metrics({"shared_period." + key: value for key, value in shared.items()})
            later = summarize(predictions.loc[predictions.date >= "2005-01-01"])
            tracker.log_metrics({"evaluation." + key: value for key, value in later.items()})
            tracker.log_dict(
                {
                    "evaluation_scope": scope,
                    "windows_identical_within_scope": True,
                    "dataset_start": str(frame.index.min()),
                    "first_prediction": str(test_start),
                    "folds": windows,
                },
                "comparison_checks.json",
            )
            mlflow.log_artifacts(str(run_output), artifact_path="diagnostics")
            row = {"variant": name, "scope": scope, "run_id": run_id, **summarize(predictions)}
            records.append(row)
            manifest["experiment_id"] = active.info.experiment_id
            manifest["runs"].append({"name": name, "run_id": run_id})
            print("FINISHED", json.dumps(row), flush=True)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pd.DataFrame(records).to_csv(output / "comparison.csv", index=False)
    for run in manifest["runs"]:
        client.log_artifact(run["run_id"], str(output / "comparison.csv"), "comparison")
    print("ALL VERIFIED", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-scope", choices=["common", "full", "both"], default="both")
    main(parser.parse_args().evaluation_scope)
