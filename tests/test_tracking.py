"""Integration checks against real MLflow SQLite storage and model serialization."""

import builtins
import json
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from xgboost import XGBClassifier

from models import run_pipeline as pipeline
from models.src.tracking import ExperimentTracker


@pytest.fixture
def store(tmp_path):
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment(
        "test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    return uri, client, experiment_id


def test_tracking_roundtrip_and_stage_artifacts(store, tmp_path):
    uri, client, experiment_id = store
    previous_uri = mlflow.get_tracking_uri()
    output = tmp_path / "metrics"
    output.mkdir()
    (output / "stale.png").write_bytes(b"previous run")
    tracker = ExperimentTracker(tracking_uri=uri, experiment_name="test")
    frame = pd.DataFrame({"f1": [0.0, 1.0], "target": [0, 1]})
    with tracker.run({"stage": "all", "horizon": 3}):
        tracker.log_dataset(frame, ["f1"])
        with tracker.stage("walk-forward", output):
            tracker.log_metrics({"auc": 0.75, "undefined": float("nan"), "infinite": float("inf")})
            tracker.log_metrics({"fold.logloss": 0.4}, step=2)
            tracker.log_frame(frame, "predictions.csv")
            (output / "new.png").write_bytes(b"current run")
    assert mlflow.active_run() is None
    assert mlflow.get_tracking_uri() == previous_uri
    runs = client.search_runs([experiment_id])
    parent = next(run for run in runs if "mlflow.parentRunId" not in run.data.tags)
    child = next(run for run in runs if "mlflow.parentRunId" in run.data.tags)
    assert parent.info.status == child.info.status == "FINISHED"
    assert child.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert child.data.metrics == {"auc": 0.75, "fold.logloss": 0.4}
    assert client.get_metric_history(child.info.run_id, "fold.logloss")[0].step == 2
    assert {item.path for item in client.list_artifacts(child.info.run_id, "diagnostics")} == {
        "diagnostics/new.png"
    }
    assert parent.data.params["observations"] == "2"
    manifest = client.download_artifacts(parent.info.run_id, "dataset.json", str(tmp_path))
    assert len(json.loads(Path(manifest).read_text())["sha256"]) == 64


def test_failed_stage_marks_parent_and_child_failed(store, tmp_path):
    uri, client, experiment_id = store
    tracker = ExperimentTracker(tracking_uri=uri, experiment_name="test")
    with pytest.raises(ValueError, match="training failed"):
        with tracker.run({"stage": "explain", "horizon": 3}):
            with tracker.stage("explain", tmp_path):
                raise ValueError("training failed")
    runs = client.search_runs([experiment_id])
    assert len(runs) == 2
    assert all(run.info.status == "FAILED" for run in runs)
    assert mlflow.active_run() is None


def test_model_reload_and_registry(store, tmp_path):
    uri, client, experiment_id = store
    X = pd.DataFrame({"f1": np.linspace(-1, 1, 30)})
    y = (X["f1"] > 0).astype(int)
    model = XGBClassifier(n_estimators=3, max_depth=2).fit(X, y)
    tracker = ExperimentTracker(
        tracking_uri=uri, experiment_name="test", registered_model_name="direction-test"
    )
    with tracker.run({"stage": "explain", "horizon": 3}):
        with tracker.stage("explain", tmp_path / "metrics"):
            tracker.log_model(model, X.head())
        loaded = mlflow.xgboost.load_model("models:/direction-test/1")
        np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
    models = client.search_model_versions("name = 'direction-test'")
    assert len(models) == 1
    assert len(client.search_runs([experiment_id])) == 2


def test_disabled_tracking_never_imports_mlflow(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("mlflow"):
            raise AssertionError("Disabled tracking imported MLflow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    tracker = ExperimentTracker(enabled=False)
    output = tmp_path / "absent"
    with tracker.run({"stage": "eda", "horizon": 3}):
        with tracker.stage("eda", output):
            tracker.log_params({"horizon": 3})
            tracker.log_metrics({"auc": 0.7})
            tracker.log_dict({}, "ignored.json")
            tracker.log_dataset(pd.DataFrame(), [])
            tracker.log_model(None, pd.DataFrame())
    assert not output.exists()


def test_environment_and_explicit_overrides(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-server:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "env-experiment")
    assert not ExperimentTracker().enabled
    tracker = ExperimentTracker(
        enabled=True, tracking_uri="sqlite:///explicit.db", experiment_name="explicit"
    )
    assert tracker.enabled
    assert tracker.tracking_uri == "sqlite:///explicit.db"
    assert tracker.experiment_name == "explicit"


def test_pipeline_tracks_data_loading_failure(store, tmp_path, monkeypatch):
    uri, client, experiment_id = store

    def fail_loading():
        raise ValueError("missing dataset")

    monkeypatch.setattr(pipeline, "load_raw_dataset", fail_loading)
    with pytest.raises(ValueError, match="missing dataset"):
        pipeline.run_pipeline(
            stage="eda",
            out_dir=tmp_path / "output",
            tracking_uri=uri,
            experiment_name="test",
            use_mlflow=True,
        )
    runs = client.search_runs([experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"


def test_pipeline_dispatch_and_predictions(store, tmp_path, monkeypatch):
    uri, client, experiment_id = store
    frame = pd.DataFrame({"f1": [0.0, 1.0], "target": [0, 1]})
    monkeypatch.setattr(pipeline, "load_raw_dataset", lambda: frame)
    monkeypatch.setattr(
        pipeline, "prepare_modeling_dataset", lambda *args, **kwargs: (frame, ["f1"])
    )

    def evaluate(df, features, **kwargs):
        kwargs["tracker"].log_metrics({"logloss_mean": 0.5})
        return frame, pd.DataFrame({"Valor": [0.5]}, index=["LogLoss"])

    monkeypatch.setattr(pipeline, "run_walk_forward_evaluation", evaluate)
    pipeline.run_pipeline(
        stage="walk-forward",
        out_dir=tmp_path / "output",
        tracking_uri=uri,
        experiment_name="test",
        use_mlflow=True,
    )
    child = next(
        run for run in client.search_runs([experiment_id]) if "mlflow.parentRunId" in run.data.tags
    )
    assert child.data.metrics["logloss_mean"] == 0.5
    assert {item.path for item in client.list_artifacts(child.info.run_id)} == {
        "predictions.csv",
        "scorecard.csv",
    }
