"""Explicit MLflow tracking for the research pipeline (no training autologging)."""

import hashlib
import math
import os
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pandas as pd

from .config import MODELS_DIR

DEFAULT_EXPERIMENT = "financial-forecasting-xgboost"
TRACKING_DIR = MODELS_DIR.parent / ".mlflow"


class ExperimentTracker:
    """Own a pipeline run and its stage/fold runs; disabled mode does no I/O."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        registered_model_name: Optional[str] = None,
    ) -> None:
        self.enabled = (
            os.getenv("MLFLOW_ENABLED", "true").lower() not in ("0", "false", "no", "off")
            if enabled is None
            else enabled
        )
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.experiment_name = (
            experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT
        )
        self.run_name = run_name or os.getenv("MLFLOW_RUN_NAME")
        self.registered_model_name = registered_model_name or os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME"
        )
        self._mlflow: Any = None

    @contextmanager
    def run(self, params: Dict[str, Any]) -> Iterator["ExperimentTracker"]:
        """Start tracking before data loading so errors mark the run as FAILED."""
        if not self.enabled:
            yield self
            return

        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow is required for tracking. Install requirements.txt or use --no-mlflow."
            ) from exc

        # Avoid changing an enclosing application's tracking store or active run.
        if mlflow.active_run() is not None:
            raise RuntimeError("Finish the active MLflow run before starting the pipeline.")
        previous_uri = mlflow.get_tracking_uri()
        self._mlflow = mlflow
        try:
            if self.tracking_uri is None:
                TRACKING_DIR.mkdir(parents=True, exist_ok=True)
                self.tracking_uri = "sqlite:///" + (TRACKING_DIR / "mlflow.db").as_posix()
                mlflow.set_tracking_uri(self.tracking_uri)
                if mlflow.get_experiment_by_name(self.experiment_name) is None:
                    mlflow.create_experiment(
                        self.experiment_name,
                        artifact_location=(TRACKING_DIR / "artifacts").as_uri(),
                    )
            else:
                mlflow.set_tracking_uri(self.tracking_uri)
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = (
                experiment.experiment_id
                if experiment is not None
                else mlflow.create_experiment(self.experiment_name)
            )
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=self.run_name or f"{params['stage']}-h{params['horizon']}",
                tags={"pipeline": DEFAULT_EXPERIMENT, "stage": params["stage"]},
            ) as active:
                self.log_params(params)
                self.log_dict(
                    {name: version(name) for name in ("mlflow", "xgboost", "pandas", "numpy")},
                    "environment.json",
                )
                print(f"MLflow run: {active.info.run_id} | Experiment: {self.experiment_name}")
                yield self
        finally:
            self._mlflow = None
            mlflow.set_tracking_uri(previous_uri)

    @contextmanager
    def stage(self, name: str, out_dir: Path) -> Iterator["ExperimentTracker"]:
        """Track only files written by this stage, excluding stale --no-clean outputs."""
        if self._mlflow is None:
            yield self
            return
        before = self._snapshot(out_dir)
        with self._mlflow.start_run(
            experiment_id=self._mlflow.active_run().info.experiment_id,
            run_name=name,
            nested=True,
            tags={"stage": name},
        ):
            yield self
            for path, stamp in self._snapshot(out_dir).items():
                if before.get(path) != stamp:
                    relative_parent = path.relative_to(out_dir).parent
                    artifact_path = (Path("diagnostics") / relative_parent).as_posix()
                    self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    @staticmethod
    def _snapshot(directory: Path) -> Dict[Path, tuple]:
        return {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in directory.rglob("*")
            if path.is_file()
        }

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_params(
                {key: value for key, value in params.items() if value is not None}
            )

    def log_metrics(self, metrics: Dict[str, Any], step: int = 0) -> None:
        if self._mlflow is not None:
            finite = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    finite[key] = float(value)
            if finite:
                self._mlflow.log_metrics(finite, step=step)

    def log_dict(self, data: Dict[str, Any], artifact_file: str) -> None:
        if self._mlflow is not None:
            self._mlflow.log_dict(data, artifact_file)

    def log_frame(self, frame: pd.DataFrame, artifact_file: str) -> None:
        if self._mlflow is not None:
            self._mlflow.log_text(frame.to_csv(index=True), artifact_file)

    def log_dataset(self, df: pd.DataFrame, features: list) -> None:
        if self._mlflow is None:
            return
        data = df[features + ["target"]]
        digest = hashlib.sha256(pd.util.hash_pandas_object(data, index=True).values.tobytes())
        digest.update(data.dtypes.to_string().encode())
        self.log_params({"observations": len(df), "n_features": len(features)})
        self.log_dict(
            {
                "features": features,
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "sha256": digest.hexdigest(),
            },
            "dataset.json",
        )

    def log_model(self, model: Any, example: pd.DataFrame) -> None:
        if self._mlflow is None:
            return
        import mlflow.xgboost
        from mlflow.models import infer_signature

        self.log_params(model.get_params())
        mlflow.xgboost.log_model(
            model,
            name="model",
            input_example=example,
            signature=infer_signature(example, model.predict(example)),
            registered_model_name=self.registered_model_name,
            pip_requirements=[
                f"{name}=={version(name)}"
                for name in ("xgboost", "scikit-learn", "numpy", "pandas", "scipy")
            ],
        )
