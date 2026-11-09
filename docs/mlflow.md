# MLflow experiment tracking

Tracking is enabled by default for `models/run_pipeline.py` and `models/predictions.py`.
Install the project dependencies, then run commands from the repository root:

```bash
python -m pip install -e ".[dev]"
python models/run_pipeline.py --stage all --run-name baseline-h36
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///.mlflow/mlflow.db
```

Open <http://127.0.0.1:5000> to compare runs. The default experiment is
`financial-forecasting-xgboost`. Metadata and the optional Model Registry use
`.mlflow/mlflow.db`; model files and diagnostics use `.mlflow/artifacts/`.
Both locations are resolved against the repository, regardless of the training
command's working directory. Start the UI command above **from the repository root**.
Cleaning `models/metrics/` does not delete previous experiments.

## What each run contains

Each invocation creates a parent run containing pipeline parameters, package versions,
the actual feature list, dataset date range, row count, types and SHA-256 fingerprint.
Only the requested stages create child runs:

| Child run | Logged information |
| --- | --- |
| `eda` | Correlation and target distribution figures |
| `walk-forward` | Mean validation metrics, baselines, ROI, metric histories indexed by fold, fold date ranges, actual model parameters and thresholds, predictions and scorecard CSVs, diagnostic figures |
| `rollout` | Out-of-sample metrics, baselines, ROI, threshold, training end date, parameters, predictions CSV and diagnostic figures; insufficient history is recorded as skipped |
| `explain` | In-sample diagnostics, latest probability, SHAP figures and final XGBoost model with input example, signature and pinned inference dependencies |

Only files created or updated by a stage are uploaded, so `--no-clean` does not attach
old figures from other stages. Undefined or infinite metrics are omitted. Exceptions
propagate and mark the affected stage and parent run as `FAILED`.
Full-dataset diagnostics in `explain` are in-sample; compare generalization using
`walk-forward` and `rollout` instead.

Tracking uses explicit logging rather than autologging. Temporal splits, purging,
hyperparameter selection and training behavior remain controlled by the existing code.
Calling the validation functions directly does not create runs; the orchestrator passes
its tracker to them.

## Configuration

Copy `.env.example` to `.env` if desired. The pipeline reads the repository's `.env`
without overriding existing environment variables. Explicit Python/CLI arguments take
precedence over environment configuration.

| Environment variable | CLI argument | Default |
| --- | --- | --- |
| `MLFLOW_ENABLED` | `--no-mlflow` disables tracking | `true` |
| `MLFLOW_TRACKING_URI` | `--tracking-uri` | Repository-local SQLite |
| `MLFLOW_EXPERIMENT_NAME` | `--experiment-name` | `financial-forecasting-xgboost` |
| `MLFLOW_RUN_NAME` | `--run-name` | Stage and horizon |
| `MLFLOW_REGISTERED_MODEL_NAME` | `--registered-model-name` | Unset: save model without registry registration |

```bash
python models/run_pipeline.py --stage walk-forward --experiment-name horizon-comparison --horizon 12
python models/run_pipeline.py --stage eda --no-mlflow
python models/run_pipeline.py --stage explain --registered-model-name financial-direction-xgboost
```

Registration creates a version of the final model; it does not promote or deploy it.
An unavailable configured tracking server fails the execution rather than silently
discarding its tracking data. Use `--no-mlflow` to run without MLflow or its installation.

## Docker tracking server

The optional Compose profile starts MLflow independently of Kafka, Spark and Airflow:

```bash
docker compose --env-file .env -f docker/docker-compose.yml --profile mlflow up -d mlflow
python models/run_pipeline.py --stage all --tracking-uri http://127.0.0.1:5000
```

If no `.env` exists, copy `.env.example` first. The UI is available at
<http://127.0.0.1:5000>. The `mlflow_data` volume persists SQLite metadata and artifacts;
the server proxies artifact uploads, so clients do not need access to its filesystem.
Docker runs use a separate store from the default local SQLite database. Always point
training at the store whose UI you intend to inspect.

## Load a registered model

After training with `--registered-model-name financial-direction-xgboost`:

```python
import mlflow
import mlflow.xgboost

# Run from the repository root, or use the remote tracking server URI.
mlflow.set_tracking_uri("sqlite:///.mlflow/mlflow.db")
model = mlflow.xgboost.load_model("models:/financial-direction-xgboost/1")
# X must contain the same ordered, engineered feature columns used for training.
probabilities = model.predict_proba(X)[:, 1]
```

The saved classifier consumes engineered features; preprocessing remains in
`models/src/features.py`. For an unregistered model, copy its model URI from the
`explain` run in the MLflow UI and pass it to `mlflow.xgboost.load_model`.

See the official [XGBoost flavor API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.xgboost.html)
and [tracking server documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).
