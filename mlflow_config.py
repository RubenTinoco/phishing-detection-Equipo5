from pathlib import Path

import mlflow


EXPERIMENT_NAME = "phishing-detection-sprint3-sprint4"
TRACKING_URI = "file:./mlruns"


def setup_mlflow(tracking_uri=TRACKING_URI, experiment_name=EXPERIMENT_NAME):
    """Configure a local MLflow tracking store and select the project experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return {
        "tracking_uri": tracking_uri,
        "tracking_path": str(Path("mlruns").resolve()),
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
    }
