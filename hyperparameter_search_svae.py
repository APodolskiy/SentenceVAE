from pathlib import Path
import tempfile
from typing import Dict, Optional

from sklearn.model_selection import ParameterGrid

import mlflow
from mlflow.entities import Run
from mlflow.tracking.client import MlflowClient

from svae.utils.training import Params
from train_svae import train


def hyperparameter_search(save_dir: str, experiment_name: str, params: Params):
    # Creating client
    save_dir = Path(save_dir)
    mlflow.set_tracking_uri(str(save_dir))
    save_dir.mkdir(parents=True)
    mlflow_client = MlflowClient(tracking_uri=str(save_dir))
    # Getting experiment
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow_client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    # Creating run under the specified experiment
    run: mlflow.entities.Run = mlflow_client.create_run(experiment_id=experiment_id)
    # Creating experiment
    # TODO: code for performing experiment
    train_dir = tempfile.mkdtemp()
    train(train_dir=train_dir, params=params)

    mlflow_client.log_artifacts(run.info.run_uuid, train_dir)
    mlflow_client.set_terminated(run.info.run_uuid)


def log_params(client: MlflowClient, run: mlflow.entities.Run, params: Dict):
    for key, value in params.items():
        client.log_param(run_id=run.info.run_uuid, key=key, value=value)


def log_metrics(client: MlflowClient, run: mlflow.entities.Run,
                metrics: Dict, step: Optional[int] = None):
    for key, value in metrics.items():
        client.log_metric(run_id=run.info.run_uuid, key=key, value=value, step=step)


if __name__ == '__main__':
    pass
