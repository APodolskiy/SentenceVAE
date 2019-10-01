from argparse import ArgumentParser
from functools import partial
import json
from pathlib import Path
import tempfile
from typing import Dict, Optional

from _jsonnet import evaluate_snippet
from sklearn.model_selection import ParameterGrid

import mlflow
from mlflow.entities import Run
from mlflow.tracking.client import MlflowClient

from svae.utils.training import Params
from train_svae import train


def hyperparameter_search(save_dir: str, experiment_name: str, params_file: str, hyper_params_file: str):
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

    with open(hyper_params_file, 'r') as fp:
        hyper_params = json.load(fp)

    # TODO: add other types of hyperparameter search
    search_type = hyper_params.pop('type')
    if not search_type == 'grid':
        raise ValueError(f"Currently only grid search is being supported.")

    hyper_params_grid = ParameterGrid(hyper_params)

    for h_params in hyper_params_grid:
        override_params = {k: json.dumps(param) for k, param in h_params.items()}

        with open(params_file) as fp:
            params = json.loads(evaluate_snippet('config', fp.read, tla_codes=override_params))

        # Creating run under the specified experiment
        run: mlflow.entities.Run = mlflow_client.create_run(experiment_id=experiment_id)
        train_dir = tempfile.mkdtemp()
        train(train_dir=train_dir,
              config=params,
              force=True,
              metric_logger=partial(log_metrics, mlflow_client, run))

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
    parser = ArgumentParser(description='Hyperparameter search')
    parser.add_argument('--params', required=True, type=str,
                        help="Path to file with default parameters")
    parser.add_argument('--hyper-params', required=True, type=str,
                        help="Path to file with hyperparameters.")
    parser.add_argument('--experiment-name', required=True, type=str,
                        help="Experiment name")
    parser.add_argument('--tracking-uri', required=True, type=str,
                        help="MlFlow tracking uri")
    args = parser.parse_args()

    hyperparameter_search(save_dir=args.tracking_uri,
                          experiment_name=args.experiment_name,
                          params_file=args.params,
                          hyper_params_file=args.hyper_params)
