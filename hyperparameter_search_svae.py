from argparse import ArgumentParser
from functools import partial
import json
from pathlib import Path
import tempfile
from typing import Dict, Optional, Tuple, Union

import git
from _jsonnet import evaluate_snippet
from sklearn.model_selection import ParameterGrid
import torch
import torch.multiprocessing as mp

import mlflow
from mlflow.entities import Run
from mlflow.tracking.client import MlflowClient

from train_svae import train
from svae.utils.mlflow_utils import get_experiment_id, get_git_tags

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def hyperparameter_search(save_dir: str, experiment_name: str, params_file: str, hyper_params_file: str):
    # Creating client
    save_dir = Path(save_dir)
    mlflow.set_tracking_uri(str(save_dir))
    mlflow_client = MlflowClient(tracking_uri=str(save_dir))
    # Getting experiment
    experiment_id = get_experiment_id(mlflow_client, experiment_name)

    with open(hyper_params_file, 'r') as fp:
        hyper_params = json.load(fp)

    # TODO: add other types of hyperparameter search
    search_type = hyper_params.pop('type')
    if not search_type == 'grid':
        raise ValueError(f"Currently only grid search is being supported.")

    hyper_params_grid = ParameterGrid(hyper_params)
    tags = get_git_tags(Path.cwd())

    # Prepare devices
    cuda_devices = hyper_params.pop('cuda_devices', None)

    if cuda_devices is not None:
        num_devices = len(cuda_devices)
    else:
        cuda_devices = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
        num_devices = 1

    if num_devices > 1:
        devices_queue = mp.Queue()
        for gpu_id in cuda_devices:
            devices_queue.put(torch.device(f'cuda:{gpu_id}'))

        # Parallel hyperparameter search
        pool = mp.Pool(processes=num_devices, initializer=init, initargs=(devices_queue,))
        pool.map(partial(run_experiment,
                         params_file=params_file,
                         tags=tags,
                         mlflow_client=mlflow_client,
                         experiment_id=experiment_id), hyper_params_grid)
        pool.close()
        pool.join()
    else:
        device = torch.device(cuda_devices)
        # Non-parallel hyperparameter search
        for h_params in hyper_params_grid:
            run_experiment(h_params=h_params,
                           params_file=params_file,
                           tags=tags,
                           mlflow_client=mlflow_client,
                           experiment_id=experiment_id)


def init(local_devices_queue):
    global global_devices_queue
    global_devices_queue = local_devices_queue


def run_experiment(h_params: Dict, params_file, mlflow_client, experiment_id, tags=None, device=None):
    if device is None:
        device = global_devices_queue.get()
    try:
        override_params = {k: json.dumps(param) for k, param in h_params.items()}

        with open(params_file) as fp:
            params = json.loads(evaluate_snippet('config', fp.read(), tla_codes=override_params))

        # Creating run under the specified experiment
        run: mlflow.entities.Run = mlflow_client.create_run(experiment_id=experiment_id, tags=tags)
        log_params(mlflow_client, run, h_params)
        status = None
        try:
            with tempfile.TemporaryDirectory() as train_dir:
                train(train_dir=train_dir,
                      config=params,
                      force=True,
                      metric_logger=partial(log_metrics, mlflow_client, run),
                      device=device,
                      verbose=False)
                mlflow_client.log_artifacts(run.info.run_uuid, train_dir)
        except Exception as e:
            print(f"Run failed! Exception occurred: {e}.")
            status = 'FAILED'
        mlflow_client.set_terminated(run.info.run_uuid, status=status)
    finally:
        global_devices_queue.put(device)


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
