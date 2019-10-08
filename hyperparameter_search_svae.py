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
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT, MLFLOW_GIT_BRANCH

from train_svae import train

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
    git_info = get_git_info(Path.cwd())

    # Prepare devices
    devices_queue = mp.Queue()
    cuda_devices = hyper_params.pop('cuda_devices', None)

    if cuda_devices is not None:
        num_devices = len(cuda_devices)
        for gpu_id in cuda_devices:
            devices_queue.put(torch.device(f'cuda:{gpu_id}'))
    else:
        devices_queue.put(torch.device('cpu'))
        num_devices = 1

    pool = mp.Pool(processes=num_devices, initializer=init, initargs=(devices_queue,))
    pool.map(partial(run_experiment,
                     params_file=params_file,
                     git_info=git_info,
                     mlflow_client=mlflow_client,
                     experiment_id=experiment_id), hyper_params_grid)
    pool.close()
    pool.join()

    # TODO: parallelize hyperparameter search on multiple GPUs
    """
    for h_params in hyper_params_grid:
        override_params = {k: json.dumps(param) for k, param in h_params.items()}

        with open(params_file) as fp:
            params = json.loads(evaluate_snippet('config', fp.read(), tla_codes=override_params))

        # Creating run under the specified experiment
        tags = None
        if git_info is not None:
            tags = {key: value for key, value in zip([MLFLOW_GIT_COMMIT, MLFLOW_GIT_BRANCH], git_info)}
        run: mlflow.entities.Run = mlflow_client.create_run(experiment_id=experiment_id, tags=tags)
        log_params(mlflow_client, run, h_params)
        status = None
        try:
            with tempfile.TemporaryDirectory() as train_dir:
                train(train_dir=train_dir,
                      config=params,
                      force=True,
                      metric_logger=partial(log_metrics, mlflow_client, run))
                mlflow_client.log_artifacts(run.info.run_uuid, train_dir)
        except Exception as e:
            print(f"Run failed! Exception occurred: {e}.")
            status = 'FAILED'
        mlflow_client.set_terminated(run.info.run_uuid, status=status)
    """


def init(local_devices_queue):
    global global_devices_queue
    global_devices_queue = local_devices_queue


def run_experiment(h_params: Dict, params_file, git_info, mlflow_client, experiment_id):
    device = global_devices_queue.get()
    try:
        override_params = {k: json.dumps(param) for k, param in h_params.items()}

        with open(params_file) as fp:
            params = json.loads(evaluate_snippet('config', fp.read(), tla_codes=override_params))

        # Creating run under the specified experiment
        tags = None
        if git_info is not None:
            tags = {key: value for key, value in zip([MLFLOW_GIT_COMMIT, MLFLOW_GIT_BRANCH], git_info)}
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


def get_git_info(path: Union[str, Path]) -> Optional[Tuple[str, str]]:
    """
    Mainly adaptation of mlflow.utils.context _get_git_commit function.
    :param path:
    :return:
    """
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        path = path.parent
    try:
        repo = git.Repo(path)
        commit = repo.head.commit.hexsha
        branch = repo.active_branch.name
        return commit, branch
    except (git.InvalidGitRepositoryError, git.GitCommandNotFound, ValueError, git.NoSuchPathError):
        return None


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
