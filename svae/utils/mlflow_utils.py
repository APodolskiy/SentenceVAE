import git
from pathlib import Path
from typing import Union, Optional, Tuple, Dict

from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT, MLFLOW_GIT_BRANCH


def get_experiment_id(mlflow_client: MlflowClient, experiment_name: str) -> int:
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow_client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


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


def get_git_tags(path: Union[str, Path]) -> Optional[Dict]:
    tags = None
    git_info = get_git_info(path)
    if git_info is not None:
        tags = {key: value for key, value in zip([MLFLOW_GIT_COMMIT, MLFLOW_GIT_BRANCH], git_info)}
    return tags
