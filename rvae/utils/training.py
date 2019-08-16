from collections import MutableMapping
from pathlib import Path
import shutil
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class Params(MutableMapping):
    DEFAULT_VALUE = object

    def __init__(self, params: Dict):
        self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__

    def get(self, key: str, default: Optional = DEFAULT_VALUE, param_type: Optional = None):
        if default is self.DEFAULT_VALUE:
            value = self.__dict__.get(key)
        else:
            value = self.__dict__.get(key, default)
        value = self._convert_value(value)
        if param_type is not None:
            value = self._convert_type(value, param_type)
        return value

    def pop(self, key: str, default: Optional = DEFAULT_VALUE, param_type: Optional = None):
        if default is self.DEFAULT_VALUE:
            value = self.__dict__.pop(key)
        else:
            value = self.__dict__.pop(key, default)
        value = self._convert_value(value)
        if param_type is not None:
            value = self._convert_value(value, param_type)
        return value

    def _convert_value(self, value):
        if isinstance(value, dict):
            return Params(value)
        if isinstance(value, list):
            value = [self._convert_value(item) for item in value]
        return value

    def _convert_type(self, value, _type):
        if value is None or value == 'None':
            return None
        if _type is bool:
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                if value == 'false':
                    return False
                if value == 'true':
                    return True
            raise ValueError(f"Can't convert {value} to type {_type}.")
        return _type(value)

    def __getitem__(self, key):
        if key in self.__dict__:
            return self._convert_value(self.__dict__[key])
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)
    """
    def __init__(self, params: Dict):
        self.params = params

    def pop(self, key: str, default: Any = DEFAULT_VALUE, param_type: Optional = None):
        if default is self.DEFAULT_VALUE:
            try:
                value = self.params.pop(key)
            except KeyError:
                raise KeyError(f"Missing parameter \"{key}\"")
        else:
            value = self.params.pop(key, default)
        value = self._convert_value(value)
        if param_type is not None:
            value = self._convert_value(value, param_type)
        return value

    def _convert_type(self, value, _type):
        if value is None or value == 'None':
            return None
        if _type is bool:
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                if value == 'false':
                    return False
                if value == 'true':
                    return True
            raise ValueError(f"Can't convert {value} to type {_type}.")
        return _type(value)

    def _convert_value(self, value):
        if isinstance(value, dict):
            return Params(value)
        if isinstance(value, list):
            value = [self._convert_value(item) for item in value]
        return value

    def __getitem__(self, key):
        if key in self.params:
            return self._convert_value(self.params[key])
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)
    """


class AggregateMetric:
    def __init__(self):
        self.total_value = 0

    def __call__(self, value: float):
        self.total_value += value

    def get_metric(self, reset: bool = False) -> float:
        total_value = self.total_value
        if reset:
            self.reset()
        return total_value

    def reset(self):
        self.total_value = 0


class AverageMetric:
    def __init__(self):
        self.steps = 0
        self.total_value = 0

    def __call__(self, value: float, num_steps: int = 1):
        self.steps += num_steps
        self.total_value += value

    def get_metric(self, reset: bool = False) -> float:
        average_value = self.total_value / self.steps if self.steps > 0 else 0
        if reset:
            self.reset()
        return average_value

    def reset(self):
        self.steps = 0
        self.total_value = 0


def save_checkpoint(state: Dict, save_dir: str, name: str = 'vae', is_best: bool = False):
    save_dir = Path(save_dir)
    checkpoint_path = save_dir / f"{name}.pt"
    if not save_dir.exists():
        print(f"There is no checkpoint directory! Creating new directory: {checkpoint_path}")
        checkpoint_path.mkdir(parents=True)
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = save_dir / f"best_{name}.pt"
        shutil.copyfile(checkpoint_path, best_model_path)


def load_checkpoint(checkpoint_path: str,
                    model: nn.Module,
                    optimizer: Optional[Optimizer] = None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ValueError(f"File doesn't exist: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optim_dict'])
    return state_dict
