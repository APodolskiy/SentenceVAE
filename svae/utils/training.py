from pathlib import Path
import shutil
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


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
