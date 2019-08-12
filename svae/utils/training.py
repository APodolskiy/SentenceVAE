from pathlib import Path
from typing import Dict

import torch


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


def save_checkpoint(state: Dict, save_dir: str, name='vae'):
    save_dir = Path(save_dir)
    checkpoint_path = save_dir / f"{name}.bin"
    if not save_dir.exists():
        print(f"There is no checkpoint directory! Creating new directory: {checkpoint_path}")
        checkpoint_path.mkdir(parents=True)
    torch.save(state, checkpoint_path)
