import math

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# TODO: flexible learning rate decay for each parameter group
class DecayLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, start_epoch: int, decay_interval: int = 1,
                 lr_multiplier: float = 0.1, last_epoch: int = -1):
        self.optimizer = optimizer
        if not lr_multiplier > 0:
            raise ValueError(f"LR multiplier must be positive.")
        self.lr_multiplier = lr_multiplier
        if not start_epoch >= 0:
            raise ValueError(f"Start epoch must be non-negative.")
        self.start_epoch = start_epoch
        if not decay_interval >= 1:
            raise ValueError(f"Decay interval must be greater than 1.")
        self.decay_interval = decay_interval
        self.last_epoch = last_epoch
        super().__init__(optimizer=self.optimizer, last_epoch=last_epoch)

    def state_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

    def get_lr(self) -> float:
        if self.start_epoch > self.last_epoch:
            return self.base_lrs
        deg = int(math.ceil((self.last_epoch - self.start_epoch + 1) / self.decay_interval))
        denom = 2 ** deg
        return [base_lr / denom for base_lr in self.base_lrs]
