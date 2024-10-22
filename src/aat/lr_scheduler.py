from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step
from torch.optim import Optimizer

from typing import Union, Callable, List

class WarmupLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: Union[List[int], int],
        start_lr_from: float=1e-6,
        last_epoch=-1,
        verbose="deprecated",
    ):
        self.optimizer = optimizer

        self.warmup_steps: List[int]

        if not isinstance(warmup_steps, list) and not isinstance(warmup_steps, tuple):
            self.warmup_steps = [warmup_steps] * len(optimizer.param_groups)
        else:
            self.warmup_steps = warmup_steps

        self.start_lr_from = start_lr_from
        if len(self.warmup_steps) != len(optimizer.param_groups):
            raise ValueError(
                f"Expected {len(optimizer.param_groups)} warmup_steps, but got {len(self.warmup_steps)}"
            )
        super().__init__(optimizer, last_epoch, verbose)


    def get_lr(self):
        """Compute learning rate."""
        _warn_get_lr_called_within_step(self)

        result_lr = []
        for warmup_steps, base_lr in zip(self.warmup_steps, self.base_lrs):
            if self._step_count > warmup_steps:
                result_lr.append(base_lr)
            else:
                result_lr.append(base_lr * self._step_count / warmup_steps)

        return result_lr

