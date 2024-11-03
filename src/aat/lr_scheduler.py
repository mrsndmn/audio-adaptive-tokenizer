from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step
from torch.optim import Optimizer

from typing import Union, Callable, List

class WarmupLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: Union[List[int], int],
        max_steps: Union[List[int], int],
        start_lr_from: float=1e-6,
        last_epoch=-1,
        verbose="deprecated",
    ):
        self.optimizer = optimizer

        self.warmup_steps: List[int]
        self.max_steps: List[int]

        if not isinstance(warmup_steps, list) and not isinstance(warmup_steps, tuple):
            self.warmup_steps = [warmup_steps] * len(optimizer.param_groups)
        else:
            self.warmup_steps = warmup_steps

        if not isinstance(max_steps, list) and not isinstance(max_steps, tuple):
            self.max_steps = [max_steps] * len(optimizer.param_groups)
        else:
            self.max_steps = max_steps


        self.start_lr_from = start_lr_from
        if len(self.warmup_steps) != len(optimizer.param_groups):
            raise ValueError(
                f"Expected {len(optimizer.param_groups)} warmup_steps, but got {len(self.warmup_steps)}"
            )
        if len(self.max_steps) != len(optimizer.param_groups):
            raise ValueError(
                f"Expected {len(optimizer.param_groups)} max_steps, but got {len(self.max_steps)}"
            )
        super().__init__(optimizer, last_epoch, verbose)


    def get_lr(self):
        """Compute learning rate."""
        _warn_get_lr_called_within_step(self)

        result_lr = []
        for warmup_steps, max_steps, base_lr in zip(self.warmup_steps, self.max_steps, self.base_lrs):
            assert warmup_steps < max_steps

            if self._step_count > max_steps:
                result_lr.append(self.start_lr_from)
            elif self._step_count > warmup_steps:
                current_annealing_step = self._step_count - warmup_steps
                annealing_steps_total = max_steps - warmup_steps
                annealing_step_decrement = (base_lr - self.start_lr_from) / annealing_steps_total
                result_lr.append(base_lr - current_annealing_step * annealing_step_decrement)
            else:
                result_lr.append(base_lr * self._step_count / warmup_steps)

        return result_lr

