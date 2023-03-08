import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

class LinearLR(nn.Module):
    def __init__(
            self, 
            optimizer,
            lr_start=0.0, 
            lr_peak=1e-4, 
            lr_end=1e-5,
            warmup_steps=1000,
            total_steps=10000
        ):
        super().__init__()
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_peak = lr_peak
        self.lr_end = lr_end
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.register_buffer('lrs', self.make_lr_steps())

    def make_lr_steps(self):
        lr_warmup = torch.linspace(self.lr_start, self.lr_peak, self.warmup_steps)
        lr_rest = torch.linspace(self.lr_peak, self.lr_end, self.total_steps - self.warmup_steps)
        return torch.cat([lr_warmup, lr_rest])

    def update(self, step: int):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lrs[min(step, len(self.lrs) - 1)]

        