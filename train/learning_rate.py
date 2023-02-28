import torch
import torch.nn as nn

class LinearLR(nn.Module):
    def __init__(self, 
        optimizer,
        lr_start=0.0, 
        lr_peak=1e-4, 
        lr_end=1e-5,
        warmup_steps=1000,
        total_steps=10000):
        super().__init__()
        self.optimizer = optimizer

        lr_warmup = torch.linspace(lr_start, lr_peak, warmup_steps)
        lr_cooldown = torch.linspace(lr_peak, lr_end, total_steps - warmup_steps)
        self.register_buffer('lrs', torch.cat([lr_warmup, lr_cooldown]))

        self.current_lr = self.lrs[0]
        self.update(0)

    def get(self):
        return self.current_lr

    def update(self, step: int):
        self.current_lr = self.lrs[step]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

        

