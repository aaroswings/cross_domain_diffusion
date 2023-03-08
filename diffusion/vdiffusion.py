import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, asdict
from tqdm import tqdm
from typing import Any, Mapping, List, Optional, Tuple

from diffusion.util import (
    t_to_alpha_sigma, 
    replace_eps_noise,
    sigma_dynamic_thresholding
)

@dataclass()
class VDiffusionConfig:
    name: str = 'vdiffusion'
    sample_steps: int = 1000
    sample_intermediates_every: int = 200
    loss_scales: int = 1
    eps_replace_variance: float = 0.0
    normalize_eps_pred: bool = True
    channel_split: int = 3
    xpred_to_x0_channels: Tuple[int] = None
    dynamic_thresholding_q_start: float = 0.999
    clip_x0_pred: bool = False
    do_sigma_dynamic_thresholding: bool = True

    def build(self):
        if self.name == 'vdiffusion':
            return VDiffusion(self)

class VDiffusion(nn.Module):
    def __init__(self, config: VDiffusionConfig):
        super().__init__()
        self.init_config_ = config
        for k, v in asdict(config).items():
            self.__setattr__(k, v)

    def loss(self, net, batch: torch.Tensor) -> torch.Tensor:
        x0 = batch.to(next(net.parameters()).device)

        t = torch.rand(x0.size(0)).cuda()
        alpha, sigma = t_to_alpha_sigma(t)
        eps = torch.randn_like(x0)

        z_t = alpha * x0 + sigma * eps
        v_target = alpha * eps - sigma * x0

        with torch.cuda.amp.autocast():
            v_pred = net(z_t, t)

            loss = F.mse_loss(v_pred, v_target)
            for _ in range(self.loss_scales - 1):
                v_pred = F.avg_pool2d(v_pred, 2)
                v_target = F.avg_pool2d(v_target, 2)
                loss += F.mse_loss(v_pred, v_target)

        return loss

    def sample(self, net, batch: torch.Tensor, guided: bool = False) -> List[torch.Tensor]:
        x0 = batch.to(next(net.parameters()).device)
        ret = []

        ts = torch.linspace(1., 0., self.sample_steps + 1).to(next(net.parameters()).device)
        ts_next = ts[1:]
        ts = ts[:-1]

        eps = torch.randn_like(x0)
        z_t = eps

        print('Running diffusion sampling process.')
        for i, (t_now, t_next) in enumerate(tqdm(zip(ts, ts_next))):
            alpha, sigma = t_to_alpha_sigma(t_now)
            alpha_next, sigma_next = t_to_alpha_sigma(t_next)
            
            with torch.cuda.amp.autocast():
                v_pred = net(z_t, t_now)

            x0_pred = alpha * z_t - sigma * v_pred
            eps_pred = sigma * z_t + alpha * v_pred

            if self.clip_x0_pred:
                x0_pred = torch.clamp(x0_pred, -1., 1.)

            if guided:
                x0_pred[:, self.xpred_to_x0_channels, :, :] = x0[:, self.xpred_to_x0_channels, :, :]

            if self.eps_replace_variance:
                eps_pred = replace_eps_noise(eps_pred, self.eps_replace_variance)

            if self.normalize_eps_pred:
                eps_pred = eps_pred / torch.clamp(eps_pred.std(axis=(1, 2, 3), keepdims=True), min=1e-9, max=None)

            z_t = alpha_next * x0_pred + sigma_next * eps_pred
            if self.do_sigma_dynamic_thresholding:
                z_t = sigma_dynamic_thresholding(z_t, sigma_next)

            if i % self.sample_intermediates_every == 0 or i == len(ts_next) - 1:
                ret.append(x0_pred)

        return ret
