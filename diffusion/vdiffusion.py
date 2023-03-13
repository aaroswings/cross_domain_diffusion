import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
from PIL import Image

from dataclasses import dataclass, asdict
from tqdm import tqdm
from typing import Any, Optional, List, Union, Tuple

def t_to_alpha_sigma(t):
    if not isinstance(t, torch.Tensor):
        t = torch.Tensor(t)
    if t.dim() == 0:
        t = t.view(1, 1, 1, 1)
    elif t.dim() == 1:
        t = t[:, None, None, None]
    else:
        raise ValueError('phi should be either a 0-dimensional float or 1-dimensional float array')
    
    clip_min = 1e-9
    
    alpha = torch.clip(torch.cos(t * math.pi / 2), clip_min, 1.)
    sigma = torch.clip(torch.sin(t * math.pi / 2), clip_min, 1.)

    return alpha, sigma

def quantile_dynamic_clip(x, q: float):
    """
    Intended as an option for z_t clipping. (Imagen)
    """
    s = torch.quantile(x.view(x.size(0), -1).abs(), q, dim=1).max()
    low_bound = torch.min(-s, -torch.ones_like(s))
    high_bound = torch.max(s, torch.ones_like(s))
    return torch.clip(x, low_bound, high_bound)

def sigma_dynamic_clip(x, sigma):
    """
    Intended as an option for x0 clipping.
    sigma = 0 at t = 0, clip to [-1, 1]
    """
    minmax = 2. * sigma + 1.
    return torch.clip(x, -minmax, minmax)

def t_to_step(t, ts):
    t_idx = (t * ts.size(0)).to(torch.long)
    return t_idx

@dataclass()
class VDiffusionConfig:
    name: str = 'vdiffusion'
    loss_type: str = 'v'
    t_steps: int = 1000
    loss_scales: int = 1 # multiscale loss
    sample_channel_split: int = 0 # concatenated samples
    sample_frozen_channels: Tuple[int] = None
    sample_do_quantile_dynamic_clip: bool = False
    sample_quantile_dynamic_clip_q: float = 0.995
    sample_intermediates_every: int = 200
    sample_clip_x0_pred_method: str = False # true/false or 'sigma'
    sample_eta: float = 0.0 
    scheduled_sampling_weight_start: float = 0.0
    scheduled_sampling_weight_end: float = 0.8 # set to 0 as well for no scheduled sampling

    def build(self):
        if self.name == 'vdiffusion':
            return VDiffusion(self)
        
class VDiffusion(nn.Module):
    def __init__(self, config: VDiffusionConfig):
        super().__init__()
        self.init_config_ = config
        for k, v in asdict(config).items():
            self.__setattr__(k, v)
        self.register_buffer('ts', torch.linspace(0, 1, self.t_steps + 1))
        
    @torch.no_grad()
    def scheduled_sampling_zt(self, net, x0, eps, t, t_idx, train_step, total_steps):
        device = net.get_device()
        batch_size = x0.size(0)
        # increasing over training
        m_current_weight = self.scheduled_sampling_weight_start + train_step / total_steps * \
                (self.scheduled_sampling_weight_end - self.scheduled_sampling_weight_start)
        
        mt_max = 1 - t.max()
        m_t = torch.clip((torch.randn(1).to(device) * mt_max * m_current_weight).abs(), 0, mt_max)
        # convert continuous m_t in [0, 1] to discrete timestep
        m_steps = (m_t * (self.ts.size(0) - 1)).to(torch.long).item() # number of steps to take

        if self.sample_frozen_channels is not None:
            # randomly choose some samples in the batch to freeze
            samples_guided_by_frozen_channels = torch.randint(low=0, high=batch_size, size=(batch_size // 2,))
        else:
            samples_guided_by_frozen_channels = None

        assert((t_idx + m_steps).max() < self.ts.size(0))
        xs, zs = self.sample(net, x0, eps, intermediates_every=m_steps + 1,
            start_step=t_idx + m_steps, end_step = t_idx, sample_eta=0.0,
            batch_subset_guided_by_frozen_channels=samples_guided_by_frozen_channels)
        z_t = zs[-1]

        return z_t

    def get_loss_params(self, x0, eps, z_t, v_pred, t_idx):
        t = self.ts[t_idx]

        alpha, sigma = t_to_alpha_sigma(t)
        if self.loss_type == 'v':
            target = alpha * eps - sigma * x0 # v
            y = v_pred
        elif self.loss_type == 'x0':
            # Predict the (maybe normalized) x0 reconstruction
            target = x0
            y = alpha * z_t - sigma * v_pred
        elif self.loss_type == 'eps':
            target = eps
            y = sigma * z_t + alpha * v_pred
        elif self.loss_type == 'v_next':
            t_next = self.ts[t_idx - 1]
            alpha_next, sigma_next = t_to_alpha_sigma(t_next)
            # predict v of the next timestep t - 1
            target = alpha_next * eps - sigma_next * x0
            x0_pred = alpha * z_t - sigma * v_pred
            eps_pred = sigma * z_t + alpha * v_pred
            y = alpha_next * eps_pred - sigma_next * x0_pred
        elif self.loss_type == 'peek':
            # v loss for timesteps t, t - 1, and t + 1
            t_next = self.ts[t_idx - 1]
            alpha_next, sigma_next = t_to_alpha_sigma(t_next)

            t_prev = self.ts[t_idx + 1]
            alpha_prev, sigma_prev = t_to_alpha_sigma(t_prev)
            
            x0_pred = alpha * z_t - sigma * v_pred
            eps_pred = sigma * z_t + alpha * v_pred

            y = torch.cat([
                v_pred,
                alpha_next * eps_pred - sigma_next * x0_pred, # v next
                alpha_prev * eps_pred - sigma_prev * x0_pred # v prev
            ], dim=1)

            target = torch.cat([
                alpha * eps - sigma * x0,
                alpha_next * eps - sigma_next * x0,
                alpha_prev * eps - sigma_prev * x0,
            ], dim=1)

        return y, target

    def loss(
        self, 
        net, 
        batch: torch.Tensor, 
        train_step: int, 
        total_steps: int
        ) -> torch.Tensor:
        device = net.get_device()
        x0 = batch.to(device)
        eps = torch.randn_like(x0)
       
        t_idx = torch.randint(1, high=self.ts.size(0) - 1, size=(x0.size(0),))
        t = self.ts[t_idx] # t in [0, 1]
        alpha, sigma = t_to_alpha_sigma(t)

        do_scheduled_sampling = self.scheduled_sampling_weight_end > 0.0
        if do_scheduled_sampling:
            z_t = self.scheduled_sampling_zt(net, x0, eps, t, t_idx, train_step, total_steps)
        else:
            z_t = alpha * x0 + sigma * eps

        with torch.cuda.amp.autocast():
            v_pred = net(z_t, t)
            y, target = self.get_loss_params(x0, eps, z_t, v_pred, t_idx)

            loss = F.mse_loss(y, target)

            for _ in range(self.loss_scales - 1):
                y = F.avg_pool2d(y, 2)
                target = F.avg_pool2d(target, 2)
                loss += F.mse_loss(y, target)

        return loss
    

    def sample_clip(self, x0, clip_x0_pred_method, sigma: Optional[torch.tensor] == None):
        if clip_x0_pred_method == 'fixed':
            x0_clip = torch.clip(x0, -1, 1)
        elif clip_x0_pred_method == 'sigma':
            assert sigma is not None
            x0_clip = sigma_dynamic_clip(x0, sigma)
        elif clip_x0_pred_method == 'none':
            x0_clip = x0
        else:
            raise ValueError
        return x0_clip
        

    @torch.no_grad()
    def sample(
        self, 
        net: nn.Module, 
        x0: torch.Tensor, 
        eps: Optional[torch.Tensor] = None,
        guide_with_frozen_channels: bool = True,
        # default overrides
        intermediates_every: Optional[int] = None,
        start_step: Optional[Union[int, torch.tensor]] = None,
        end_step: Optional[Union[int, torch.tensor]] = None,
        frozen_channels: Optional[Tuple[int]] = None,
        do_quantile_dynamic_clip: Optional[bool] = None,
        clip_x0_pred_method: Optional[str] = None,
        sample_eta: Optional[float] = None,
        batch_subset_guided_by_frozen_channels: Optional[torch.tensor] = None,
        use_tqdm: bool = False
    ) -> Tuple[List[torch.tensor], List[torch.tensor]]:
        device = net.get_device()
        x0 = x0.to(device)
        eps = eps.to(device) if eps is not None else torch.randn_like(x0)

        intermediates_every = intermediates_every or self.sample_intermediates_every
        frozen_channels = frozen_channels or self.sample_frozen_channels
        do_quantile_dynamic_clip = do_quantile_dynamic_clip or self.sample_do_quantile_dynamic_clip
        clip_x0_pred_method = clip_x0_pred_method or self.sample_clip_x0_pred_method
        sample_eta = sample_eta or self.sample_eta

        # make sure that start_step, end_step is tensor of size (B,)
        start_step = start_step if start_step is not None else self.t_steps # countdown
        end_step = end_step if end_step is not None else 0

        if isinstance(start_step, int):
            start_step = torch.ones(x0.size(0)).long() * start_step
        if isinstance(end_step, int):
            end_step = torch.ones(x0.size(0)).long() * end_step

        assert start_step.size(0) == x0.size(0) and end_step.size(0) == x0.size(0)
        assert (start_step > 0).all()
        assert (start_step >= end_step).all()
        assert ((start_step - end_step) == (start_step - end_step)[0]).all() # same difference between start, end across batch

        n_steps = (start_step - end_step)[0]

        t = self.ts[start_step]
        alpha, sigma = t_to_alpha_sigma(t)
        z_t = alpha * x0 + sigma * eps

        ret_xs = [x0]
        ret_zs = [z_t]

        for i in tqdm(range(n_steps, 0, -1)) if use_tqdm else range(n_steps, 0, -1):
            step = end_step + i
            t = self.ts[step]
            alpha, sigma = t_to_alpha_sigma(t)

            with torch.cuda.amp.autocast():
                v_pred = net(z_t, t)
                x0_pred = alpha * z_t - sigma * v_pred
                eps_pred = sigma * z_t + alpha * v_pred

            x0_pred = self.sample_clip(x0_pred, clip_x0_pred_method, sigma)

            t_next = self.ts[step - 1]
            alpha_next, sigma_next = t_to_alpha_sigma(t_next)
            
            if guide_with_frozen_channels:
                assert frozen_channels is not None
                if batch_subset_guided_by_frozen_channels is not None:
                    x0_pred[batch_subset_guided_by_frozen_channels, frozen_channels, :, :] = x0[batch_subset_guided_by_frozen_channels, frozen_channels, :, :]
                else:
                    x0_pred[:, frozen_channels, :, :] = x0[:, frozen_channels, :, :]

            z_t = alpha_next * x0_pred + sigma_next * eps_pred

            if quantile_dynamic_clip:
                z_t = quantile_dynamic_clip(z_t, self.sample_quantile_dynamic_clip_q)  

            if i % self.sample_intermediates_every == 0 or i - 1 == 0:
                ret_xs.append(x0_pred)
                ret_zs.append(z_t)

        return ret_xs, ret_zs
        