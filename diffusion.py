import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from typing import List, Optional, Tuple


@torch.no_grad()
def get_noise(x, repeat_noise: int ) -> torch.Tensor:
    assert x.size(1) % repeat_noise == 0
    noise = torch.randn(x.size(0), int(x.size(1) / repeat_noise), x.size(2), x.size(3)).to(x.device)
    noise = noise.repeat(1, repeat_noise, 1, 1)
    return noise


def get_alphas_sigmas(t):
    # On the Importance of Noise Scheduling for Diffusion Models - Ting Chen
    if not isinstance(t, torch.Tensor):
        t = torch.Tensor(t)
    clip_min = 1e-9
    linear_gamma = torch.clip(1 - t, clip_min, 1.)
    alpha = linear_gamma.sqrt()
    sigma = (1 - linear_gamma).sqrt()

    # Have to get the correct dimensions to broadcast and multiply with x

    if alpha.dim() == 0:
        alpha = alpha.view(1, 1, 1, 1)
        sigma = sigma.view(1, 1, 1, 1)
    else:
        alpha = alpha[:, None, None, None]
        sigma = sigma[:, None, None, None]

    return alpha, sigma


def v_loss(
        net: nn.Module, 
        x0: torch.Tensor, 
        c: torch.Tensor, 
        # repeat_noise: int = 1, # shared latent
        loss_scales: int = 1, # multiscale loss
        x0_scale: float = 1.,
        normalize_x = True
    ) -> torch.Tensor:
    t = torch.rand(x0.size(0)).to(x0.device)
    alpha, sigma = get_alphas_sigmas(t)

    noise = torch.randn_like(x0)
    x = alpha * x0_scale * x0 + noise * sigma

    x = x / x.std(axis=(1, 2, 3), keepdims=True) if normalize_x else x

    with torch.cuda.amp.autocast():
        v_pred = net(x, t, c)
        # v-objective, epsilon loss as in "simple diffusion"
        noise_pred = x * sigma + v_pred * alpha
        loss = torch.zeros(1).to(x.device)

        for _ in range(loss_scales):
            loss += F.mse_loss(noise_pred, noise) / loss_scales
            noise_pred = F.avg_pool2d(noise_pred, 2)
            noise = F.avg_pool2d(noise, 2)

    return loss


def dynamic_thresholding(x, q=0.995):
    s = torch.quantile(x.view(x.size(0), -1).abs(), q, dim=1).max()
    x = torch.clip(x, -s, s,) / s
    return x


@torch.no_grad()
def v_sample(
    net: nn.Module,
    x0: torch.tensor,
    steps: int = 1000,
    t_start: float = 1.,
    ignore_pred_channels: Optional[Tuple[int]] = None,
    normalize_x = True,
    x0_scale: float = 1.,
    eta = 0.0,
    dynamic_thresholding_quantile: float = 1.0,
    intermediates_every: int = None,
    freeze_dims: Optional[Tuple[int]] = None
) -> List[torch.Tensor]:
    """
    net: the denoising unet
    x0: x0 to base sample generation on, set to 0 if starting from t=1
    ignore_pred_channels: replace the neural network's predictions with x0 on these channels
    normalize_x: should be True if the network was trained on loss with normalize_x = True
    x0_scale: should match the x0_scale the net was trained with
    eta: sample eta, fresh noise added at each timestep during sampling
    intermediates_every: return a List of Images every k timesteps during sampling
    """

    ts = torch.linspace(t_start, 0., steps).to(x0.device)
    ts_next = ts[1:]

    alpha, sigma = get_alphas_sigmas(ts[0])
    noise = torch.randn_like(x0)
    x = alpha * x0_scale * x0 + noise * sigma

    results = []

    print(f'\nSampling from t={t_start}')
    for i, (t_now, t_next) in enumerate(tqdm(zip(ts[:-1], ts_next))):
        x = x / x.std(axis=(1, 2, 3), keepdims=True) if normalize_x else x
        alpha, sigma = get_alphas_sigmas(t_now)
        alpha_next, sigma_next = get_alphas_sigmas(t_next)

        with torch.cuda.amp.autocast():
            v_pred = net(x, t_now, None)

        pred_x = x * alpha - v_pred * sigma
        pred_noise = x * sigma + v_pred * alpha

        ddim_sigma = eta * (sigma_next**2 / sigma**2).sqrt() * \
                    (1 - alpha**2 / alpha_next**2).sqrt()
        adjusted_sigma = (sigma_next**2 - ddim_sigma**2).sqrt()

        if freeze_dims is not None:
            pred_x[:, freeze_dims, :, :] = x0[:, freeze_dims, :, :]
            #pred_noise[:, freeze_dims, :, :] = noise[:, freeze_dims, :, :]

        x = pred_x * alpha_next + pred_noise * adjusted_sigma

        x = dynamic_thresholding(x, dynamic_thresholding_quantile)

        if eta:
            new_noise = torch.randn_like(x0) * ddim_sigma
            x += new_noise

        if i % intermediates_every == 0 or i == len(ts_next) - 1:
            results.append(pred_x / x0_scale)

    return results