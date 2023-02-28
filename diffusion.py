import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import chunk_and_cat

import math
import numpy as np
import os

from typing import Optional, Tuple


@torch.no_grad()
def get_noise(x, repeat_noise) -> torch.Tensor:
    assert x.size(1) % repeat_noise == 0
    noise = torch.randn(x.size(0), int(x.size(1) / repeat_noise), x.size(2), x.size(3)).to(x.device)
    noise = noise.repeat(1, repeat_noise, 1, 1)
    return noise


"""On the Importans of Noise Scheduling for Diffusion Models - Ting Chen"""
def get_alphas_sigmas(t):
    if not isinstance(t, torch.Tensor):
        t = torch.Tensor(t)
    clip_min = 1e-9
    linear_gamma = torch.clip(1 - t, clip_min, 1.)
    return linear_gamma.sqrt(), (1 - linear_gamma).sqrt()


def loss(
        net: nn.Module, 
        x0: torch.Tensor, 
        c: torch.Tensor, 
        # repeat_noise: int = 1, # shared latent
        loss_scales: int = 1, # multiscale loss
        x0_scale: float = 1.,
        normalize_x = True
    ) -> torch.Tensor:
    t = torch.rand(x.size(0)).to(x.device)
    alpha, sigma = get_alphas_sigmas(t)

    noise = torch.randn_like(x0)
    x = alpha * x0_scale * x0 + noise * sigma

    x = x / x.std(axis=(1, 2, 3), keepdims=True) if normalize_x else x

    with torch.cuda.amp.autocast():
        v_pred = net(x, t, c)
        # v-objective, epsilon loss as in simple diffusion
        noise_pred = x * sigma + v_pred * alpha
        loss = torch.zeros(1).to(x.device)

        for _ in range(loss_scales):
            loss += F.mse_loss(noise_pred, noise) / loss_scales
            noise_pred = F.avg_pool2d(noise_pred, 2)
            noise = F.avg_pool2d(noise, 2)

    return loss


def sample(
    net: nn.Module,
    x0: torch.tensor,
    steps: int = 1000,
    t_start: float = 1.0,
    ignore_pred_channels: Optional[Tuple[int]] = None,
    normalize_x = True,
    x0_scale: float = 1,
    eta = 0.0,
    return_intermediates_every: int = None
):
    """
    net: the denoising unet
    x0: x0 to base sample generation on, set to 0 if starting from t=1
    ignore_pred_channels: replace the neural network's predictions with x0 on these channels
    normalize_x: should be True if the network was trained on loss with normalize_x = True
    x0_scale: should match the x0_scale the net was trained with
    eta: sample eta, fresh noise added at each timestep during sampling
    return_intermediates_every: return a List of Images every k timesteps during sampling
    """

    ts = np.linspace(t_start, 0., steps)
    ts_next = ts[1:]

    alpha, sigma = get_alphas_sigmas(ts[0])
    noise = torch.randn_like(x0)
    x = alpha * x0_scale * x0 + noise * sigma


    for t_now, t_next in zip(ts[:-1], ts_next):
        x = x / x.std(axis=(1, 2, 3), keepdims=True) if normalize_x else x
        alpha, sigma = get_alphas_sigmas(t_now)
        alpha_next, sigma_next = get_alphas_sigmas(t_next)

        with torch.cuda.amp.autocast():
            v_pred = net(x, t_now, None)

        pred = x * alpha - v_pred * sigma
        eps = x * sigma + v_pred * alpha

        ddim_sigma = eta * (sigma_next**2 / sigma**2).sqrt() * \
                    (1 - alpha**2 / alpha_next**2).sqrt()
        adjusted_sigma = (sigma_next**2 - ddim_sigma**2).sqrt()
        x = pred * alpha_next + eps * adjusted_sigma

        if eta:
            new_noise = torch.randn_like(x0) * ddim_sigma
            x += new_noise

    return pred