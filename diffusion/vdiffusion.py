import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import chunk_and_cat

import math
import numpy as np
import os

from typing import Optional, Tuple

def get_alphas_sigmas(logsnr_t: torch.Tensor):
    return torch.sigmoid(logsnr_t).sqrt(), torch.sigmoid(-logsnr_t).sqrt()

"""
Referenceing the paper "On the Importance of Noise Scheduling for Diffusion Models"
"""
@torch.no_grad()
def get_noise(x, repeat_noise) -> torch.Tensor:
    assert x.size(1) % repeat_noise == 0
    noise = torch.randn(x.size(0), int(x.size(1) / repeat_noise), x.size(2), x.size(3)).to(x.device)
    noise = noise.repeat(1, repeat_noise, 1, 1)
    return noise


def cosine_schedule(t: torch.Tensor, start=0, end=1, tau=1, clip_min=1e-9):
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.clip(output, clip_min, 1.)


def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    if t.dim() == 0:
        t = t * torch.ones(1).to(t.device)
    return torch.cos(t * math.pi / 2)[:, None, None, None], torch.sin(t * math.pi / 2)[:, None, None, None]


def v_loss(
        net: nn.Module, 
        x: torch.Tensor, 
        c: torch.Tensor, 
        repeat_noise: int = 1, # shared latent
        loss_scales: int = 1, # multiscale loss
        normalize=True
    ) -> torch.Tensor:

    t = cosine_schedule(torch.rand(x.size(0)).to(x.device)) # batch of timesteps
    alpha, sigma = get_alphas_sigmas(t)

    noise = get_noise(x, repeat_noise)
    inputs = alpha * x + noise * sigma
    inputs = inputs / inputs.std(axis=(1, 2, 3), keepdims=True) if normalize else inputs
    targets = alpha * noise - x * sigma

    with torch.cuda.amp.autocast():
        pred = net(inputs, t, c)
        loss = torch.zeros(1).to(x.device)

        for _ in range(loss_scales):
            loss += F.mse_loss(pred, targets) / loss_scales
            pred = F.avg_pool2d(pred, 2)
            targets = F.avg_pool2d(targets, 2)

    return loss


def v_paired_sample(
    net: nn.Module,
    size: int = 256,
    a_guide: Optional[torch.tensor] = None,
    b_guide: Optional[torch.tensor] = None,
    t_start: float = 1.0,
    unfreeze_b_at: float = 1.0,
    batch_size: int = 4,
    num_steps: int = 1000,
    normalize=True,
    eta=0.0,
    save_dir=None,
    device=None
):
    if a_guide is not None and b_guide is not None:
        assert a_guide.shape == b_guide.shape

    if a_guide is not None:
        batch_size = a_guide.size(0)
    else:
        a_guide = torch.zeros(batch_size, 3, size, size).to(device)

    if b_guide is not None:
        batch_size = a_guide.size(0)
    else:
        b_guide = torch.zeros(batch_size, 3, size, size).to(device)

    x = torch.cat([a_guide, b_guide], dim=1)
    noise = get_noise(x, 2)

    ts = cosine_schedule(torch.linspace(t_start, 0, num_steps).to(device))
    ts_next = ts[1:]
    ts = ts[:-1]

    for i, (t, t_next) in enumerate(zip(ts, ts_next)):
        alpha, sigma = get_alphas_sigmas(t)
        alpha_next, sigma_next = get_alphas_sigmas(t_next)

        inputs = alpha * x + noise * sigma
        inputs = inputs / inputs.std(axis=(1, 2, 3), keepdims=True) if normalize else inputs

        with torch.cuda.amp.autocast():
            v = net(x, t, None)

        pred = x * alpha - v * sigma
        eps = x * sigma + v * alpha

        if t > unfreeze_b_at:
            # Ignore the model's predictions for these channels.
            pred[:, 3:] = b_guide
            eps[:, 3:] = noise[:, 3:]

        if i < ts.size(0) - 1:
            ddim_sigma = eta * (sigma_next**2 / sigma**2).sqrt() * \
                    (1 - alpha**2 / alpha_next**2).sqrt()
            adjusted_sigma = (sigma_next**2 - ddim_sigma**2).sqrt()
            x = pred * alpha_next + eps * adjusted_sigma
            if eta:
                new_noise = get_noise(x, 2) * ddim_sigma
                x += new_noise
                
                # double check this? 
                # it's for the "ignore models predictions if unfreeze b" step just above
                noise = noise * (alpha + adjusted_sigma) + new_noise

        # It's clunky to depend on chunk_and_cat in the dataset file, but otherwise memory would blow up
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            for batch_i in range(x.size(0)):
                img = chunk_and_cat(x[batch_i])
                img.save(os.path.join(save_dir, f"sample_{batch_i}_at_timestep_index_{i}.jpeg"))

    return pred



