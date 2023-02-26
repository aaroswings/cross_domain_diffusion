import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import chunk_and_cat

import math
import os

from typing import Optional, Tuple

def get_alphas_sigmas(t: torch.Tensor):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def alpha_sigma_to_t(alpha, sigma) -> torch.Tensor:
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


def get_t_schedule(t) -> torch.Tensor:
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)


@torch.no_grad()
def get_noise(x, repeat_noise) -> torch.Tensor:
    assert x.size(1) % repeat_noise == 0
    noise = torch.randn(x.size(0), int(x.size(1) / repeat_noise), x.size(2), x.size(3)).to(x.device)
    noise = noise.repeat(1, repeat_noise, 1, 1)
    return noise


def v_loss(
        net: nn.Module, 
        x: torch.Tensor, 
        c: torch.Tensor, 
        repeat_noise: int = 1, # shared latent
        loss_scales: int = 1 # multiscale loss
    ) -> torch.Tensor:

    t = get_t_schedule(torch.rand(x.size(0), 1).to(x.device)) # batch of timesteps
    alpha, sigma = get_alphas_sigmas(t)
    noise = get_noise(x, repeat_noise)
    inputs = alpha * x + noise * sigma
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
    eta=0.0,
    save_dir=None
):
    if a_guide is not None and b_guide is not None:
        assert a_guide.shape == b_guide.shape

    if a_guide is not None:
        batch_size = a_guide.size(0)
    else:
        a_guide = torch.zeros(batch_size, 3, size, size).to(net.device)

    if b_guide is not None:
        batch_size = a_guide.size(0)
    else:
        b_guide = torch.zeros(batch_size, 3, size, size).to(net.device)

    x = torch.cat([a_guide, b_guide], dim=1)
    noise = get_noise(x, 2)

    ts = torch.linspace(t_start, 0, num_steps).to(net.device)
    ts_next = ts[1:]
    ts = ts[:-1]

    for i, (t, t_next) in enumerate(zip(ts, ts_next)):
        t = get_t_schedule(t * torch.ones(batch_size, 1))
        t_next = get_t_schedule(t * torch.ones(batch_size, 1))

        alpha, sigma = get_alphas_sigmas(t)
        alpha_next, sigma_next = get_alphas_sigmas(t_next)

        inputs = alpha * x + noise * sigma

        with torch.cuda.amp.autocast():
            v = net(x, t, None)

        pred = x * alpha - v * sigma
        eps = x * sigma + v * alpha

        if t > unfreeze_b_at:
            # Ignore the model's predictions for these channels.
            pred[:, 3:] = b_guide
            eps[:, 3:] = noise[:, 3:]

        if i < size.size(0) - 1:
            ddim_sigma = eta * (sigma_next**2 / sigma**2).sqrt() * \
                    (1 - alpha**2 / alpha_next**2).sqrt()
            adjusted_sigma = (sigma_next**2 - ddim_sigma**2).sqrt()
            x = pred * alpha_next + eps * adjusted_sigma
            if eta:
                new_noise = get_noise(x, 2) * ddim_sigma
                x += new_noise
                
                # double check this?
                noise = noise * (alpha + adjusted_sigma) + new_noise

        # It's clunky to depend on chunk_and_cat in the dataset file, but otherwise memory would blow up
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            for batch_i in range(x.size(0)):
                img = chunk_and_cat(x[batch_i])
                img.save(os.path.join(save_dir, f"sample_{batch_i}_at_timestep_{t}.jpeg"))

    return pred



