import torch
import math
import os
import numpy as np
from PIL import Image
from typing import List, Tuple

def t_to_phi(t):
    return t * math.pi / 2

def phi_to_alpha_sigma(phi):
    if not isinstance(phi, torch.Tensor):
        phi = torch.Tensor(phi)

    clip_min = 1e-9

    if phi.dim() == 0:
        phi = phi.view(1, 1, 1, 1)
    elif phi.dim() == 1:
        phi = phi[:, None, None, None]
    else:
        raise ValueError('phi should be either a 0-dimensional float or 1-dimensional float array')
    
    alpha = torch.clip(torch.cos(phi), clip_min, 1.)
    sigma = torch.clip(torch.sin(phi), clip_min, 1.)

    return alpha, sigma

def t_to_alpha_sigma(t):
    phi = t_to_phi(t)
    return phi_to_alpha_sigma(phi)

def dynamic_thresholding(x, t, q=0.999):
    """
    At t=0, clip x to [-1, 1].
    """
    s = torch.quantile(x.view(x.size(0), -1).abs(), q, dim=1).max()
    low_bound = -s * t + -1. * (1. - t)
    high_bound = s * t + 1. * (1. - t)
    return torch.clip(x, low_bound, high_bound)

def sigma_dynamic_thresholding(x, sigma):
    minmax = 2. * sigma + 1.
    return torch.clip(x, -minmax, minmax)

def alpha_sigma_to_phi(alpha, sigma):
    return torch.atan2(sigma, alpha)

def replace_eps_noise(gaussian_eps, alpha: float = 0.5) -> torch.Tensor:
    """
    At alpha = 1, returns entirely new noise. At alpha = 0, returns the original gaussian noise.
    In between, return a blend of the gaussians, keeping standard deviation fixed to 1.
    """
    return torch.randn_like(gaussian_eps) * math.sqrt(alpha) + gaussian_eps * math.sqrt(1 - alpha)

@torch.no_grad()
def tensor_to_image(y: torch.Tensor) -> torch.Tensor:
    y = torch.clamp(y, -1, 1).cpu()
    y = (y / 2.0 + 0.5) * 255.
    arr = y.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if arr.shape[2] == 3:
        im = Image.fromarray(arr)
    else:
        im = Image.fromarray(arr.squeeze(2), 'L')
    return im

@torch.no_grad()
def cat_images(sequence_of_images: List[object]) -> Image:
    # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    widths, heights = zip(*(i.size for i in sequence_of_images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in sequence_of_images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im

@torch.no_grad()
def chunk_and_cat_pair(x: torch.Tensor, channels=3) -> Image:
    assert x.size(0) > channels
    assert x.size(0) - channels == 1 or x.size(0) - channels == 3
    return cat_images([tensor_to_image(x[:channels]), tensor_to_image(x[channels:])])

@torch.no_grad()
def save_samples(
    sample_intermediates: List[torch.Tensor], 
    dir: str,
    channel_split: int) -> None:
    os.makedirs(dir, exist_ok=True)
    for step_i, sample_batch in enumerate(sample_intermediates):
        for sample_i, sample in enumerate(sample_batch):
            sample_dir = os.path.join(dir, f"sample_{sample_i}")
            os.makedirs(sample_dir, exist_ok=True)
            out_image = chunk_and_cat_pair(sample.cpu(), channel_split)
            out_image.save(f"{sample_dir}/step_{step_i}.jpeg")

