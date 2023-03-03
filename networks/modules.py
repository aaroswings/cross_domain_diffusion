import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import List, Optional
from abc import abstractmethod


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class FourierFeatures(nn.Module):
    def __init__(self, out_features, std=16.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(out_features // 2) * std, requires_grad=False)

    def forward(self, input):
        if input.dim() == 0:
            input = torch.ones(1).to(input.device) * input
        f = 2 * math.pi * torch.log(input)[:, None] @ self.weight[None, :]
        return torch.cat([f.cos(), f.sin()], dim=-1)
    

class GroupNorm(nn.Module):
    def __init__(self, num_groups: int, channels: int,
            actvn_type: Optional[nn.Module] = nn.SiLU, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, eps, affine=True)
        self.actvn = actvn_type() if actvn_type is not None else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = self.actvn(x)
        return x
    

class ScaleShiftNorm(nn.Module):
    def __init__(
        self, 
        num_groups: int, out_dim: int, emb_dim: int,
        actvn_type: Optional[nn.Module] = nn.SiLU, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.proj = nn.Sequential(
            actvn_type() if actvn_type else nn.Identity(),
            zero_module(nn.Linear(emb_dim, out_dim * 2))
        )
        self.actvn = actvn_type() if actvn_type else nn.Identity()
    
    def forward(self, x, emb):
        assert x is not None
        emb = self.proj(emb)[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)
        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        x = self.actvn(x)
        return x
    

class SelfAttention2d(nn.Module):
    def __init__(self, num_channels, num_heads=1):
        super().__init__()
        assert num_channels % num_heads == 0
        self.n_head = num_heads
        self.qkv_proj = nn.Conv2d(num_channels, num_channels * 3, 1)
        self.out_proj = nn.Conv2d(num_channels, num_channels, 1)

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(input)
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3]**-0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.out_proj(y)


class ResBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            emb_dim: int,
            p_dropout: float = 0.0,
            conv2d_type: nn.Module = nn.Conv2d, 
            actvn_type: nn.Module = nn.SiLU):
        super().__init__()
        self.group_norm = GroupNorm(32, in_channels, actvn_type)
        self.conv_hidden1 = conv2d_type(in_channels, out_channels, 3, padding=1)
        self.scaleshift_norm = ScaleShiftNorm(32, out_channels, emb_dim, actvn_type)
        self.dropout = nn.Dropout(p_dropout) if p_dropout else nn.Identity()
        self.conv_hidden2 = zero_module(conv2d_type(out_channels, out_channels, 3, padding=1))
        self.skip = nn.Identity() if in_channels == out_channels else conv2d_type(in_channels, out_channels, 1, bias=False)
        
    def forward(self, x, emb):
        h = x
        h = self.group_norm(h)
        h = self.conv_hidden1(h)
        h = self.scaleshift_norm(h, emb)
        h = self.dropout(h)
        h = self.conv_hidden2(h)

        return h + self.skip(x)


# export types without creating an import dependency on on this file when creating objects using these modules
NormalizationLayers = {
    'groupnorm': GroupNorm,
    'scale_shift_norm': ScaleShiftNorm,
}

Activations = {
    'silu': nn.SiLU,
    'relu': nn.ReLU,
}

Conv2DLayers = {
    'conv2d': nn.Conv2d
}