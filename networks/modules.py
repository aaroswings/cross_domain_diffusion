import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange

import math
from typing import List, Optional
from networks.util import zero_module

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
            nn.Linear(emb_dim, out_dim * 2)
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


class ResBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            emb_dim: int,
            p_dropout: float = 0.0,
            conv2d_type: nn.Module = nn.Conv2d, 
            actvn_type: nn.Module = nn.SiLU
    ):
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
    
class MLPConv(nn.Module):
    def __init__(
            self,
            in_channels: int, 
            out_channels: int, 
            emb_dim: int,
            p_dropout: float = 0.0,
            conv2d_type: nn.Module = nn.Conv2d, 
            actvn_type: nn.Module = nn.SiLU
    ):
        super().__init__()
        self.scaleshift_norm = ScaleShiftNorm(32, out_channels, emb_dim, actvn_type)
        self.dropout = nn.Dropout(p_dropout) if p_dropout else nn.Identity()
        self.conv_out = conv2d_type(in_channels, out_channels, 1, bias=False)

    def forward(self, x, emb):
        x = self.scaleshift_norm(x, emb)
        x = self.dropout(x)
        x = self.conv_out(x)
        return x

class DWConv2d(nn.Module):
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size: int = 3, 
            stride: int = 1, 
            padding: int = 0,
            bias=True
    ):
        super().__init__()
        self.spacewise = nn.Conv2d(in_channels, max(in_channels, out_channels), kernel_size, stride, padding, bias=False, groups=min(in_channels, out_channels))
        self.pointwise = nn.Conv2d(max(in_channels, out_channels), out_channels, 1, stride, 0, bias=bias)

    def forward(self, x):
        x = self.spacewise(x)
        x = F.silu(x)
        x = self.pointwise(x)
        return x
    

# based on Phil Wang's implementation of simple diffusion from github.com/lucidrains/denoising-diffusion-pytorch
class UpsampleShuffle(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        factor: int = 2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim_out or dim
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)
    
def DownsampleRearrange(
    dim: int,
    dim_out: Optional[int] = None,
    factor: int = 2
):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = factor, p2 = factor),
        nn.Conv2d(dim * (factor ** 2), dim_out or dim, 1)
    )


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
    'conv2d': nn.Conv2d,
    'depthwise_conv2d': DWConv2d
}