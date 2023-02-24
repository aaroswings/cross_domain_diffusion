import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
import math
from typing import List, Optional

# via github.com/crowsonkb/v-diffusion-pytorch/
# via github.com/openai/guided-diffusion
# via github.com/VSehwag/minimal-diffusion/

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class CondBlock(nn.Module):
    @abstractmethod
    def forward(self, x, temb, c):
        raise NotImplementedError


class ConditionedSequential(nn.Sequential, CondBlock):
    def forward(self, x, temb, c):
        for layer in self:
            x = layer(x, temb, c) if isinstance(layer, CondBlock) else layer(x)
        return x


class CondIdentity(CondBlock):
    def __init__(self):
        self.identity = nn.Identity()

    def forward(self, x, temb, c):
        return self.identity(x)


class SkipBlock(CondBlock):
    def __init__(self, main: List[CondBlock], skip: Optional[CondBlock]=None):
        super().__init__()
        self.main = ConditionedSequential(*main)
        self.skip = skip if skip else CondIdentity()

    def forward(self, x, temb, c):
        return torch.cat([self.skip(x, temb, c), self.main(x, temb, c)], dim=1)


class NormLayer(nn.Module):
    def __init__(
        self,
        num_groups: int, out_dim: int, emb_dim: int,
        actvn_type: Optional[nn.Module] = nn.SiLU, eps: float = 1e-5
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.num_groups = num_groups
        self.actvn = actvn_type(inplace=True) if actvn_type else None
        self.eps = eps 

    @abstractmethod
    def forward(self, x, emb):
        "Base class for normalization layer."


class GroupNorm(NormLayer):
    def __init__(self, *args):
        super().__init__(*args)
        self.norm = nn.GroupNorm(self.num_groups, self.out_dim, self.eps, affine=True)

    def forward(self, x, emb):
        x = self.norm(x)
        if super().actvn:
            x = super().actvn(x)
        return x


class ScaleShiftNorm(NormLayer):
    def __init__(
        self, 
        num_groups: int, out_dim: int, emb_dim: int,
        actvn_type: Optional[nn.Module] = None, eps: float = 1e-5
    ):
        super().__init__(num_groups, out_dim, emb_dim, actvn_type, eps)
        self.proj = nn.Sequential(
            actvn_type(inplace=True) if actvn_type else nn.Identity(),
            nn.Linear(emb_dim, out_dim * 2)
        )
    
    def forward(self, x, emb):
        emb = self.proj(emb)[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)
        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        if super().actvn:
            x = super().actvn(x)
        return x


class TimestepAdaGroupNorm(CondBlock, ScaleShiftNorm):
    def __init__(self, *args):
        super().__init__(*args)
    
    def forward(self, x, temb, c):
        return super().forward(x, temb)
    

class ContextAdaGroupNorm(CondBlock, ScaleShiftNorm):
    def __init__(self, *args):
        super().__init__(*args)
    
    def forward(self, x, temb, c):
        return super().forward(x, c)
    

class ResConvBlock(CondBlock):
    """
    OpenAI-style residual convolution block. In the OpenAI model,
    the scale shift norm option allows adaptive group normalization using the timestep embedding
    in the "out_layers" subnet only, but in this version, the in_layers and out_layers
    subnets can either be a vanilla group norm or an adaptive group norm from either the c
    vector or the timestep embedding.
    """
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            temb_dim: int,
            c_dim: int,
            p_dropout: float = 0.0,
            norm_in_type: NormLayer = TimestepAdaGroupNorm, 
            norm_out_type: NormLayer = TimestepAdaGroupNorm,
            conv2d_type: nn.Module = nn.Conv2d, 
            actvn_type: nn.Module = nn.SiLU):

        # 0. Normalization layer types.
        if norm_in_type == GroupNorm or norm_in_type == TimestepAdaGroupNorm:
            norm_in = norm_in_type(32, in_channels, temb_dim)
        elif norm_in_type == ContextAdaGroupNorm:
            norm_in = norm_in_type(32, in_channels, c_dim)
        else:
            raise NotImplementedError("Only GroupNorm, TimestepAdaGroupNorm, or ContextAdaGroupNorm types are supported")

        if norm_out_type == GroupNorm or norm_out_type == TimestepAdaGroupNorm:
            norm_out = norm_in_type(32, out_channels, temb_dim)
        elif norm_out_type == ContextAdaGroupNorm:
            norm_out = norm_in_type(32, out_channels, c_dim)
        else:
            raise NotImplementedError("Only GroupNorm, TimestepAdaGroupNorm, or ContextAdaGroupNorm types are supported")

        # 1. Hidden blocks.
        self.main = ConditionedSequential(
            norm_in,
            actvn_type(inplace=True),
            conv2d_type(in_channels, out_channels, 3, padding=1),
            norm_out,
            nn.Dropout(p_dropout),
            actvn_type(inplace=True),
            zero_module(conv2d_type(out_channels, out_channels, 3, padding=1))
        )

        # 2. Skip connection.
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = conv2d_type(in_channels, out_channels, 1, bias=False)

    def forward(self, x, temb, c):
        return self.main(x, temb, c) + self.skip(x)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])