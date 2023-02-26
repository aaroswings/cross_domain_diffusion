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


class ConditionedSequential(nn.Sequential, CondBlock):
    def forward(self, x, temb, c):
        assert x is not None
        for layer in self:
            assert x is not None
            x = layer(x, temb, c) if isinstance(layer, CondBlock) else layer(x)
            assert x is not None
        return x


class CondIdentity(CondBlock):
    def forward(self, x, temb, c):
        return x


class SkipBlock(CondBlock):
    def __init__(self, main: List[CondBlock], skip: Optional[CondBlock]=None):
        super().__init__()
        self.main = ConditionedSequential(*main)

    def forward(self, x, temb, c):
        # in skip connects, scale the skip by root(2) as in SimpleDiffusion, Imagen net
        return x / torch.sqrt(torch.tensor(2)).to(x.device) + self.main(x, temb, c)


class GroupNorm(CondBlock):
    def __init__(self, num_groups: int, out_dim: int, emb_dim: int,
            actvn_type: Optional[nn.Module] = nn.SiLU, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, out_dim, eps, affine=True)
        self.actvn = actvn_type(inplace=True) if actvn_type is not None else nn.Identity()

    def forward(self, x, emb, c):
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
            actvn_type(inplace=True) if actvn_type else nn.Identity(),
            zero_module(nn.Linear(emb_dim, out_dim * 2))
        )
        self.actvn = actvn_type(inplace=True) if actvn_type else nn.Identity()
    
    def forward(self, x, emb):
        assert x is not None
        emb = self.proj(emb)[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)
        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        x = self.actvn(x)
        return x


class TimestepAdaGroupNorm(CondBlock):
    def __init__(self, *args):
        super().__init__()
        self.norm = ScaleShiftNorm(*args)
    
    def forward(self, x, temb, c):
        return self.norm(x, temb)
    

class ContextAdaGroupNorm(CondBlock):
    def __init__(self, *args):
        super().__init__()
        self.norm = ScaleShiftNorm(*args)
    
    def forward(self, x, temb, c):
        return self.norm (x, c)
    

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
            emb_dim: int,
            p_dropout: float = 0.0,
            norm_in_type: CondBlock = GroupNorm, 
            norm_out_type: CondBlock = GroupNorm,
            conv2d_type: nn.Module = nn.Conv2d, 
            actvn_type: nn.Module = nn.SiLU):
        super().__init__()
        # 0. Normalization layer types.
        norm_in = norm_in_type(32, in_channels, emb_dim, actvn_type)
        norm_out = norm_out_type(32, out_channels, emb_dim, actvn_type)

        # 1. Hidden blocks.
        self.main = ConditionedSequential(
            norm_in,
            conv2d_type(in_channels, out_channels, 3, padding=1),
            norm_out,
            nn.Dropout(p_dropout),
            zero_module(conv2d_type(out_channels, out_channels, 3, padding=1))
        )

        # 2. Skip connection.
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = conv2d_type(in_channels, out_channels, 1, bias=False)

    def forward(self, x, temb, c):
        return self.main(x, temb, c) + self.skip(x)


class ConvBlock(CondBlock):
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
            norm_in_type: CondBlock = GroupNorm, 
            norm_out_type: CondBlock = GroupNorm,
            conv2d_type: nn.Module = nn.Conv2d, 
            actvn_type: nn.Module = nn.SiLU):
        super().__init__()
        # 0. Normalization layer types.
        if norm_in_type == GroupNorm or norm_in_type == TimestepAdaGroupNorm:
            norm_in = norm_in_type(32, in_channels, temb_dim, actvn_type)
        elif norm_in_type == ContextAdaGroupNorm:
            norm_in = norm_in_type(32, in_channels, c_dim, actvn_type)
        else:
            raise NotImplementedError("Only GroupNorm, TimestepAdaGroupNorm, or ContextAdaGroupNorm types are supported")

        if norm_out_type == GroupNorm or norm_out_type == TimestepAdaGroupNorm:
            norm_out = norm_in_type(32, out_channels, temb_dim, actvn_type)
        elif norm_out_type == ContextAdaGroupNorm:
            norm_out = norm_in_type(32, out_channels, c_dim, actvn_type)
        else:
            raise NotImplementedError("Only GroupNorm, TimestepAdaGroupNorm, or ContextAdaGroupNorm types are supported")

        # 1. Hidden blocks.
        self.main = ConditionedSequential(
            norm_in,
            conv2d_type(in_channels, out_channels, 3, padding=1),
            norm_out,
            nn.Dropout(p_dropout),
            zero_module(conv2d_type(out_channels, out_channels, 3, padding=1))
        )

        # 2. Skip connection.
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = conv2d_type(in_channels, out_channels, 1, bias=False)

    def forward(self, x, temb, c):
        assert x is not None
        x = self.main(x, temb, c) + self.skip(x)
        assert x is not None
        return x


"""
References: 
https://github.com/huggingface/diffusers/blob/589faa8c881de32932faebf92d2d8007135294d6/src/diffusers/models/unet_2d.py#L126
https://github.com/crowsonkb/v-diffusion-pytorch/blob/master/diffusion/models/danbooru_128.py#L59
"""
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


# export types without creating an import dependency on on this file when creating objects using these modules
NormalizationLayers = {
    'groupnorm': GroupNorm,
    'ada_groupnorm_temb': TimestepAdaGroupNorm,
    'ada_groupnorm_cond': ContextAdaGroupNorm
}

Activations = {
    'silu': nn.SiLU,
    'relu': nn.ReLU,
}

Conv2DLayers = {
    'conv2d': nn.Conv2d
}