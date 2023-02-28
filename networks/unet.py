import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional
from abc import abstractmethod

"""
If a 512x512 version of the model is added, use the patching technique (strided conv in, transposed conv out) described in Simple Diffusion.
Todo: 
    - combine c, t embeddings by adding them (see 2.5, Neural Network Architecture, imagen paper)
    - add crossattn in the mid blocks for c vector
"""


from networks.modules import (
    CondBlock,
    FourierFeatures,
    SelfAttention2d,
    GroupNorm,
    ResBlock,
    Activations,
    Conv2DLayers,
    NormalizationLayers,
    zero_module
)

class MidBlock(CondBlock):
    def __init__(        
        self,
        channels: int,
        emb_dim: int,
        p_dropout: float = 0.0,
        attn_heads: int = 0,
        actvn_type: nn.Module = nn.SiLU,
        res_norm_in_type: CondBlock = GroupNorm,
        res_norm_out_type: CondBlock = GroupNorm,
        conv2d_type: nn.Module = nn.Conv2d,
        c_dim: Optional[int] = None,
        ):
        super().__init__()
        self.channels = channels
        self.emb_dim = emb_dim
        self.p_dropout = p_dropout
        self.attn_heads = attn_heads
        self.actvn_type = actvn_type
        self.res_norm_in_type = res_norm_in_type
        self.res_norm_out_type = res_norm_out_type
        self.conv2d_type = conv2d_type
        self.c_dim = c_dim

        self.res_in_block = ResBlock(self.channels, self.channels, self.emb_dim,
                        self.p_dropout, self.res_norm_in_type, self.res_norm_out_type,
                        self.conv2d_type, self.actvn_type)

        self.attn = SelfAttention2d(self.channels, self.attn_heads)

    def forward(self, x, t, c):
        x = self.res_in_block(x, t, c)
        x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            emb_dim: int = 1024,
            channels: List[int] = [128, 256, 512, 1024, 1024],
            p_dropouts: List[float] = [0, 0, 0, 0.1, 0.1],
            attn_heads: List[int] = [0, 0, 0, 4, 4],
            res_norm_in_names: List[str] = ['groupnorm'] * 5,
            res_norm_out_names: List[str] = ['ada_groupnorm_temb'] * 5,
            num_res_blocks: List[int] = [1, 2, 3, 6],
            actvn_name: str = 'silu',
            conv2d_in_name: str = 'conv2d',
            conv2d_name: str = 'conv2d',
            conv2d_out_name: str = 'conv2d',
            c_dim_in: Optional[int] = None, # dimension of c, if c is batch of 1d vectors (not class labels)
            c_num_classes: Optional[int] = None, # if c is class labels, setting this to a value other than None will create an embedding layer
            fuse_t_c_with_projection: bool = False
        ) -> None:
        super().__init__()

        assert len(channels) == len(p_dropouts) == len(attn_heads) == len(res_norm_in_names) \
            == len(res_norm_out_names) == (len(num_res_blocks) + 1)
        
        self.num_res_blocks = num_res_blocks
        
        self.updown_levels = len(channels)

        self.fuse_t_c_with_projection = fuse_t_c_with_projection

        # c vector
        if c_dim_in is not None: assert c_num_classes is None, "c input must be class embeddings or a vector"
        if c_num_classes is not None: assert c_dim_in is None, "c input must be class embeddings or a vector"

        self.c_is_vector = c_dim_in is not None
        self.c_is_labels = c_num_classes is not None

        if self.c_is_labels:
            self.c_embed = nn.Embedding(c_num_classes, emb_dim, max_norm=True)
        elif self.c_is_vector:
            self.c_embed = nn.Sequential(nn.Linear(c_dim_in, emb_dim), nn.SiLU())

        # time - embed layers same as CompVis latent-diffusion openaimodel, with fourier features
        self.t_embed = FourierFeatures(channels[0], std=16.)

        self.t_project = nn.Sequential(
            nn.Linear(channels[0] + emb_dim if fuse_t_c_with_projection else channels[0], emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # x features
        self.embed_input = nn.Sequential(
            Conv2DLayers[conv2d_in_name](in_channels, channels[0], 3, padding=1, bias=False)
        )

        down_blocks = []
        up_blocks = []

        for i in range(len(channels) - 1):
            resblocks = [
                ResBlock(channels[i], 
                        channels[i] if block_ri > 0 else channels[i + 1], 
                        emb_dim,
                        p_dropouts[i], 
                        NormalizationLayers[res_norm_in_names[i]], 
                        NormalizationLayers[res_norm_out_names[i]], 
                        Conv2DLayers[conv2d_name], 
                        Activations[actvn_name])
                for block_ri in reversed(range(num_res_blocks[i]))]
                
            down_blocks.append(nn.ModuleList(resblocks))

        self.middle_block = MidBlock(channels[-1], emb_dim, p_dropouts[-1],
                attn_heads[-1], Activations[actvn_name], NormalizationLayers[res_norm_in_names[-1]],
                NormalizationLayers[res_norm_out_names[-1]], Conv2DLayers[conv2d_name], 
                emb_dim)

        for i in range(len(channels) - 1):
            resblocks = [
                ResBlock(channels[-i - 2] if block_i > 0 else channels[-i - 1], 
                         channels[-i - 2],
                         emb_dim,
                         p_dropouts[-i - 2], 
                         NormalizationLayers[res_norm_in_names[-i - 2]], 
                         NormalizationLayers[res_norm_out_names[-i - 2]], 
                         Conv2DLayers[conv2d_name], Activations[actvn_name])
                for block_i in range(num_res_blocks[-i - 1])]
            up_blocks.append(nn.ModuleList(resblocks))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)

        self.project_output = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            Activations[actvn_name](),
            Conv2DLayers[conv2d_out_name](channels[0], out_channels, 3, padding=1, bias=False)
        )

        self.register_buffer('root2', torch.sqrt(torch.tensor(2)))


    def forward(self, x, t, c: Optional[torch.tensor] = None):
        """
        x : (B, C, H, W) tensor
        t : (B,) tensor
        c: (B,) or (B, c) tensor, if c is labels or vector respectively
        """
        # context vector projection
        if c is not None:
            c = self.c_embed(c)

        # time embedding and projection
        temb = self.t_embed(t)
        if self.fuse_t_c_with_projection:
            temb = torch.cat([temb, c], dim=1)
        temb = self.t_project(temb)

        # evaluate x
        h0 = self.embed_input(x)
        hs = []

        last_h = h0

        for resblocks in self.down_blocks:
            for resblock in resblocks:
                last_h = resblock(last_h, temb, c)
                hs.append(last_h)
            
            last_h = F.avg_pool2d(last_h, 2)

        last_h = self.middle_block(last_h, temb, c)

        for resblocks in self.up_blocks:
            last_h = F.interpolate(last_h, scale_factor=2)
            for resblock in resblocks:
                skip_h = hs.pop()
                last_h = last_h + skip_h / self.root2 # skip
                last_h = resblock(last_h, temb, c)
        
        last_h = self.project_output(last_h)

        return last_h