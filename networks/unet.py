import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Tuple, Optional

from networks.modules import FourierFeatures, ResBlock, SelfAttention2d, Conv2DLayers, Activations

@dataclass()
class UNetConfig:
    in_channels: int = 3
    out_channels: int = 3
    emb_dim: int = 512
    channels: Tuple[int] = (128, 256, 512, 1024, 512)
    p_dropouts: Tuple[float] = (0, 0, 0, 0.1, 0.1)
    attn_heads: int = 4
    num_res_blocks: Tuple[int] = (1, 1, 2, 8, 1)
    actvn_name: str = 'silu'
    conv2d_in_name: str = 'conv2d'
    conv2d_name: str = 'conv2d'
    conv2d_out_name: str = 'conv2d'


class UNet(nn.Module):
    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config
        self.num_feature_map_sizes = len(self.config.channels)

        self.register_buffer('root2', torch.sqrt(torch.tensor(2))) 

        self.t_embed = FourierFeatures(self.config.channels[0], std=16.)
        self.t_project = nn.Sequential(
            nn.Linear(self.config.channels[0], self.config.emb_dim),
            nn.SiLU(),
            nn.Linear(self.config.emb_dim, self.config.emb_dim)
        )

        self.embed_input = Conv2DLayers[self.config.conv2d_in_name](self.config.in_channels, self.config.channels[0], 3, padding=1, bias=False)
        self.project_output = nn.Sequential(
            nn.GroupNorm(32, self.config.channels[0]),
            Activations[self.config.actvn_name](),
            Conv2DLayers[self.config.conv2d_out_name](self.config.channels[0], self.config.out_channels, 3, padding=1, bias=False)
        )

        down_blocks = []
        for i in range(self.num_feature_map_sizes - 1):
            resblocks = [
                ResBlock(self.config.channels[i], 
                        self.config.channels[i] if block_ri > 0 else self.config.channels[i + 1], 
                        self.config.emb_dim,
                        self.config.p_dropouts[i], 
                        Conv2DLayers[self.config.conv2d_name], 
                        Activations[self.config.actvn_name])
                for block_ri in reversed(range(self.config.num_res_blocks[i]))]
            down_blocks.append(nn.ModuleList(resblocks))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.middle_resblocks = nn.ModuleList([
            ResBlock(self.config.channels[-1], 
                        self.config.channels[-1], 
                        self.config.emb_dim,
                        self.config.p_dropouts[-1], 
                        Conv2DLayers[self.config.conv2d_name], 
                        Activations[self.config.actvn_name])
            for _ in range(self.config.num_res_blocks[-1])])
        self.middle_attn_layers = nn.ModuleList([SelfAttention2d(self.config.channels[-1], self.config.attn_heads)
            for _ in range(self.config.num_res_blocks[-1])])

        up_blocks = []
        for i in range(self.num_feature_map_sizes - 1):
            resblocks = [
                ResBlock(self.config.channels[-i - 2] if block_i > 0 else self.config.channels[-i - 1], 
                         self.config.channels[-i - 2],
                         self.config.emb_dim,
                         self.config.p_dropouts[-i - 2], 
                         Conv2DLayers[self.config.conv2d_name], Activations[self.config.actvn_name])
                for block_i in range(self.config.num_res_blocks[-i - 2])]
            up_blocks.append(nn.ModuleList(resblocks))
        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(self, x, t, c: Optional[torch.tensor] = None):
        temb = self.t_embed(t)
        temb = self.t_project(temb)

        h0 = self.embed_input(x)
        hs = []
        last_h = h0

        for resblocks in self.down_blocks:
            for resblock in resblocks:
                last_h = resblock(last_h, temb)
                hs.append(last_h)
            last_h = F.avg_pool2d(last_h, 2)

        for resblock, attn_layer in zip(self.middle_resblocks, self.middle_attn_layers):
            last_h = resblock(last_h, temb)
            last_h = attn_layer(last_h)

        for resblocks in self.up_blocks:
            last_h = F.interpolate(last_h, scale_factor=2)
            for resblock in resblocks:
                skip_h = hs.pop()
                last_h = last_h + skip_h / self.root2
                last_h = resblock(last_h, temb)
        
        last_h = self.project_output(last_h)

        return last_h


