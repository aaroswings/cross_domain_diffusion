import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, asdict
from typing import Tuple, Optional

from networks.modules import (
    FourierFeatures, 
    ResBlock,
    MLPConv,
    UpsampleShuffle,
    DownsampleRearrange,
    Conv2DLayers, 
    Activations
)

from networks.attention import (
    SpatialTransformer,
    SelfAttention2d
)

@dataclass()
class UNetConfig:
    name: str = 'unet'
    in_channels: int = 3
    out_channels: int = 3
    emb_dim: int = 512
    channels: Tuple[int] = (128, 256, 512, 512, 512)
    p_dropouts: Tuple[float] = (0, 0, 0, 0.1, 0.1)
    attn_heads: int = 4
    block_depth: Tuple[int] = (2, 4, 8, 16, 4)
    crossattn_context: str = 'timestep' # or c_vector or none
    context_dim: Optional[int] = None
    actvn_name: str = 'silu'
    conv2d_in_name: str = 'conv2d'
    conv2d_name: str = 'conv2d'
    conv2d_out_name: str = 'conv2d'
    downsample_name: str = 'avg_pool' # or pixel_shuffle

    def build(self):
        if self.name == 'unet':
            return UNet(self)

class UNet(nn.Module):
    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.init_config_ = config
        [self.__setattr__(k, v) for k, v in asdict(config).items()]
        self.num_feature_map_sizes = len(self.channels)

        self.register_buffer('root2', torch.sqrt(torch.tensor(2))) 

        self.t_embed = FourierFeatures(self.channels[0], std=16.)
        self.t_project = nn.Sequential(
            nn.Linear(self.channels[0], self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )

        if self.crossattn_context == 'c_vector':
            assert  self.context_dim is not None
            self.c_embed_project = nn.Sequential(
                nn.Linear(self.context_dim, self.emb_dim),
                nn.SiLU(),
                nn.Linear(self.emb_dim, self.emb_dim)
            )



        self.embed_input = Conv2DLayers[self.conv2d_in_name](self.in_channels, self.channels[0], 3, padding=1, bias=False)
        self.project_output = nn.Sequential(
            nn.GroupNorm(32, self.channels[0]),
            Activations[self.actvn_name](),
            Conv2DLayers[self.conv2d_out_name](self.channels[0], self.out_channels, 3, padding=1, bias=False)
        )

        down_blocks = []
        for i in range(self.num_feature_map_sizes - 1):
            resblocks = [
                ResBlock(self.channels[i], 
                        self.channels[i], 
                        self.emb_dim,
                        self.p_dropouts[i], 
                        Conv2DLayers[self.conv2d_name], 
                        Activations[self.actvn_name])
                for _ in range(self.block_depth[i])]
            downsample = [DownsampleRearrange(self.channels[i], self.channels[i + 1])]
            down_blocks.append(nn.ModuleList(resblocks + downsample))
            
        self.down_blocks = nn.ModuleList(down_blocks)

        self.attention = SpatialTransformer(
            self.channels[-1],
            self.attn_heads,
            self.block_depth[-1],
            self.p_dropouts[-1],
            self.emb_dim
        )


        # self.middle_block = Tr
        # self.middle_mlp_blocks = nn.ModuleList([
        #     MLPConv(self.channels[-1], 
        #             self.channels[-1], 
        #             self.emb_dim,
        #             self.p_dropouts[-1], 
        #             Conv2DLayers[self.conv2d_name], 
        #             Activations[self.actvn_name])
        #     for _ in range(self.block_depth[-1])])
        # self.middle_attn_layers = nn.ModuleList([SelfAttention2d(self.channels[-1], self.attn_heads)
        #     for _ in range(self.block_depth[-1])])

        up_blocks = []
        for i in range(self.num_feature_map_sizes - 1):
            upsample = [UpsampleShuffle(self.channels[-i - 1], self.channels[-i - 2])]
            resblocks = [
                ResBlock(self.channels[-i - 2], 
                         self.channels[-i - 2],
                         self.emb_dim,
                         self.p_dropouts[-i - 2], 
                         Conv2DLayers[self.conv2d_name], Activations[self.actvn_name])
                for block_i in range(self.block_depth[-i - 2])]
            up_blocks.append(nn.ModuleList(upsample + resblocks))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.print_param_count()

    def print_param_count(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([p.numel() for p in model_parameters])
        print(f'Num params in network: {n_params}\n({int(n_params / 1000000)}M params)', )

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, x, t, c: Optional[torch.tensor] = None) -> torch.Tensor:
        t_embedded = self.t_embed(t)
        t_embedded = self.t_project(t_embedded)

        h0 = self.embed_input(x)
        hs = []
        last_h = h0

        for down_block in self.down_blocks:
            for resblock in down_block[:-1]:
                last_h = resblock(last_h, t_embedded)
                hs.append(last_h)
            downsample = down_block[-1]
            last_h = downsample(last_h)

        if self.crossattn_context == 'c_vector':
            assert c is not None
            c_embedded = self.c_embed_project(c)
            last_h = self.attention(last_h, t_embedded)
        elif self.crossattn_context == 'timestep':
            last_h = self.attention(last_h, c_embedded)
        elif self.crossattn_context == 'none':
            last_h = self.attention(last_h, None)
        

        # for mlp_block, attn_layer in zip(self.middle_mlp_blocks, self.middle_attn_layers):
        #     last_h = mlp_block(last_h, emb)
        #     last_h = attn_layer(last_h)

        for up_block in self.up_blocks:
            upsample = up_block[0]
            last_h = upsample(last_h)
            for resblock in up_block[1:]:
                skip_h = hs.pop()
                last_h = (last_h + skip_h) / self.root2
                last_h = resblock(last_h, t_embedded)
        
        last_h = self.project_output(last_h)

        return last_h

