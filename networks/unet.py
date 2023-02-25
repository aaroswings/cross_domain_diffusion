import torch
import torch.nn as nn

from typing import List, Optional

from networks.resnet import (
    CondBlock,
    CondIdentity,
    ConditionedSequential,
    FourierFeatures,
    SelfAttention2d,
    GroupNorm,
    ResConvBlock,
    SkipBlock,
    Activations,
    Conv2DLayers,
    NormalizationLayers
)

class UpDownBlock(CondBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        p_dropout: float = 0.0,
        attn_heads: int = 0,
        actvn_type: nn.Module = nn.SiLU,
        res_norm_in_type: CondBlock = GroupNorm,
        res_norm_out_type: CondBlock = GroupNorm,
        conv2d_type: nn.Module = nn.Conv2d,
        updown_mode: str = 'mid',
        c_dim: Optional[int] = None,
        num_res_blocks: int = 1
        ):
        super().__init__()
        self.net = ConditionedSequential(
            *([ResConvBlock(
                    in_channels, out_channels, temb_dim, c_dim,
                    p_dropout, res_norm_in_type, res_norm_out_type,
                    conv2d_type, actvn_type)
            ]  + [
                ResConvBlock(
                    out_channels, out_channels, temb_dim, c_dim,
                    p_dropout, res_norm_in_type, res_norm_out_type,
                    conv2d_type, actvn_type)
            ]* num_res_blocks),
        )
        self.use_attn = attn_heads > 0
        if self.use_attn:
            self.attn = SelfAttention2d(out_channels, attn_heads)

        if updown_mode == 'down':
            self.updown = nn.AvgPool2d(2)
        elif updown_mode == 'up':
            self.updown = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.updown = nn.Identity()

    def forward(self, x, temb, c):
        x = self.net(x, temb, c)
        if self.use_attn:
            x = self.attn(x)
        x = self.updown(x)
        return x


class UNet256(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_dim: int = 1024,
            channels: List[int] = [128, 256, 512, 1024, 1024, 1024],
            p_dropouts: List[float] = [0, 0, 0, 0.1, 0.1, 0.0],
            attn_heads: List[int] = [0, 0, 0, 4, 4, 8],
            res_norm_in_names: List[str] = ['groupnorm'] * 6, # openai style
            res_norm_out_names: List[str] = ['ada_groupnorm_temb'] * 6,
            num_res_blocks: List[int] = [1, 2, 2, 4, 12, 2],
            actvn_name: str = 'silu',
            conv2d_name: str = 'conv2d',
            c_dim_in: Optional[int] = None,
            c_proj_dim: Optional[int] = None,

        ) -> None:
        super().__init__()
        # time
        self.timestep_features = FourierFeatures(1, temb_dim, std=0.2)

        self.c_dim_in = c_dim_in

        if c_dim_in is not None:
            if c_proj_dim is not None:
                c_dim = c_proj_dim
                self.c_proj = nn.Sequential(
                    nn.Linear(c_dim_in, c_dim),
                    nn.SiLU(),
                    nn.Linear(c_dim, c_dim),
                )
            else:
                c_dim = c_dim_in
                self.c_proj = nn.Identity()
        else:
            self.c_proj = None
            c_dim = None

        # X features
        self.feature_extractor = nn.Sequential(
            Conv2DLayers[conv2d_name](in_channels, channels[0], 3, padding=1, bias=False)
        )

        self.skip_blocks = ConditionedSequential(
            SkipBlock([ # 256 -> 128
                UpDownBlock(channels[0], channels[1], temb_dim, p_dropouts[0], 
                    attn_heads[0], Activations[actvn_name],
                    NormalizationLayers[res_norm_in_names[0]], NormalizationLayers[res_norm_out_names[0]],
                    Conv2DLayers[conv2d_name], 'down', c_dim, num_res_blocks[0]),
                SkipBlock([ # 128 -> 64
                    UpDownBlock(channels[1], channels[2], temb_dim, p_dropouts[1], 
                        attn_heads[1], Activations[actvn_name],
                        NormalizationLayers[res_norm_in_names[1]], NormalizationLayers[res_norm_out_names[1]],
                        Conv2DLayers[conv2d_name], 'down', c_dim, num_res_blocks[1]),
                    SkipBlock([ #64 - > 32
                        UpDownBlock(channels[2], channels[3], temb_dim, p_dropouts[2], 
                            attn_heads[2], Activations[actvn_name],
                            NormalizationLayers[res_norm_in_names[2]], NormalizationLayers[res_norm_out_names[2]],
                            Conv2DLayers[conv2d_name], 'down', c_dim, num_res_blocks[2]),
                        SkipBlock([ #32 - > 16
                            UpDownBlock(channels[3], channels[4], temb_dim, p_dropouts[3], 
                                attn_heads[3], Activations[actvn_name],
                                NormalizationLayers[res_norm_in_names[3]], NormalizationLayers[res_norm_out_names[3]],
                                Conv2DLayers[conv2d_name], 'down', c_dim, num_res_blocks[2]),
                            SkipBlock([ # 16 -> 8
                                UpDownBlock(channels[4], channels[5], temb_dim, p_dropouts[4], 
                                    attn_heads[4], Activations[actvn_name],
                                    NormalizationLayers[res_norm_in_names[4]], NormalizationLayers[res_norm_out_names[4]],
                                    Conv2DLayers[conv2d_name], 'down', c_dim, num_res_blocks[4]),

                                # Middle block - https://github.com/VSehwag/minimal-diffusion/blob/ea0321eba164bfabe5562487cb5d1267b4881dad/unets.py#L665
                                # attention sandwhiched between two res blocks
                                UpDownBlock(channels[5], channels[5], temb_dim, p_dropouts[5], 
                                    attn_heads[5], Activations[actvn_name],
                                    NormalizationLayers[res_norm_in_names[5]], NormalizationLayers[res_norm_out_names[5]],
                                    Conv2DLayers[conv2d_name], 'mid', c_dim, num_res_blocks[4]),

                                UpDownBlock(channels[5], channels[5], temb_dim, p_dropouts[5], 
                                    0, Activations[actvn_name],
                                    NormalizationLayers[res_norm_in_names[5]], NormalizationLayers[res_norm_out_names[5]],
                                    Conv2DLayers[conv2d_name], 'mid', c_dim, num_res_blocks[4]),

                                ####
                                UpDownBlock(channels[5], channels[4], temb_dim, p_dropouts[4], # 8 -> 16
                                    attn_heads[4], Activations[actvn_name],
                                    NormalizationLayers[res_norm_in_names[4]], NormalizationLayers[res_norm_out_names[4]],
                                    Conv2DLayers[conv2d_name], 'up', c_dim, num_res_blocks[4]),
                            ]),
                            UpDownBlock(channels[4] * 2, channels[3], temb_dim, p_dropouts[3],  # 16 -> 32
                                attn_heads[3], Activations[actvn_name],
                                NormalizationLayers[res_norm_in_names[3]], NormalizationLayers[res_norm_out_names[3]],
                                Conv2DLayers[conv2d_name], 'up', c_dim, num_res_blocks[2])
                        ]),
                        UpDownBlock(channels[3] * 2, channels[2], temb_dim, p_dropouts[2], # 32 -> 64
                            attn_heads[2], Activations[actvn_name],
                            NormalizationLayers[res_norm_in_names[2]], NormalizationLayers[res_norm_out_names[2]],
                            Conv2DLayers[conv2d_name], 'up', c_dim, num_res_blocks[2])
                    ]),
                    UpDownBlock(channels[2] * 2, channels[1], temb_dim, p_dropouts[1],  # 64 -> 128
                        attn_heads[1], Activations[actvn_name],
                        NormalizationLayers[res_norm_in_names[1]], NormalizationLayers[res_norm_out_names[1]],
                        Conv2DLayers[conv2d_name], 'up', c_dim, num_res_blocks[1])
                ]),
                UpDownBlock(channels[1] * 2, channels[0], temb_dim, p_dropouts[0], # 128 -> 256
                    attn_heads[0], Activations[actvn_name],
                    NormalizationLayers[res_norm_in_names[0]], NormalizationLayers[res_norm_out_names[0]],
                    Conv2DLayers[conv2d_name], 'up', c_dim, num_res_blocks[0])
            ])
        )

        self.postprocess_block = nn.Sequential(
            nn.GroupNorm(32, channels[0] * 2),
            Activations[actvn_name](inplace=True),
            Conv2DLayers[conv2d_name](channels[0] * 2, out_channels, 3, padding=1, bias=False)
        )


    def forward(self, x, t, c: Optional[torch.tensor] = None):
        assert t.size(1) == 1
        # time
        time_proj = self.timestep_features(t) # ????? psnr?


        if c is not None:
            assert self.c_proj is not None
            assert c.size(1) == self.c_dim_in
            c = self.c_proj(c)

        x = self.feature_extractor(x)
        x = self.skip_blocks(x, t, c)
        x = self.postprocess_block(x)

        return x