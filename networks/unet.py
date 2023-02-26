import torch
import torch.nn as nn

from typing import List, Optional

"""
todo: implement c cross attn in Unet
"""


from networks.modules import (
    CondBlock,
    ConditionedSequential,
    FourierFeatures,
    SelfAttention2d,
    GroupNorm,
    ResConvBlock,
    SkipBlock,
    Activations,
    Conv2DLayers,
    NormalizationLayers,
    zero_module
)

class UpDownBlock(CondBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
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
        self.updown_mode = updown_mode

        if self.updown_mode == 'down':
            self.conv = conv2d_type(in_channels, in_channels, 3, padding=1, bias=False)
            self.net = ConditionedSequential(
                *([ResConvBlock(
                        in_channels, in_channels, emb_dim,
                        p_dropout, res_norm_in_type, res_norm_out_type,
                        conv2d_type, actvn_type)
                ]* num_res_blocks + [
                    ResConvBlock(
                        in_channels, out_channels, emb_dim,
                        p_dropout, res_norm_in_type, res_norm_out_type,
                        conv2d_type, actvn_type)
                    ]),
            )
        else:
            self.net = ConditionedSequential(
                *([ResConvBlock(
                        in_channels, out_channels, emb_dim,
                        p_dropout, res_norm_in_type, res_norm_out_type,
                        conv2d_type, actvn_type)
                ]* num_res_blocks + [
                    ResConvBlock(
                        out_channels, out_channels, emb_dim,
                        p_dropout, res_norm_in_type, res_norm_out_type,
                        conv2d_type, actvn_type)
                    ]),
            )
            self.conv = conv2d_type(out_channels, out_channels, 3, padding=1, bias=False)

        self.use_attn = attn_heads > 0
        if self.use_attn:
            self.attn = SelfAttention2d(out_channels, attn_heads)

        if self.updown_mode == 'down':
            self.updown = nn.AvgPool2d(2)
        elif self.updown_mode == 'up':
            self.updown = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.updown = nn.Identity()

    def forward(self, x, temb, c):
        if self.updown_mode == 'down':
            x = self.conv(x)
            
        x = self.net(x, temb, c)

        if self.use_attn:
            x = self.attn(x)

        if self.updown_mode != 'down':
            x = self.conv(x)
        x = self.updown(x)
        
        return x


class UNet256(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            emb_dim: int = 1024,
            channels: List[int] = [128, 256, 512, 1024, 1024, 1024],
            p_dropouts: List[float] = [0, 0, 0, 0.1, 0.1, 0.0],
            attn_heads: List[int] = [0, 0, 0, 4, 4, 8],
            res_norm_in_names: List[str] = ['groupnorm'] * 6, # openai style
            res_norm_out_names: List[str] = ['ada_groupnorm_temb'] * 6,
            num_res_blocks: List[int] = [1, 2, 2, 4, 12, 2],
            actvn_name: str = 'silu',
            conv2d_in_name: str = 'conv2d',
            conv2d_name: str = 'conv2d',
            conv2d_out_name: str = 'conv2d',
            c_dim_in: Optional[int] = None, # dimension of c, if c is batch of 1d vectors (not class labels)
            c_num_classes: Optional[int] = None, # if c is class labels, setting this to a value other than None will create an embedding layer
            fuse_t_c_with_projection: bool = False
        ) -> None:
        super().__init__()

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
        self.t_embed = FourierFeatures(1, channels[0], std=0.2)

        self.t_project = nn.Sequential(
            nn.Linear(channels[0] + emb_dim if fuse_t_c_with_projection else channels[0], emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

        # x features
        self.feature_extractor = nn.Sequential(
            Conv2DLayers[conv2d_in_name](in_channels, channels[0], 3, padding=1, bias=False)
        )

        self.skip_blocks = ConditionedSequential(
            SkipBlock([ # 256 -> 128
                UpDownBlock(channels[0], channels[1], emb_dim, p_dropouts[0], 
                    attn_heads[0], Activations[actvn_name],
                    NormalizationLayers[res_norm_in_names[0]], NormalizationLayers[res_norm_out_names[0]],
                    Conv2DLayers[conv2d_name], 'down', emb_dim, num_res_blocks[0]),
                SkipBlock([ # 128 -> 64
                    UpDownBlock(channels[1], channels[2], emb_dim, p_dropouts[1], 
                        attn_heads[1], Activations[actvn_name],
                        NormalizationLayers[res_norm_in_names[1]], NormalizationLayers[res_norm_out_names[1]],
                        Conv2DLayers[conv2d_name], 'down', emb_dim, num_res_blocks[1]),
                    SkipBlock([ #64 - > 32
                        UpDownBlock(channels[2], channels[3], emb_dim, p_dropouts[2], 
                            attn_heads[2], Activations[actvn_name],
                            NormalizationLayers[res_norm_in_names[2]], NormalizationLayers[res_norm_out_names[2]],
                            Conv2DLayers[conv2d_name], 'down', emb_dim, num_res_blocks[2]),
                        SkipBlock([ #32 - > 16
                            UpDownBlock(channels[3], channels[4], emb_dim, p_dropouts[3], 
                                attn_heads[3], Activations[actvn_name],
                                NormalizationLayers[res_norm_in_names[3]], NormalizationLayers[res_norm_out_names[3]],
                                Conv2DLayers[conv2d_name], 'down', emb_dim, num_res_blocks[2]),
                            SkipBlock([ # 16 -> 8
                                UpDownBlock(channels[4], channels[5], emb_dim, p_dropouts[4], 
                                    attn_heads[4], Activations[actvn_name],
                                    NormalizationLayers[res_norm_in_names[4]], NormalizationLayers[res_norm_out_names[4]],
                                    Conv2DLayers[conv2d_name], 'down', emb_dim, num_res_blocks[4]),

                                # Middle block - https://github.com/VSehwag/minimal-diffusion/blob/ea0321eba164bfabe5562487cb5d1267b4881dad/unets.py#L665
                                # attention sandwhiched between two res blocks
                                UpDownBlock(channels[5], channels[5], emb_dim, p_dropouts[5], 
                                    attn_heads[5], Activations[actvn_name],
                                    NormalizationLayers[res_norm_in_names[5]], NormalizationLayers[res_norm_out_names[5]],
                                    Conv2DLayers[conv2d_name], 'mid', emb_dim, num_res_blocks[4]),

                                UpDownBlock(channels[5], channels[5], emb_dim, p_dropouts[5], 
                                    0, Activations[actvn_name],
                                    NormalizationLayers[res_norm_in_names[5]], NormalizationLayers[res_norm_out_names[5]],
                                    Conv2DLayers[conv2d_name], 'mid', emb_dim, num_res_blocks[4]),

                                ####
                                UpDownBlock(channels[5], channels[4], emb_dim, p_dropouts[4], # 8 -> 16
                                    attn_heads[4], Activations[actvn_name],
                                    NormalizationLayers[res_norm_in_names[4]], NormalizationLayers[res_norm_out_names[4]],
                                    Conv2DLayers[conv2d_name], 'up', emb_dim, num_res_blocks[4]),
                            ]),
                            UpDownBlock(channels[4], channels[3], emb_dim, p_dropouts[3],  # 16 -> 32
                                attn_heads[3], Activations[actvn_name],
                                NormalizationLayers[res_norm_in_names[3]], NormalizationLayers[res_norm_out_names[3]],
                                Conv2DLayers[conv2d_name], 'up', emb_dim, num_res_blocks[2])
                        ]),
                        UpDownBlock(channels[3], channels[2], emb_dim, p_dropouts[2], # 32 -> 64
                            attn_heads[2], Activations[actvn_name],
                            NormalizationLayers[res_norm_in_names[2]], NormalizationLayers[res_norm_out_names[2]],
                            Conv2DLayers[conv2d_name], 'up', emb_dim, num_res_blocks[2])
                    ]),
                    UpDownBlock(channels[2], channels[1], emb_dim, p_dropouts[1],  # 64 -> 128
                        attn_heads[1], Activations[actvn_name],
                        NormalizationLayers[res_norm_in_names[1]], NormalizationLayers[res_norm_out_names[1]],
                        Conv2DLayers[conv2d_name], 'up', emb_dim, num_res_blocks[1])
                ]),
                UpDownBlock(channels[1], channels[0], emb_dim, p_dropouts[0], # 128 -> 256
                    attn_heads[0], Activations[actvn_name],
                    NormalizationLayers[res_norm_in_names[0]], NormalizationLayers[res_norm_out_names[0]],
                    Conv2DLayers[conv2d_name], 'up', emb_dim, num_res_blocks[0])
            ])
        )

        self.postprocess_block = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            Activations[actvn_name](inplace=True),
            zero_module(Conv2DLayers[conv2d_out_name](channels[0], out_channels, 3, padding=1, bias=False))
        )


    def forward(self, x, t, c: Optional[torch.tensor] = None):
        # context vector projection
        if c is not None:
            c = self.c_embed(c)

        # time embedding and projection
        temb = self.t_embed(t)
        if self.fuse_t_c_with_projection:
            temb = torch.cat([temb, c], dim=1)
        temb = self.t_project(temb)

        # evaluate x
        x = self.feature_extractor(x)
        x = self.skip_blocks(x, temb, c)
        x = self.postprocess_block(x)

        return x