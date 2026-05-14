from dataclasses import dataclass
from math import gcd
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import apply_freeu
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    Downsample2D,
    ResnetBlock2D,
    Transformer2DModel,
    UNetMidBlock2DCrossAttn,
    Upsample2D,
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from diffusers.models.controlnets.controlnet import ControlNetConditioningEmbedding


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# =========================
# ====== Base Output ======
# =========================


@dataclass
class DualBranchXSOutput(BaseOutput):
    sample: Tensor = None


# ==============================
# == Adapter building blocks  ==
# ==============================


class DualBranchXSAdapterDownBlock(nn.Module):
    """
    Single-branch adapter components that, together with corresponding components from the base model, form a
    fused dual-branch down block.

    NOTE: this is the template for one XS branch. In the dual-branch UNet we clone it twice
    (spa / spe) when fusing into the base UNet.
    """

    def __init__(
        self,
        resnets: nn.ModuleList,
        base_to_ctrl: nn.ModuleList,
        ctrl_to_base: nn.ModuleList,
        attentions: Optional[nn.ModuleList] = None,
        downsampler: Optional[nn.Conv2d] = None,
    ):
        super().__init__()
        self.resnets = resnets
        self.base_to_ctrl = base_to_ctrl
        self.ctrl_to_base = ctrl_to_base
        self.attentions = attentions
        self.downsamplers = downsampler


class DualBranchXSAdapterMidBlock(nn.Module):
    """
    Single-branch adapter components that, together with corresponding components from the base model, form a
    fused dual-branch mid block.
    """

    def __init__(self, midblock: UNetMidBlock2DCrossAttn, base_to_ctrl: nn.ModuleList, ctrl_to_base: nn.ModuleList):
        super().__init__()
        self.midblock = midblock
        self.base_to_ctrl = base_to_ctrl
        self.ctrl_to_base = ctrl_to_base


class DualBranchXSAdapterUpBlock(nn.Module):
    """
    Single-branch adapter components that, together with corresponding components from the base model, form a
    fused dual-branch up block.
    """

    def __init__(self, ctrl_to_base: nn.ModuleList):
        super().__init__()
        self.ctrl_to_base = ctrl_to_base


def get_down_block_adapter(
    base_in_channels: int,
    base_out_channels: int,
    ctrl_in_channels: int,
    ctrl_out_channels: int,
    temb_channels: int,
    max_norm_num_groups: Optional[int] = 32,
    has_crossattn=True,
    transformer_layers_per_block: Optional[Union[int, Tuple[int]]] = 1,
    num_attention_heads: Optional[int] = 1,
    cross_attention_dim: Optional[int] = 1024,
    add_downsample: bool = True,
    upcast_attention: Optional[bool] = False,
    use_linear_projection: Optional[bool] = True,
):
    """
    Build a single-branch XS adapter down block (template). In the fused model this template will be duplicated
    for spa / spe branches.
    """
    num_layers = 2  # only support sd + sdxl

    resnets = []
    attentions = []
    ctrl_to_base = []
    base_to_ctrl = []

    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block] * num_layers

    for i in range(num_layers):
        base_in_channels = base_in_channels if i == 0 else base_out_channels
        ctrl_in_channels = ctrl_in_channels if i == 0 else ctrl_out_channels

        base_to_ctrl.append(make_zero_conv(base_in_channels, base_in_channels))

        resnets.append(
            ResnetBlock2D(
                in_channels=ctrl_in_channels + base_in_channels,
                out_channels=ctrl_out_channels,
                temb_channels=temb_channels,
                groups=find_largest_factor(ctrl_in_channels + base_in_channels, max_factor=max_norm_num_groups),
                groups_out=find_largest_factor(ctrl_out_channels, max_factor=max_norm_num_groups),
                eps=1e-5,
            )
        )

        if has_crossattn:
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    ctrl_out_channels // num_attention_heads,
                    in_channels=ctrl_out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    norm_num_groups=find_largest_factor(ctrl_out_channels, max_factor=max_norm_num_groups),
                )
            )

        ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))

    if add_downsample:
        base_to_ctrl.append(make_zero_conv(base_out_channels, base_out_channels))

        downsamplers = Downsample2D(
            ctrl_out_channels + base_out_channels, use_conv=True, out_channels=ctrl_out_channels, name="op"
        )

        ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))
    else:
        downsamplers = None

    down_block_components = DualBranchXSAdapterDownBlock(
        resnets=nn.ModuleList(resnets),
        base_to_ctrl=nn.ModuleList(base_to_ctrl),
        ctrl_to_base=nn.ModuleList(ctrl_to_base),
    )

    if has_crossattn:
        down_block_components.attentions = nn.ModuleList(attentions)
    if downsamplers is not None:
        down_block_components.downsamplers = downsamplers

    return down_block_components


def get_mid_block_adapter(
    base_channels: int,
    ctrl_channels: int,
    temb_channels: Optional[int] = None,
    max_norm_num_groups: Optional[int] = 32,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = 1,
    cross_attention_dim: Optional[int] = 1024,
    upcast_attention: bool = False,
    use_linear_projection: bool = True,
):
    base_to_ctrl = make_zero_conv(base_channels, base_channels)

    midblock = UNetMidBlock2DCrossAttn(
        transformer_layers_per_block=transformer_layers_per_block,
        in_channels=ctrl_channels + base_channels,
        out_channels=ctrl_channels,
        temb_channels=temb_channels,
        resnet_groups=find_largest_factor(gcd(ctrl_channels, ctrl_channels + base_channels), max_norm_num_groups),
        cross_attention_dim=cross_attention_dim,
        num_attention_heads=num_attention_heads,
        use_linear_projection=use_linear_projection,
        upcast_attention=upcast_attention,
    )

    ctrl_to_base = make_zero_conv(ctrl_channels, base_channels)

    return DualBranchXSAdapterMidBlock(base_to_ctrl=base_to_ctrl, midblock=midblock, ctrl_to_base=ctrl_to_base)


def get_up_block_adapter(
    out_channels: int,
    prev_output_channel: int,
    ctrl_skip_channels: List[int],
):
    ctrl_to_base = []
    num_layers = 3  # only support sd + sdxl
    for i in range(num_layers):
        resnet_in_channels = prev_output_channel if i == 0 else out_channels
        ctrl_to_base.append(make_zero_conv(ctrl_skip_channels[i], resnet_in_channels))

    return DualBranchXSAdapterUpBlock(ctrl_to_base=nn.ModuleList(ctrl_to_base))


# ============================
# == Single XS Adapter body ==
# ============================


class DualBranchXSAdapter(ModelMixin, ConfigMixin):
    """
    Single-branch adapter template. In the dual-branch UNet we duplicate this
    template into a spatial branch and a spectral branch that both couple into the same base UNet.

    (GLU scheme-2):
      - All "zero conv" projections are replaced by GLU-gated 1x1 projections (still zero-initialized).
      - Scalar gates are removed in the fused dual-branch UNet.
    """

    @register_to_config
    def __init__(
        self,
        conditioning_channels: int = 3,
        conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
        time_embedding_mix: float = 1.0,
        learn_time_embedding: bool = False,
        num_attention_heads: Union[int, Tuple[int]] = 4,
        block_out_channels: Tuple[int] = (4, 8, 16, 16),
        base_block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        cross_attention_dim: int = 1024,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        sample_size: Optional[int] = 96,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        upcast_attention: bool = True,
        max_norm_num_groups: int = 32,
        use_linear_projection: bool = True,
        in_channels: int = 4,
    ):
        super().__init__()

        time_embedding_input_dim = base_block_out_channels[0]
        time_embedding_dim = base_block_out_channels[0] * 4

        if conditioning_channel_order not in ["rgb", "bgr"]:
            raise ValueError(f"unknown `conditioning_channel_order`: {conditioning_channel_order}")

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. "
                f"`block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(transformer_layers_per_block, (list, tuple)):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
        if not isinstance(cross_attention_dim, (list, tuple)):
            cross_attention_dim = [cross_attention_dim] * len(down_block_types)
        if not isinstance(num_attention_heads, (list, tuple)):
            num_attention_heads = [num_attention_heads] * len(down_block_types)

        if len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. "
                f"`num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        if learn_time_embedding:
            self.time_embedding = TimestepEmbedding(time_embedding_input_dim, time_embedding_dim)
        else:
            self.time_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.up_connections = nn.ModuleList([])

        self.conv_in = nn.Conv2d(int(in_channels), block_out_channels[0], kernel_size=3, padding=1)
        self.control_to_base_for_conv_in = make_zero_conv(block_out_channels[0], base_block_out_channels[0])

        base_out_channels = base_block_out_channels[0]
        ctrl_out_channels = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            base_in_channels = base_out_channels
            base_out_channels = base_block_out_channels[i]
            ctrl_in_channels = ctrl_out_channels
            ctrl_out_channels = block_out_channels[i]
            has_crossattn = "CrossAttn" in down_block_type
            is_final_block = i == len(down_block_types) - 1

            self.down_blocks.append(
                get_down_block_adapter(
                    base_in_channels=base_in_channels,
                    base_out_channels=base_out_channels,
                    ctrl_in_channels=ctrl_in_channels,
                    ctrl_out_channels=ctrl_out_channels,
                    temb_channels=time_embedding_dim,
                    max_norm_num_groups=max_norm_num_groups,
                    has_crossattn=has_crossattn,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim[i],
                    add_downsample=not is_final_block,
                    upcast_attention=upcast_attention,
                    use_linear_projection=use_linear_projection,
                )
            )

        self.mid_block = get_mid_block_adapter(
            base_channels=base_block_out_channels[-1],
            ctrl_channels=block_out_channels[-1],
            temb_channels=time_embedding_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            upcast_attention=upcast_attention,
            use_linear_projection=use_linear_projection,
        )

        ctrl_skip_channels = [block_out_channels[0]]
        for i, out_channels in enumerate(block_out_channels):
            number_of_subblocks = 3 if i < len(block_out_channels) - 1 else 2
            ctrl_skip_channels.extend([out_channels] * number_of_subblocks)

        reversed_base_block_out_channels = list(reversed(base_block_out_channels))

        base_out_channels = reversed_base_block_out_channels[0]
        for i in range(len(down_block_types)):
            prev_base_output_channel = base_out_channels
            base_out_channels = reversed_base_block_out_channels[i]
            ctrl_skip_channels_ = [ctrl_skip_channels.pop() for _ in range(3)]

            self.up_connections.append(
                get_up_block_adapter(
                    out_channels=base_out_channels,
                    prev_output_channel=prev_base_output_channel,
                    ctrl_skip_channels=ctrl_skip_channels_,
                )
            )

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        size_ratio: Optional[float] = None,
        block_out_channels: Optional[List[int]] = None,
        num_attention_heads: Optional[List[int]] = None,
        learn_time_embedding: bool = False,
        time_embedding_mix: float = 1.0,
        conditioning_channels: int = 3,
        conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        fixed_size = block_out_channels is not None
        relative_size = size_ratio is not None
        if not (fixed_size ^ relative_size):
            raise ValueError(
                "Pass exactly one of `block_out_channels` (for absolute sizing) or `size_ratio` (for relative sizing)."
            )

        block_out_channels = block_out_channels or [int(b * size_ratio) for b in unet.config.block_out_channels]
        if num_attention_heads is None:
            num_attention_heads = unet.config.attention_head_dim

        model = cls(
            conditioning_channels=conditioning_channels,
            conditioning_channel_order=conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            time_embedding_mix=time_embedding_mix,
            learn_time_embedding=learn_time_embedding,
            num_attention_heads=num_attention_heads,
            block_out_channels=block_out_channels,
            base_block_out_channels=unet.config.block_out_channels,
            cross_attention_dim=unet.config.cross_attention_dim,
            down_block_types=unet.config.down_block_types,
            sample_size=unet.config.sample_size,
            transformer_layers_per_block=unet.config.transformer_layers_per_block,
            upcast_attention=unet.config.upcast_attention,
            max_norm_num_groups=unet.config.norm_num_groups,
            use_linear_projection=unet.config.use_linear_projection,
            in_channels=unet.config.in_channels,
        )

        model.to(unet.dtype)
        return model

    def forward(self, *args, **kwargs):
        raise ValueError(
            "A DualBranchXSAdapter cannot be run by itself. Use it together with a UNet2DConditionModel to "
            "instantiate a UNetDualBranchXSModel."
        )


# ==========================================
# == Dual-branch UNet + XS (spa & spe)   ==
# ==========================================


class UNetDualBranchXSModel(ModelMixin, ConfigMixin):
    r"""
    A UNet fused with two XS branches (spa & spe).

    GLU scheme-2 changes:
      - Removes scalar gates (input_gate_spa/spe and per-block gate_spa/gate_spe).
      - Uses GLU-gated "zero conv" projections to provide channel-wise gating.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        sample_size: Optional[int] = 96,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        norm_num_groups: Optional[int] = 32,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = 8,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        upcast_attention: bool = True,
        use_linear_projection: bool = True,
        time_cond_proj_dim: Optional[int] = None,
        projection_class_embeddings_input_dim: Optional[int] = None,
        time_embedding_mix: float = 1.0,
        ctrl_conditioning_channels: int = 3,
        ctrl_conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
        ctrl_conditioning_channel_order: str = "rgb",
        ctrl_learn_time_embedding: bool = False,
        ctrl_block_out_channels: Tuple[int] = (4, 8, 16, 16),
        ctrl_num_attention_heads: Union[int, Tuple[int]] = 4,
        ctrl_max_norm_num_groups: int = 32,
    ):
        super().__init__()

        if time_embedding_mix < 0 or time_embedding_mix > 1:
            raise ValueError("`time_embedding_mix` needs to be between 0 and 1.")
        if time_embedding_mix < 1 and not ctrl_learn_time_embedding:
            raise ValueError("To use `time_embedding_mix` < 1, `ctrl_learn_time_embedding` must be `True`")

        if addition_embed_type is not None and addition_embed_type != "text_time":
            raise ValueError(
                "As `UNetDualBranchXSModel` currently only supports StableDiffusion and StableDiffusion-XL, "
                "`addition_embed_type` must be `None` or `'text_time'`."
            )

        if not isinstance(transformer_layers_per_block, (list, tuple)):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
        if not isinstance(cross_attention_dim, (list, tuple)):
            cross_attention_dim = [cross_attention_dim] * len(down_block_types)
        if not isinstance(num_attention_heads, (list, tuple)):
            num_attention_heads = [num_attention_heads] * len(down_block_types)
        if not isinstance(ctrl_num_attention_heads, (list, tuple)):
            ctrl_num_attention_heads = [ctrl_num_attention_heads] * len(down_block_types)

        base_num_attention_heads = num_attention_heads

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.base_conv_in = nn.Conv2d(self.in_channels, block_out_channels[0], kernel_size=3, padding=1)

        total_cond_ch = int(ctrl_conditioning_channels)
        if total_cond_ch < 2:
            raise ValueError(f"`ctrl_conditioning_channels` must be >= 2 (MS + PAN), got {total_cond_ch}.")
        self.ctrl_ms_channels = total_cond_ch - 1
        self.ctrl_pan_channels = 1

        self.controlnet_cond_embedding_spa = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=ctrl_block_out_channels[0],
            block_out_channels=ctrl_conditioning_embedding_out_channels,
            conditioning_channels=self.ctrl_pan_channels,
        )
        self.controlnet_cond_embedding_spe = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=ctrl_block_out_channels[0],
            block_out_channels=ctrl_conditioning_embedding_out_channels,
            conditioning_channels=self.ctrl_ms_channels,
        )

        self.ctrl_conv_in_spa = nn.Conv2d(self.in_channels, ctrl_block_out_channels[0], kernel_size=3, padding=1)
        self.ctrl_conv_in_spe = nn.Conv2d(self.in_channels, ctrl_block_out_channels[0], kernel_size=3, padding=1)

        self.control_to_base_for_conv_in_spa = make_zero_conv(ctrl_block_out_channels[0], block_out_channels[0])
        self.control_to_base_for_conv_in_spe = make_zero_conv(ctrl_block_out_channels[0], block_out_channels[0])

        time_embed_input_dim = block_out_channels[0]
        time_embed_dim = block_out_channels[0] * 4

        self.base_time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
        self.base_time_embedding = TimestepEmbedding(
            time_embed_input_dim,
            time_embed_dim,
            cond_proj_dim=time_cond_proj_dim,
        )
        if ctrl_learn_time_embedding:
            self.ctrl_time_embedding = TimestepEmbedding(in_channels=time_embed_input_dim, time_embed_dim=time_embed_dim)
        else:
            self.ctrl_time_embedding = None

        if addition_embed_type is None:
            self.base_add_time_proj = None
            self.base_add_embedding = None
        else:
            self.base_add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.base_add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        down_blocks = []
        base_out_channels = block_out_channels[0]
        ctrl_out_channels = ctrl_block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            base_in_channels = base_out_channels
            base_out_channels = block_out_channels[i]
            ctrl_in_channels = ctrl_out_channels
            ctrl_out_channels = ctrl_block_out_channels[i]
            has_crossattn = "CrossAttn" in down_block_type
            is_final_block = i == len(down_block_types) - 1

            down_blocks.append(
                DualBranchXSCrossAttnDownBlock2D(
                    base_in_channels=base_in_channels,
                    base_out_channels=base_out_channels,
                    ctrl_in_channels=ctrl_in_channels,
                    ctrl_out_channels=ctrl_out_channels,
                    temb_channels=time_embed_dim,
                    norm_num_groups=norm_num_groups,
                    ctrl_max_norm_num_groups=ctrl_max_norm_num_groups,
                    has_crossattn=has_crossattn,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    base_num_attention_heads=base_num_attention_heads[i],
                    ctrl_num_attention_heads=ctrl_num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim[i],
                    add_downsample=not is_final_block,
                    upcast_attention=upcast_attention,
                    use_linear_projection=use_linear_projection,
                )
            )

        self.mid_block = DualBranchXSCrossAttnMidBlock2D(
            base_channels=block_out_channels[-1],
            ctrl_channels=ctrl_block_out_channels[-1],
            temb_channels=time_embed_dim,
            norm_num_groups=norm_num_groups,
            ctrl_max_norm_num_groups=ctrl_max_norm_num_groups,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            base_num_attention_heads=base_num_attention_heads[-1],
            ctrl_num_attention_heads=ctrl_num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            upcast_attention=upcast_attention,
            use_linear_projection=use_linear_projection,
        )

        up_blocks = []
        rev_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        rev_num_attention_heads = list(reversed(base_num_attention_heads))
        rev_cross_attention_dim = list(reversed(cross_attention_dim))

        ctrl_skip_channels = [ctrl_block_out_channels[0]]
        for i, out_channels_ in enumerate(ctrl_block_out_channels):
            number_of_subblocks = 3 if i < len(ctrl_block_out_channels) - 1 else 2
            ctrl_skip_channels.extend([out_channels_] * number_of_subblocks)

        reversed_block_out_channels = list(reversed(block_out_channels))

        out_channels_ = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = out_channels_
            out_channels_ = reversed_block_out_channels[i]
            in_channels_ = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            ctrl_skip_channels_ = [ctrl_skip_channels.pop() for _ in range(3)]

            has_crossattn = "CrossAttn" in up_block_type
            is_final_block = i == len(block_out_channels) - 1

            up_blocks.append(
                DualBranchXSCrossAttnUpBlock2D(
                    in_channels=in_channels_,
                    out_channels=out_channels_,
                    prev_output_channel=prev_output_channel,
                    ctrl_skip_channels=ctrl_skip_channels_,
                    temb_channels=time_embed_dim,
                    resolution_idx=i,
                    has_crossattn=has_crossattn,
                    transformer_layers_per_block=rev_transformer_layers_per_block[i],
                    num_attention_heads=rev_num_attention_heads[i],
                    cross_attention_dim=rev_cross_attention_dim[i],
                    add_upsample=not is_final_block,
                    upcast_attention=upcast_attention,
                    norm_num_groups=norm_num_groups,
                    use_linear_projection=use_linear_projection,
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)

        self.base_conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups)
        self.base_conv_act = nn.SiLU()
        self.base_conv_out = nn.Conv2d(block_out_channels[0], self.out_channels, kernel_size=3, padding=1)

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        controlnet: Optional[DualBranchXSAdapter] = None,
        adapter: Optional[DualBranchXSAdapter] = None,
        size_ratio: Optional[float] = None,
        ctrl_block_out_channels: Optional[List[float]] = None,
        time_embedding_mix: Optional[float] = None,
        ctrl_optional_kwargs: Optional[Dict] = None,
    ):
        if adapter is not None and controlnet is not None and adapter is not controlnet:
            raise ValueError("Pass only one of `adapter` or `controlnet`.")
        adapter_module = adapter if adapter is not None else controlnet

        if adapter_module is None:
            adapter_module = DualBranchXSAdapter.from_unet(
                unet, size_ratio, ctrl_block_out_channels, **(ctrl_optional_kwargs or {})
            )
        else:
            if any(
                o is not None for o in (size_ratio, ctrl_block_out_channels, time_embedding_mix, ctrl_optional_kwargs)
            ):
                raise ValueError(
                    "When an adapter is passed, none of these parameters should be passed: "
                    "size_ratio, ctrl_block_out_channels, time_embedding_mix, ctrl_optional_kwargs."
                )

        params_for_unet = [
            "in_channels",
            "out_channels",
            "sample_size",
            "down_block_types",
            "up_block_types",
            "block_out_channels",
            "norm_num_groups",
            "cross_attention_dim",
            "transformer_layers_per_block",
            "addition_embed_type",
            "addition_time_embed_dim",
            "upcast_attention",
            "use_linear_projection",
            "time_cond_proj_dim",
            "projection_class_embeddings_input_dim",
        ]
        params_for_unet = {k: v for k, v in unet.config.items() if k in params_for_unet}
        params_for_unet["num_attention_heads"] = unet.config.attention_head_dim

        params_for_controlnet = [
            "conditioning_channels",
            "conditioning_embedding_out_channels",
            "conditioning_channel_order",
            "learn_time_embedding",
            "block_out_channels",
            "num_attention_heads",
            "max_norm_num_groups",
        ]
        params_for_controlnet = {"ctrl_" + k: v for k, v in adapter_module.config.items() if k in params_for_controlnet}
        params_for_controlnet["time_embedding_mix"] = adapter_module.config.time_embedding_mix

        model = cls.from_config({**params_for_unet, **params_for_controlnet})

        modules_from_unet = [
            "time_embedding",
            "conv_in",
            "conv_norm_out",
            "conv_out",
        ]
        for m in modules_from_unet:
            getattr(model, "base_" + m).load_state_dict(getattr(unet, m).state_dict())

        optional_modules_from_unet = [
            "add_time_proj",
            "add_embedding",
        ]
        for m in optional_modules_from_unet:
            if hasattr(unet, m) and getattr(unet, m) is not None:
                getattr(model, "base_" + m).load_state_dict(getattr(unet, m).state_dict())

        total_c = int(getattr(model.config, "ctrl_conditioning_channels", adapter_module.config.conditioning_channels))
        if total_c < 2:
            raise ValueError(f"`ctrl_conditioning_channels` must be >= 2, got {total_c}.")
        ms_ch = total_c - 1

        src_emb = adapter_module.controlnet_cond_embedding
        spe_map = list(range(0, ms_ch))
        spa_map = [ms_ch]

        load_controlnet_conditioning_embedding_from_template(
            target=model.controlnet_cond_embedding_spe,
            template=src_emb,
            channel_map=spe_map,
        )
        load_controlnet_conditioning_embedding_from_template(
            target=model.controlnet_cond_embedding_spa,
            template=src_emb,
            channel_map=spa_map,
        )

        model.ctrl_conv_in_spa.load_state_dict(adapter_module.conv_in.state_dict())
        model.ctrl_conv_in_spe.load_state_dict(adapter_module.conv_in.state_dict())

        if adapter_module.time_embedding is not None:
            model.ctrl_time_embedding.load_state_dict(adapter_module.time_embedding.state_dict())

        model.control_to_base_for_conv_in_spa.load_state_dict(adapter_module.control_to_base_for_conv_in.state_dict())
        model.control_to_base_for_conv_in_spe.load_state_dict(adapter_module.control_to_base_for_conv_in.state_dict())

        model.down_blocks = nn.ModuleList(
            DualBranchXSCrossAttnDownBlock2D.from_modules(b, c)
            for b, c in zip(unet.down_blocks, adapter_module.down_blocks)
        )
        model.mid_block = DualBranchXSCrossAttnMidBlock2D.from_modules(unet.mid_block, adapter_module.mid_block)
        model.up_blocks = nn.ModuleList(
            DualBranchXSCrossAttnUpBlock2D.from_modules(b, c)
            for b, c in zip(unet.up_blocks, adapter_module.up_connections)
        )

        model.to(unet.dtype)
        return model

    def freeze_unet_params(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

        base_parts = [
            "base_time_proj",
            "base_time_embedding",
            "base_add_time_proj",
            "base_add_embedding",
            "base_conv_in",
            "base_conv_norm_out",
            "base_conv_act",
            "base_conv_out",
        ]
        base_parts = [getattr(self, part) for part in base_parts if getattr(self, part) is not None]
        for part in base_parts:
            for param in part.parameters():
                param.requires_grad = False

        for d in self.down_blocks:
            d.freeze_base_params()
        self.mid_block.freeze_base_params()
        for u in self.up_blocks:
            u.freeze_base_params()

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(f"{name}", module, processor)

    def set_default_attn_processor(self):
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                "Cannot call `set_default_attn_processor` when attention processors are of mixed types "
                f"{next(iter(self.attn_processors.values()))}"
            )
        self.set_attn_processor(processor)

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    def disable_freeu(self):
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)

    def fuse_qkv_projections(self):
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        if getattr(self, "original_attn_processors", None) is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        sample: Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: Optional[torch.Tensor] = None,
        adapter_cond: Optional[torch.Tensor] = None,
        conditioning_scale: Optional[float] = None,
        conditioning_scale_spa: Optional[float] = None,
        conditioning_scale_spe: Optional[float] = None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: bool = True,
        apply_control: bool = True,
    ) -> Union[DualBranchXSOutput, Tuple]:
        if adapter_cond is not None and controlnet_cond is not None and adapter_cond is not controlnet_cond:
            raise ValueError("Pass only one of `adapter_cond` or `controlnet_cond`.")
        if adapter_cond is not None:
            controlnet_cond = adapter_cond

        if controlnet_cond is not None and self.config.ctrl_conditioning_channel_order == "bgr":
            if controlnet_cond.shape[1] == 3:
                controlnet_cond = torch.flip(controlnet_cond, dims=[1])
            else:
                logger.warning(
                    "ctrl_conditioning_channel_order='bgr' is only supported for 3-channel conditioning. "
                    f"Got {controlnet_cond.shape[1]} channels; skip channel flip to avoid breaking MS/PAN ordering."
                )

        if conditioning_scale_spa is None and conditioning_scale_spe is None:
            base_scale = 1.0 if conditioning_scale is None else float(conditioning_scale)
            conditioning_scale_spa = base_scale
            conditioning_scale_spe = base_scale
        else:
            base_scale = 1.0 if conditioning_scale is None else float(conditioning_scale)
            if conditioning_scale_spa is None:
                conditioning_scale_spa = base_scale
            if conditioning_scale_spe is None:
                conditioning_scale_spe = base_scale

        conditioning_scale_spa = float(conditioning_scale_spa)
        conditioning_scale_spe = float(conditioning_scale_spe)

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.base_time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)

        if self.config.ctrl_learn_time_embedding and apply_control:
            ctrl_temb = self.ctrl_time_embedding(t_emb, timestep_cond)
            base_temb = self.base_time_embedding(t_emb, timestep_cond)
            interpolation_param = self.config.time_embedding_mix**0.3
            temb = ctrl_temb * interpolation_param + base_temb * (1 - interpolation_param)
        else:
            temb = self.base_time_embedding(t_emb)

        aug_emb = None
        if self.config.addition_embed_type is None:
            pass
        elif self.config.addition_embed_type == "text_time":
            if added_cond_kwargs is None:
                raise ValueError("`added_cond_kwargs` must be provided when addition_embed_type='text_time'.")
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has `addition_embed_type='text_time'` which requires `text_embeds` "
                    "in `added_cond_kwargs`."
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has `addition_embed_type='text_time'` which requires `time_ids` "
                    "in `added_cond_kwargs`."
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.base_add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(temb.dtype)
            aug_emb = self.base_add_embedding(add_embeds)
        else:
            raise ValueError(
                "This model currently only supports StableDiffusion and StableDiffusion-XL, "
                f"so addition_embed_type = {self.config.addition_embed_type} is not supported."
            )

        temb = temb + aug_emb if aug_emb is not None else temb
        cemb = encoder_hidden_states

        guided_hint_spa = None
        guided_hint_spe = None

        if controlnet_cond is not None:
            b, c, h, w = controlnet_cond.shape

            expected_c = int(getattr(self.config, "ctrl_conditioning_channels", c))
            if expected_c < 2:
                raise ValueError(f"ctrl_conditioning_channels must be >= 2 (MS + PAN), got {expected_c}.")

            if c < expected_c:
                pad = torch.zeros(b, expected_c - c, h, w, device=controlnet_cond.device, dtype=controlnet_cond.dtype)
                cond = torch.cat([controlnet_cond, pad], dim=1)
            elif c > expected_c:
                cond = controlnet_cond[:, :expected_c]
            else:
                cond = controlnet_cond

            ms_ch = expected_c - 1
            cond_spe = cond[:, :ms_ch]
            cond_spa = cond[:, ms_ch:ms_ch+1]

            guided_hint_spe = self.controlnet_cond_embedding_spe(cond_spe)
            guided_hint_spa = self.controlnet_cond_embedding_spa(cond_spa)

        h_base = self.base_conv_in(sample)
        h_spa = self.ctrl_conv_in_spa(sample)
        h_spe = self.ctrl_conv_in_spe(sample)

        if guided_hint_spa is not None:
            h_spa = h_spa + guided_hint_spa
        if guided_hint_spe is not None:
            h_spe = h_spe + guided_hint_spe

        if apply_control:
            delta_spa = self.control_to_base_for_conv_in_spa(h_spa)
            delta_spe = self.control_to_base_for_conv_in_spe(h_spe)
            h_base = alternating_lf_hf_update(
                h_base,
                delta_spa=delta_spa,
                delta_spe=delta_spe,
                scale_spa=conditioning_scale_spa,
                scale_spe=conditioning_scale_spe,
            )

        hs_base, hs_spa, hs_spe = [], [], []
        hs_base.append(h_base)
        hs_spa.append(h_spa)
        hs_spe.append(h_spe)

        for down in self.down_blocks:
            h_base, h_spa, h_spe, residual_hb, residual_hspa, residual_hspe = down(
                hidden_states_base=h_base,
                hidden_states_spa=h_spa,
                hidden_states_spe=h_spe,
                temb=temb,
                encoder_hidden_states=cemb,
                conditioning_scale_spa=conditioning_scale_spa,
                conditioning_scale_spe=conditioning_scale_spe,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                apply_control=apply_control,
            )
            hs_base.extend(residual_hb)
            hs_spa.extend(residual_hspa)
            hs_spe.extend(residual_hspe)

        h_base, h_spa, h_spe = self.mid_block(
            hidden_states_base=h_base,
            hidden_states_spa=h_spa,
            hidden_states_spe=h_spe,
            temb=temb,
            encoder_hidden_states=cemb,
            conditioning_scale_spa=conditioning_scale_spa,
            conditioning_scale_spe=conditioning_scale_spe,
            cross_attention_kwargs=cross_attention_kwargs,
            attention_mask=attention_mask,
            apply_control=apply_control,
        )

        for up in self.up_blocks:
            n_resnets = len(up.resnets)
            skips_hb = hs_base[-n_resnets:]
            skips_hspa = hs_spa[-n_resnets:]
            skips_hspe = hs_spe[-n_resnets:]
            hs_base = hs_base[:-n_resnets]
            hs_spa = hs_spa[:-n_resnets]
            hs_spe = hs_spe[:-n_resnets]
            h_base = up(
                hidden_states=h_base,
                res_hidden_states_tuple_base=skips_hb,
                res_hidden_states_tuple_spa=skips_hspa,
                res_hidden_states_tuple_spe=skips_hspe,
                temb=temb,
                encoder_hidden_states=cemb,
                conditioning_scale_spa=conditioning_scale_spa,
                conditioning_scale_spe=conditioning_scale_spe,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                apply_control=apply_control,
            )

        h_base = self.base_conv_norm_out(h_base)
        h_base = self.base_conv_act(h_base)
        h_base = self.base_conv_out(h_base)

        if not return_dict:
            return (h_base,)

        return DualBranchXSOutput(sample=h_base)


# =========================================
# == Dual-branch fused Down / Mid / Up  ==
# =========================================


class DualBranchXSCrossAttnDownBlock2D(nn.Module):
    """
    Fused down block:
        base UNet path + XS_spa branch + XS_spe branch
    """

    def __init__(
        self,
        base_in_channels: int,
        base_out_channels: int,
        ctrl_in_channels: int,
        ctrl_out_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        ctrl_max_norm_num_groups: int = 32,
        has_crossattn: bool = True,
        transformer_layers_per_block: Optional[Union[int, Tuple[int]]] = 1,
        base_num_attention_heads: Optional[int] = 1,
        ctrl_num_attention_heads: Optional[int] = 1,
        cross_attention_dim: Optional[int] = 1024,
        add_downsample: bool = True,
        upcast_attention: Optional[bool] = False,
        use_linear_projection: Optional[bool] = True,
    ):
        super().__init__()

        num_layers = 2

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        base_resnets = []
        base_attentions = []

        spa_resnets = []
        spa_attentions = []
        spa_base_to_ctrl = []
        spa_ctrl_to_base = []

        spe_resnets = []
        spe_attentions = []
        spe_base_to_ctrl = []
        spe_ctrl_to_base = []

        for i in range(num_layers):
            _base_in = base_in_channels if i == 0 else base_out_channels
            _ctrl_in = ctrl_in_channels if i == 0 else ctrl_out_channels

            spa_base_to_ctrl.append(make_zero_conv(_base_in, _base_in))
            spa_resnets.append(
                ResnetBlock2D(
                    in_channels=_ctrl_in + _base_in,
                    out_channels=ctrl_out_channels,
                    temb_channels=temb_channels,
                    groups=find_largest_factor(_ctrl_in + _base_in, max_factor=ctrl_max_norm_num_groups),
                    groups_out=find_largest_factor(ctrl_out_channels, max_factor=ctrl_max_norm_num_groups),
                    eps=1e-5,
                )
            )

            spe_base_to_ctrl.append(make_zero_conv(_base_in, _base_in))
            spe_resnets.append(
                ResnetBlock2D(
                    in_channels=_ctrl_in + _base_in,
                    out_channels=ctrl_out_channels,
                    temb_channels=temb_channels,
                    groups=find_largest_factor(_ctrl_in + _base_in, max_factor=ctrl_max_norm_num_groups),
                    groups_out=find_largest_factor(ctrl_out_channels, max_factor=ctrl_max_norm_num_groups),
                    eps=1e-5,
                )
            )

            base_resnets.append(
                ResnetBlock2D(
                    in_channels=_base_in,
                    out_channels=base_out_channels,
                    temb_channels=temb_channels,
                    groups=norm_num_groups,
                )
            )

            if has_crossattn:
                base_attentions.append(
                    Transformer2DModel(
                        base_num_attention_heads,
                        base_out_channels // base_num_attention_heads,
                        in_channels=base_out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        norm_num_groups=norm_num_groups,
                    )
                )
                spa_attentions.append(
                    Transformer2DModel(
                        ctrl_num_attention_heads,
                        ctrl_out_channels // ctrl_num_attention_heads,
                        in_channels=ctrl_out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        norm_num_groups=find_largest_factor(ctrl_out_channels, max_factor=ctrl_max_norm_num_groups),
                    )
                )
                spe_attentions.append(
                    Transformer2DModel(
                        ctrl_num_attention_heads,
                        ctrl_out_channels // ctrl_num_attention_heads,
                        in_channels=ctrl_out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        norm_num_groups=find_largest_factor(ctrl_out_channels, max_factor=ctrl_max_norm_num_groups),
                    )
                )

            spa_ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))
            spe_ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))

        if add_downsample:
            self.base_downsamplers = Downsample2D(
                base_out_channels, use_conv=True, out_channels=base_out_channels, name="op"
            )

            self.spa_downsamplers = Downsample2D(
                ctrl_out_channels + base_out_channels, use_conv=True, out_channels=ctrl_out_channels, name="op"
            )
            self.spe_downsamplers = Downsample2D(
                ctrl_out_channels + base_out_channels, use_conv=True, out_channels=ctrl_out_channels, name="op"
            )

            spa_base_to_ctrl.append(make_zero_conv(base_out_channels, base_out_channels))
            spe_base_to_ctrl.append(make_zero_conv(base_out_channels, base_out_channels))
            spa_ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))
            spe_ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))
        else:
            self.base_downsamplers = None
            self.spa_downsamplers = None
            self.spe_downsamplers = None

        self.base_resnets = nn.ModuleList(base_resnets)
        self.base_attentions = nn.ModuleList(base_attentions) if has_crossattn else [None] * num_layers

        self.spa_resnets = nn.ModuleList(spa_resnets)
        self.spa_attentions = nn.ModuleList(spa_attentions) if has_crossattn else [None] * num_layers
        self.spa_base_to_ctrl = nn.ModuleList(spa_base_to_ctrl)
        self.spa_ctrl_to_base = nn.ModuleList(spa_ctrl_to_base)

        self.spe_resnets = nn.ModuleList(spe_resnets)
        self.spe_attentions = nn.ModuleList(spe_attentions) if has_crossattn else [None] * num_layers
        self.spe_base_to_ctrl = nn.ModuleList(spe_base_to_ctrl)
        self.spe_ctrl_to_base = nn.ModuleList(spe_ctrl_to_base)

        self.gradient_checkpointing = False

    @classmethod
    def from_modules(cls, base_downblock: CrossAttnDownBlock2D, ctrl_downblock: DualBranchXSAdapterDownBlock):
        def get_first_cross_attention(block):
            return block.attentions[0].transformer_blocks[0].attn2

        base_in_channels = base_downblock.resnets[0].in_channels
        base_out_channels = base_downblock.resnets[0].out_channels
        ctrl_in_channels = ctrl_downblock.resnets[0].in_channels - base_in_channels
        ctrl_out_channels = ctrl_downblock.resnets[0].out_channels
        temb_channels = base_downblock.resnets[0].time_emb_proj.in_features
        num_groups = base_downblock.resnets[0].norm1.num_groups
        ctrl_num_groups = ctrl_downblock.resnets[0].norm1.num_groups

        if hasattr(base_downblock, "attentions"):
            has_crossattn = True
            transformer_layers_per_block = len(base_downblock.attentions[0].transformer_blocks)
            base_num_attention_heads = get_first_cross_attention(base_downblock).heads
            ctrl_num_attention_heads = get_first_cross_attention(ctrl_downblock).heads
            cross_attention_dim = get_first_cross_attention(base_downblock).cross_attention_dim
            upcast_attention = get_first_cross_attention(base_downblock).upcast_attention
            use_linear_projection = base_downblock.attentions[0].use_linear_projection
        else:
            has_crossattn = False
            transformer_layers_per_block = None
            base_num_attention_heads = None
            ctrl_num_attention_heads = None
            cross_attention_dim = None
            upcast_attention = None
            use_linear_projection = None

        add_downsample = base_downblock.downsamplers is not None

        model = cls(
            base_in_channels=base_in_channels,
            base_out_channels=base_out_channels,
            ctrl_in_channels=ctrl_in_channels,
            ctrl_out_channels=ctrl_out_channels,
            temb_channels=temb_channels,
            norm_num_groups=num_groups,
            ctrl_max_norm_num_groups=ctrl_num_groups,
            has_crossattn=has_crossattn,
            transformer_layers_per_block=transformer_layers_per_block,
            base_num_attention_heads=base_num_attention_heads,
            ctrl_num_attention_heads=ctrl_num_attention_heads,
            cross_attention_dim=cross_attention_dim,
            add_downsample=add_downsample,
            upcast_attention=upcast_attention,
            use_linear_projection=use_linear_projection,
        )

        model.base_resnets.load_state_dict(base_downblock.resnets.state_dict())
        if has_crossattn:
            model.base_attentions.load_state_dict(base_downblock.attentions.state_dict())
        if add_downsample:
            model.base_downsamplers.load_state_dict(base_downblock.downsamplers[0].state_dict())

        model.spa_resnets.load_state_dict(ctrl_downblock.resnets.state_dict())
        model.spe_resnets.load_state_dict(ctrl_downblock.resnets.state_dict())
        if has_crossattn:
            model.spa_attentions.load_state_dict(ctrl_downblock.attentions.state_dict())
            model.spe_attentions.load_state_dict(ctrl_downblock.attentions.state_dict())

        model.spa_base_to_ctrl.load_state_dict(ctrl_downblock.base_to_ctrl.state_dict())
        model.spa_ctrl_to_base.load_state_dict(ctrl_downblock.ctrl_to_base.state_dict())
        model.spe_base_to_ctrl.load_state_dict(ctrl_downblock.base_to_ctrl.state_dict())
        model.spe_ctrl_to_base.load_state_dict(ctrl_downblock.ctrl_to_base.state_dict())

        if add_downsample:
            model.spa_downsamplers.load_state_dict(ctrl_downblock.downsamplers.state_dict())
            model.spe_downsamplers.load_state_dict(ctrl_downblock.downsamplers.state_dict())

        return model

    def freeze_base_params(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

        base_parts = [self.base_resnets]
        if isinstance(self.base_attentions, nn.ModuleList):
            base_parts.append(self.base_attentions)
        if self.base_downsamplers is not None:
            base_parts.append(self.base_downsamplers)

        for part in base_parts:
            for p in part.parameters():
                p.requires_grad = False

    def forward(
        self,
        hidden_states_base: Tensor,
        temb: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        hidden_states_spa: Optional[Tensor] = None,
        hidden_states_spe: Optional[Tensor] = None,
        conditioning_scale_spa: float = 1.0,
        conditioning_scale_spe: float = 1.0,
        attention_mask: Optional[Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        apply_control: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated and ignored here.")

        h_base = hidden_states_base
        h_spa = hidden_states_spa
        h_spe = hidden_states_spe

        base_output_states = ()
        spa_output_states = ()
        spe_output_states = ()

        base_blocks = list(zip(self.base_resnets, self.base_attentions))
        spa_blocks = list(zip(self.spa_resnets, self.spa_attentions))
        spe_blocks = list(zip(self.spe_resnets, self.spe_attentions))

        for (b_res, b_attn), (s_res, s_attn), (p_res, p_attn), b2s, s2b, b2p, p2b in zip(
            base_blocks,
            spa_blocks,
            spe_blocks,
            self.spa_base_to_ctrl,
            self.spa_ctrl_to_base,
            self.spe_base_to_ctrl,
            self.spe_ctrl_to_base,
        ):
            if apply_control:
                h_spa = torch.cat([h_spa, b2s(h_base)], dim=1)
                h_spe = torch.cat([h_spe, b2p(h_base)], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                h_base = self._gradient_checkpointing_func(b_res, h_base, temb)
            else:
                h_base = b_res(h_base, temb)

            if b_attn is not None:
                h_base = b_attn(
                    h_base,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

            if apply_control:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    h_spa = self._gradient_checkpointing_func(s_res, h_spa, temb)
                else:
                    h_spa = s_res(h_spa, temb)
                if s_attn is not None:
                    h_spa = s_attn(
                        h_spa,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    h_spe = self._gradient_checkpointing_func(p_res, h_spe, temb)
                else:
                    h_spe = p_res(h_spe, temb)
                if p_attn is not None:
                    h_spe = p_attn(
                        h_spe,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

                delta_spa = s2b(h_spa)
                delta_spe = p2b(h_spe)

                h_base = alternating_lf_hf_update(
                    h_base,
                    delta_spa=delta_spa,
                    delta_spe=delta_spe,
                    scale_spa=conditioning_scale_spa,
                    scale_spe=conditioning_scale_spe,
                )

            base_output_states = base_output_states + (h_base,)
            spa_output_states = spa_output_states + (h_spa,)
            spe_output_states = spe_output_states + (h_spe,)

        if self.base_downsamplers is not None:
            b2s = self.spa_base_to_ctrl[-1]
            s2b = self.spa_ctrl_to_base[-1]
            b2p = self.spe_base_to_ctrl[-1]
            p2b = self.spe_ctrl_to_base[-1]

            if apply_control:
                h_spa = torch.cat([h_spa, b2s(h_base)], dim=1)
                h_spe = torch.cat([h_spe, b2p(h_base)], dim=1)

            h_base = self.base_downsamplers(h_base)

            if apply_control:
                h_spa = self.spa_downsamplers(h_spa)
                h_spe = self.spe_downsamplers(h_spe)

                delta_spa = s2b(h_spa)
                delta_spe = p2b(h_spe)

                h_base = alternating_lf_hf_update(
                    h_base,
                    delta_spa=delta_spa,
                    delta_spe=delta_spe,
                    scale_spa=conditioning_scale_spa,
                    scale_spe=conditioning_scale_spe,
                )

            base_output_states = base_output_states + (h_base,)
            spa_output_states = spa_output_states + (h_spa,)
            spe_output_states = spe_output_states + (h_spe,)

        return h_base, h_spa, h_spe, base_output_states, spa_output_states, spe_output_states


class DualBranchXSCrossAttnMidBlock2D(nn.Module):
    """
    Fused mid block:
        base_midblock + XS_spa_midblock + XS_spe_midblock
    """

    def __init__(
        self,
        base_channels: int,
        ctrl_channels: int,
        temb_channels: Optional[int] = None,
        norm_num_groups: int = 32,
        ctrl_max_norm_num_groups: int = 32,
        transformer_layers_per_block: int = 1,
        base_num_attention_heads: Optional[int] = 1,
        ctrl_num_attention_heads: Optional[int] = 1,
        cross_attention_dim: Optional[int] = 1024,
        upcast_attention: bool = False,
        use_linear_projection: Optional[bool] = True,
    ):
        super().__init__()

        self.base_to_spa = make_zero_conv(base_channels, base_channels)
        self.base_to_spe = make_zero_conv(base_channels, base_channels)

        self.base_midblock = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=base_channels,
            temb_channels=temb_channels,
            resnet_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=base_num_attention_heads,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

        self.spa_midblock = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=ctrl_channels + base_channels,
            out_channels=ctrl_channels,
            temb_channels=temb_channels,
            resnet_groups=find_largest_factor(gcd(ctrl_channels, ctrl_channels + base_channels), ctrl_max_norm_num_groups),
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=ctrl_num_attention_heads,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

        self.spe_midblock = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=ctrl_channels + base_channels,
            out_channels=ctrl_channels,
            temb_channels=temb_channels,
            resnet_groups=find_largest_factor(gcd(ctrl_channels, ctrl_channels + base_channels), ctrl_max_norm_num_groups),
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=ctrl_num_attention_heads,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

        self.spa_to_base = make_zero_conv(ctrl_channels, base_channels)
        self.spe_to_base = make_zero_conv(ctrl_channels, base_channels)

        self.gradient_checkpointing = False

    @classmethod
    def from_modules(
        cls,
        base_midblock: UNetMidBlock2DCrossAttn,
        ctrl_midblock: DualBranchXSAdapterMidBlock,
    ):
        base_to_ctrl = ctrl_midblock.base_to_ctrl
        ctrl_to_base = ctrl_midblock.ctrl_to_base
        ctrl_midblock_core = ctrl_midblock.midblock

        def get_first_cross_attention(midblock):
            return midblock.attentions[0].transformer_blocks[0].attn2

        base_channels = ctrl_to_base.out_channels
        ctrl_channels = ctrl_to_base.in_channels
        transformer_layers_per_block = len(base_midblock.attentions[0].transformer_blocks)
        temb_channels = base_midblock.resnets[0].time_emb_proj.in_features
        num_groups = base_midblock.resnets[0].norm1.num_groups
        ctrl_num_groups = ctrl_midblock_core.resnets[0].norm1.num_groups
        base_num_attention_heads = get_first_cross_attention(base_midblock).heads
        ctrl_num_attention_heads = get_first_cross_attention(ctrl_midblock_core).heads
        cross_attention_dim = get_first_cross_attention(base_midblock).cross_attention_dim
        upcast_attention = get_first_cross_attention(base_midblock).upcast_attention
        use_linear_projection = base_midblock.attentions[0].use_linear_projection

        model = cls(
            base_channels=base_channels,
            ctrl_channels=ctrl_channels,
            temb_channels=temb_channels,
            norm_num_groups=num_groups,
            ctrl_max_norm_num_groups=ctrl_num_groups,
            transformer_layers_per_block=transformer_layers_per_block,
            base_num_attention_heads=base_num_attention_heads,
            ctrl_num_attention_heads=ctrl_num_attention_heads,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_linear_projection=use_linear_projection,
        )

        model.base_midblock.load_state_dict(base_midblock.state_dict())

        model.base_to_spa.load_state_dict(base_to_ctrl.state_dict())
        model.spa_midblock.load_state_dict(ctrl_midblock_core.state_dict())
        model.spa_to_base.load_state_dict(ctrl_to_base.state_dict())

        model.base_to_spe.load_state_dict(base_to_ctrl.state_dict())
        model.spe_midblock.load_state_dict(ctrl_midblock_core.state_dict())
        model.spe_to_base.load_state_dict(ctrl_to_base.state_dict())

        return model

    def freeze_base_params(self) -> None:
        for p in self.base_midblock.parameters():
            p.requires_grad = False

    def forward(
        self,
        hidden_states_base: Tensor,
        temb: Tensor,
        encoder_hidden_states: Tensor,
        hidden_states_spa: Optional[Tensor] = None,
        hidden_states_spe: Optional[Tensor] = None,
        conditioning_scale_spa: float = 1.0,
        conditioning_scale_spe: float = 1.0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        apply_control: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated and ignored.")

        h_base = hidden_states_base
        h_spa = hidden_states_spa
        h_spe = hidden_states_spe

        joint_args = {
            "temb": temb,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": attention_mask,
            "cross_attention_kwargs": cross_attention_kwargs,
            "encoder_attention_mask": encoder_attention_mask,
        }

        if apply_control:
            h_spa = torch.cat([h_spa, self.base_to_spa(h_base)], dim=1)
            h_spe = torch.cat([h_spe, self.base_to_spe(h_base)], dim=1)

        h_base = self.base_midblock(h_base, **joint_args)
        if apply_control:
            h_spa = self.spa_midblock(h_spa, **joint_args)
            h_spe = self.spe_midblock(h_spe, **joint_args)

            delta_spa = self.spa_to_base(h_spa)
            delta_spe = self.spe_to_base(h_spe)

            h_base = alternating_lf_hf_update(
                h_base,
                delta_spa=delta_spa,
                delta_spe=delta_spe,
                scale_spa=conditioning_scale_spa,
                scale_spe=conditioning_scale_spe,
            )

        return h_base, h_spa, h_spe


class DualBranchXSCrossAttnUpBlock2D(nn.Module):
    """
    Fused up block:
        base UpBlock + XS_spa_skip + XS_spe_skip
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        ctrl_skip_channels: List[int],
        temb_channels: int,
        norm_num_groups: int = 32,
        resolution_idx: Optional[int] = None,
        has_crossattn: bool = True,
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1024,
        add_upsample: bool = True,
        upcast_attention: bool = False,
        use_linear_projection: Optional[bool] = True,
    ):
        super().__init__()
        num_layers = 3

        self.has_cross_attention = has_crossattn
        self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        resnets = []
        attentions = []
        ctrl_to_base_spa = []
        ctrl_to_base_spe = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            ctrl_to_base_spa.append(make_zero_conv(ctrl_skip_channels[i], resnet_in_channels))
            ctrl_to_base_spe.append(make_zero_conv(ctrl_skip_channels[i], resnet_in_channels))

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=norm_num_groups,
                )
            )

            if has_crossattn:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        norm_num_groups=norm_num_groups,
                    )
                )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions) if has_crossattn else [None] * num_layers
        self.ctrl_to_base_spa = nn.ModuleList(ctrl_to_base_spa)
        self.ctrl_to_base_spe = nn.ModuleList(ctrl_to_base_spe)

        if add_upsample:
            self.upsamplers = Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    @classmethod
    def from_modules(cls, base_upblock: CrossAttnUpBlock2D, ctrl_upblock: DualBranchXSAdapterUpBlock):
        ctrl_to_base_skip_connections = ctrl_upblock.ctrl_to_base

        def get_first_cross_attention(block):
            return block.attentions[0].transformer_blocks[0].attn2

        out_channels = base_upblock.resnets[0].out_channels
        in_channels = base_upblock.resnets[-1].in_channels - out_channels
        prev_output_channels = base_upblock.resnets[0].in_channels - out_channels
        ctrl_skip_channels = [c.in_channels for c in ctrl_to_base_skip_connections]
        temb_channels = base_upblock.resnets[0].time_emb_proj.in_features
        num_groups = base_upblock.resnets[0].norm1.num_groups
        resolution_idx = base_upblock.resolution_idx

        if hasattr(base_upblock, "attentions"):
            has_crossattn = True
            transformer_layers_per_block = len(base_upblock.attentions[0].transformer_blocks)
            num_attention_heads = get_first_cross_attention(base_upblock).heads
            cross_attention_dim = get_first_cross_attention(base_upblock).cross_attention_dim
            upcast_attention = get_first_cross_attention(base_upblock).upcast_attention
            use_linear_projection = base_upblock.attentions[0].use_linear_projection
        else:
            has_crossattn = False
            transformer_layers_per_block = None
            num_attention_heads = None
            cross_attention_dim = None
            upcast_attention = None
            use_linear_projection = None
        add_upsample = base_upblock.upsamplers is not None

        model = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channels,
            ctrl_skip_channels=ctrl_skip_channels,
            temb_channels=temb_channels,
            norm_num_groups=num_groups,
            resolution_idx=resolution_idx,
            has_crossattn=has_crossattn,
            transformer_layers_per_block=transformer_layers_per_block,
            num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim,
            add_upsample=add_upsample,
            upcast_attention=upcast_attention,
            use_linear_projection=use_linear_projection,
        )

        model.resnets.load_state_dict(base_upblock.resnets.state_dict())
        if has_crossattn:
            model.attentions.load_state_dict(base_upblock.attentions.state_dict())
        if add_upsample:
            model.upsamplers.load_state_dict(base_upblock.upsamplers[0].state_dict())

        model.ctrl_to_base_spa.load_state_dict(ctrl_to_base_skip_connections.state_dict())
        model.ctrl_to_base_spe.load_state_dict(ctrl_to_base_skip_connections.state_dict())

        return model

    def freeze_base_params(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

        base_parts = [self.resnets]
        if isinstance(self.attentions, nn.ModuleList):
            base_parts.append(self.attentions)
        if self.upsamplers is not None:
            base_parts.append(self.upsamplers)

        for part in base_parts:
            for p in part.parameters():
                p.requires_grad = False

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple_base: Tuple[Tensor, ...],
        res_hidden_states_tuple_spa: Tuple[Tensor, ...],
        res_hidden_states_tuple_spe: Tuple[Tensor, ...],
        temb: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        conditioning_scale_spa: float = 1.0,
        conditioning_scale_spe: float = 1.0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[Tensor] = None,
        upsample_size: Optional[int] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        apply_control: bool = True,
    ) -> Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated and ignored.")

        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        def maybe_apply_freeu_to_subblock(hidden_states_, res_h_base_):
            if is_freeu_enabled:
                return apply_freeu(
                    self.resolution_idx,
                    hidden_states_,
                    res_h_base_,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )
            else:
                return hidden_states_, res_h_base_

        for resnet, attn, c2b_spa, c2b_spe, res_h_base, res_h_spa, res_h_spe in zip(
            self.resnets,
            self.attentions,
            self.ctrl_to_base_spa,
            self.ctrl_to_base_spe,
            reversed(res_hidden_states_tuple_base),
            reversed(res_hidden_states_tuple_spa),
            reversed(res_hidden_states_tuple_spe),
        ):
            if apply_control:
                delta_spa = c2b_spa(res_h_spa)
                delta_spe = c2b_spe(res_h_spe)

                hidden_states = alternating_lf_hf_update(
                    hidden_states,
                    delta_spa=delta_spa,
                    delta_spe=delta_spe,
                    scale_spa=conditioning_scale_spa,
                    scale_spe=conditioning_scale_spe,
                )

            hidden_states, res_h_base = maybe_apply_freeu_to_subblock(hidden_states, res_h_base)
            hidden_states = torch.cat([hidden_states, res_h_base], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

            if attn is not None:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            hidden_states = self.upsamplers(hidden_states, upsample_size)

        return hidden_states


# ======================
# == Helper functions ==
# ======================


class GLUZeroConv(nn.Module):
    """
    1x1 projection with channel-wise gating (GLU-style), zero-initialized.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.proj = nn.Conv2d(self.in_channels, self.out_channels * 2, kernel_size=1, padding=0)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        v, g = self.proj(x).chunk(2, dim=1)
        return v * torch.sigmoid(g)


def make_zero_conv(in_channels, out_channels=None):
    if out_channels is None:
        out_channels = in_channels
    return GLUZeroConv(in_channels, out_channels)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def find_largest_factor(number, max_factor):
    factor = max_factor
    if factor >= number:
        return number
    while factor != 0:
        residual = number % factor
        if residual == 0:
            return factor
        factor -= 1
    return 1


def split_lf_hf(x: Tensor, kernel_size: int = 3) -> Tuple[Tensor, Tensor]:
    if kernel_size <= 1:
        return x, torch.zeros_like(x)

    pad = kernel_size // 2
    _, c, _, _ = x.shape
    weight = torch.ones(c, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
    weight = weight / (kernel_size * kernel_size)
    lf = F.conv2d(x, weight, padding=pad, groups=c)
    hf = x - lf
    return lf, hf


def alternating_lf_hf_update(
    h_base: Tensor,
    delta_spa: Tensor,
    delta_spe: Tensor,
    scale_spa: float,
    scale_spe: float,
    kernel_size: int = 3,
) -> Tensor:
    lf, hf = split_lf_hf(h_base, kernel_size=kernel_size)

    delta_spe_lf, _ = split_lf_hf(delta_spe, kernel_size=kernel_size)
    lf = lf + scale_spe * delta_spe_lf

    h_tmp = lf + hf
    lf2, hf2 = split_lf_hf(h_tmp, kernel_size=kernel_size)

    _, delta_spa_hf = split_lf_hf(delta_spa, kernel_size=kernel_size)
    hf2 = hf2 + scale_spa * delta_spa_hf

    return lf2 + hf2


def _adapt_conv_in_weight(weight: Tensor, target_in_channels: int) -> Tensor:
    if weight.ndim != 4:
        raise ValueError(f"Expected conv weight with shape (out,in,k,k), got {tuple(weight.shape)}")

    out_c, in_c, kh, kw = weight.shape
    if target_in_channels == in_c:
        return weight

    if target_in_channels == 1:
        return weight.mean(dim=1, keepdim=True)

    if target_in_channels < in_c:
        return weight[:, :target_in_channels, :, :].contiguous()

    reps = (target_in_channels + in_c - 1) // in_c
    w_rep = weight.repeat(1, reps, 1, 1)[:, :target_in_channels, :, :].contiguous()
    w_rep = w_rep * (in_c / float(target_in_channels))
    return w_rep


def load_controlnet_conditioning_embedding_from_template(
    target: ControlNetConditioningEmbedding,
    template: ControlNetConditioningEmbedding,
    channel_map: Optional[List[int]] = None,
) -> None:
    t_sd = target.state_dict()
    s_sd = template.state_dict()

    new_sd = {}
    for k in t_sd.keys():
        if k not in s_sd:
            continue

        v = s_sd[k]
        if k.endswith("conv_in.weight"):
            if channel_map is not None and v.shape[1] >= (max(channel_map) + 1):
                idx = torch.as_tensor(channel_map, device=v.device, dtype=torch.long)
                v = v.index_select(1, idx).contiguous()

            tgt_in = int(getattr(target.conv_in, "in_channels", t_sd[k].shape[1]))
            if v.shape[1] != tgt_in:
                v = _adapt_conv_in_weight(v, tgt_in)

            new_sd[k] = v
        else:
            if v.shape == t_sd[k].shape:
                new_sd[k] = v

    missing, unexpected = target.load_state_dict(new_sd, strict=False)
    if len(missing) > 0:
        logger.warning(f"[cond-embed] missing keys when loading template into target: {missing}")
    if len(unexpected) > 0:
        logger.warning(f"[cond-embed] unexpected keys when loading template into target: {unexpected}")
