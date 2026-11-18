"""
Conditioning variants for ablation:

1. shift_linear  — z := z + W g + b   (Linear, zero-init W,b)
2. shift_mlp     — z := z + MLP(g)    (small MLP, last layer zero-init)
3. film_linear   — z := (1 + U g) * z + W g   (FiLM scale+shift; U,W zero-init
                                               so starts as identity)
4. none          — stock A1 WaveNet (knobs ignored; baseline single-condition)

All applied per WaveNet layer, after dilated-conv + input_mixer, before activation.
For gated layers mid_channels = 2*channels so both gate halves get the shift/scale.

Also: option use_bias=False on the shift Linear to test redundancy of b vs conv bias.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn

from .parametric_wavenet import (
    ParametricWaveNet,
    _iter_layers,
    _patch_layer_forward,
    build_a1_parametric,
)
from .a1_config import a1_standard_config
from nam.models.wavenet import WaveNet as NamWaveNet


VariantName = Literal["shift_linear", "shift_mlp", "film_linear", "none"]


class KnobMLP(nn.Module):
    """Small MLP g -> mid_channels; final Linear zero-init so starts as 0 shift."""

    def __init__(self, knob_dim: int, mid: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(knob_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, mid),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)


class FiLMLinear(nn.Module):
    """
    FiLM: scale = 1 + U g, shift = W g + b
    U,W zero-init => identity at start.
    """

    def __init__(self, knob_dim: int, mid: int, use_bias: bool = True):
        super().__init__()
        self.scale = nn.Linear(knob_dim, mid, bias=False)
        self.shift = nn.Linear(knob_dim, mid, bias=use_bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        if use_bias:
            nn.init.zeros_(self.shift.bias)

    def forward(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # z: (B, C, L), g: (B, K)
        s = 1.0 + self.scale(g).unsqueeze(-1)
        t = self.shift(g).unsqueeze(-1)
        return s * z + t


def _patch_layer_forward_variant(layer, variant: VariantName) -> None:
    """Like _patch_layer_forward but dispatches on layer._cond_variant."""

    def forward(x: torch.Tensor, h: torch.Tensor, out_length: int):
        def _c(t_len: int, tensor: torch.Tensor = h) -> torch.Tensor:
            return tensor[:, :, -t_len:]

        conv_input = x
        if layer._conv_pre_film is not None:
            pre_conv_length = min(conv_input.shape[2], h.shape[2])
            conv_input = layer._conv_pre_film(
                _c(pre_conv_length, tensor=conv_input), _c(pre_conv_length)
            )
        zconv = layer.conv(conv_input)
        if layer._conv_post_film is not None:
            post_conv_length = min(zconv.shape[2], h.shape[2])
            zconv = layer._conv_post_film(
                _c(post_conv_length, tensor=zconv), _c(post_conv_length)
            )

        mixin_input = h
        if layer._input_mixin_pre_film is not None:
            mixin_input = layer._input_mixin_pre_film(mixin_input, h)
        mix_out = layer._input_mixer(mixin_input)[:, :, -zconv.shape[2] :]
        if layer._input_mixin_post_film is not None:
            mix_out = layer._input_mixin_post_film(mix_out, _c(mix_out.shape[2]))

        z1len = min(zconv.shape[2], mix_out.shape[2])
        z1 = zconv[:, :, -z1len:] + mix_out[:, :, -z1len:]

        g = getattr(layer, "_knob_g", None)
        if g is not None:
            v = getattr(layer, "_cond_variant", "shift_linear")
            if v == "none":
                pass
            elif v == "film_linear":
                z1 = layer.knob_proj(z1, g)
            else:
                # shift_linear / shift_mlp: additive
                z1 = z1 + layer.knob_proj(g).unsqueeze(-1)

        if layer._activation_pre_film is not None:
            z1 = layer._activation_pre_film(z1, _c(z1.shape[2]))

        post_activation = layer._activation(z1)
        if layer._activation_post_film is not None:
            post_activation = layer._activation_post_film(
                post_activation, _c(post_activation.shape[2])
            )

        layer_output = post_activation
        if layer._layer1x1 is not None:
            layer_output = layer._layer1x1(layer_output)
            if layer._layer1x1_post_film is not None:
                layer_output = layer._layer1x1_post_film(
                    layer_output, _c(layer_output.shape[2])
                )

        head_output = post_activation
        if layer.head1x1 is not None:
            head_output = layer.head1x1(head_output)[:, :, -out_length:]
            if layer._head1x1_post_film is not None:
                head_output = layer._head1x1_post_film(
                    head_output, _c(head_output.shape[2])
                )
        else:
            head_output = head_output[:, :, -out_length:]

        residual = x[:, :, -layer_output.shape[2] :] + layer_output
        return (residual, head_output)

    layer.forward = forward  # type: ignore[method-assign]


class ParametricWaveNetVariant(ParametricWaveNet):
    """ParametricWaveNet with selectable conditioning variant."""

    def __init__(
        self,
        nam_wavenet: NamWaveNet,
        knob_dim: int = 2,
        knob_names: Optional[Sequence[str]] = None,
        variant: VariantName = "shift_linear",
        use_bias: bool = True,
        mlp_hidden: int = 16,
    ):
        # Skip ParametricWaveNet.__init__ attachment; do our own
        nn.Module.__init__(self)
        self.base = nam_wavenet
        self.knob_dim = int(knob_dim)
        self.knob_names = list(knob_names) if knob_names is not None else [
            f"k{i}" for i in range(self.knob_dim)
        ]
        if len(self.knob_names) != self.knob_dim or self.knob_dim < 1:
            raise ValueError("knob_names length must match positive knob_dim")
        self.variant = variant
        self.use_bias = use_bias
        self.mlp_hidden = mlp_hidden

        self.knob_projections = nn.ModuleList()
        for layer in _iter_layers(self.base):
            mid = layer.conv.out_channels
            if variant == "none":
                # Parameter-free placeholder; forward ignores knobs.
                proj = nn.Identity()
            elif variant == "shift_linear":
                proj = nn.Linear(self.knob_dim, mid, bias=use_bias)
                nn.init.zeros_(proj.weight)
                if use_bias:
                    nn.init.zeros_(proj.bias)
            elif variant == "shift_mlp":
                proj = KnobMLP(self.knob_dim, mid, hidden=mlp_hidden)
            elif variant == "film_linear":
                proj = FiLMLinear(self.knob_dim, mid, use_bias=use_bias)
            else:
                raise ValueError(variant)
            layer.add_module("knob_proj", proj)
            layer._cond_variant = variant
            _patch_layer_forward_variant(layer, variant)
            self.knob_projections.append(proj)


def build_a1_parametric_variant(
    variant: VariantName = "shift_linear",
    knob_dim: int = 2,
    knob_names: Optional[Sequence[str]] = None,
    sample_rate: float = 48000.0,
    use_bias: bool = True,
    mlp_hidden: int = 16,
    # Optional smaller architecture for CPU budget
    channels: Optional[tuple] = None,
) -> ParametricWaveNetVariant:
    """
    :param channels: if set, e.g. (8, 4) for a smaller-than-A1 net for fast CPU trials.
                     dilations remain A1's 1..512 so RF stays 4093.
    """
    cfg = a1_standard_config(sample_rate=sample_rate)
    if channels is not None:
        c0, c1 = channels
        cfg["layers_configs"][0]["channels"] = c0
        cfg["layers_configs"][0]["head"] = {
            "out_channels": max(c1, 1),
            "kernel_size": 1,
            "bias": False,
        }
        cfg["layers_configs"][1]["input_size"] = c0
        cfg["layers_configs"][1]["channels"] = c1
        cfg["layers_configs"][1]["head"] = {
            "out_channels": 1,
            "kernel_size": 1,
            "bias": True,
        }
    nam = NamWaveNet.init_from_config(cfg)
    return ParametricWaveNetVariant(
        nam,
        knob_dim=knob_dim,
        knob_names=knob_names or (["gain", "treble"] if knob_dim == 2 else None),
        variant=variant,
        use_bias=use_bias,
        mlp_hidden=mlp_hidden,
    )
