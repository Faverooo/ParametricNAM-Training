"""
Bake (fold) a fixed knob vector g into a standard NAM .nam file.

Shift variants (shift_linear / shift_mlp last Linear):
    z := z + t(g)  with t = W g + b
    => conv.bias <- conv.bias + t

FiLM (film_linear):
    z := s(g) * z + t(g)  with s = 1 + U g, t = W g + b
    and z = conv(x) + mixer(cond)   (mixer has no bias in A1)
    => scale every output channel of conv.weight, conv.bias, mixer.weight by s,
       then add t to conv.bias. Zero the FiLM projections afterward.

Variant "none": no folding; export base WaveNet as-is.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from .parametric_wavenet import ParametricWaveNet, _iter_layers, build_a1_parametric


def _infer_channels(model: ParametricWaveNet) -> Optional[tuple]:
    widths = tuple(int(array._layers[0].conv.in_channels)
                   for array in model.base._net._layer_arrays)
    return None if widths == (16, 8) else widths


def clone_parametric(model: ParametricWaveNet) -> ParametricWaveNet:
    """Deep-copy via state_dict round-trip (preserves variant / channels / bias)."""
    from nam.models import init_from_nam
    from .variants import ParametricWaveNetVariant
    # Rebind forward closures after copying so they refer to the copied layers.
    twin = ParametricWaveNetVariant(
        init_from_nam(model.base._get_export_dict()), knob_dim=model.knob_dim,
        knob_names=model.knob_names, variant=getattr(model, "variant", "shift_linear"),
        use_bias=getattr(model, "use_bias", True), mlp_hidden=getattr(model, "mlp_hidden", 16),
    )
    param = next(model.parameters())
    twin.to(device=param.device, dtype=param.dtype)
    twin.load_state_dict(deepcopy(model.state_dict()))
    twin.train(model.training)
    return twin


def fold_knobs_into_conv_bias(model: ParametricWaveNet, g: torch.Tensor) -> ParametricWaveNet:
    """Fold shift-only conditioning into dilated convolution bias in place.

    This legacy entry point also delegates FiLM to fold_knobs.
    """
    return fold_knobs(model, g)


def fold_knobs(model: ParametricWaveNet, g: torch.Tensor) -> ParametricWaveNet:
    """
    In-place: fold fixed g into stock WaveNet weights; zero conditioning modules.
    Returns the same model (mutated) for chaining.
    """
    variant = getattr(model, "variant", "shift_linear")
    if g.ndim == 1:
        g = g.unsqueeze(0)
    if g.shape != (1, model.knob_dim):
        raise ValueError(
            f"Bake expects g shape (1,{model.knob_dim}) or ({model.knob_dim},), got {tuple(g.shape)}"
        )
    param = next(model.parameters())
    g = g.to(device=param.device, dtype=param.dtype)
    # Simple affine folding is incompatible with audio FiLM after convolution or mixing.
    if any(layer._conv_post_film is not None or layer._input_mixin_post_film is not None
           for layer in _iter_layers(model.base)):
        raise NotImplementedError("Bake with audio post-conv/mixer FiLM is not supported")

    if variant == "none":
        return model

    for layer in _iter_layers(model.base):
        if variant == "film_linear":
            _fold_film_layer(layer, g)
        elif variant == "shift_mlp":
            _fold_shift_module(layer, g, layer.knob_proj)
        else:
            # shift_linear (default ParametricWaveNet)
            _fold_shift_module(layer, g, layer.knob_proj)
    return model


def _fold_shift_module(layer, g: torch.Tensor, proj: nn.Module) -> None:
    """Add proj(g) into conv.bias; zero proj parameters."""
    with torch.no_grad():
        if isinstance(proj, nn.Linear):
            shift = proj(g).squeeze(0)
        else:
            # KnobMLP
            shift = proj(g).squeeze(0)
        if layer.conv.bias is None:
            raise RuntimeError("Layer dilated conv has no bias; cannot fold knobs")
        layer.conv.bias.add_(shift)
        for p in proj.parameters():
            p.zero_()


def _fold_film_layer(layer, g: torch.Tensor) -> None:
    """
    Fold FiLM: z' = s * z + t with z = conv(x) + mixer(cond) (+ conv.bias inside conv).

    Per output channel c:
      conv.weight[c]  *= s[c]
      mixer.weight[c] *= s[c]
      conv.bias[c]     = s[c] * conv.bias[c] + t[c]
    Then zero scale (U) and shift (W,b) so subsequent forwards are identity FiLM.
    """
    film = layer.knob_proj  # FiLMLinear
    with torch.no_grad():
        s = (1.0 + film.scale(g)).squeeze(0)  # (C,)
        t = film.shift(g).squeeze(0)  # (C,)
        if layer.conv.bias is None:
            raise RuntimeError("Layer dilated conv has no bias; cannot fold FiLM")
        C = s.numel()
        # conv.weight: (C, Cin, K)
        layer.conv.weight.data.mul_(s.view(C, 1, 1))
        layer.conv.bias.data.mul_(s).add_(t)
        mix = layer._input_mixer
        # mixer.weight: (C, Cin_cond, 1)
        if mix.weight.shape[0] != C:
            raise RuntimeError(
                f"FiLM bake: mixer out_channels {mix.weight.shape[0]} != scale {C}"
            )
        mix.weight.data.mul_(s.view(C, 1, 1))
        if mix.bias is not None:
            mix.bias.mul_(s)  # The convolution bias already includes t once.
        # Zero FiLM so s=1, t=0
        film.scale.weight.zero_()
        film.shift.weight.zero_()
        if film.shift.bias is not None:
            film.shift.bias.zero_()


@torch.no_grad()
def bake_knobs_into_nam(
    model: ParametricWaveNet,
    g: Sequence[float] | torch.Tensor,
    out_path: Union[str, Path],
    metadata: Optional[dict] = None,
) -> dict:
    """
    Export a standard WaveNet .nam with knobs folded into stock weights.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(list(g), dtype=torch.float32)
    g = g.float().view(-1)
    if g.numel() != model.knob_dim:
        raise ValueError(f"g must have {model.knob_dim} elems, got {g.numel()}")

    baked = clone_parametric(model)
    fold_knobs(baked, g)

    export = baked.base._get_export_dict()
    if metadata:
        export.setdefault("metadata", {}).update(metadata)
    export.setdefault("metadata", {})["baked_knobs"] = {
        name: float(v) for name, v in zip(model.knob_names, g.tolist())
    }
    export.setdefault("metadata", {})["bake_variant"] = getattr(model, "variant", "shift_linear")
    with open(out_path, "w") as f:
        json.dump(export, f)
    return export


@torch.no_grad()
def verify_bake_equivalence(
    model: ParametricWaveNet,
    g: torch.Tensor,
    x: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> float:
    """
    Max abs diff between conditioned model @ g and baked stock model @ same g.
    Raises AssertionError if not close.
    """
    model.eval()
    param = next(model.parameters())
    x = x.to(device=param.device, dtype=param.dtype)
    g = g.to(device=param.device, dtype=param.dtype)
    if g.ndim == 1:
        g = g.unsqueeze(0)
    g = g.expand(x.shape[0], -1)
    y_cond = model(x, g, pad_start=True)

    baked = clone_parametric(model)
    fold_knobs(baked, g[:1])
    y_baked = baked.stock_forward(x, pad_start=True)

    diff = (y_cond - y_baked).abs().max().item()
    if not torch.allclose(y_cond, y_baked, rtol=rtol, atol=atol):
        raise AssertionError(f"Bake mismatch: max abs diff={diff}")
    return diff
