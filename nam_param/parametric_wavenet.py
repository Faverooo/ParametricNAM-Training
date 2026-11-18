"""
Parametric A1 WaveNet: per-layer additive knob conditioning before activation.

Stock NAM layer (simplified):
    z1 = conv_dilated(x) + input_mixer(condition_audio)
    h  = act(z1)

Ours:
    z1 = conv_dilated(x) + input_mixer(condition_audio) + W_l g + b_l
    h  = act(z1)

where g is a constant-per-file knob vector (e.g. [gain, treble] in [0,1]^2).
W_l, b_l are zero-initialized so the model starts identical to stock A1.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from . import _tk_stub  # noqa: F401  — ensure nam imports on headless
from .a1_config import A1_STANDARD_CONFIG, a1_standard_config

from nam.models.wavenet import WaveNet as NamWaveNet
from nam.models.wavenet._layer_array import _Layer
from nam.models import init_from_nam


def _iter_layers(nam_wavenet: NamWaveNet):
    """Yield each _Layer inside the wrapped WaveNet."""
    inner = nam_wavenet._net
    for layer_array in inner._layer_arrays:
        for layer in layer_array._layers:
            yield layer


def _patch_layer_forward(layer: _Layer) -> None:
    """
    Replace layer.forward so that after forming z1 (= conv + input_mixer [+ films]),
    we add knob_proj(g) before the activation.
    """
    # Bind original unbound logic by capturing the current bound method's code path
    # via a thin wrapper that reuses the original implementation pieces.
    # We re-implement the critical middle section by calling into the original
    # modules, mirroring nam.models.wavenet._layer_array._Layer.forward.

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

        # ---- parametric knob shift: W g + b (broadcast over time) ----
        g = getattr(layer, "_knob_g", None)
        if g is not None:
            # g: (B, knob_dim) -> (B, mid_channels, 1)
            shift = layer.knob_proj(g).unsqueeze(-1)
            z1 = z1 + shift

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


class ParametricWaveNet(nn.Module):
    """
    A1 WaveNet + per-layer Linear(knob_dim -> mid_channels) before activation.

    forward(x, g):
        x: (B, L) audio
        g: (B, knob_dim) knobs in [0,1]
        returns (B, L) with pad_start=True (same length as input)
    """

    def __init__(
        self,
        nam_wavenet: NamWaveNet,
        knob_dim: int = 2,
        knob_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.base = nam_wavenet
        self.knob_dim = int(knob_dim)
        self.knob_names = list(knob_names) if knob_names is not None else [
            f"k{i}" for i in range(self.knob_dim)
        ]
        if len(self.knob_names) != self.knob_dim:
            raise ValueError("knob_names length must match knob_dim")

        # Attach zero-init projections and patch forwards.
        self.knob_projections = nn.ModuleList()
        for layer in _iter_layers(self.base):
            mid = layer.conv.out_channels  # 2*channels if gated
            lin = nn.Linear(self.knob_dim, mid)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            layer.add_module("knob_proj", lin)
            _patch_layer_forward(layer)
            self.knob_projections.append(lin)
            # Keep a live reference so ModuleList owns parameters even if layer
            # registration is odd; layer.knob_proj IS the same module.
            assert layer.knob_proj is lin

    @property
    def receptive_field(self) -> int:
        return self.base.receptive_field

    @property
    def sample_rate(self) -> Optional[float]:
        return self.base.sample_rate

    @sample_rate.setter
    def sample_rate(self, val: Optional[float]):
        self.base.sample_rate = val

    def _set_knobs(self, g: torch.Tensor) -> None:
        if g.ndim != 2 or g.shape[1] != self.knob_dim:
            raise ValueError(
                f"Expected g shape (B, {self.knob_dim}), got {tuple(g.shape)}"
            )
        for layer in _iter_layers(self.base):
            layer._knob_g = g

    def _clear_knobs(self) -> None:
        for layer in _iter_layers(self.base):
            layer._knob_g = None

    def forward(
        self, x: torch.Tensor, g: torch.Tensor, pad_start: Optional[bool] = True
    ) -> torch.Tensor:
        """
        :param x: (B, L)
        :param g: (B, knob_dim)
        :param pad_start: if True (default), left-pad RF-1 zeros so output length = L
        """
        self._set_knobs(g)
        try:
            return self.base(x, pad_start=pad_start)
        finally:
            self._clear_knobs()

    def stock_forward(self, x: torch.Tensor, pad_start: Optional[bool] = True) -> torch.Tensor:
        """Forward with knobs cleared (equivalent to g-path with W=0,b=0)."""
        self._clear_knobs()
        return self.base(x, pad_start=pad_start)

    def export_parametric(self, path: Union[str, Path], metadata: Optional[Dict] = None) -> None:
        """
        Save parametric weights + architecture as .pt + sidecar .json.
        Standard NAM players will NOT load this; use bake_knobs_into_nam for .nam.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        from .bake import _infer_channels

        cfg = {
            "format": "parametric_nam_a1",
            "version": "0.1.1",
            "nam_tag": "v0.13.0",
            "architecture": "ParametricWaveNet",
            "base_architecture": "WaveNet",
            "base_config": self.base._export_config(),
            "knob_dim": self.knob_dim,
            "knob_names": self.knob_names,
            "sample_rate": self.sample_rate,
            "receptive_field": self.receptive_field,
            "n_knob_layers": len(self.knob_projections),
            "variant": getattr(self, "variant", "shift_linear"),
            "use_bias": getattr(self, "use_bias", True),
            "mlp_hidden": getattr(self, "mlp_hidden", 16),
            "channels": _infer_channels(self),
            "metadata": metadata or {},
        }
        json_path = path.with_suffix(".json") if path.suffix == ".pt" else Path(str(path) + ".json")
        pt_path = path if path.suffix == ".pt" else Path(str(path) + ".pt")
        payload = {"config": cfg, "state_dict": self.state_dict(),
                   "base_nam": self.base._get_export_dict()}
        # File temporanei nella stessa cartella: riduce checkpoint troncati.
        tmp_json = json_path.with_name(json_path.name + ".tmp")
        tmp_pt = pt_path.with_name(pt_path.name + ".tmp")
        with open(tmp_json, "w") as f:
            json.dump(cfg, f, indent=2)
        torch.save(payload, tmp_pt)
        tmp_pt.replace(pt_path)
        tmp_json.replace(json_path)

    def load_stock_nam_weights(self, nam_path: Union[str, Path]) -> None:
        """Warm-start base WaveNet weights from a standard A1 .nam; leave knob_proj at 0."""
        with open(nam_path, "r") as f:
            nam_cfg = json.load(f)
        if nam_cfg.get("architecture") != "WaveNet":
            raise ValueError(f"Expected WaveNet .nam, got {nam_cfg.get('architecture')}")
        donor = init_from_nam(nam_cfg)
        own_cfg, donor_cfg = self.base._export_config(), donor._export_config()
        own_cfg.pop("head_scale", None)
        donor_cfg.pop("head_scale", None)
        if own_cfg != donor_cfg:
            raise ValueError("Warm-start architecture/config mismatch")
        if donor.sample_rate is not None and donor.sample_rate != self.sample_rate:
            raise ValueError("Warm-start sample rate mismatch")
        self.base._net._head_scale = donor._net._head_scale
        # Copy matching parameters from donor into self.base
        donor_sd = donor.state_dict()
        base_sd = self.base.state_dict()
        matched = {k: v for k, v in donor_sd.items() if k in base_sd and base_sd[k].shape == v.shape}
        missing = [k for k in base_sd if k not in matched]
        # knob_proj lives under base.*.knob_proj — those won't be in donor
        unexpected_missing = [k for k in missing if "knob_proj" not in k]
        if unexpected_missing:
            raise RuntimeError(
                f"Warm-start shape/name mismatch for: {unexpected_missing[:10]}"
            )
        base_sd.update(matched)
        self.base.load_state_dict(base_sd, strict=True)
        # Ensure knob projections remain zero
        for proj in self.knob_projections:
            # MLP: azzera solo l'uscita, non lo strato nascosto (altrimenti morto).
            output = proj.net[-1] if hasattr(proj, "net") else proj
            for param in output.parameters():
                nn.init.zeros_(param)


def build_a1_parametric(
    knob_dim: int = 2,
    knob_names: Optional[Sequence[str]] = None,
    sample_rate: float = 48000.0,
) -> ParametricWaveNet:
    cfg = a1_standard_config(sample_rate=sample_rate)
    nam = NamWaveNet.init_from_config(cfg)
    return ParametricWaveNet(
        nam,
        knob_dim=knob_dim,
        knob_names=knob_names or (["gain", "treble"] if knob_dim == 2 else None),
    )


def load_parametric(path: Union[str, Path]) -> ParametricWaveNet:
    """Load a parametric checkpoint (.pt). Supports shift_linear and variants."""
    path = Path(path)
    pt_path = path if path.suffix == ".pt" else Path(str(path) + ".pt")
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    variant = cfg.get("variant", "shift_linear")
    if payload.get("base_nam") is not None:
        from .variants import ParametricWaveNetVariant
        model = ParametricWaveNetVariant(
            init_from_nam(payload["base_nam"]), knob_dim=cfg["knob_dim"],
            knob_names=cfg.get("knob_names"), variant=variant,
            use_bias=cfg.get("use_bias", True), mlp_hidden=cfg.get("mlp_hidden", 16),
        )
        model.load_state_dict(payload["state_dict"])
        return model
    channels = cfg.get("channels")
    if channels is not None:
        channels = tuple(channels)
    use_bias = cfg.get("use_bias", True)
    mlp_hidden = cfg.get("mlp_hidden", 16)
    if variant == "shift_linear" and use_bias and channels is None:
        # Classic ParametricWaveNet (also works for variant-built shift_linear A1)
        try:
            model = build_a1_parametric(
                knob_dim=cfg["knob_dim"],
                knob_names=cfg.get("knob_names"),
                sample_rate=cfg.get("sample_rate") or 48000.0,
            )
            model.load_state_dict(payload["state_dict"])
            return model
        except RuntimeError:
            pass
    from .variants import build_a1_parametric_variant

    model = build_a1_parametric_variant(
        variant=variant,
        knob_dim=cfg["knob_dim"],
        knob_names=cfg.get("knob_names"),
        sample_rate=cfg.get("sample_rate") or 48000.0,
        use_bias=use_bias,
        mlp_hidden=mlp_hidden,
        channels=channels,
    )
    model.load_state_dict(payload["state_dict"])
    return model
