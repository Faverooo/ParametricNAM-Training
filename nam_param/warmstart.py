"""Warm-start a model with more knobs from one with fewer knobs.

For example, expand exp3's gain and treble to exp4's gain, treble, mid, and bass.
Each knob projection has shape (C_out, K): FiLM scale U and shift W, the
shift_linear matrix W, or the first shift_mlp linear layer. Copy existing knob
columns by name and initialize new columns to zero. For example:

    U_new [gain, treble, mid, bass] = U_old [gain, treble] + 0*mid + 0*bass

Copy all other weights and head_scale unchanged. At step zero, the expanded
model produces the same output for each gain and treble regardless of mid and
bass, as verified by test_warm_start_expand_equivalence.
"""

from __future__ import annotations

import re
from typing import Dict

import torch


def expand_knob_warm_start(model, source) -> Dict:
    """Copy source weights into model while expanding the knob columns.

    :param model: Destination ParametricWaveNet or variant with at least as many knobs.
    :param source: Source model returned by load_parametric.
    :return: Copied and expanded keys and new knob names for logging.
    """
    src_variant = getattr(source, "variant", "shift_linear")
    dst_variant = getattr(model, "variant", "shift_linear")
    if src_variant != dst_variant:
        raise ValueError(f"Warm-start: variante {src_variant} != {dst_variant}")
    src_names = [str(n).lower() for n in source.knob_names]
    dst_names = [str(n).lower() for n in model.knob_names]
    missing = [n for n in src_names if n not in dst_names]
    if missing:
        raise ValueError(f"Warm-start: knob {missing} del checkpoint non esistono nel modello nuovo {dst_names}")
    src_cfg = source.base._export_config()
    dst_cfg = model.base._export_config()
    src_cfg.pop("head_scale", None)
    dst_cfg.pop("head_scale", None)
    if src_cfg != dst_cfg:
        raise ValueError("Warm-start: architettura WaveNet diversa (canali/dilatazioni)")
    col_map = [dst_names.index(n) for n in src_names]
    new_knobs = [n for n in dst_names if n not in src_names]

    src_sd = source.state_dict()
    dst_sd = model.state_dict()
    if set(src_sd) != set(dst_sd):
        raise ValueError(f"Warm-start: parametri diversi: {sorted(set(src_sd) ^ set(dst_sd))[:6]}")
    # Only linear layers that consume g have one column per knob. The state dict
    # lists each projection twice, under base...knob_proj and the knob_projections.N alias.
    knob_input = re.compile(r"(^|\.)(knob_proj|knob_projections\.\d+)\.(scale\.weight|shift\.weight|weight|net\.0\.weight)$")
    out, expanded, copied = {}, [], []
    for key, dst in dst_sd.items():
        src = src_sd[key]
        if knob_input.search(key):
            if not (src.ndim == dst.ndim == 2 and src.shape[0] == dst.shape[0]
                    and src.shape[1] == len(src_names) and dst.shape[1] == len(dst_names)):
                raise ValueError(f"Warm-start: forma incompatibile per {key}: {tuple(src.shape)} -> {tuple(dst.shape)}")
            w = torch.zeros_like(dst)
            w[:, col_map] = src.to(dst.dtype)
            out[key] = w
            expanded.append(key)
        elif src.shape == dst.shape:
            out[key] = src.clone()
            copied.append(key)
        else:
            raise ValueError(f"Warm-start: forma incompatibile per {key}: {tuple(src.shape)} -> {tuple(dst.shape)}")
    model.load_state_dict(out, strict=True)
    # Copy head_scale explicitly because it is a plain float attribute.
    model.base._net._head_scale = source.base._net._head_scale
    return {"copied": len(copied), "expanded": expanded, "new_knobs": new_knobs,
            "source_knobs": src_names, "target_knobs": dst_names}
