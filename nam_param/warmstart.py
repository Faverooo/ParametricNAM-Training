"""Warm-start di un modello con PIÙ knob dai pesi di un modello con meno knob.

Esempio exp4: exp3 (g = [gain, treble]) -> exp4 (g = [gain, treble, mid, bass]).

Ogni proiezione dei knob (FiLM: scale U e shift W; shift_linear: W; shift_mlp:
primo Linear) è una matrice (C_out, K). Si copiano le colonne dei knob che
esistono già (abbinati per NOME, non per posizione) e le colonne nuove partono
a ZERO. Poiché U g e W g sono somme colonna per colonna,

    U_new [gain, treble, mid, bass] = U_old [gain, treble] + 0*mid + 0*bass

quindi al passo 0 il modello nuovo produce esattamente l'uscita di exp3 per ogni
(gain, treble), qualunque siano mid e bass. Tutti gli altri pesi (WaveNet,
bias FiLM, head_scale) vengono copiati identici. Il test
test_warm_start_expand_equivalence lo verifica numericamente.
"""

from __future__ import annotations

import re
from typing import Dict

import torch


def expand_knob_warm_start(model, source) -> Dict:
    """Copia i pesi di ``source`` in ``model`` espandendo le colonne dei knob.

    :param model: ParametricWaveNet(Variant) di destinazione (knob_dim >= source)
    :param source: modello caricato con load_parametric
    :return: dizionario con chiavi copiate/espanse e knob nuovi (per log/report)
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
    # Solo i Linear che ricevono g hanno una colonna per knob. Lo state_dict ha
    # ogni proiezione due volte (base...knob_proj e alias knob_projections.N).
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
    # head_scale è un attributo float, non un parametro: va copiato a mano.
    model.base._net._head_scale = source.base._net._head_scale
    return {"copied": len(copied), "expanded": expanded, "new_knobs": new_knobs,
            "source_knobs": src_names, "target_knobs": dst_names}
