"""
NAM-aligned losses for ParametricNAM.

Error-to-Signal Ratio (ESR), as in Neural Amp Modeler / nam.models.losses.esr:

    ESR(y_hat, y) = || y_hat - y ||_2^2  /  || y ||_2^2

For one signal (T,) or one item of a batch (B,T):

    ESR_b = sum_t (y_hat_bt - y_bt)^2 / sum_t y_bt^2; ESR = mean_b ESR_b

Optional first-order pre-emphasis (NAM default coef ≈ 0.85) applied to both
prediction and target before the MSE that feeds ESR / MSE training:

    y'_t = y_t - coef * y_{t-1}     (first sample trimmed, as in NAM)

This boosts high-frequency error so the optimizer cares about treble/attack.

Multi-resolution STFT (MRSTFT, auraloss-style, used by NAM when enabled):
sum over resolutions r of spectral convergence + log-magnitude terms:

    L_MRSTFT = sum_r [ || |STFT_r(y)| - |STFT_r(y_hat)| ||_F / || |STFT_r(y)| ||_F
                      + mean( | log|STFT_r(y)| - log|STFT_r(y_hat)| | ) ]

Training objective (v2, "file-normalized ESR", default --loss esr):

    L = mean_b[  MSE( pre(y_hat_b), pre(y_b) ) / E_file(b)  ]
      + λ_MRSTFT * L_MRSTFT(y_hat, y)

    E_file(b) = mean_t pre(y_file)^2 over the WHOLE train region of the file
                (condition) that crop b comes from; precomputed once.

Why not the crop energy (v1, now --loss esr_crop)? A 1 s random crop can land
in a near-silent passage of the v3 signal (e.g. 10.0-12.0 s: the target is the
-99 dBFS noise floor). Its denominator is ~1e-8 of the file mean and that single
item's ESR reaches ~1.5e4 at init (measured), dominating the batch mean and the
gradient. With E_file the denominator is a fixed per-condition constant:
inside one file the loss is exactly NAM's (pre-emphasized) MSE up to a constant,
across files quiet conditions still weigh like loud ones, and the expected value
over random crops is the train-region ESR of each file.

Opt-in v3 (--loss esr_dual): add raw_weight * MSE(pred, target) / E_file_raw
(no pre-emphasis in this second term). Each term uses its OWN whole-train-file
energy. A DC error d survives pre-emphasis with amplitude (1-0.85)*d; adding
the raw term makes the objective constrain DC/low-frequency errors too, without
changing inference or the standard .nam bake. --loss esr_dc is an ablation:
add dc_weight * mean_t(pred-target)^2 / E_file_raw instead. Both are implemented
by compute_batch_loss in train_param.py. Default --loss esr remains v2.

NAM v0.13 itself trains on MSE (+ optional pre-emph MSE / MRSTFT) and uses
nam.models.losses.esr (mean over batch items of per-item ratios, no epsilon)
only for validation, where the batch is ONE whole validation signal.
"""

from __future__ import annotations

import torch
from nam.models.losses import esr, apply_pre_emphasis_filter


def mse_loss(pred: torch.Tensor, target: torch.Tensor, pre_emph_coef: float | None = None) -> torch.Tensor:
    if pre_emph_coef is not None:
        pred = apply_pre_emphasis_filter(pred, pre_emph_coef)
        target = apply_pre_emphasis_filter(target, pre_emph_coef)
    return torch.mean((pred - target) ** 2)


def esr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute NAM ESR as the mean of per-item energy ratios."""
    return esr(pred, target)


def esr_per_item(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-batch-item ESR for training (shape (B,) or (B, C)).
    pred/target: (B, ..., T) — reduce over all dims except batch.
    """
    dims = tuple(range(1, pred.ndim))
    num = (pred - target).pow(2).sum(dim=dims)
    den = target.pow(2).sum(dim=dims).clamp_min(eps)
    return num / den


def preemphasized_esr_mean(
    pred: torch.Tensor,
    target: torch.Tensor,
    pre_emph_coef: float | None = 0.85,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean over batch of per-item ESR after optional pre-emphasis (training loss)."""
    if pre_emph_coef is not None:
        pred = apply_pre_emphasis_filter(pred, pre_emph_coef)
        target = apply_pre_emphasis_filter(target, pre_emph_coef)
    return esr_per_item(pred, target, eps=eps).mean()


def file_normalized_esr(
    pred: torch.Tensor,
    target: torch.Tensor,
    file_energy: torch.Tensor,
    pre_emph_coef: float | None = 0.85,
) -> torch.Tensor:
    """Training loss v2: per-item MSE (after pre-emphasis) / energy of its FILE.

    :param file_energy: (B,) mean pre-emphasized target power of the whole train
        region of the condition each item comes from (see file_train_energies).
    """
    if pre_emph_coef is not None:
        pred = apply_pre_emphasis_filter(pred, pre_emph_coef)
        target = apply_pre_emphasis_filter(target, pre_emph_coef)
    if file_energy.shape != pred.shape[:1]:
        raise ValueError(f"file_energy shape {tuple(file_energy.shape)} != batch {pred.shape[:1]}")
    if not bool((file_energy > 0).all()):
        raise ValueError("file_energy must be positive (silent train file?)")
    dims = tuple(range(1, pred.ndim))
    mse_b = (pred - target).pow(2).mean(dim=dims)
    return (mse_b / file_energy.to(mse_b.dtype)).mean()


def file_train_energies(y_list, start: int, stop: int, pre_emph_coef: float | None = 0.85):
    """Mean (pre-emphasized) target power per file over [start, stop), in float64.

    Same filter as the loss, so for a crop that matches the file average the
    loss equals that crop's ESR.
    """
    import numpy as np
    out = []
    for y in y_list:
        seg = np.asarray(y[start:stop], dtype=np.float64)
        if pre_emph_coef is not None:
            seg = seg[1:] - pre_emph_coef * seg[:-1]
        out.append(float(np.mean(seg * seg)))
    return torch.tensor(out, dtype=torch.float32)


def floor_file_energies(table: torch.Tensor, floor_db_below_median: float | None):
    """Floor per-file energy at median(E) * 10**(-floor_db / 10).

    Per-file normalization can let nearly silent files dominate the objective.
    In the four-knob dataset, the quietest setting is about 59 dB below the median,
    so even a -70 dBFS model offset can yield an ESR of about 70. A floor 30 dB below
    the median affects only the three recordings below about -60 dBFS. Because it is
    relative to the median, the floor is independent of calibration gain. Compute
    it separately for raw and pre-emphasized tables when calling this function.

    Return the floored table, floor value, and indices of the affected files.
    """
    if floor_db_below_median is None:
        return table, None, []
    if not (floor_db_below_median > 0 and floor_db_below_median < 200):
        raise ValueError("energy floor dB must be in (0, 200)")
    if table.ndim != 1 or not bool((table >= 0).all()):
        raise ValueError("energy table must be 1-D and nonnegative")
    median = float(table.double().median())
    if not median > 0:
        raise ValueError("median file energy is zero: dataset silenzioso?")
    floor = median * 10.0 ** (-floor_db_below_median / 10.0)
    floored = [i for i, e in enumerate(table.tolist()) if e < floor]
    return table.clamp_min(floor), floor, floored


def floor_file_energies_paired(raw_table: torch.Tensor, pre_table: torch.Tensor, floor_db_below_median: float | None):
    """Floor per-file raw energy relative to its median for the V4 training loss.

    For raw energy below the floor, multiply raw and pre-emphasized energy by
    the same factor, floor / E_raw. This preserves their ratio and avoids flooring
    a file solely because it has little high-frequency energy.

    Return the floored raw and pre-emphasized tables, raw floor, and affected indices.
    """
    raw_f, floor, idx = floor_file_energies(raw_table, floor_db_below_median)
    if floor is None:
        return raw_table, pre_table, None, []
    if pre_table.shape != raw_table.shape:
        raise ValueError("raw/pre tables must have the same shape")
    factor = raw_f / raw_table.clamp_min(1e-30)
    return raw_f, pre_table * factor, floor, idx
