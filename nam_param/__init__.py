"""Parametric NAM: A1 WaveNet with per-layer knob conditioning (W g + b / FiLM)."""

from .a1_config import A1_STANDARD_CONFIG, A1_STANDARD_NOTES, NAM_TAG, NAM_COMMIT
from .parametric_wavenet import ParametricWaveNet, build_a1_parametric, load_parametric
from .bake import bake_knobs_into_nam, fold_knobs_into_conv_bias, fold_knobs, verify_bake_equivalence
from .dataset import ParametricNAMDataset, V3_TRAIN_START, V3_VALIDATION_START, V3_T_VALIDATE

__all__ = [
    "A1_STANDARD_CONFIG",
    "A1_STANDARD_NOTES",
    "NAM_TAG",
    "NAM_COMMIT",
    "ParametricWaveNet",
    "build_a1_parametric",
    "load_parametric",
    "bake_knobs_into_nam",
    "fold_knobs_into_conv_bias",
    "fold_knobs",
    "verify_bake_equivalence",
    "ParametricNAMDataset",
    "V3_TRAIN_START",
    "V3_VALIDATION_START",
    "V3_T_VALIDATE",
]

__version__ = "0.1.1"
