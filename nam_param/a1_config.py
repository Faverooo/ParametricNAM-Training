"""
NAM A1-standard WaveNet architecture definition.

A1 is the historical NAM WaveNet family (standard / lite / feather / nano).
After Architecture A2 (packed WaveNet) became the Colab/GUI default, those
classic configs are referred to as A1-*. See:
  https://www.neuralampmodeler.com/post/architecture-a2

This file pins A1-standard to the config shipped with neural-amp-modeler
at tag v0.13.0 as nam_full_configs/models/wavenet.json.
"""

from copy import deepcopy

# Recorded from the cloned repo at install time.
NAM_TAG = "v0.13.0"
NAM_COMMIT = "f26112906de06ec6b796ad6d1982e29eed83144e"

A1_STANDARD_NOTES = """
A1-standard (aka historical "standard WaveNet"):
  - Two LayerArrays, condition_size=1 (condition = input audio via input_mixer)
  - Array 0: channels=16, Tanh, kernel=3, dilations 1..512, head -> 8 ch
  - Array 1: channels=8,  Tanh, kernel=3, dilations 1..512, head -> 1 ch
  - head_scale=0.02, no top-level head
  - Receptive field = 4093 samples @ 48 kHz
  - ~13.8k parameters (exact 13801 without knob projections)

Existing NAM condition path (input_mixer): each layer does
  z1 = conv_dilated(x) + Conv1x1_biasFalse(condition_audio)
before the activation. That condition is a *time-varying audio* signal
(typically the raw input), NOT a constant knob vector. We therefore add an
explicit per-layer Linear(knob_dim -> mid_channels) path for knobs, zero-init
so the model starts identical to stock A1 (warm-start friendly).
"""

A1_STANDARD_CONFIG = {
    "layers_configs": [
        {
            "condition_size": 1,
            "input_size": 1,
            "channels": 16,
            "head": {"out_channels": 8, "kernel_size": 1, "bias": False},
            "kernel_size": 3,
            "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            "activation": "Tanh",
        },
        {
            "condition_size": 1,
            "input_size": 16,
            "channels": 8,
            "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
            "kernel_size": 3,
            "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            "activation": "Tanh",
        },
    ],
    "head_scale": 0.02,
}


def a1_standard_config(sample_rate: float = 48000.0) -> dict:
    cfg = deepcopy(A1_STANDARD_CONFIG)
    cfg["sample_rate"] = sample_rate
    return cfg
