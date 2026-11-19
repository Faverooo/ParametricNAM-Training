import json

import numpy as np

from nam_param.dataset import ParametricNAMDataset, _write_wav_mono_f32_pcm24
from run_training import stage_dataset


def test_capture_inputs_survive_staging_and_pair_with_their_targets(tmp_path):
    source = tmp_path / "capture"
    staged = tmp_path / "staged"
    source.mkdir()
    staged.mkdir()
    for filename, amplitude in {
        "reference.wav": 0.1,
        "input-a.wav": 0.2,
        "input-b.wav": 0.3,
        "loopback.wav": 0.4,
        "target-b.wav": 0.5,
    }.items():
        _write_wav_mono_f32_pcm24(source / filename, np.full(10000, amplitude, dtype=np.float32), 48000)
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({
        "x": "reference.wav",
        "sample_rate": 48000,
        "knob_names": ["gain"],
        "train": [{"x": "input-a.wav", "y": "loopback.wav", "g": [0.2]}],
        "test": [{"x": "input-b.wav", "y": "target-b.wav", "g": [0.8]}],
    }))

    stage_dataset(manifest, staged)
    training = ParametricNAMDataset(staged, segment_samples=512, receptive_field=1)
    x, y, _ = training[0]
    assert np.allclose(x, 0.2) and np.allclose(y, 0.4)

    test = ParametricNAMDataset(staged, split="full", load_full=True)
    x, y, _ = test[0]
    assert np.allclose(x, 0.3) and np.allclose(y, 0.5)
