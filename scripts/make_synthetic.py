#!/usr/bin/env python3
"""Generate a tiny synthetic parametric dataset from input.wav for unit tests / fallback."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nam_param.dataset import _read_wav_mono_f32, _write_wav_mono_f32_pcm24

def biquad_treble(x, amount, sr=48000):
    # simple 1st-order shelf-ish high emphasis via pre-emphasis blend
    # amount in [0,1]: 0 = dark (lowpass-ish), 1 = bright
    y = np.empty_like(x)
    # one-pole lowpass
    a = 0.05 + 0.9 * (1.0 - amount)  # more smoothing when dark
    acc = 0.0
    for i, s in enumerate(x):
        acc = a * acc + (1 - a) * s
        # blend dry/wet toward bright (highpassed) as amount increases
        hp = s - acc
        y[i] = (1 - amount) * acc + amount * (acc + 1.5 * hp)
    return y.astype(np.float32)

def process(x, g0, g1):
    # gain (soft clip) * treble filter
    drive = 0.5 + 8.0 * g0
    y = np.tanh(drive * x)
    y = biquad_treble(y, g1)
    return (0.7 * y).astype(np.float32)

def make(out: Path, source: Path | None = None):
    out.mkdir(parents=True, exist_ok=True)
    (out / "train").mkdir(exist_ok=True)
    (out / "test").mkdir(exist_ok=True)
    if source is None:
        sr = 48000
        time = np.arange(sr, dtype=np.float32) / sr
        rng = np.random.default_rng(42)
        x = (0.18 * np.sin(2 * np.pi * 117 * time)
             + 0.07 * np.sin(2 * np.pi * 607 * time)
             + 0.025 * rng.normal(size=sr)).astype(np.float32)
    else:
        x, sr = _read_wav_mono_f32(source)
        x = x[:sr * 6].copy()
    n = len(x)
    _write_wav_mono_f32_pcm24(out / "input.wav", x, sr)

    train_grid = [0.0, 0.5, 1.0]
    train = []
    for g0 in train_grid:
        for g1 in train_grid:
            y = process(x, g0, g1)
            name = f"train/gain{g0:.2f}_treble{g1:.2f}.wav"
            _write_wav_mono_f32_pcm24(out / name, y, sr)
            train.append({"y": name, "g": [g0, g1]})

    test_pts = [(0.25, 0.75), (0.75, 0.25), (0.4, 0.4)]
    test = []
    for g0, g1 in test_pts:
        y = process(x, g0, g1)
        name = f"test/gain{g0:.2f}_treble{g1:.2f}.wav"
        _write_wav_mono_f32_pcm24(out / name, y, sr)
        test.append({"y": name, "g": [g0, g1]})

    manifest = {
        "sample_rate": sr,
        "channels": 1,
        "bit_depth": 24,
        "amp": "synthetic_tanh",
        "cab": None,
        "parameters": {"gain": 0, "treble": 1},
        "x": "input.wav",
        "train": train,
        "test": test,
        "synthetic": True,
        "n_samples": n,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote synthetic dataset to {out} ({n} samples, {len(train)} train, {len(test)} test)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    make(args.out, args.source)
