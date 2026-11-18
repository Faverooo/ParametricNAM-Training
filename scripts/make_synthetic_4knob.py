#!/usr/bin/env python3
"""Dataset sintetico a 4 knob con la STESSA struttura di dataset_4knob (per test/smoke).

81 train (griglia {0,0.5,1}^4) + 8 test fuori griglia, nomi
train/gain0.00_treble0.50_mid1.00_bass0.50.wav, manifest.json senza knob_names
(come quello reale), input.wav dentro la cartella, 48 kHz mono 24 bit.
Il target è un "amp" giocattolo: drive tanh(gain) -> tone stack (bass/mid/treble
come pesi di tre bande) -> livello. Con treble=mid=bass=0 l'uscita è ~60 dB più
bassa, come nel tone stack Fender reale (serve a provare il pavimento della loss).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nam_param.dataset import _write_wav_mono_f32_pcm24, _read_wav_mono_f32  # noqa: E402

TEST_G = [(0.25, 0.25, 0.25, 0.25), (0.25, 0.75, 0.5, 0.75), (0.75, 0.25, 0.75, 0.25),
          (0.75, 0.75, 0.25, 0.5), (0.15, 0.4, 0.65, 0.85), (0.35, 0.8, 0.2, 0.3),
          (0.55, 0.2, 0.8, 0.65), (0.9, 0.3, 0.45, 0.7)]
NAMES = ("gain", "treble", "mid", "bass")


def cond_name(g):
    return "_".join(f"{n}{v:.2f}" for n, v in zip(NAMES, g)) + ".wav"


def _onepole(x, a):
    try:
        from scipy.signal import lfilter
        return lfilter([1 - a], [1, -a], x)
    except ImportError:
        pass
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):  # lento ma semplice; i test usano file corti
        acc = a * acc + (1 - a) * x[i]
        y[i] = acc
    return y


def toy_amp(x, g):
    gain, treble, mid, bass = g
    d = np.tanh((1.0 + 8.0 * gain) * x) / (1.0 + 2.0 * gain)
    lo = _onepole(d, 0.97)
    lo2 = _onepole(d, 0.75)
    band_lo, band_mid, band_hi = lo, lo2 - lo, d - lo2
    tone = 1e-4 + 0.9 * bass * band_lo + 0.9 * mid * band_mid + 0.9 * treble * band_hi
    return 0.5 * np.tanh(3.0 * tone)


def make(out: Path, seconds: float = 2.0, input_wav: Path | None = None, seed: int = 0):
    out = Path(out)
    rng = np.random.default_rng(seed)
    n = int(round(seconds * 48000))
    if input_wav is not None:
        x, sr = _read_wav_mono_f32(input_wav)
        if sr != 48000 or len(x) < n:
            raise ValueError("input_wav too short or wrong rate")
        x = x[:n].astype(np.float64)
    else:
        x = 0.3 * rng.standard_normal(n) * np.sin(np.linspace(0, 20, n)) ** 2
    _write_wav_mono_f32_pcm24(out / "input.wav", x)
    xq, _ = _read_wav_mono_f32(out / "input.wav")
    grid = [0.0, 0.5, 1.0]
    train = [(a, b, c, d) for a in grid for b in grid for c in grid for d in grid]
    manifest = {"amp": "synthetic_toy_4knob", "bit_depth": 24, "cab": "none", "channels": 1, "knobs": 4,
                "parameters": {"gain": 0, "treble": 1, "mid": 2, "bass": 3}, "sample_rate": 48000,
                "train_values": grid, "x": "input.wav", "train": [], "test": []}
    for split, gs in (("train", train), ("test", TEST_G)):
        for g in gs:
            rel = f"{split}/{cond_name(g)}"
            _write_wav_mono_f32_pcm24(out / rel, toy_amp(xq.astype(np.float64), g))
            manifest[split].append({"g": [float(v) for v in g], "y": rel})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--input-wav", default=None)
    a = ap.parse_args()
    make(Path(a.out), a.seconds, Path(a.input_wav) if a.input_wav else None)
    print("ok", a.out)
