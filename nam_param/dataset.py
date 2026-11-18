"""
Parametric NAM dataset: shared input.wav + per-knob target WAVs from manifest.json.

NAM v3_0_0 layout (from nam.train.core._V3_DATA_INFO):
  train_start       = 480_000
  validation_start  = -432_000   (last 9 s)
  t_validate        = 432_000
  first_blips_start = 480_000

For short/synthetic datasets (n <= train_start + t_validate), falls back to
using most of the file for train and the last 20% for val.
"""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

V3_SAMPLE_RATE = 48000
V3_TRAIN_START = 480_000
V3_VALIDATION_START = -432_000
V3_T_VALIDATE = 432_000


def _read_wav_mono_f32(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            nch = w.getnchannels()
            sr = w.getframerate()
            sw = w.getsampwidth()
            n = w.getnframes()
            raw = w.readframes(n)
    except wave.Error:
        # wave legge PCM, non IEEE float: niente euristiche sui bit di un int32.
        from scipy.io import wavfile
        sr, audio = wavfile.read(path)
        if audio.ndim != 1 or not audio.size or audio.dtype.kind not in ("f", "i"):
            raise ValueError(f"Expected mono PCM or IEEE-float WAV: {path}")
        if not np.isfinite(audio).all():
            raise ValueError(f"Non-finite WAV samples: {path}")
        if audio.dtype.kind == "i":
            audio = audio.astype(np.float32) / float(2 ** (audio.dtype.itemsize * 8 - 1))
        return audio.astype(np.float32), sr
    if len(raw) != n * sw * nch:
        raise ValueError(f"Truncated WAV data: {path}")
    if nch != 1 or n == 0:
        raise ValueError(f"Expected non-empty mono WAV: {path}")
    if sw == 3:
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        b = (
            a[:, 0].astype(np.int32)
            | (a[:, 1].astype(np.int32) << 8)
            | (a[:, 2].astype(np.int32) << 16)
        )
        b = np.where(b >= 0x800000, b - 0x1000000, b).astype(np.float32) / 8388608.0
    elif sw == 2:
        b = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        b = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sampwidth {sw} for {path}")
    return b, sr


def _write_wav_mono_f32_pcm24(path: Union[str, Path], audio: np.ndarray, sr: int = 48000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(audio.astype(np.float64), -1.0, 1.0)
    i = np.round(x * 8388607.0).astype(np.int32)
    i = np.clip(i, -8388608, 8388607)
    u = i.astype(np.uint32) & 0xFFFFFF
    packed = np.empty(len(u) * 3, dtype=np.uint8)
    packed[0::3] = u & 0xFF
    packed[1::3] = (u >> 8) & 0xFF
    packed[2::3] = (u >> 16) & 0xFF
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(packed.tobytes())


_COND_TOKEN = re.compile(r"^([A-Za-z][A-Za-z]*?)(-?\d+(?:\.\d+)?)$")


def parse_condition_name(name: Union[str, Path]):
    """'train/gain0.00_treble0.50_mid1.00_bass0.50.wav' -> (['gain','treble','mid','bass'], [0,0.5,1,0.5]).

    Solleva ValueError se il nome non è nel formato nome+valore separati da '_'.
    """
    stem = Path(str(name).replace("\\", "/")).stem
    names, values = [], []
    for token in stem.split("_"):
        m = _COND_TOKEN.match(token)
        if not m:
            raise ValueError(f"Nome file non nel formato knob+valore: {name!r} (token {token!r})")
        names.append(m.group(1).lower())
        values.append(float(m.group(2)))
    if len(set(names)) != len(names):
        raise ValueError(f"Knob ripetuto nel nome file: {name!r}")
    return names, values


def infer_knob_names(manifest: dict, atol: float = 1e-3) -> List[str]:
    """Ordine dei knob in g: manifest['knob_names'] se presente, altrimenti dai nomi file.

    Ogni entry train/test deve avere gli stessi nomi nello stesso ordine e valori
    uguali a g (entro atol): così un manifest scritto a mano o rinominato male si
    ferma qui invece di allenare con le manopole scambiate.
    Dataset legacy a 2 knob senza nomi leggibili: ['gain', 'treble'].
    """
    explicit = manifest.get("knob_names")
    entries = list(manifest.get("train", [])) + list(manifest.get("test", []))
    if not entries:
        raise ValueError("Manifest senza entry train/test")
    parsed = None
    try:
        parsed = [parse_condition_name(e["y"]) for e in entries]
    except ValueError:
        parsed = None
    if parsed is not None:
        names = parsed[0][0]
        for e, (n, v) in zip(entries, parsed):
            g = list(e["g"])
            if n != names:
                raise ValueError(f"Ordine knob incoerente: {e['y']} ha {n}, attesi {names}")
            if len(g) != len(v) or any(abs(a - b) > atol for a, b in zip(g, v)):
                raise ValueError(f"g={g} non corrisponde al nome file {e['y']}")
        if explicit is not None and [str(x).lower() for x in explicit] != names:
            raise ValueError(f"manifest knob_names={explicit} ma i nomi file dicono {names}")
        return names
    if explicit is not None:
        return list(explicit)
    if all(len(e["g"]) == 2 for e in entries):
        return ["gain", "treble"]
    raise ValueError("Impossibile ricavare i nomi dei knob: aggiungi 'knob_names' al manifest")


def read_knob_names(dataset_dir: Union[str, Path]) -> List[str]:
    with open(Path(dataset_dir) / "manifest.json") as f:
        return infer_knob_names(json.load(f))


def resolve_regions(n: int, v3: bool = True):
    """Return (train_start, train_stop, val_start, val_stop)."""
    if n < 2:
        raise ValueError("Audio too short for disjoint train/val regions")
    if v3 and n > V3_TRAIN_START + V3_T_VALIDATE:
        return V3_TRAIN_START, n + V3_VALIDATION_START, n + V3_VALIDATION_START, n
    # short / synthetic fallback
    val_len = max(1, n // 5, min(n // 4, 48000))
    train_start = 0
    train_stop = n - val_len
    return train_start, train_stop, train_stop, n


class ParametricNAMDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: str = "train",
        segment_samples: int = 48000,
        receptive_field: int = 4093,
        entries: Optional[str] = None,
        num_segments: Optional[int] = None,
        seed: int = 0,
        load_full: bool = False,
        balanced: bool = True,
        fixed_g: Optional[Sequence[float]] = None,
        fixed_g_atol: float = 1e-5,
        return_file_index: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir)
        with open(self.dataset_dir / "manifest.json") as f:
            self.manifest = json.load(f)
        self.sr = int(self.manifest.get("sample_rate", V3_SAMPLE_RATE))
        self.split = split
        self.segment_samples = int(segment_samples)
        self.receptive_field = int(receptive_field)
        if self.segment_samples < 1 or self.receptive_field < 1:
            raise ValueError("segment_samples and receptive_field must be positive")
        self.rng = np.random.default_rng(seed)
        self._worker_seed = None
        self.knob_names = infer_knob_names(self.manifest)
        if len(set(self.knob_names)) != len(self.knob_names) or not self.knob_names:
            raise ValueError("Invalid manifest knob_names")
        self.balanced = bool(balanced)
        # Se True i crop restituiscono anche l'indice del file (condizione):
        # serve al trainer per la loss normalizzata sull'energia del file.
        self.return_file_index = bool(return_file_index)
        self._rr_counter = 0

        entries_key = entries or ("test" if split == "full" else "train")
        self.entries: List[dict] = list(self.manifest[entries_key])
        if not self.entries:
            raise ValueError(f"No entries in manifest['{entries_key}']")
        for e in self.entries:
            g = np.asarray(e["g"], dtype=np.float32)
            if g.shape != (len(self.knob_names),) or not np.isfinite(g).all():
                raise ValueError(f"Invalid g / knob_names for {e['y']}")
        if fixed_g is not None:
            target = np.asarray(list(fixed_g), dtype=np.float32)
            if target.shape != (len(self.knob_names),):
                raise ValueError("fixed_g length must match knob_names")
            filtered = [
                e for e in self.entries
                if np.allclose(np.asarray(e["g"], dtype=np.float32), target, atol=fixed_g_atol)
            ]
            if not filtered:
                raise ValueError(
                    f"fixed_g={list(target)} matched no entries in manifest['{entries_key}']"
                )
            self.entries = filtered

        x_name = self.manifest.get("x", "input.wav")
        x_path = self.dataset_dir / x_name
        if not x_path.exists():
            alt = self.dataset_dir.parent / x_name
            x_path = alt if alt.exists() else x_path
        self.x_full, sr_x = _read_wav_mono_f32(x_path)
        if sr_x != self.sr:
            raise ValueError(f"input.wav sr {sr_x} != manifest {self.sr}")

        self.x_list: List[np.ndarray] = []
        self.y_list: List[np.ndarray] = []
        self.g_list: List[np.ndarray] = []
        for e in self.entries:
            x = self.x_full
            if "x" in e:
                take_input = self.dataset_dir / e["x"]
                x, take_sr = _read_wav_mono_f32(take_input)
                if take_sr != self.sr or len(x) != len(self.x_full):
                    raise ValueError(f"{take_input} differs from the reference sample rate or length")
            self.x_list.append(x)
            y_path = self.dataset_dir / e["y"]
            y, sr_y = _read_wav_mono_f32(y_path)
            if sr_y != self.sr:
                raise ValueError(f"{y_path} sr {sr_y} != {self.sr}")
            if len(y) != len(self.x_full):
                raise ValueError(
                    f"{y_path} length {len(y)} != input {len(self.x_full)}"
                )
            self.y_list.append(y)
            self.g_list.append(np.asarray(e["g"], dtype=np.float32))

        n = len(self.x_full)
        tr_s, tr_e, va_s, va_e = resolve_regions(n)
        if split == "train":
            self.region_start, self.region_stop = tr_s, tr_e
        elif split == "val":
            self.region_start, self.region_stop = va_s, va_e
        elif split == "full":
            self.region_start, self.region_stop = 0, n
        else:
            raise ValueError(split)

        self.region_len = self.region_stop - self.region_start
        self.load_full = load_full
        if not load_full:
            self.segment_samples = min(self.segment_samples, self.region_len)

        if num_segments is None:
            self.num_segments = max(
                1, (self.region_len // max(self.segment_samples, 1)) * len(self.y_list)
            )
        else:
            self.num_segments = int(num_segments)
        if self.num_segments < 1:
            raise ValueError("num_segments must be positive")
        if self.balanced:
            n_files = len(self.y_list)
            self.num_segments = ((self.num_segments + n_files - 1) // n_files) * n_files

        self.val_start_abs = va_s  # for ESR on val region of full files

    def input_for(self, index: int) -> np.ndarray:
        return self.x_list[index]

    def __len__(self) -> int:
        if self.load_full:
            return len(self.y_list)
        return self.num_segments

    def __getitem__(self, idx: int):
        if self.load_full:
            file_i = idx % len(self.y_list)
            x = torch.from_numpy(self.input_for(file_i).copy())
            y = torch.from_numpy(self.y_list[file_i].copy())
            g = torch.from_numpy(self.g_list[file_i].copy())
            return x, y, g

        # Worker diversi non devono ereditare lo stesso stream NumPy via fork.
        worker = torch.utils.data.get_worker_info()
        if worker is not None and self._worker_seed != worker.seed:
            self.rng = np.random.default_rng(worker.seed)
            self._worker_seed = worker.seed
        n_files = len(self.y_list)
        if self.balanced:
            # Round-robin over conditions so every batch epoch covers all knobs evenly.
            file_i = int((idx if idx is not None else self._rr_counter) % n_files)
            self._rr_counter += 1
        else:
            file_i = int(self.rng.integers(0, n_files))
        max_start = self.region_stop - self.segment_samples
        if max_start < self.region_start:
            start = self.region_start
        else:
            start = int(self.rng.integers(self.region_start, max_start + 1))
        stop = start + self.segment_samples
        # Il target inizia a start; x contiene anche RF-1 campioni di passato.
        # Il trainer usa pad_start=False: niente falsi transitori a ogni crop.
        ctx = self.receptive_field - 1
        left = max(0, start - ctx)
        x_np = self.input_for(file_i)[left:stop].copy()
        if start < ctx:
            x_np = np.pad(x_np, (ctx - start, 0))
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(self.y_list[file_i][start:stop].copy())
        g = torch.from_numpy(self.g_list[file_i].copy())
        if self.return_file_index:
            return x, y, g, file_i
        return x, y, g
