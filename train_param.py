#!/usr/bin/env python3
"""Train a parametric A1 WaveNet on a ParametricNAM dataset (manifest.json).

The order of g comes from manifest['knob_names'] or from filenames via
nam_param.dataset.infer_knob_names. Exp1–3 use two knobs, and exp4 uses
four (gain, treble, mid, bass).

Supports:
  - variants: none | shift_linear | shift_mlp | film_linear
  - step-based training with exponential learning rate decay
  - resume from last or best checkpoint after a Colab interruption
  - ESR and pre-emphasis, with optional MRSTFT
  - balanced sampling across conditions or a fixed_g subset for the baseline
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import warnings
import numpy as np
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Headless-safe before any nam.train import path
import nam_param._tk_stub  # noqa: F401

from nam_param.device import resolve_device, device_label
from nam_param.parametric_wavenet import build_a1_parametric, load_parametric
from nam_param.variants import build_a1_parametric_variant
from nam_param.dataset import ParametricNAMDataset, read_knob_names, reference_audio_path
from nam_param.losses_util import (mse_loss, esr_loss, esr_per_item, preemphasized_esr_mean,
                                   file_normalized_esr, file_train_energies, floor_file_energies,
                                   floor_file_energies_paired)
from nam_param.warmstart import expand_knob_warm_start
from nam_param.progress import ProgressWriter
from nam_param.bake import bake_knobs_into_nam


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train parametric A1 NAM (N knobs)")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--variant", type=str, default="film_linear",
                   choices=["none", "shift_linear", "shift_mlp", "film_linear"])
    p.add_argument("--channels", type=str, default=None,
                   help="e.g. 16,8 for A1 width; default = A1 standard")
    p.add_argument("--no-bias", action="store_true", help="use_bias=False on knob Linear/FiLM shift")
    p.add_argument("--epochs", type=int, default=None, help="If set (and --steps unset), train this many epochs")
    p.add_argument("--steps", type=int, default=None, help="Total optimizer steps (preferred for Colab)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.004)
    p.add_argument("--lr-gamma", type=float, default=0.993,
                   help="ExponentialLR gamma applied every --lr-step-every steps (or each epoch)")
    p.add_argument("--lr-step-every", type=int, default=1000,
                   help="Apply LR decay every N steps when using --steps")
    p.add_argument("--segment-samples", type=int, default=48000)
    p.add_argument("--segments-per-epoch", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--init-parametric", type=str, default=None,
                   help="Warm-start weights only from a parametric .pt; fresh Adam/history, unlike resume")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="Global gradient norm limit; 0 disables clipping (legacy behaviour)")
    p.add_argument("--raw-weight", type=float, default=1.0,
                   help="Weight of unfiltered file-normalized MSE for --loss esr_dual")
    p.add_argument("--dc-weight", type=float, default=1.0,
                   help="Weight of file-normalized squared mean error for --loss esr_dc (ablation)")
    p.add_argument("--pre-emph", type=float, default=0.85)
    p.add_argument("--no-pre-emph", action="store_true")
    p.add_argument("--mrstft-weight", type=float, default=0.0,
                   help="NAM docs suggest ~2e-4 when enabled")
    p.add_argument("--loss", type=str, default="esr", choices=["esr", "esr_dual", "esr_dc", "esr_crop", "mse"],
                   help="esr (default, v2): MSE pre-enfatizzato / energia del FILE della condizione; "
                        "esr_crop: vecchia v1, ESR per crop (esplode sui crop silenziosi, solo confronto); "
                        "mse: MSE semplice; esr_dual: v2 + raw-weight * MSE non filtrato / energia FILE raw; "
                        "esr_dc: v2 + dc-weight * mean(error)^2 / energia FILE raw")
    p.add_argument("--energy-floor-db", type=float, default=None,
                   help="Loss v4: normalizzatore per file = max(E_file, mediana(E) - X dB). "
                        "Exp4 usa 30. Default None = loss v3 invariata (exp3).")
    p.add_argument("--film-lr-mult", type=float, default=1.0,
                   help="LR dei moduli di condizionamento (knob_proj) = lr x questo; 1 = un solo gruppo (exp1-3)")
    p.add_argument("--knob-names", type=str, default=None,
                   help="es. gain,treble,mid,bass; default: da manifest/nomi file (verificato)")
    p.add_argument("--log-every", type=int, default=50, help="Stampa loss/velocità ogni N step")
    p.add_argument("--progress-every", type=int, default=500,
                   help="Riscrive OUT/progress.txt ogni N step (anche a val, fine, errore)")
    p.add_argument("--early-stop-steps", type=int, default=0,
                   help="Stop after this many steps without a better validation score (0 disables)")
    p.add_argument("--val-every", type=int, default=500, help="Validate every N steps (step mode)")
    p.add_argument("--checkpoint-every", type=int, default=1000,
                   help="Write last.pt / train_state.pt every N steps")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to train_state.pt (or out dir). Default: auto-resume from --out/train_state.pt if present")
    p.add_argument("--no-auto-resume", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bake-g", type=str, default=None, help='e.g. "0.5,0.5"')
    p.add_argument("--fixed-g", type=str, default=None,
                   help='Train on a single condition only, e.g. "0.5,0.5" (baseline)')
    p.add_argument("--balanced", action="store_true", default=True)
    p.add_argument("--no-balanced", action="store_true")
    p.add_argument("--max-files-eval", type=int, default=None)
    p.add_argument("--no-cudnn-benchmark", action="store_true",
                   help="Su CUDA cudnn.benchmark è attivo di default (shape fisse: sceglie l'algoritmo conv più veloce)")
    return p.parse_args(argv)


class StatefulSegmentSampler(Sampler):
    """Resumable sampler that visits every condition in each balanced block.

    Globally shuffling the DataLoader only balances full epochs. Balanced blocks
    also cover the conditions evenly when training stops after a few steps.
    """
    def __init__(self, ds, seed=0):
        self.ds = ds
        self.generator = torch.Generator().manual_seed(seed)
        self.cursor = 0
        self.order = []
        self.new_epoch()

    def new_epoch(self):
        n = len(self.ds)
        if self.ds.balanced:
            k = len(self.ds.y_list)
            self.order = [base + i for base in range(0, n, k)
                          for i in torch.randperm(k, generator=self.generator).tolist()]
        else:
            self.order = torch.randperm(n, generator=self.generator).tolist()
        self.cursor = 0

    def __iter__(self):
        while self.cursor < len(self.order):
            idx = self.order[self.cursor]
            self.cursor += 1
            yield idx

    def __len__(self):
        return len(self.order)

    def state_dict(self):
        return {"order": self.order, "cursor": self.cursor,
                "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        if len(state["order"]) != len(self.ds):
            raise ValueError("Resume dataset length changed; use a fresh run directory")
        self.order = state["order"]
        self.cursor = state["cursor"]
        self.generator.set_state(state["generator"].cpu())


def capture_rng_state(ds, sampler, loader_generator):
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(), "dataset": ds.rng.bit_generator.state,
            "sampler": sampler.state_dict(), "loader": loader_generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state, ds, sampler, loader_generator):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    ds.rng.bit_generator.state = state["dataset"]
    sampler.load_state_dict(state["sampler"])
    loader_generator.set_state(state["loader"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def build_model(args, knob_names=None):
    knob_names = list(knob_names or ["gain", "treble"])
    channels = None
    if args.channels:
        channels = tuple(int(x) for x in args.channels.split(","))
    use_bias = not args.no_bias
    if args.variant == "shift_linear" and use_bias and channels is None:
        # Keep classic builder path for backward compatibility
        model = build_a1_parametric(knob_dim=len(knob_names), knob_names=knob_names, sample_rate=48000.0)
        model.variant = "shift_linear"
        model.use_bias = True
        model.mlp_hidden = 16
        return model
    return build_a1_parametric_variant(
        variant=args.variant,
        knob_dim=len(knob_names),
        knob_names=knob_names,
        sample_rate=48000.0,
        use_bias=use_bias,
        channels=channels,
    )


def parse_g(s: Optional[str]):
    if not s:
        return None
    return [float(x) for x in s.split(",")]


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate crops with pre-roll and equal weight per item."""
    was_training = model.training
    model.eval()
    esrs = []
    try:
        for x, y, g in loader:
            x, y, g = x.to(device), y.to(device), g.to(device)
            pred = model(x, g, pad_start=False)
            if pred.shape != y.shape:
                raise ValueError("Crop evaluation alignment mismatch")
            esrs.extend(esr_per_item(pred, y).cpu().tolist())
    finally:
        model.train(was_training)
    return sum(esrs) / max(len(esrs), 1), esrs


def validation_view(train_ds):
    """Vista deterministica: tutti i file train selezionati, intera regione val.

    Riusa gli array già caricati, senza duplicare in RAM i 25 WAV.
    """
    ds = copy.copy(train_ds)
    ds.split = "val"
    ds.region_start = ds.val_start_abs
    ds.region_stop = len(ds.x_full)
    ds.region_len = ds.region_stop - ds.region_start
    ds.load_full = True
    ds.return_file_index = False
    return ds


@torch.no_grad()
def evaluate_region_esr(model, x, y, g, device, start=0, stop=None, chunk_samples=48000,
                        return_energies=False):
    """Compute ESR over the whole region using signal and error energies.

    Each chunk uses RF-1 real pre-roll samples. Causal zero padding is used at
    the start of the file. Each target sample contributes exactly once.
    Chunking bounds memory use for long test recordings.
    """
    model.eval()
    stop = len(y) if stop is None else stop
    if not (0 <= start < stop <= len(y)) or chunk_samples <= 0:
        raise ValueError("Regione/chunk di valutazione non validi")
    ctx = model.receptive_field - 1
    g_t = torch.as_tensor(g, dtype=torch.float32).unsqueeze(0).to(device)
    error_energy = signal_energy = prediction_peak = 0.0
    for pos in range(start, stop, chunk_samples):
        end = min(pos + chunk_samples, stop)
        left = max(0, pos - ctx)
        x_t = torch.as_tensor(x[left:end]).unsqueeze(0).to(device)
        pred = model(x_t, g_t, pad_start=True)[..., pos-left:]
        target = torch.as_tensor(y[pos:end]).unsqueeze(0)
        if pred.shape != target.shape:
            raise RuntimeError(f"Eval alignment: {pred.shape} != {target.shape}")
        # Accumulate in float64 on the CPU for DirectML compatibility.
        pred = pred.cpu().double()
        target = target.double()
        error_energy += (pred - target).square().sum().item()
        signal_energy += target.square().sum().item()
        if return_energies:
            prediction_peak = max(prediction_peak, pred.abs().max().item())
    esr = error_energy / max(signal_energy, 1e-8)
    if return_energies:
        return esr, error_energy, signal_energy, prediction_peak
    return esr


@torch.no_grad()
def evaluate_per_file(model, ds_full, device, max_files=None, region="full", chunk_samples=48000,
                      floor_power=None):
    """Compute one ESR per condition over the training validation region or full test file.

    Held-out tests include all samples, including initial transients and silence.
    The region helper handles padding and pre-roll.
    """
    if region not in ("val", "full"):
        raise ValueError(region)
    model.eval()
    results = []
    n_files = len(ds_full.y_list) if max_files is None else min(len(ds_full.y_list), max_files)
    start = ds_full.val_start_abs if region == "val" else 0
    for i in range(n_files):
        y, g = ds_full.y_list[i], ds_full.g_list[i]
        x = ds_full.input_for(i) if hasattr(ds_full, "input_for") else ds_full.x_full
        e, err_e, sig_e, prediction_peak = evaluate_region_esr(model, x, y, g, device, start=start,
                                              chunk_samples=chunk_samples, return_energies=True)
        n_samp = len(y) - start
        row = {
            "index": i,
            "file": ds_full.entries[i]["y"],
            "g": g.tolist(),
            "region": region,
            "start_sample": start,
            "stop_sample": len(y),
            "num_samples": len(y) - start,
            "esr_val_region" if region == "val" else "esr_full": e,
            "prediction_peak": prediction_peak,
            "signal_power_dbfs": 10.0 * np.log10(max(sig_e / max(n_samp, 1), 1e-30)),
        }
        if floor_power is not None:
            # Use the same per-sample raw power floor as the v4 loss so nearly silent files
            # cannot dominate the mean or determine the best checkpoint by themselves.
            row["esr_floored"] = err_e / max(sig_e, floor_power * n_samp, 1e-8)
            row["floored"] = sig_e < floor_power * n_samp
        results.append(row)
    return results


def mean_validation_esr(model, val_ds, device, floor_power=None):
    """Media aritmetica per file: ogni condizione pesa uguale, nessun RNG."""
    rows = evaluate_per_file(model, val_ds, device, region="val", floor_power=floor_power)
    value = sum(r["esr_val_region"] for r in rows) / len(rows)
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite validation ESR")
    return value, rows


def floored_mean(rows, key="esr_val_region"):
    """Average the floored ESR if available, or the raw ESR otherwise."""
    vals = [r.get("esr_floored", r[key]) for r in rows]
    value = sum(vals) / len(vals)
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite floored ESR")
    return value


def knob_marginals(rows, knob_names, key):
    """Media ESR per ogni valore di ogni knob (es. tutte le condizioni con mid=0)."""
    out = {}
    for k, name in enumerate(knob_names):
        groups = {}
        for r in rows:
            groups.setdefault(round(float(r["g"][k]), 4), []).append(r.get("esr_floored", r[key]))
        out[name] = {f"{v:g}": sum(x) / len(x) for v, x in sorted(groups.items())}
    return out


def write_rows_csv(path, rows, knob_names, key):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file"] + list(knob_names) + [key, "esr_floored", "floored", "signal_power_dbfs"])
        for r in rows:
            w.writerow([r["file"]] + [f"{v:g}" for v in r["g"]] +
                       [f"{r[key]:.6g}", f"{r.get('esr_floored', r[key]):.6g}",
                        int(bool(r.get("floored", False))), f"{r.get('signal_power_dbfs', float('nan')):.2f}"])


def copy_checkpoint(source: Path, destination: Path):
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def save_train_state(path: Path, model, opt, sched, step, epoch, best_val, args_dict, history, rng_state=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Model weights via export_parametric side file + full train state
    payload = {
        "rng_state": rng_state,
        "step": step,
        "epoch": epoch,
        "best_val": best_val,
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "args": args_dict,
        "history": history,
        "model_state_dict": model.state_dict(),
        "config": {
            "variant": getattr(model, "variant", "shift_linear"),
            "use_bias": getattr(model, "use_bias", True),
            "mlp_hidden": getattr(model, "mlp_hidden", 16),
            "knob_dim": model.knob_dim,
            "knob_names": model.knob_names,
            "sample_rate": model.sample_rate or 48000.0,
            "channels": args_dict.get("channels"),
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    # Keep the previous checkpoint as a backup in case Colab stops while Drive
    # syncs the new file. Resume can then use train_state.prev.pt.
    prev = previous_state_path(path)
    if path.exists():
        copy_checkpoint(path, prev)
    tmp.replace(path)


def previous_state_path(path: Path) -> Path:
    return path.with_name(path.stem + ".prev" + path.suffix)


def train_state_exists(path: Path) -> bool:
    return path.is_file() or previous_state_path(path).is_file()


def load_train_state(path: Path, device, *, with_source=False):
    """Load the requested checkpoint or its backup, optionally with its source path."""
    path = Path(path)
    errors = []
    for cand in (path, previous_state_path(path)):
        if not cand.exists():
            continue
        try:
            state = torch.load(cand, map_location=device, weights_only=False)
            if not isinstance(state, dict) or "model_state_dict" not in state:
                raise ValueError("contenuto non valido")
            if cand != path:
                reason = errors[-1] if errors else "file assente"
                warnings.warn(f"{reason}: riprendo da {cand.name}")
                print(f"ATTENZIONE: {reason}, riprendo dal checkpoint precedente {cand.name}")
            return (state, cand) if with_source else state
        except Exception as e:  # The file may be truncated after an interruption.
            errors.append(f"{cand.name}: {e}")
    raise RuntimeError("Nessun train_state leggibile: " + "; ".join(errors))


# Checkpoints record the loss definition. V1 uses per-crop ESR
# and is incompatible with the V2 objective.
LOSS_DEF = "v2_file_normalized_esr"  # legacy API/default retained
LOSS_DEF_V3 = "v3_dual_file_normalized_esr"
LOSS_DEF_DC = "v3_dc_file_normalized_esr"
# V4 adds a per-file normalization floor to the V3 dual loss.
# Use --init-parametric to import V3 weights into a new V4 run.
LOSS_DEF_V4 = "v4_dual_floored_file_normalized_esr"


def loss_definition(loss_kind, energy_floor_db=None):
    if energy_floor_db is not None:
        if loss_kind != "esr_dual":
            raise ValueError("--energy-floor-db è supportato solo con --loss esr_dual (v4)")
        return LOSS_DEF_V4
    return {"esr_dual": LOSS_DEF_V3, "esr_dc": LOSS_DEF_DC}.get(loss_kind, LOSS_DEF)


def _fmt_g(g):
    return "(" + ",".join(f"{v:.2f}" for v in g) + ")"


def compact_validation_summary(rows):
    key = lambda r: r.get("esr_floored", r["esr_val_region"])
    worst = sorted(rows, key=key, reverse=True)[:3]
    low = [key(r) for r in rows if abs(r["g"][0]) < 1e-6]
    gain0 = sum(low) / len(low) if low else None
    text = "; ".join(f"{_fmt_g(r['g'])}={key(r):.5g}" for r in worst)
    return {"worst3": worst, "gain0_mean_esr": gain0}, "worst3 " + text + (f" | gain0_mean={gain0:.6g}" if gain0 is not None else "")


@torch.no_grad()
def evaluate_zero_input(model, g_list, device, measure_samples=4096, batch_size=8):
    """Steady zero-input response: RF-1 zero pre-roll, then measured output.

    No filtering or correction is applied; this does not alter the baked NAM.
    """
    if measure_samples < 1 or batch_size < 1:
        raise ValueError("positive measure_samples/batch_size required")
    was_training = model.training
    model.eval()
    rows = []
    try:
        for start in range(0, len(g_list), batch_size):
            gs = np.asarray(g_list[start:start+batch_size], dtype=np.float32)
            g = torch.as_tensor(gs, device=device)
            x = torch.zeros(len(gs), model.receptive_field - 1 + measure_samples, device=device)
            p = model(x, g, pad_start=False).cpu().double()
            for gi, pred in zip(gs, p):
                rms = pred.square().mean().sqrt().item()
                rows.append({"g": gi.tolist(), "mean": pred.mean().item(), "rms": rms,
                             "rms_dbfs": 20.0 * np.log10(max(rms, 1e-15)),
                             "std": pred.std(unbiased=False).item(), "peak_abs": pred.abs().max().item(),
                             "measure_samples": measure_samples})
    finally:
        model.train(was_training)
    return rows


def compute_batch_loss(pred, y, pre_emph, loss_kind, mrstft_weight, mrstft_mod, file_energy=None,
                       raw_file_energy=None, raw_weight=1.0, dc_weight=1.0):
    if pred.shape != y.shape:
        raise ValueError(f"Training alignment: {pred.shape} != {y.shape}")
    if pre_emph is not None and y.shape[-1] < 2:
        raise ValueError("Pre-emphasis requires at least two target samples")
    pred_n, y_n = pred, y
    if loss_kind in ("esr", "esr_dual", "esr_dc"):
        if file_energy is None:
            raise ValueError("loss 'esr' (v2) needs file_energy: per-item energy of the item's train file")
        loss = file_normalized_esr(pred_n, y_n, file_energy, pre_emph_coef=pre_emph)
        if loss_kind in ("esr_dual", "esr_dc"):
            if raw_file_energy is None:
                raise ValueError("v3 loss requires raw_file_energy")
            if raw_file_energy.shape != y.shape[:1] or not bool((raw_file_energy > 0).all()):
                raise ValueError("raw_file_energy must be positive with shape (B,)")
            if loss_kind == "esr_dual":
                if not np.isfinite(raw_weight) or raw_weight <= 0:
                    raise ValueError("raw_weight must be finite and positive")
                loss = loss + raw_weight * file_normalized_esr(pred_n, y_n, raw_file_energy, None)
            else:
                if not np.isfinite(dc_weight) or dc_weight <= 0:
                    raise ValueError("dc_weight must be finite and positive")
                dims = tuple(range(1, y.ndim))
                loss = loss + dc_weight * (((pred_n - y_n).mean(dim=dims).square()) / raw_file_energy).mean()
    elif loss_kind == "esr_crop":
        loss = preemphasized_esr_mean(pred_n, y_n, pre_emph_coef=pre_emph)
    elif loss_kind == "mse":
        loss = mse_loss(pred_n, y_n, pre_emph_coef=pre_emph)
    else:
        raise ValueError(loss_kind)
    if mrstft_mod is not None and mrstft_weight > 0:
        from nam.models.losses import multi_resolution_stft_loss
        spectral = multi_resolution_stft_loss(
            pred_n, y_n, loss_func=mrstft_mod,
            device=torch.device("cpu") if "privateuseone" in str(pred.device) else pred.device
        )
        loss = loss + mrstft_weight * spectral.to(loss.device)
    return loss


def dataset_fingerprint(dataset_dir):
    return hashlib.sha256((Path(dataset_dir) / "manifest.json").read_bytes()).hexdigest()


def audio_fingerprint(dataset_dir, manifest):
    root = Path(dataset_dir)
    names = set()
    for split in ("train", "test"):
        for entry in manifest.get(split) or []:
            names.add(entry["y"])
            if "x" in entry:
                names.add(entry["x"])
    fingerprint = hashlib.sha256()
    paths = [reference_audio_path(root, manifest), *(root / name for name in sorted(names))]
    for path in paths:
        with path.open("rb") as file:
            fingerprint.update(hashlib.file_digest(file, "sha256").digest())
    return fingerprint.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    if args.steps is None and args.epochs is None:
        args.steps = 50000
    if args.early_stop_steps < 0 or (args.early_stop_steps and args.steps is None):
        raise ValueError("Early stopping requires step-based training and a nonnegative patience")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prog = ProgressWriter(out / "progress.txt", run_name=out.name, total_steps=args.steps)
    try:
        return _main(args, out, prog)
    except BaseException as e:
        prog.error(e)
        raise


def _main(args, out, prog):
    for name in ("batch_size", "segment_samples", "lr_step_every", "checkpoint_every", "val_every", "log_every",
                 "progress_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or (args.steps is not None and args.steps <= 0) or (args.epochs is not None and args.epochs <= 0):
        raise ValueError("Invalid workers/steps/epochs")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    balanced = not args.no_balanced
    if not np.isfinite(args.grad_clip) or args.grad_clip < 0:
        raise ValueError("grad_clip must be finite and nonnegative")
    if args.init_from and args.init_parametric:
        raise ValueError("choose --init-from OR --init-parametric")
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            resume_path = resume_path / "train_state.pt"
        if not train_state_exists(resume_path):
            raise FileNotFoundError(resume_path)
    elif not args.no_auto_resume and train_state_exists(out / "train_state.pt"):
        resume_path = out / "train_state.pt"
    if args.loss == "esr_dual" and (not np.isfinite(args.raw_weight) or args.raw_weight <= 0):
        raise ValueError("raw_weight must be finite and positive")
    if args.loss == "esr_dc" and (not np.isfinite(args.dc_weight) or args.dc_weight <= 0):
        raise ValueError("dc_weight must be finite and positive")
    fixed_g = parse_g(args.fixed_g)
    if args.energy_floor_db is not None and not (0 < args.energy_floor_db < 200):
        raise ValueError("--energy-floor-db must be in (0, 200)")
    loss_definition(args.loss, args.energy_floor_db)  # Validate the combination before training.
    knob_names = read_knob_names(args.dataset)
    if args.knob_names:
        wanted = [k.strip().lower() for k in args.knob_names.split(",")]
        if wanted != knob_names:
            raise ValueError(f"--knob-names {wanted} != dataset {knob_names}")
    if fixed_g is not None and len(fixed_g) != len(knob_names):
        raise ValueError(f"--fixed-g needs {len(knob_names)} values for {knob_names}")
    channels_tuple = tuple(int(x) for x in args.channels.split(",")) if args.channels else None

    device = resolve_device(args.device)
    print(f"Device: {device_label(device)}")
    is_cuda = device.type == "cuda"
    if is_cuda and not args.no_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    print(f"Variant: {args.variant}  channels={channels_tuple or 'A1(16,8)'}  fixed_g={fixed_g}")
    print(f"Knob ({len(knob_names)}): {knob_names}")

    model = build_model(args, knob_names)
    warm_info = None
    if args.init_from and resume_path is None:
        print(f"Warm-start from {args.init_from}")
        model.load_stock_nam_weights(args.init_from)  # Fail on a weight mismatch.
    if args.init_parametric and resume_path is None:
        source = load_parametric(args.init_parametric)
        if getattr(source, "variant", "shift_linear") != args.variant:
            raise ValueError("parametric warm-start variant mismatch")
        if list(source.knob_names) == list(model.knob_names):
            model.load_state_dict(source.state_dict(), strict=True)
        else:
            # For example, new mid and bass columns start at zero when expanding exp3 to exp4.
            warm_info = expand_knob_warm_start(model, source)
            print(f"Warm-start ESPANSO {warm_info['source_knobs']} -> {warm_info['target_knobs']}: "
                  f"{len(warm_info['expanded'])} matrici FiLM con colonne nuove {warm_info['new_knobs']} a zero")
        print(f"Warm-start parametric weights only: {args.init_parametric} (fresh Adam unless resuming)")
    model.to(device)

    rf = model.receptive_field
    train_ds = ParametricNAMDataset(
        args.dataset,
        split="train",
        segment_samples=args.segment_samples,
        receptive_field=rf,
        entries="train",
        num_segments=args.segments_per_epoch,
        seed=args.seed,
        balanced=balanced,
        fixed_g=fixed_g,
        return_file_index=True,
    )
    # The V2 denominator is each file's full training-region energy.
    # Compute it once with the same pre-emphasis as the loss.
    pre_emph = None if args.no_pre_emph else args.pre_emph
    file_energy_table = file_train_energies(
        train_ds.y_list, train_ds.region_start, train_ds.region_stop, pre_emph
    ).to(device)
    raw_file_energy_table = (file_train_energies(
        train_ds.y_list, train_ds.region_start, train_ds.region_stop, None
    ).to(device) if args.loss in ("esr_dual", "esr_dc") else None)
    # Keep held-out tests out of best checkpoint selection.
    # A deterministic pass covers the full validation region of each training condition,
    # with equal weight per condition and only fixed_g for the baseline.
    raw_floor, floored_idx = None, []
    if args.energy_floor_db is not None:
        # V4 requires esr_dual, so the raw energy table is available.
        raw_file_energy_table, file_energy_table, raw_floor, floored_idx = floor_file_energies_paired(
            raw_file_energy_table, file_energy_table, args.energy_floor_db)
        names = [train_ds.entries[i]["y"] for i in floored_idx]
        print(f"Loss v4: pavimento normalizzatore = mediana - {args.energy_floor_db:g} dB = "
              f"{10*np.log10(raw_floor):.1f} dBFS (potenza raw); file pavimentati {len(names)}/{len(train_ds.y_list)}: {names}")
    if train_ds.sr != 48000 or train_ds.knob_names != model.knob_names:
        raise ValueError(f"Trainer expects 48 kHz and g ordered as {model.knob_names}")
    val_ds = validation_view(train_ds)

    sampler = StatefulSegmentSampler(train_ds, seed=args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed + 12345)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, generator=loader_generator,
        pin_memory=is_cuda,
    )
    if args.num_workers:
        warnings.warn("num_workers>0: RNG worker indipendenti, ma prefetch impedisce resume identico. "
                      "Usa --num-workers 0 (default Colab) per ripresa esatta.")

    if not (np.isfinite(args.film_lr_mult) and args.film_lr_mult > 0):
        raise ValueError("--film-lr-mult must be finite and positive")
    if args.film_lr_mult == 1.0:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        knob_ids = {id(q) for proj in model.knob_projections for q in proj.parameters()}
        base_params = [q for q in model.parameters() if id(q) not in knob_ids]
        knob_params = [q for q in model.parameters() if id(q) in knob_ids]
        opt = torch.optim.Adam([{"params": base_params, "lr": args.lr},
                                {"params": knob_params, "lr": args.lr * args.film_lr_mult}])
        print(f"LR WaveNet {args.lr:g}, LR condizionamento FiLM {args.lr * args.film_lr_mult:g}")
    # Step-mode: decay every lr_step_every steps; epoch-mode: every epoch
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_gamma)

    mrstft_mod = None
    if args.mrstft_weight and args.mrstft_weight > 0:
        from nam._dependencies.auraloss.freq import MultiResolutionSTFTLoss
        # DirectML can lack STFT support. Autograd works across the CPU transfer.
        mrstft_device = torch.device("cpu") if "privateuseone" in str(device) else device
        mrstft_mod = MultiResolutionSTFTLoss().to(mrstft_device)

    args_dict = vars(args).copy()
    args_dict["run_id"] = uuid.uuid4().hex
    args_dict["channels"] = list(channels_tuple) if channels_tuple else None
    args_dict["loss_def"] = loss_definition(args.loss, args.energy_floor_db)
    args_dict["knob_names_resolved"] = knob_names
    # Compare dataset contents because the path can change between sessions.
    args_dict["dataset_fingerprint"] = dataset_fingerprint(args.dataset)
    args_dict["audio_fingerprint"] = audio_fingerprint(args.dataset, train_ds.manifest)
    args_dict["warm_start_info"] = warm_info
    select_floored = args.energy_floor_db is not None
    val_floor_power = raw_floor if select_floored else None
    print(f"Loss: {args_dict['loss_def']} | grad_clip={args.grad_clip:g}")
    samples_per_step = args.batch_size * train_ds.segment_samples
    region_samples = train_ds.region_len * len(train_ds.y_list)
    if args.steps is not None:
        print(f"Budget: {args.steps} steps x {samples_per_step} target samples = "
              f"{args.steps * samples_per_step / region_samples:.3g} passate sulla regione train "
              f"({len(train_ds.y_list)} file x {train_ds.region_len / train_ds.sr:.0f} s)")

    history = []
    best_val = float("inf")
    step = 0
    epoch = 0
    t0 = time.time()

    # Resume
    resume_in_place = False
    checkpoint_current = False
    best_step = None
    if resume_path is not None:
        resume_in_place = resume_path.parent.resolve() == out.resolve()
        checkpoint_current = resume_in_place and resume_path.name == "train_state.pt"
        print(f"Resuming from {resume_path}")
        state, loaded_state_path = load_train_state(resume_path, device, with_source=True)
        old = state.get("args", {})
        prior_run_id = old.get("run_id")
        if prior_run_id is not None:
            args_dict["run_id"] = prior_run_id
        if old.get("loss_def") != args_dict["loss_def"]:
            raise ValueError(
                f"Checkpoint {resume_path} usa la loss {old.get('loss_def', 'v1 (ESR per crop)')}, "
                f"non {args_dict['loss_def']}: non riprenderlo, usa una cartella --out nuova (o cancella la vecchia)."
            )
        if old.get("dataset_fingerprint", args_dict["dataset_fingerprint"]) != args_dict["dataset_fingerprint"]:
            raise ValueError(f"Resume incompatible option dataset: manifest diverso ({old.get('dataset')} -> {args.dataset})")
        if old.get("audio_fingerprint", args_dict["audio_fingerprint"]) != args_dict["audio_fingerprint"]:
            raise ValueError("Resume incompatible option dataset: WAV content changed")
        if old.get("dataset") != args_dict["dataset"]:
            print(f"  dataset in un percorso nuovo ({args.dataset}), stesso manifest: ok")
        # Keep the original warm-start information when resuming a run.
        if old.get("init_parametric"):
            args.init_parametric = old["init_parametric"]
            args_dict["init_parametric"] = old["init_parametric"]
            warm_info = old.get("warm_start_info")
            args_dict["warm_start_info"] = warm_info
            print(f"  run partito dal warm-start {old['init_parametric']} (conservato nel report)")
        for key in ("variant", "channels", "fixed_g", "batch_size", "segment_samples",
                    "segments_per_epoch", "no_balanced", "num_workers", "lr_step_every",
                    "pre_emph", "no_pre_emph", "loss", "mrstft_weight", "lr", "lr_gamma", "no_bias", "seed"):
            if key in old and old[key] != args_dict[key]:
                raise ValueError(f"Resume incompatible option {key}: {old[key]} != {args_dict[key]}")
        for key, default in (("raw_weight", 1.0), ("dc_weight", 1.0), ("grad_clip", 0.0),
                             ("energy_floor_db", None), ("knob_names_resolved", ["gain", "treble"]),
                             ("film_lr_mult", 1.0)):
            if old.get(key, default) != args_dict[key]:
                raise ValueError(f"Resume incompatible option {key}: {old.get(key, default)} != {args_dict[key]}")
        model.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["opt"])
        try:
            sched.load_state_dict(state["sched"])
        except Exception as e:
            print(f"  scheduler state skipped: {e}")
        step = int(state.get("step", 0))
        epoch = int(state.get("epoch", 0))
        best_val = float(state.get("best_val", best_val))
        history = list(state.get("history", []))
        source_best = resume_path.with_name("best.pt")
        best_matches_run = False
        if source_best.is_file():
            payload = torch.load(source_best, map_location="cpu", weights_only=False)
            metadata = payload.get("config", {}).get("metadata", {})
            del payload
            saved_score = metadata.get("best_val_esr", best_val)
            saved_run_id = metadata.get("run_id")
            best_matches_run = saved_run_id == prior_run_id or (
                saved_run_id is None and np.isfinite(best_val) and saved_score == best_val
            )
            if best_matches_run and (not np.isfinite(saved_score) or saved_score > best_val):
                raise ValueError(f"{source_best} does not match the best score in the training checkpoint")
            if best_matches_run:
                checkpoint_current = checkpoint_current and saved_score == best_val
                best_val = saved_score
                best_step = metadata.get("step")
        if np.isfinite(best_val):
            if not best_matches_run:
                raise ValueError(f"Resume requires the matching best model at {source_best}")
            if not resume_in_place:
                for source in (source_best.with_suffix(".json"), source_best):
                    if source.is_file():
                        copy_checkpoint(source, out / source.name)
        if state.get("rng_state"):
            restore_rng_state(state["rng_state"], train_ds, sampler, loader_generator)
        else:
            warnings.warn("Legacy checkpoint lacks RNG/sampler state: resume is not exact")
        if resume_in_place and resume_path.name == "train_state.pt" and loaded_state_path != resume_path:
            copy_checkpoint(loaded_state_path, resume_path)
        print(f"  resumed at step={step} epoch={epoch} best_val={best_val:.6g}")

    start_step = step
    if best_step is None:
        for h in history:
            sel = h.get("val_esr_floored") if h.get("val_esr_floored") is not None else h.get("val_esr")
            if sel is not None and sel == best_val:
                best_step = h["step"]
                break
    last_h = history[-1] if history else {}
    prog.update(step=step, total=args.steps, best=(best_val if np.isfinite(best_val) else None), best_step=best_step,
                val=last_h.get("val_esr"), val_floored=last_h.get("val_esr_floored"), val_step=last_h.get("step"))
    prog.write("RUNNING", "ripreso dal checkpoint" if step > 0 else "primo avvio")
    val_seconds_total, val_count = 0.0, 0
    last_val_step = history[-1].get("step", -1) if history else -1
    last_ckpt_step = step if checkpoint_current else -1

    def write_checkpoint():
        nonlocal last_ckpt_step
        model.export_parametric(out / "last.pt", metadata={"step": step, "epoch": epoch})
        save_train_state(
            out / "train_state.pt", model, opt, sched, step, epoch, best_val, args_dict, history,
            capture_rng_state(train_ds, sampler, loader_generator)
        )
        last_ckpt_step = step

    def maybe_validate_and_ckpt(force=False):
        nonlocal best_val, last_val_step, best_step, val_seconds_total, val_count
        do_val = (force or step % args.val_every == 0) and step > 0 and step != last_val_step
        do_ckpt = (force or step % args.checkpoint_every == 0) and step > 0 and (step != last_ckpt_step or do_val)
        row = None
        if do_val:
            was_training = model.training
            t_val = time.time()
            val_esr, validation_rows = mean_validation_esr(model, val_ds, device, floor_power=val_floor_power)
            model.train(was_training)
            last_val_step = step
            val_sel = floored_mean(validation_rows) if select_floored else val_esr
            row = {
                "step": step,
                "epoch": epoch,
                "val_esr": val_esr,
                "val_esr_floored": val_sel if select_floored else None,
                "val_esr_median": float(np.median([r["esr_val_region"] for r in validation_rows])),
                "lr": sched.get_last_lr()[0],
            }
            val_seconds = time.time() - t_val  # Omit timing from history so resumed runs can compare it exactly.
            val_seconds_total += val_seconds
            val_count += 1
            summary, summary_text = compact_validation_summary(validation_rows)
            row["per_condition"] = validation_rows
            row.update(summary)
            extra = (f"  val_ESR_pavimentata={val_sel:.6g} (sceglie best)" if select_floored else "")
            print(f"step {step:7d}  val_ESR={val_esr:.6g}{extra}  mediana={row['val_esr_median']:.6g}  "
                  f"lr={row['lr']:.6g}  ({val_seconds:.1f} s val)")
            print("  " + summary_text)
            history.append(row)
            if val_sel < best_val:
                best_val = val_sel
                best_step = step
                model.export_parametric(out / "best.pt", metadata={
                    "best_val_esr": best_val, "step": step, "val_esr_raw_mean": val_esr,
                    "run_id": args_dict["run_id"],
                    "selection_metric": "floored_mean" if select_floored else "mean",
                    "loss_def": args_dict["loss_def"], "knob_names": knob_names})
                print(f"  saved best -> {out / 'best.pt'}")
        if do_ckpt:
            write_checkpoint()
            print(f"  checkpoint -> {out / 'train_state.pt'}")
        if row is not None:
            prog.update(step=step, val=row["val_esr"], val_floored=row["val_esr_floored"], val_step=step,
                        best=best_val, best_step=best_step)
            prog.write("RUNNING")
        return row

    def patience_expired():
        return args.early_stop_steps > 0 and best_step is not None and step - best_step >= args.early_stop_steps

    stopped_early = False

    # ---- step-based training (preferred) ----
    if args.steps is not None:
        target_steps = int(args.steps)
        model.train()
        data_iter = iter(train_loader)
        last_log = time.time()
        running = 0.0
        n_run = 0
        train_time = 0.0  # Use training step time for the estimate.
        train_steps_timed = 0
        if step < target_steps and patience_expired():
            maybe_validate_and_ckpt(force=True)
            stopped_early = patience_expired()
        while step < target_steps and not stopped_early:
            t_step = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                sampler.new_epoch()
                data_iter = iter(train_loader)
                batch = next(data_iter)
            x, y, g, fi = batch
            x, y, g = (t.to(device, non_blocking=is_cuda) for t in (x, y, g))
            opt.zero_grad(set_to_none=True)
            pred = model(x, g, pad_start=False)
            loss = compute_batch_loss(
                pred, y, pre_emph, args.loss, args.mrstft_weight, mrstft_mod,
                file_energy=file_energy_table[fi.to(device)],
                raw_file_energy=(raw_file_energy_table[fi.to(device)] if raw_file_energy_table is not None else None),
                raw_weight=args.raw_weight, dc_weight=args.dc_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            opt.step()
            step += 1
            running += float(loss.item())
            n_run += 1
            if step % args.lr_step_every == 0:
                sched.step()
            if step - start_step > 3:  # Skip the initial cuDNN benchmark and warmup steps.
                train_time += time.time() - t_step
                train_steps_timed += 1
            if step % args.log_every == 0:
                dt = time.time() - last_log
                sps = n_run / max(dt, 1e-9)
                sec_per_step = train_time / max(train_steps_timed, 1)
                eta_h = sec_per_step * (target_steps - step) / 3600.0
                n_val_left = max(0, (target_steps - step) // args.val_every)
                print(
                    f"step {step:7d}/{target_steps}  loss={running/max(n_run,1):.6g}  "
                    f"{sps:.2f} steps/s ({sps * samples_per_step / 1e6:.2f} M campioni/s)  "
                    f"lr={sched.get_last_lr()[0]:.6g}  | ~{sec_per_step*1000/60:.1f} min/1000 step, "
                    f"ETA ~{eta_h:.1f} h (+{n_val_left} validazioni)"
                )
                prog.update(loss=running / max(n_run, 1))
                running = 0.0
                n_run = 0
                last_log = time.time()
            if step % args.progress_every == 0 or step == target_steps:
                sec_per_step = train_time / max(train_steps_timed, 1)
                n_val_left = max(0, (target_steps - step) // args.val_every)
                avg_val = val_seconds_total / val_count if val_count else 0.0
                if n_run:
                    prog.update(loss=running / n_run)
                prog.update(step=step, sps=(1.0 / sec_per_step if train_steps_timed else None),
                            eta_h=(sec_per_step * (target_steps - step) + n_val_left * avg_val) / 3600.0
                            if train_steps_timed else None)
                prog.write("RUNNING")
            maybe_validate_and_ckpt(force=patience_expired())
            stopped_early = step < target_steps and patience_expired()
        maybe_validate_and_ckpt(force=True)
        if stopped_early:
            print(f"Early stop at step {step}: no better validation score in {step - best_step} steps "
                  f"(best checkpoint at step {best_step}).")

    # ---- epoch-based (legacy) ----
    else:
        for ep in range(epoch + 1, args.epochs + 1):
            epoch = ep
            if sampler.cursor >= len(sampler):
                sampler.new_epoch()
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            t_epoch = time.time()
            for x, y, g, fi in train_loader:
                x, y, g = (t.to(device, non_blocking=is_cuda) for t in (x, y, g))
                opt.zero_grad(set_to_none=True)
                pred = model(x, g, pad_start=False)
                loss = compute_batch_loss(
                    pred, y, pre_emph, args.loss, args.mrstft_weight, mrstft_mod,
                    file_energy=file_energy_table[fi.to(device)],
                    raw_file_energy=(raw_file_energy_table[fi.to(device)] if raw_file_energy_table is not None else None),
                    raw_weight=args.raw_weight, dc_weight=args.dc_weight,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                opt.step()
                epoch_loss += float(loss.item())
                n_batches += 1
                step += 1
            sched.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - t_epoch
            steps_per_s = n_batches / max(elapsed, 1e-9)
            val_esr, validation_rows = mean_validation_esr(model, val_ds, device, floor_power=val_floor_power)
            val_sel = floored_mean(validation_rows) if select_floored else val_esr
            row = {
                "epoch": epoch,
                "step": step,
                "train_loss": avg_loss,
                "val_esr": val_esr,
                "val_esr_floored": val_sel if select_floored else None,
                "steps_per_s": steps_per_s,
                "lr": sched.get_last_lr()[0],
            }
            summary, summary_text = compact_validation_summary(validation_rows)
            row["per_condition"] = validation_rows
            row.update(summary)
            print("  " + summary_text)
            history.append(row)
            print(
                f"epoch {epoch:4d}  loss={avg_loss:.6g}  val_ESR={val_esr:.6g}  "
                f"{steps_per_s:.2f} steps/s  lr={row['lr']:.6g}"
            )
            if val_sel < best_val:
                best_val = val_sel
                best_step = step
                model.export_parametric(out / "best.pt", metadata={
                    "best_val_esr": best_val, "epoch": epoch, "step": step, "run_id": args_dict["run_id"],
                })
            write_checkpoint()
        if step > 0 and step != last_ckpt_step:
            write_checkpoint()

    total_time = time.time() - t0
    prog.update(step=step, eta_h=0.0)
    prog.write("RUNNING", "training finito, valutazione finale e test in corso")
    steps_this_run = step - start_step
    print(f"Training done in {total_time:.1f}s, {steps_this_run} new steps ({step} total), {steps_this_run/max(total_time,1e-9):.2f} steps/s overall")

    # Prefer best weights for final eval if present
    best_pt = out / "best.pt"
    if np.isfinite(best_val):
        model = load_parametric(best_pt).to(device)
        print("  (loaded best.pt for eval)")
    final_val_esr, val_rows = mean_validation_esr(model, val_ds, device, floor_power=val_floor_power)
    final_val_floored = floored_mean(val_rows) if select_floored else None
    center_rows = [r for r in val_rows if len(r["g"]) == len(knob_names)
                   and all(abs(a - 0.5) < 1e-5 for a in r["g"])]
    center_esr = center_rows[0]["esr_val_region"] if center_rows else None
    print(f"Final mean val ESR ({len(val_rows)} train conditions): {final_val_esr:.6g}")
    if select_floored:
        print(f"Final mean val ESR con pavimento (metrica di scelta): {final_val_floored:.6g}")
    print(f"Final median val ESR: {float(np.median([r['esr_val_region'] for r in val_rows])):.6g}")
    if center_esr is not None:
        print(f"Val ESR @ g={_fmt_g([0.5]*len(knob_names))}, classic NAM comparison: {center_esr:.6g}")
    # Load held-out tests after model selection for the final report.
    per_file, zero_test = [], []
    mean_test = None
    if train_ds.manifest.get("test"):
        test_full = ParametricNAMDataset(
            args.dataset, split="full", entries="test", load_full=True,
            receptive_field=rf, seed=args.seed,
        )
        print("Per-test-file ESR (whole file, zero padding only at file start):")
        if fixed_g is not None:
            print("  Baseline: fuori dalla condizione allenata, solo riferimento; non un test parametrico.")
        per_file = evaluate_per_file(model, test_full, device, max_files=args.max_files_eval,
                                     floor_power=val_floor_power)
        for r in per_file:
            print(f"  {r['file']}  g={_fmt_g(r['g'])}  ESR_full={r['esr_full']:.6g}")
        if per_file:
            mean_test = sum(r["esr_full"] for r in per_file) / len(per_file)
            print(f"Mean test ESR ({len(per_file)} file): {mean_test:.6g}")

        zero_test = evaluate_zero_input(model, test_full.g_list, device)
    else:
        print("No held-out takes. Reporting temporal validation only.")

    zero_train = evaluate_zero_input(model, train_ds.g_list, device)
    print("Zero-input DC (best checkpoint used for final eval), 5 condizioni con |DC| maggiore:")
    for r in sorted(zero_train + zero_test, key=lambda r: -abs(r["mean"]))[:5]:
        print(f"  g={_fmt_g(r['g'])} DC={r['mean']:.6g} RMS={r['rms_dbfs']:.2f} dBFS")
    worst_dc = max(abs(r["mean"]) for r in zero_train + zero_test)
    print(f"  max |DC| su {len(zero_train) + len(zero_test)} condizioni: {worst_dc:.3g} "
          f"({20*np.log10(max(worst_dc, 1e-15)):.1f} dBFS)")
    val_marg = knob_marginals(val_rows, knob_names, "esr_val_region")
    print("Val ESR media per valore di ogni knob:")
    for name, d in val_marg.items():
        print(f"  {name}: " + "  ".join(f"{v}={e:.5g}" for v, e in d.items()))
    write_rows_csv(out / "val_per_condition.csv", val_rows, knob_names, "esr_val_region")
    write_rows_csv(out / "test_per_file.csv", per_file, knob_names, "esr_full")
    report = {
        "knob_names": knob_names,
        "mean_test_esr": mean_test,
        "validation_prediction_peak": max(row["prediction_peak"] for row in val_rows),
        "mean_val_esr_floored": final_val_floored,
        "selection_metric": "floored_mean" if select_floored else "mean",
        "energy_floor_db": args.energy_floor_db,
        "energy_floor_raw_power": raw_floor,
        "energy_floor_raw_dbfs": (10 * np.log10(raw_floor) if raw_floor else None),
        "floored_train_files": [train_ds.entries[i]["y"] for i in floored_idx],
        "val_knob_marginals": val_marg,
        "warm_start": warm_info,
        "max_abs_zero_input_dc": worst_dc,
        "zero_input_train_conditions": zero_train,
        "zero_input_test_conditions": zero_test,
        "zero_input_protocol": "RF-1 zero pre-roll then 4096 measured samples; no output filtering",
        "grad_clip": args.grad_clip,
        "raw_weight": args.raw_weight,
        "dc_weight": args.dc_weight,
        "init_parametric": args.init_parametric,
        "history": history,
        "per_file_esr": per_file,
        "validation_per_file_esr": val_rows,
        "mean_val_esr": final_val_esr,
        "center_val_esr": center_esr,
        "validation_protocol": "Full val region of train entries, same fixed_g filter; equal mean per condition",
        "test_protocol": "Whole unseen test files including initial blips/silence; causal zero padding at start, RF pre-roll per chunk",
        "test_out_of_trained_condition": fixed_g is not None,
        "total_seconds": total_time,
        "steps_this_run": steps_this_run,
        "stopped_early": stopped_early,
        "early_stop_steps": args.early_stop_steps,
        "best_step": best_step,
        "steps": step,
        "device": device_label(device),
        "best_val_esr": best_val,
        "variant": args.variant,
        "fixed_g": fixed_g,
        "mrstft_weight": args.mrstft_weight,
        "pre_emph": pre_emph,
        "loss": args.loss,
        "loss_def": args_dict["loss_def"],
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    if args.bake_g:
        gvals = parse_g(args.bake_g)
        bake_path = out / f"baked_g{'_'.join(f'{v:.2f}' for v in gvals)}.nam"
        bake_knobs_into_nam(model.cpu(), gvals, bake_path)
        print(f"Baked .nam -> {bake_path}")

    print(f"Artifacts in {out}")
    prog.update(step=step, eta_h=0.0, best=best_val, best_step=best_step)
    prog.write("FINISHED", f"val finale {final_val_esr:.6g}"
               + (f" (pavimento {final_val_floored:.6g})" if final_val_floored is not None else "")
               + (f", test medio {mean_test:.6g} su {len(per_file)} file" if mean_test is not None else "")
               + ". Esegui la cella report/.nam")
    return report


if __name__ == "__main__":
    main()
