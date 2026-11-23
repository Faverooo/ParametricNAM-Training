"""Train an A1 FiLM model with the Night Train S01 recipe and export PNAM."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def stage_dataset(manifest: Path, destination: Path) -> None:
    """Stage an aligned dataset in a fresh directory, preserving target filenames."""
    document = json.loads(manifest.read_text(encoding="utf-8"))
    root = manifest.parent
    input_name = Path(document.get("x", "input.wav"))
    input_wav = root / input_name
    if not input_wav.is_file() and input_name == Path("input_corto.wav"):
        # Captures from the Night Train exporter share this WAV across subsets.
        input_wav = root.parent / input_name
    print(f"Copying reference audio: {input_wav}", flush=True)
    shutil.copyfile(input_wav, destination / "input.wav")
    document["x"] = "input.wav"

    for split in ("train", "test"):
        entries = document.get(split) or []
        if not entries and split == "train":
            raise ValueError("Manifest needs at least one training take")
        if entries:
            print(f"Copying {len(entries)} {split} takes from Drive to local storage.", flush=True)
        for index, entry in enumerate(entries):
            source = root / entry["y"]
            relative = Path(split) / str(index) / source.name
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entry["y"] = relative.as_posix()
            if "x" in entry:
                take_input = relative.with_name("input.wav" if source.name == "loopback.wav" else "loopback.wav")
                shutil.copyfile(root / entry["x"], destination / take_input)
                entry["x"] = take_input.as_posix()
            copied = index + 1
            if copied % 10 == 0 or copied == len(entries):
                print(f"Copied {copied}/{len(entries)} {split} takes.", flush=True)
    (destination / "manifest.json").write_text(json.dumps(document, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", default=60_000, type=int, help="Total optimizer steps, including resumed steps")
    parser.add_argument("--early-stop-steps", default=30_000, type=int,
                        help="Optimizer steps without a better validation score (0 disables)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scratch", type=Path, help="Local parent folder for staged WAV files")
    parser.add_argument("--batch-size", default=12, type=int)
    parser.add_argument("--val-every", default=1000, type=int)
    parser.add_argument("--checkpoint-every", default=1000, type=int)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.val_every, args.checkpoint_every) < 1:
        parser.error("Step counts and batch size must be positive")
    if args.early_stop_steps < 0:
        parser.error("Early-stop steps must be nonnegative")

    print("Loading training dependencies.", flush=True)
    from export_pnam import export, suggested_output_trim
    from nam_param.dataset import read_knob_names
    from train_param import train_state_exists

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pnam-data-", dir=args.scratch) as stage:
        dataset = Path(stage)
        stage_dataset(args.manifest, dataset)
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        if [control["id"] for control in profile["controls"]] != read_knob_names(dataset):
            parser.error("Profile control IDs must match the dataset knob order")

        audio_paths = sorted(dataset.rglob("*.wav"))
        print(f"Verifying {len(audio_paths)} staged WAV files.", flush=True)
        audio_sha256 = {
            path.relative_to(dataset).as_posix(): file_sha256(path) for path in audio_paths
        }
        print("Audio verification complete.", flush=True)
        signature = {
            "preset": "nighttrain-s01-v1",
            "manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
            "audio_sha256": audio_sha256,
            "batch_size": args.batch_size,
            "validation_interval": args.val_every,
        }
        run_config = args.out / "run_config.json"
        if run_config.exists() and json.loads(run_config.read_text(encoding="utf-8")) != signature:
            parser.error("This output directory belongs to a different configuration or manifest")
        if not run_config.exists():
            if train_state_exists(args.out / "train_state.pt"):
                parser.error("Use a new output directory for a run without run_config.json")
            temporary = run_config.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
            temporary.replace(run_config)

        print(f"Starting training on {args.device} for {args.steps} total steps.", flush=True)
        trainer = Path(__file__).with_name("train_param.py")
        subprocess.run([
            sys.executable, str(trainer), "--dataset", str(dataset), "--out", str(args.out),
            "--device", args.device, "--steps", str(args.steps), "--variant", "film_linear",
            "--channels", "16,8", "--batch-size", str(args.batch_size), "--segment-samples", "48000",
            "--seed", "0", "--lr", "0.002", "--lr-gamma", "0.95", "--lr-step-every", "1000",
            "--grad-clip", "1.0", "--film-lr-mult", "1.0", "--loss", "esr_dual",
            "--pre-emph", "0.85", "--raw-weight", "1.0", "--mrstft-weight", "0.0005",
            "--energy-floor-db", "30", "--val-every", str(args.val_every),
            "--checkpoint-every", str(args.checkpoint_every),
            "--early-stop-steps", str(args.early_stop_steps),
        ], check=True)
    report = json.loads((args.out / "report.json").read_text(encoding="utf-8"))
    trim = suggested_output_trim(report["validation_prediction_peak"])
    export(args.out / "best.pt", args.profile, args.out / "model.pnam", output_trim_db=trim)
    print(f"Exported model.pnam with default output trim {trim:+.1f} dB.")


if __name__ == "__main__":
    main()
