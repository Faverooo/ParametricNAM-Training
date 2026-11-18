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
    shutil.copyfile(input_wav, destination / "input.wav")
    document["x"] = "input.wav"

    for split in ("train", "test"):
        entries = document.get(split)
        if not entries:
            raise ValueError(f"Manifest needs a nonempty {split!r} split")
        for index, entry in enumerate(entries):
            source = root / entry["y"]
            relative = Path(split) / str(index) / source.name
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entry["y"] = relative.as_posix()
            if "x" in entry:
                take_input = relative.with_name("loopback.wav")
                shutil.copyfile(root / entry["x"], destination / take_input)
                entry["x"] = take_input.as_posix()
    (destination / "manifest.json").write_text(json.dumps(document, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int, help="Total optimizer steps, including resumed steps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scratch", type=Path, help="Local parent folder for staged WAV files")
    parser.add_argument("--batch-size", default=12, type=int)
    parser.add_argument("--val-every", default=1000, type=int)
    parser.add_argument("--checkpoint-every", default=1000, type=int)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.val_every, args.checkpoint_every) < 1:
        parser.error("Step counts and batch size must be positive")

    from export_pnam import export
    from nam_param.dataset import read_knob_names

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pnam-data-", dir=args.scratch) as stage:
        dataset = Path(stage)
        stage_dataset(args.manifest, dataset)
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        if [control["id"] for control in profile["controls"]] != read_knob_names(dataset):
            parser.error("Profile control IDs must match the dataset knob order")

        signature = {
            "preset": "nighttrain-s01-v1",
            "manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
            "audio_sha256": {
                path.relative_to(dataset).as_posix(): file_sha256(path)
                for path in sorted(dataset.rglob("*.wav"))
            },
            "batch_size": args.batch_size,
            "validation_interval": args.val_every,
        }
        run_config = args.out / "run_config.json"
        if run_config.exists() and json.loads(run_config.read_text(encoding="utf-8")) != signature:
            parser.error("This output directory belongs to a different configuration or manifest")
        if not run_config.exists() and (args.out / "train_state.pt").exists():
            parser.error("Use a new output directory for a run without run_config.json")
        run_config.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")

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
        ], check=True)
    export(args.out / "best.pt", args.profile, args.out / "model.pnam")


if __name__ == "__main__":
    main()
