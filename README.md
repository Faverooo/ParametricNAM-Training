# ParametricNAM Training

Train an A1 FiLM model for the ParametricNAM VST3. The trainer keeps the Night Train S01 recipe and uses a fixed optimizer step budget. It selects the best checkpoint on validation regions and reports the held-out test separately.

## Colab

1. Put this project and the complete `dataset/` export from ParametricNAM Capture on Drive.
2. Open `ParametricNAM_Colab.ipynb` in a GPU runtime. Set the project, dataset and output paths before running the cells.
3. Download `model.pnam` from the output folder. Use the same output folder and configuration to resume, or increase the total step budget to continue training.

The launcher stages WAVs locally and leaves checkpoints and reports in the output directory. It validates the dataset fingerprint when resuming. Legacy datasets with a shared input WAV are supported alongside Capture exports with a loopback WAV for each take. An existing Night Train dataset uses `profiles/nighttrain.json`.

## Command line

```sh
python run_training.py --manifest /path/to/dataset/manifest.json \
  --profile /path/to/dataset/profile.json --out runs/amp-v1 --steps 20000
```

To export an existing compatible checkpoint, run `python export_pnam.py best.pt profile.json model.pnam`.

## GitHub Actions

The workflow runs the Python tests and exports a model from a one-step CPU training run. Successful builds upload an archive of the notebook and trainer. Pushing a `v*` tag publishes this archive as a GitHub Release asset.
