# ParametricNAM Training

Turn a dataset recorded with [ParametricNAM Capture](https://github.com/Faverooo/ParametricNAM-Capture) into a `.pnam` amp model you can play in your DAW.

## Get started on Google Colab

1. Download the trainer from [Releases](https://github.com/Faverooo/ParametricNAM-Training/releases/latest) and extract it into a Google Drive folder named `ParametricNAM-Training`. Upload the whole dataset exported by Capture to Drive.
2. Open [ParametricNAM_Colab.ipynb](ParametricNAM_Colab.ipynb) in Colab and select a **GPU** runtime. Set `DATASET_NAME` to the folder name on MyDrive (for example, `VOX`) and set the output folder path. The notebook looks for `manifest.json` and `profile.json` in that folder and its `dataset/` subfolder. Then run the cells. The number and names of the knobs come from the dataset. If the dataset includes test takes, the trainer saves their results in `test_per_file.csv`.
3. When training finishes, download `model.pnam` from the output folder and open it in the [ParametricNAM VST3](https://github.com/Faverooo/ParametricNeuralAmpModelerPlugin).

Training defaults to 60,000 steps. Try 80,000 if validation is still improving. Training stops early after 30,000 steps without a better validation score and exports the best model.

If a run is interrupted, reopen the notebook and use the same dataset and output folder. To start a different run, choose a new output folder.

## License

The project code and notebook are [MIT licensed](LICENSE). Dependencies and recordings retain their own terms.
