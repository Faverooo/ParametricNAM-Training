# ParametricNAM Training

Turn a dataset recorded with [ParametricNAM Capture](https://github.com/Faverooo/ParametricNAM-Capture) into a `.pnam` amp model you can play in your DAW.

## Get started on Google Colab

1. Download the trainer from [Releases](https://github.com/Faverooo/ParametricNAM-Training/releases/latest) and extract it into a Google Drive folder named `ParametricNAM-Training`. Upload the whole dataset exported by Capture to Drive.
2. Open [ParametricNAM_Colab.ipynb](ParametricNAM_Colab.ipynb) in Colab and select a **GPU** runtime. Set the dataset and output folder paths in the notebook, then run the cells. The number and names of the knobs come from the dataset.
3. When training finishes, download `model.pnam` from the output folder and open it in the [ParametricNAM VST3](https://github.com/Faverooo/ParametricNeuralAmpModelerPlugin).

If a run is interrupted, reopen the notebook and use the same dataset and output folder. To start a different run, choose a new output folder.

## License

The project code and notebook are [MIT licensed](LICENSE). Dependencies and recordings retain their own terms.
