# CS 445 Final Project

This James Madison University CS 445 research project investigates whether
spectral augmentation improves the robustness of AI-generated music detection.
We plan to compare a baseline residual CNN inspired by ResNet18 with a classifier
trained on mixed original and spectrally augmented audio, measuring balanced
accuracy, robustness, precision, recall, and F1. CNN development and training are
future work; the current script only inspects dataset metadata.

## Dataset

We use [FakeMusicCaps on Hugging Face](https://huggingface.co/datasets/DeepFense/FakeMusicCaps).
The `DeepFense/FakeMusicCaps` repository provides a small Parquet metadata table
containing recording IDs, audio file paths, labels, and dataset names. Loading
this table does not download the audio corpus. Its current metadata contains only
the `spoof` label, so real recordings will also be needed for future binary
classification experiments.

## Installation

Requires Python 3.10 or newer, internet access for the first dataset download,
and the Hugging Face `datasets` package. A GPU and TensorFlow are not required.

```sh
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install the dependency:

```sh
python -m pip install datasets
```

## Run the dataset inspection

From the repository directory, with the environment activated:

```sh
python model.py
```

The script loads and caches the metadata, then prints the available splits,
example counts for each split, columns and feature types, label distributions
(when a label column exists), and up to three sample records per split. It does
not open the referenced audio files or train a model.
