# Frequency-Weighted Training Losses for Phoneme-Level DNN-based Speech Enhancement

This repository provides the experimental framework and code used in the paper:

> **Frequency-Weighted Training Losses for Phoneme-Level DNN-based Speech Enhancement**  
> *Nasser-Eddine Monir, Paul Magron, Romain Serizel*  
> [arXiv:2506.18714](https://doi.org/10.48550/arXiv.2506.18714), submitted to the 26th IEEE International Workshop on Multimedia Signal Processing (MMSP 2025).

---

## Overview

Recent advances in deep learning have significantly improved multichannel speech enhancement, yet standard loss functions like the **scale-invariant signal-to-distortion ratio (SI-SDR)** do not optimally preserve perceptually important spectral details.  
This repository introduces **frequency-weighted SDR losses** designed to better capture **phoneme-level intelligibility** and fine spectral cues.

We propose several **perceptually informed loss functions**, formulated in the **time-frequency domain**, that emphasize regions where speech energy or perceptual importance is higher.  
The repository provides:

- Implementations of **custom loss functions** (e.g., frequency-weighted SI-SDR variants).  
- Training and evaluation scripts for **FaSNet-TAC** (a multichannel speech enhancement model).  
- Tools for **phoneme-level intelligibility analysis** and **spectral diagnostics**.

## Repository Structure

```
SELoss/
│
├── configs/
│ └── phoneme-classes.json # Phoneme categories for phoneme-level analysis
│
├── external_repo/
│ └── asteroid/ # Local copy of the Asteroid toolkit (modified)
│ └── losses/
│ └── sdr_xp.py # Implementation of SI-SDR and perceptually weighted spectral losses
│
├── egs/
│ └── TAC/ # Example experiment: Target-Aware Convolutions (FaSNet-TAC)
│ ├── custom_inference.py # Inference and metric computation pipeline
│ ├── run_.sh # Main training launcher
│ └── train.py # Training loop (PyTorch Lightning-based)
│
├── scripts/
│ ├── evaluate/
│ │ ├── evaluate_fasnet.sh # Local evaluation script
│ │ └── evaluate_fasnet_job.sh # Cluster submission for evaluation
│ │
│ └── train/
│ └── train_fasnet.sh # Cluster submission for training
│
├── src/
│ ├── auditus/
│ │ └── evaluation.py # Unified evaluation metrics (SI-SDR, SIR, SAR, PESQ, STOI)
│ │
│ └── evaluation/
│ ├── ph_eval.py # Phoneme-level intelligibility evaluation
│ ├── ph_retrieve.py # Phoneme alignment retrieval utilities
│ └── utt_eval.py # Utterance-level metric analysis
│
├── .gitignore
└── README.md
```

## Installation

### Clone the repository

```bash
git clone https://github.com/<username>/SELoss.git
cd fw-se-loss
```

### Install dependencies

Create a new Conda environment (recommended):
```bash
conda create -n fwseloss python=3.10
conda activate fw-se-loss
```

Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a FaSNet-TAC model with a given loss configuration:

```bash
bash scripts/train/train_fasnet.sh XP1 neg_sisdr gres 2 48 1024 256 mel 18 speech_norm
```

This will start a job that:
* Loads the dataset
* Trains with the selected loss and frequency weighting strategy
* Logs results in experiments/<EXPERIMENT_TAG>/

Evaluation

Run inference and compute metrics:

```bash
bash scripts/evaluate/evaluate_fasnet.sh <STAGE> <EXPERIMENT_TAG>
```

Or submit a cluster job:

```bash
bash scripts/evaluate/evaluate_fasnet_job.sh <EXPERIMENT_TAG> <STAGE> <WALLTIME> <CLUSTER>
```

## Implemented Losses

Implemented in external_repo/asteroid/losses/sdr_xp.py: Each loss can be configured by scale (mel, critical, or linear), weighting strategy, and STFT parameters (n_fft, hop_length, n_bins).

## Evaluation Metrics

Computed in src/auditus/evaluation.py and used in `custom_inference.py`: 
* SI-SDR, SI-SIR, SI-SAR
* Frequency-weighted SDR, SIR and SAR denoted as FW-SDR, FW-SIR and FW-SAR
* STOI (Short-Time Objective Intelligibility)

## Citation

If you use this codebase, please cite:

```
@article{Monir2025FreqWeightedSE,
  title={Frequency-Weighted Training Losses for Phoneme-Level DNN-based Speech Enhancement},
  author={Nasser-Eddine Monir and Paul Magron and Romain Serizel},
  journal={arXiv preprint arXiv:2506.18714},
  year={2025}
}
```

## Acknowledgments

This repository extends the Asteroid speech enhancement toolkit with additional perceptual loss functions and evaluation tools.
We gratefully acknowledge the support of Université de Lorraine, CNRS, and the Multispeech team at Inria Nancy.