# ImageCLEF 2026 — Medical Concept Detection (DeepLens, IUST)

This repository contains the DeepLens team's code for the **ImageCLEFmedical 2026 Concept Detection** task — multi-label classification of medical
concepts (CUIs) from radiology images (ROCOv2 dataset).

## Overview

Given a medical image, the goal is to predict the set of UMLS Concept Unique Identifiers (CUIs)
that describe its clinical content. We evaluate several CNN/transformer backbones, both fine-tuned
on the 2026 data and reused frozen from our 2025 pipeline, and combine them via a simple
logical-OR ensemble.

## Results

| Model | Primary F1 | Secondary F1 |
|---|---|---|
| DenseNet-201 (fine-tuned, 2026) | 0.5542 | 0.9276 |
| EfficientNet-B0 (frozen, 2025) | 0.5632 | 0.9323 |
| EfficientNet-B1 (frozen, 2025) | **0.5648** | **0.9336** |
| Ensemble (logical union) | 0.5632 | 0.9323 |

Our best single model, the frozen 2025 EfficientNet-B1 baseline, placed 7th of 11 teams on the
official leaderboard.

<!-- ## Repository structure

```
.
├── train_densenet201.ipynb        # Data prep + DenseNet-201 training loop
├── generate_submissions.ipynb     # Load all trained models, run inference, build ensembles
├── fix_submission_sorting.ipynb   # Utility: sort a submission CSV by ID for the eval server
├── docs/
│   └── paper.pdf                  # Working notes describing the full method (optional)
└── README.md
``` -->

<!-- ## Setup

1. Clone the repo and open the notebooks in Google Colab (they assume a Colab + Google Drive
   environment; adjust the `CHECKPOINT_DIR` / path variables near the top of each notebook if
   running elsewhere).
2. Install dependencies (already handled in-notebook via `pip install`), main ones being:
   `torch`, `torchvision`, `timm`, `scikit-learn`, `pandas`, `gdown`.
3. Download the ROCOv2 / ImageCLEFmedical 2026 dataset from the official challenge page and update
   the data paths in `train_densenet201.ipynb`.
4. Train (or download our checkpoints) and run `generate_submissions.ipynb` to produce a
   submission CSV.
5. If the eval server rejects your CSV due to ID ordering, run `fix_submission_sorting.ipynb`.

## Citation

If you use this code, please cite our working notes:

> A. Salimi Rudsari, B. Kavousi Nejad, M. Hajihosseini, S. Eetemadi, "DeepLens at ImageCLEFmedical
> 2026: Concept Detection," CLEF 2026 Working Notes, CEUR Workshop Proceedings, 2026. -->

<!-- ## License

Add a license of your choice (e.g. MIT) here. -->
