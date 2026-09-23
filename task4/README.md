# Task 4 — Open-Set Recognition

This directory contains the Open-Set Recognition experiments using CIFAR-10 as the known-class dataset and fixed CIFAR-100 test classes as unknowns.

## Experimental Protocol

- Known classes: all ten CIFAR-10 classes.
- CIFAR-10 training split: 45,000 training and 5,000 validation images.
- Split seed: 6304.
- Known test set: complete CIFAR-10 test partition.
- Near unknowns: bus, pickup truck, motorcycle, tractor, wolf, fox, leopard, and camel.
- Far unknowns: bottle, bowl, chair, clock, keyboard, mushroom, sunflower, and wardrobe.
- Unknown samples: 800 near and 800 far.
- CIFAR-100 training data was not used.
- Rejection thresholds were fixed at the 95th percentile of CIFAR-10 validation unknownness.
- Checkpoints were selected using CIFAR-10 validation accuracy only.

## Required Methods

1. Vanilla CIFAR ResNet-18.
2. Post-hoc MSP, MLS, Energy, and Mahalanobis scores.
3. GCSC using RandAugment and MLS.
4. PROSER using five dummy classifiers and manifold mixup after layer2.

Reciprocal Point Learning was not implemented because it was an optional extension.

## Notebook Order

1. `01_vanilla.ipynb`
2. `02_posthoc_scores.ipynb`
3. `03_gcsc.ipynb`
4. `04_proser.ipynb`
5. `05_final_evaluation.ipynb`

## Main Results

### Vanilla Post-hoc Scores

| Score | Near AUROC | Far AUROC | All AUROC | Near Rejection | Far Rejection |
|---|---:|---:|---:|---:|---:|
| MSP | 0.8037 | 0.8940 | 0.8488 | 0.2825 | 0.4375 |
| MLS | 0.7856 | 0.8911 | 0.8383 | 0.3188 | 0.5400 |
| Energy | 0.7856 | 0.8919 | 0.8387 | 0.3150 | 0.5450 |
| Mahalanobis | 0.7974 | 0.9105 | 0.8540 | 0.2925 | 0.4600 |

### Trained-Model Comparison

| Model and Score | CSA | Near AUROC | Far AUROC | Near Rejection | Far Rejection |
|---|---:|---:|---:|---:|---:|
| Vanilla + MLS | 0.9494 | 0.7856 | 0.8911 | 0.3188 | 0.5400 |
| GCSC + MLS | 0.9509 | 0.8106 | 0.9002 | 0.3613 | 0.5988 |
| PROSER + MLS | 0.9465 | 0.7879 | 0.8751 | 0.3238 | 0.4975 |
| PROSER + Placeholder | 0.9465 | 0.7825 | 0.8757 | 0.3175 | 0.4850 |

## Repository Contents

- `configs/`: fixed training and method configurations.
- `data/`: CIFAR-10 split and CIFAR-100 unknown-set construction.
- `models/`: CIFAR-appropriate ResNet-18.
- `methods/`: Vanilla, GCSC, and PROSER implementations.
- `scores/`: post-hoc and placeholder unknownness scores.
- `evaluation/`: closed-set evaluation utilities.
- `notebooks/`: experiment notebooks and outputs.
- `results/`: CSV and JSON experimental results.
- `figures/`: training, ROC, comparison, and failure-analysis figures.
- `splits/`: reproducible CIFAR-10 train/validation split.
- `scripts/`: Task 4 validation utilities.

Datasets and model checkpoints are intentionally excluded from Git. Checkpoints are stored outside the repository under the Task 4 working directory in Google Drive.
