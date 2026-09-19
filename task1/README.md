# Task 1: Inductive Biases and Feature Representations

This directory contains the code, metadata, results, and figures for Task 1.

## Models

- ResNet-50 pretrained on ImageNet
- ViT-B/16 pretrained on ImageNet
- CLIP ViT-B/32 pretrained by OpenAI

The pretrained backbones remain frozen. Linear classification heads are trained on STL10 features.

## Notebook order

1. `notebooks/01_clean_baseline.ipynb`
2. `notebooks/02_color_translation_patch.ipynb`
3. `notebooks/03_cue_conflicts.ipynb`
4. `notebooks/04_representation_analysis.ipynb`

The notebooks contain the complete experiment sequence and retained outputs.

## Directory structure

- `configs/`: Task 1 experiment settings
- `data/`: Dataset wrappers, interventions, and cue-conflict metadata
- `models/`: Frozen backbone and linear-probe utilities
- `analysis/`: Bias and representation-analysis utilities
- `splits/`: Fixed data indices and patch permutations
- `results/`: Final CSV result tables
- `figures/`: Final figures used for analysis
- `notebooks/`: Executable Colab experiment notebooks

## Reproducibility

The experiments use seed 6304. Fixed split files are included so every model is evaluated on exactly the same images.

Install dependencies from the repository root with:

    pip install -r requirements.txt

A GPU-enabled Colab runtime is recommended.

## Excluded generated artifacts

Raw datasets, model checkpoints, cached feature tensors, external AdaIN weights, and the 300 generated cue-conflict candidate images are excluded from Git.

The metadata recording candidate generation, manual review decisions, and the final balanced 200-image selection is included.

## Repository validation

From the repository root, validate the committed Task 1 artifacts with:

    python task1/scripts/validate_task1.py
