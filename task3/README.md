# Task 3 — Domain Generalization

This directory contains the Task 3 domain-generalization experiments on PACS.

## Protocol

The source domains are:

- Photo
- Art Painting
- Cartoon

Sketch is reserved as the unseen target domain.

Task 3 reuses the exact source train/validation splits from Task 2 with seed 6304. All model training, checkpoint selection, controlled-study decisions, source diagnostics, and hyperparameter choices are completed without using Sketch images or labels.

Sketch labels are accessed only in the final evaluation notebook.

## Main Methods

- ERM: the Task 2 source-only checkpoint, reused without retraining.
- DAN-DG: pairwise multi-kernel MMD across the three source domains with the required main weight of 1.0.
- SAM: non-adaptive sharpness-aware minimization with radius 0.05 and AdamW as the base optimizer.

All methods use an ImageNet-pretrained ResNet-18, domain-balanced source batches, frozen batch-normalization running statistics, and source-validation checkpoint selection.

## Controlled Study

DAN-DG is evaluated with MMD weights:

- 0.1
- 1.0
- 10.0

The required main DAN-DG result remains the model trained with weight 1.0. The other values are controlled-study variants.

## Notebooks

1. `00_task3_setup.ipynb` — repository and implementation setup
2. `01_erm_baseline.ipynb` — reuse and source evaluation of ERM
3. `02_dan_dg.ipynb` — main DAN-DG experiment
4. `03_sam.ipynb` — SAM experiment
5. `04_controlled_study.ipynb` — DAN-DG alignment-weight study
6. `05_source_diagnostics.ipynb` — source-domain separability and sharpness
7. `06_final_evaluation.ipynb` — final unseen-Sketch evaluation and failure analysis

## Main Source-Validation Results

| Method | Mean source macro-F1 |
|---|---:|
| ERM | 0.9367 |
| DAN-DG, weight 1.0 | 0.0507 |
| SAM, radius 0.05 | 0.9515 |

## Final Sketch Results

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| ERM | 0.6483 | 0.6240 |
| DAN-DG, weight 1.0 | 0.0407 | 0.0112 |
| SAM, radius 0.05 | 0.6042 | 0.6301 |
| DAN-DG, weight 0.1 | 0.7147 | 0.7133 |
| DAN-DG, weight 10.0 | 0.0407 | 0.0112 |

## Reproduction Order

Run the notebooks in numerical order. Checkpoints are saved outside the Git repository under:

`MyDrive/ATML/PA1/task3/checkpoints/`

Datasets and model checkpoints are intentionally excluded from version control.
