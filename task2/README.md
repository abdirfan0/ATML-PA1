# Task 2 — Unsupervised Domain Adaptation

This directory contains the PACS unsupervised domain-adaptation experiments.

## Protocol

- Source domains: Photo, Art Painting, and Cartoon.
- Target domain: Sketch.
- Source split: stratified 80/20 training-validation split.
- Random seed: 6304.
- Target images are used without class labels during adaptation.
- Target labels are accessed only during final evaluation.
- Backbone: ImageNet-pretrained ResNet-18 with full fine-tuning.
- Batch-normalization running statistics remain frozen.
- Optimizer: AdamW with learning rate 1e-4 and weight decay 1e-4.
- Maximum training length: 30 epochs.
- Early-stopping patience: 5 epochs.
- Selection metric: mean macro-F1 across the three source validation domains.
- Gradient clipping norm: 1.0.

## Methods

- Source-only baseline.
- DAN using multi-kernel maximum mean discrepancy.
- DANN using gradient reversal and a domain discriminator.
- CDAN using class-conditional feature-probability outer products.

DANN and CDAN use a maximum gradient-reversal strength of 0.1. This value was fixed before target-label evaluation.

## Controlled Study

DAN was evaluated with MMD weights 0.1, 1, and 10. The weight 1 was selected using source-validation macro-F1. The weight 10 caused classification collapse because the alignment objective dominated.

## Notebook Order

1. `01_source_only.ipynb` — source-only baseline.
2. `02_dan.ipynb` — DAN with the selected MMD weight.
3. `03_dann.ipynb` — DANN.
4. `04_cdan.ipynb` — CDAN.
5. `05_controlled_study.ipynb` — DAN MMD-weight study.
6. `06_final_evaluation.ipynb` — final Sketch evaluation and domain-separability diagnostic.

## Final Sketch Results

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Source-only | 0.6483 | 0.6240 |
| DAN | 0.5798 | 0.5503 |
| DANN | 0.0422 | 0.0135 |
| CDAN | 0.8045 | 0.8119 |

## Saved Artifacts

- Experiment configurations are stored in `configs/`.
- Reusable implementations are stored in `models/`, `methods/`, `evaluation/`, and `selection/`.
- CSV and JSON outputs are stored in `results/`.
- Generated plots are stored in `figures/`.
- Model checkpoints are stored outside the Git repository under `ATML/PA1/task2/checkpoints/` because of their size.
- The PACS dataset is also stored outside the Git repository.

## Validation

From the repository root, run:

```bash
python task2/scripts/validate_task2.py
```
