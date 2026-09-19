# ATML Programming Assignment 1

Code and experimental results for Programming Assignment 1 of
EE-5102/CS-6304: Advanced Topics in Machine Learning, Fall 2026.

## Tasks

1. Inductive Biases and Feature Representations
2. Unsupervised Domain Adaptation
3. Domain Generalization
4. Open-Set Recognition

## Repository Structure

    common/     Shared utilities
    shared/     Shared PACS protocol for Tasks 2 and 3
    task1/      Inductive-bias experiments
    task2/      Unsupervised domain adaptation
    task3/      Domain generalization
    task4/      Open-set recognition
    report/     Report figures and source files

## Environment

The experiments were developed using Python and PyTorch in Google Colab with GPU acceleration.

Install the required packages using:

    pip install -r requirements.txt

## Reproducibility

The experiments use random seed 6304 unless otherwise specified.

Raw datasets, pretrained weights, cached features, and large model checkpoints are not included in this repository.
Dataset-preparation and experiment scripts reproduce the required artifacts.
