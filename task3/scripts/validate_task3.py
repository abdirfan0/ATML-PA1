from pathlib import Path

import json
import subprocess
import sys

import pandas as pd


REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[2]
)

TASK3_ROOT = (
    REPOSITORY_ROOT / "task3"
)

REQUIRED_FILES = [
    "README.md",
    "configs/base.yaml",
    "configs/controlled_study.yaml",
    "configs/dan_dg.yaml",
    "configs/erm.yaml",
    "configs/sam.yaml",
    "data/source_loaders.py",
    "methods/erm.py",
    "methods/dan_dg.py",
    "methods/sam.py",
    "train.py",
    "notebooks/00_task3_setup.ipynb",
    "notebooks/01_erm_baseline.ipynb",
    "notebooks/02_dan_dg.ipynb",
    "notebooks/03_sam.ipynb",
    "notebooks/04_controlled_study.ipynb",
    "notebooks/05_source_diagnostics.ipynb",
    "notebooks/06_final_evaluation.ipynb",
    "results/erm_source_validation.csv",
    "results/dan_dg_source_validation.csv",
    "results/sam_source_validation.csv",
    "results/dan_dg_controlled_study_summary.csv",
    "results/source_domain_separability.csv",
    "results/source_sharpness_proxy.csv",
    "results/target_sketch_summary.csv",
    "results/target_sketch_per_class.csv",
    "results/target_sketch_predictions.csv",
    "results/target_sketch_failure_summary.csv",
    "results/final_evaluation_metadata.json",
    "figures/erm_source_validation.png",
    "figures/dan_dg_training_curves.png",
    "figures/sam_training_curves.png",
    "figures/dan_dg_controlled_study.png",
    "figures/source_diagnostics.png",
    "figures/target_sketch_performance.png",
    "figures/target_sketch_per_class.png",
    "figures/target_sketch_confusion_matrices.png",
    "figures/target_sketch_failure_examples.png",
]

missing_files = [
    relative_path
    for relative_path in REQUIRED_FILES
    if not (
        TASK3_ROOT / relative_path
    ).is_file()
]

if missing_files:
    print(
        "Missing required files:"
    )

    for relative_path in missing_files:
        print(
            " -",
            relative_path,
        )

    sys.exit(1)


target_summary = pd.read_csv(
    TASK3_ROOT
    / "results/target_sketch_summary.csv"
)

required_methods = {
    "erm",
    "dan_dg",
    "sam",
    "dan_dg_lambda_0p1",
    "dan_dg_lambda_10p0",
}

observed_methods = set(
    target_summary["method"]
)

if observed_methods != required_methods:
    raise ValueError(
        "Unexpected target-summary methods: "
        f"{sorted(observed_methods)}"
    )

if not (
    target_summary[
        "number_of_target_samples"
    ]
    == 3929
).all():
    raise ValueError(
        "Target sample count must be 3929."
    )


controlled_summary = pd.read_csv(
    TASK3_ROOT
    / "results/"
    "dan_dg_controlled_study_summary.csv"
)

observed_weights = set(
    controlled_summary[
        "mmd_weight"
    ].astype(float)
)

if observed_weights != {
    0.1,
    1.0,
    10.0,
}:
    raise ValueError(
        "Controlled-study weights are incorrect."
    )


separability = pd.read_csv(
    TASK3_ROOT
    / "results/"
    "source_domain_separability.csv"
)

sharpness = pd.read_csv(
    TASK3_ROOT
    / "results/"
    "source_sharpness_proxy.csv"
)

main_methods = {
    "erm",
    "dan_dg",
    "sam",
}

if set(
    separability["method"]
) != main_methods:
    raise ValueError(
        "Domain-separability methods are incorrect."
    )

if set(
    sharpness["method"]
) != main_methods:
    raise ValueError(
        "Sharpness methods are incorrect."
    )


metadata_path = (
    TASK3_ROOT
    / "results/"
    "final_evaluation_metadata.json"
)

with metadata_path.open(
    "r",
    encoding="utf-8",
) as metadata_file:
    metadata = json.load(
        metadata_file
    )

if metadata[
    "target_domain"
] != "sketch":
    raise ValueError(
        "Final target domain must be Sketch."
    )

if metadata[
    "number_of_target_samples"
] != 3929:
    raise ValueError(
        "Incorrect target sample count "
        "in final metadata."
    )


tracked_files = subprocess.run(
    [
        "git",
        "ls-files",
        "task3",
    ],
    cwd=REPOSITORY_ROOT,
    capture_output=True,
    text=True,
    check=True,
).stdout.splitlines()

tracked_checkpoints = [
    path
    for path in tracked_files
    if path.endswith(
        (
            ".pt",
            ".pth",
            ".ckpt",
            ".safetensors",
        )
    )
]

if tracked_checkpoints:
    raise ValueError(
        "Checkpoint files must not be tracked: "
        f"{tracked_checkpoints}"
    )


print("Task 3 validation passed.")

print(
    "Methods:",
    sorted(
        required_methods
    ),
)

print(
    "Target samples:",
    int(
        target_summary[
            "number_of_target_samples"
        ].iloc[0]
    ),
)

print(
    "Tracked checkpoints inside task3/:",
    len(tracked_checkpoints),
)
