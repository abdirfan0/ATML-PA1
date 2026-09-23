import json
import py_compile

from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

TASK4_ROOT = (
    REPOSITORY_ROOT
    / "task4"
)


required_files = [
    "README.md",
    "configs/base.yaml",
    "configs/vanilla.yaml",
    "configs/gcsc.yaml",
    "configs/proser.yaml",
    "data/cifar10.py",
    "data/cifar100_open_set.py",
    "data/make_splits.py",
    "models/resnet_cifar.py",
    "methods/vanilla.py",
    "methods/gcsc.py",
    "methods/proser.py",
    "scores/msp.py",
    "scores/mls.py",
    "scores/energy.py",
    "scores/mahalanobis.py",
    "scores/proser.py",
    "train.py",
    "train_proser.py",
    "notebooks/01_vanilla.ipynb",
    "notebooks/02_posthoc_scores.ipynb",
    "notebooks/03_gcsc.ipynb",
    "notebooks/04_proser.ipynb",
    "notebooks/05_final_evaluation.ipynb",
    "results/vanilla_posthoc_osr_comparison.csv",
    "results/trained_model_osr_comparison.csv",
    "results/open_set_per_class_results.csv",
    "results/open_set_unknown_predictions.csv",
    "results/vanilla_mls_selected_failures.csv",
    "results/final_evaluation_metadata.json",
    "figures/vanilla_score_roc_curves.png",
    "figures/trained_model_osr_comparison.png",
    "figures/vanilla_mls_failure_examples.png",
    "splits/cifar10_train_val_seed6304.json",
]


missing_files = [
    relative_path
    for relative_path
    in required_files
    if not (
        TASK4_ROOT
        / relative_path
    ).exists()
]

if missing_files:
    raise FileNotFoundError(
        "Missing required Task 4 files:\n"
        + "\n".join(missing_files)
    )


split_path = (
    TASK4_ROOT
    / "splits/"
    "cifar10_train_val_seed6304.json"
)

with split_path.open(
    "r",
    encoding="utf-8",
) as split_file:
    split_data = json.load(
        split_file
    )

train_indices = split_data[
    "train_indices"
]

validation_indices = split_data[
    "validation_indices"
]

if len(train_indices) != 45000:
    raise ValueError(
        "Expected 45,000 training indices."
    )

if len(validation_indices) != 5000:
    raise ValueError(
        "Expected 5,000 validation indices."
    )

if set(train_indices).intersection(
    validation_indices
):
    raise ValueError(
        "Training and validation splits overlap."
    )


vanilla_results = pd.read_csv(
    TASK4_ROOT
    / "results/"
    "vanilla_posthoc_osr_comparison.csv"
)

expected_vanilla_scores = {
    "msp",
    "mls",
    "energy",
    "mahalanobis",
}

if set(
    vanilla_results["score"]
) != expected_vanilla_scores:
    raise ValueError(
        "Vanilla score rows are incomplete."
    )


trained_results = pd.read_csv(
    TASK4_ROOT
    / "results/"
    "trained_model_osr_comparison.csv"
)

expected_trained_rows = {
    ("vanilla", "mls"),
    ("gcsc", "mls"),
    ("proser", "mls"),
    ("proser", "placeholder"),
}

actual_trained_rows = set(
    zip(
        trained_results["model"],
        trained_results["score"],
    )
)

if (
    actual_trained_rows
    != expected_trained_rows
):
    raise ValueError(
        "Trained-model comparison rows "
        "are incomplete."
    )


bounded_columns = [
    "near_auroc",
    "far_auroc",
    "all_unknown_auroc",
    "known_test_acceptance_rate",
    "near_rejection_rate",
    "far_rejection_rate",
]

for table_name, table in [
    (
        "vanilla comparison",
        vanilla_results,
    ),
    (
        "trained comparison",
        trained_results,
    ),
]:
    for column_name in (
        bounded_columns
    ):
        if not table[
            column_name
        ].between(
            0.0,
            1.0,
        ).all():
            raise ValueError(
                f"{table_name}: "
                f"{column_name} is outside "
                "[0, 1]."
            )


predictions = pd.read_csv(
    TASK4_ROOT
    / "results/"
    "open_set_unknown_predictions.csv"
)

prediction_counts = (
    predictions
    .groupby(
        [
            "model",
            "score",
            "group",
        ]
    )
    .size()
)

for model_name, score_name in (
    expected_trained_rows
):
    for group_name in [
        "near",
        "far",
    ]:
        count = prediction_counts.loc[
            (
                model_name,
                score_name,
                group_name,
            )
        ]

        if count != 800:
            raise ValueError(
                "Each model-score-group "
                "must contain 800 samples."
            )


per_class_results = pd.read_csv(
    TASK4_ROOT
    / "results/"
    "open_set_per_class_results.csv"
)

if not (
    per_class_results[
        "number_of_samples"
    ]
    == 100
).all():
    raise ValueError(
        "Every CIFAR-100 class must "
        "contain 100 test images."
    )


selected_failures = pd.read_csv(
    TASK4_ROOT
    / "results/"
    "vanilla_mls_selected_failures.csv"
)

failure_counts = (
    selected_failures[
        "group"
    ].value_counts()
)

if (
    failure_counts.get(
        "near",
        0,
    )
    != 3
    or failure_counts.get(
        "far",
        0,
    )
    != 3
):
    raise ValueError(
        "Expected three near and "
        "three far failure examples."
    )


metadata_path = (
    TASK4_ROOT
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
    "cifar100_train_used"
]:
    raise ValueError(
        "CIFAR-100 training data "
        "must not be used."
    )

if metadata[
    "number_of_near_samples"
] != 800:
    raise ValueError(
        "Incorrect near-unknown count."
    )

if metadata[
    "number_of_far_samples"
] != 800:
    raise ValueError(
        "Incorrect far-unknown count."
    )


checkpoint_files = [
    file_path
    for file_path in (
        TASK4_ROOT.rglob("*")
    )
    if (
        file_path.is_file()
        and file_path.suffix.lower()
        in {
            ".pt",
            ".pth",
            ".ckpt",
        }
    )
]

if checkpoint_files:
    raise ValueError(
        "Checkpoint files found inside "
        "the Task 4 repository folder."
    )


python_files = list(
    TASK4_ROOT.rglob("*.py")
)

for python_file in python_files:
    py_compile.compile(
        str(python_file),
        doraise=True,
    )


print("Task 4 validation passed.")
print(
    "Vanilla scores:",
    sorted(
        expected_vanilla_scores
    ),
)

print(
    "Trained comparisons:",
    sorted(
        expected_trained_rows
    ),
)

print(
    "Unknown samples:",
    len(predictions),
)

print(
    "Python files compiled:",
    len(python_files),
)

print(
    "Checkpoint files inside task4/:",
    len(checkpoint_files),
)
