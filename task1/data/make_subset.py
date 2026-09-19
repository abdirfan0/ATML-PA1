
import json

from pathlib import Path

import numpy as np

from sklearn.model_selection import (
    train_test_split,
)


def create_train_validation_split(
    labels,
    validation_fraction=0.20,
    seed=6304,
):
    labels = np.asarray(labels)
    all_indices = np.arange(len(labels))

    train_indices, validation_indices = (
        train_test_split(
            all_indices,
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
    )

    return (
        train_indices.astype(int).tolist(),
        validation_indices.astype(int).tolist(),
    )


def create_balanced_subset(
    labels,
    total_size=500,
    number_of_classes=10,
    seed=6304,
):
    if total_size % number_of_classes != 0:
        raise ValueError(
            "total_size must be divisible by "
            "number_of_classes"
        )

    labels = np.asarray(labels)

    samples_per_class = (
        total_size // number_of_classes
    )

    random_generator = (
        np.random.default_rng(seed)
    )

    selected_indices = []

    for class_index in range(
        number_of_classes
    ):
        class_indices = np.flatnonzero(
            labels == class_index
        )

        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Class {class_index} has only "
                f"{len(class_indices)} samples."
            )

        class_selection = (
            random_generator.choice(
                class_indices,
                size=samples_per_class,
                replace=False,
            )
        )

        selected_indices.extend(
            class_selection.astype(int).tolist()
        )

    random_generator.shuffle(
        selected_indices
    )

    return selected_indices


def save_split_file(
    output_path,
    train_indices,
    validation_indices,
    test_indices,
    seed=6304,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_data = {
        "seed": int(seed),
        "train_indices": [
            int(index)
            for index in train_indices
        ],
        "validation_indices": [
            int(index)
            for index in validation_indices
        ],
        "test_indices": [
            int(index)
            for index in test_indices
        ],
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            split_data,
            output_file,
            indent=2,
        )


def load_split_file(split_path):
    split_path = Path(split_path)

    with split_path.open(
        "r",
        encoding="utf-8",
    ) as split_file:
        split_data = json.load(split_file)

    return split_data
