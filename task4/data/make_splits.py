import json

import numpy as np

from sklearn.model_selection import (
    train_test_split,
)


def create_cifar10_split(
    targets,
    output_path,
    validation_fraction=0.10,
    seed=6304,
):
    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    all_indices = np.arange(
        len(targets)
    )

    train_indices, validation_indices = (
        train_test_split(
            all_indices,
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=targets,
        )
    )

    split = {
        "seed": int(seed),
        "validation_fraction": float(
            validation_fraction
        ),
        "number_of_examples": int(
            len(targets)
        ),
        "train_indices": sorted(
            int(index)
            for index in train_indices
        ),
        "validation_indices": sorted(
            int(index)
            for index
            in validation_indices
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            split,
            output_file,
            indent=2,
        )

    return split


def load_cifar10_split(
    split_path,
):
    with split_path.open(
        "r",
        encoding="utf-8",
    ) as split_file:
        split = json.load(
            split_file
        )

    train_indices = set(
        split["train_indices"]
    )

    validation_indices = set(
        split["validation_indices"]
    )

    if train_indices & validation_indices:
        raise ValueError(
            "Training and validation "
            "indices overlap."
        )

    if (
        len(train_indices)
        + len(validation_indices)
        != split["number_of_examples"]
    ):
        raise ValueError(
            "Split does not cover the "
            "complete training partition."
        )

    return split
