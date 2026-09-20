
import json

from pathlib import Path

import numpy as np

from sklearn.model_selection import (
    train_test_split,
)


SOURCE_DOMAINS = [
    "photo",
    "art_painting",
    "cartoon",
]

TARGET_DOMAIN = "sketch"

PACS_CLASSES = [
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person",
]


def create_pacs_protocol(
    dataset,
    validation_fraction=0.20,
    seed=6304,
):
    all_domains = np.asarray(
        dataset["domain"]
    )

    source_splits = {}

    for domain_name in SOURCE_DOMAINS:
        domain_indices = np.flatnonzero(
            all_domains == domain_name
        )

        source_subset = dataset.select(
            domain_indices.tolist()
        )

        source_labels = np.asarray(
            source_subset["label"]
        )

        relative_indices = np.arange(
            len(domain_indices)
        )

        (
            relative_train_indices,
            relative_validation_indices,
        ) = train_test_split(
            relative_indices,
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=source_labels,
        )

        train_indices = domain_indices[
            relative_train_indices
        ]

        validation_indices = domain_indices[
            relative_validation_indices
        ]

        source_splits[domain_name] = {
            "train_indices": (
                train_indices.astype(
                    int
                ).tolist()
            ),
            "validation_indices": (
                validation_indices.astype(
                    int
                ).tolist()
            ),
        }

    target_indices = np.flatnonzero(
        all_domains == TARGET_DOMAIN
    )

    protocol = {
        "seed": int(seed),
        "validation_fraction": float(
            validation_fraction
        ),
        "source_domains": (
            SOURCE_DOMAINS
        ),
        "target_domain": TARGET_DOMAIN,
        "class_names": PACS_CLASSES,
        "source_splits": source_splits,
        "target_adaptation_indices": (
            target_indices.astype(
                int
            ).tolist()
        ),
    }

    return protocol


def validate_pacs_protocol(
    protocol,
):
    problems = []

    if protocol["seed"] != 6304:
        problems.append(
            "The protocol seed is not 6304."
        )

    if (
        protocol["source_domains"]
        != SOURCE_DOMAINS
    ):
        problems.append(
            "The source-domain order is incorrect."
        )

    if (
        protocol["target_domain"]
        != TARGET_DOMAIN
    ):
        problems.append(
            "The target domain is incorrect."
        )

    all_source_indices = set()

    for domain_name in SOURCE_DOMAINS:
        domain_split = (
            protocol["source_splits"][
                domain_name
            ]
        )

        train_indices = set(
            domain_split[
                "train_indices"
            ]
        )

        validation_indices = set(
            domain_split[
                "validation_indices"
            ]
        )

        if train_indices & validation_indices:
            problems.append(
                f"{domain_name} has overlapping "
                "training and validation sets."
            )

        if all_source_indices & (
            train_indices
            | validation_indices
        ):
            problems.append(
                "Different source domains have "
                "overlapping indices."
            )

        all_source_indices.update(
            train_indices
        )

        all_source_indices.update(
            validation_indices
        )

    target_indices = set(
        protocol[
            "target_adaptation_indices"
        ]
    )

    if all_source_indices & target_indices:
        problems.append(
            "Source and target indices overlap."
        )

    if problems:
        raise ValueError(
            "\n".join(problems)
        )

    return True


def save_pacs_protocol(
    protocol,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            protocol,
            output_file,
            indent=2,
        )


def load_pacs_protocol(
    protocol_path,
):
    protocol_path = Path(
        protocol_path
    )

    with protocol_path.open(
        "r",
        encoding="utf-8",
    ) as protocol_file:
        protocol = json.load(
            protocol_file
        )

    validate_pacs_protocol(
        protocol
    )

    return protocol
