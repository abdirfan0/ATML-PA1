import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from shared.pacs import (
    PACSLabeledDataset,
    build_pacs_evaluation_transform,
    build_pacs_train_transform,
)


def _seed_worker(worker_id):
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_task3_source_datasets(
    dataset,
    protocol,
):
    source_domains = protocol[
        "source_domains"
    ]

    reserved_target = protocol[
        "target_domain"
    ]

    if reserved_target in source_domains:
        raise ValueError(
            "The reserved target domain "
            "cannot be a source domain."
        )

    train_transform = (
        build_pacs_train_transform()
    )

    evaluation_transform = (
        build_pacs_evaluation_transform()
    )

    source_train = {}
    source_validation = {}

    for domain_name in source_domains:
        domain_split = protocol[
            "source_splits"
        ][domain_name]

        source_train[domain_name] = (
            PACSLabeledDataset(
                dataset=dataset,
                indices=domain_split[
                    "train_indices"
                ],
                transform=train_transform,
                domain_name=domain_name,
            )
        )

        source_validation[
            domain_name
        ] = PACSLabeledDataset(
            dataset=dataset,
            indices=domain_split[
                "validation_indices"
            ],
            transform=(
                evaluation_transform
            ),
            domain_name=domain_name,
        )

    return {
        "source_train": source_train,
        "source_validation": (
            source_validation
        ),
    }


def build_task3_source_loaders(
    datasets,
    source_batch_size=8,
    evaluation_batch_size=64,
    number_of_workers=2,
    seed=6304,
):
    source_train_loaders = {}
    source_validation_loaders = {}

    for domain_index, (
        domain_name,
        domain_dataset,
    ) in enumerate(
        datasets[
            "source_train"
        ].items()
    ):
        generator = torch.Generator()

        generator.manual_seed(
            seed + domain_index
        )

        source_train_loaders[
            domain_name
        ] = DataLoader(
            domain_dataset,
            batch_size=source_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=(
                number_of_workers
            ),
            pin_memory=True,
            worker_init_fn=_seed_worker,
            generator=generator,
        )

    for domain_name, domain_dataset in (
        datasets[
            "source_validation"
        ].items()
    ):
        source_validation_loaders[
            domain_name
        ] = DataLoader(
            domain_dataset,
            batch_size=(
                evaluation_batch_size
            ),
            shuffle=False,
            drop_last=False,
            num_workers=(
                number_of_workers
            ),
            pin_memory=True,
            worker_init_fn=_seed_worker,
        )

    steps_per_epoch = max(
        len(loader)
        for loader in (
            source_train_loaders.values()
        )
    )

    return {
        "source_train": (
            source_train_loaders
        ),
        "source_validation": (
            source_validation_loaders
        ),
        "steps_per_epoch": (
            steps_per_epoch
        ),
    }


def cycle_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch
