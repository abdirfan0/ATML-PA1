
import torch

from torch.utils.data import DataLoader

from common.seed import (
    make_generator,
    seed_worker,
)

from shared.pacs import (
    PACSLabeledDataset,
    PACSUnlabeledTargetDataset,
    build_pacs_evaluation_transform,
    build_pacs_train_transform,
)

from shared.pacs_protocol import (
    SOURCE_DOMAINS,
)


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def build_pacs_datasets(
    dataset,
    protocol,
):
    train_transform = (
        build_pacs_train_transform()
    )

    evaluation_transform = (
        build_pacs_evaluation_transform()
    )

    source_train_datasets = {}
    source_validation_datasets = {}

    for domain_name in SOURCE_DOMAINS:
        domain_split = (
            protocol["source_splits"][
                domain_name
            ]
        )

        source_train_datasets[
            domain_name
        ] = PACSLabeledDataset(
            dataset=dataset,
            indices=domain_split[
                "train_indices"
            ],
            transform=train_transform,
            domain_name=domain_name,
        )

        source_validation_datasets[
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

    target_adaptation_dataset = (
        PACSUnlabeledTargetDataset(
            dataset=dataset,
            indices=protocol[
                "target_adaptation_indices"
            ],
            transform=train_transform,
            domain_name=protocol[
                "target_domain"
            ],
        )
    )

    return {
        "source_train": (
            source_train_datasets
        ),
        "source_validation": (
            source_validation_datasets
        ),
        "target_adaptation": (
            target_adaptation_dataset
        ),
    }


def build_pacs_loaders(
    datasets,
    source_batch_size=8,
    target_batch_size=24,
    evaluation_batch_size=64,
    number_of_workers=2,
    seed=6304,
):
    use_pinned_memory = (
        torch.cuda.is_available()
    )

    use_persistent_workers = (
        number_of_workers > 0
    )

    source_train_loaders = {}

    for domain_offset, domain_name in enumerate(
        SOURCE_DOMAINS
    ):
        source_train_loaders[
            domain_name
        ] = DataLoader(
            datasets[
                "source_train"
            ][domain_name],
            batch_size=source_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=number_of_workers,
            pin_memory=use_pinned_memory,
            persistent_workers=(
                use_persistent_workers
            ),
            worker_init_fn=seed_worker,
            generator=make_generator(
                seed + domain_offset
            ),
        )

    source_validation_loaders = {}

    for domain_name in SOURCE_DOMAINS:
        source_validation_loaders[
            domain_name
        ] = DataLoader(
            datasets[
                "source_validation"
            ][domain_name],
            batch_size=(
                evaluation_batch_size
            ),
            shuffle=False,
            drop_last=False,
            num_workers=number_of_workers,
            pin_memory=use_pinned_memory,
            persistent_workers=(
                use_persistent_workers
            ),
            worker_init_fn=seed_worker,
        )

    target_adaptation_loader = DataLoader(
        datasets[
            "target_adaptation"
        ],
        batch_size=target_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=number_of_workers,
        pin_memory=use_pinned_memory,
        persistent_workers=(
            use_persistent_workers
        ),
        worker_init_fn=seed_worker,
        generator=make_generator(
            seed + 100
        ),
    )

    steps_per_epoch = max(
        len(loader)
        for loader
        in source_train_loaders.values()
    )

    return {
        "source_train": (
            source_train_loaders
        ),
        "source_validation": (
            source_validation_loaders
        ),
        "target_adaptation": (
            target_adaptation_loader
        ),
        "steps_per_epoch": (
            steps_per_epoch
        ),
    }
