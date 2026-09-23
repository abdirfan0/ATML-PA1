import random

import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    Dataset,
)

from torchvision import datasets
from torchvision import transforms


CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STANDARD_DEVIATION = (
    0.2470,
    0.2435,
    0.2616,
)


class CIFAR10IndexedSubset(
    Dataset
):
    def __init__(
        self,
        dataset,
        indices,
    ):
        self.dataset = dataset

        self.indices = [
            int(index)
            for index in indices
        ]

    def __len__(
        self,
    ):
        return len(
            self.indices
        )

    def __getitem__(
        self,
        position,
    ):
        dataset_index = (
            self.indices[position]
        )

        image, label = self.dataset[
            dataset_index
        ]

        return {
            "image": image,
            "label": int(label),
            "dataset_index": int(
                dataset_index
            ),
        }


def build_cifar10_train_transform(
    use_randaugment=False,
):
    operations = [
        transforms.RandomCrop(
            32,
            padding=4,
        ),
        transforms.RandomHorizontalFlip(),
    ]

    if use_randaugment:
        operations.append(
            transforms.RandAugment(
                num_ops=2,
                magnitude=9,
            )
        )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=(
                    CIFAR10_STANDARD_DEVIATION
                ),
            ),
        ]
    )

    return transforms.Compose(
        operations
    )


def build_cifar10_evaluation_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=(
                    CIFAR10_STANDARD_DEVIATION
                ),
            ),
        ]
    )


def build_cifar10_datasets(
    data_root,
    split,
    use_randaugment=False,
    download=True,
):
    training_base = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=download,
        transform=(
            build_cifar10_train_transform(
                use_randaugment=(
                    use_randaugment
                )
            )
        ),
    )

    validation_base = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=download,
        transform=(
            build_cifar10_evaluation_transform()
        ),
    )

    test_base = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=download,
        transform=(
            build_cifar10_evaluation_transform()
        ),
    )

    return {
        "train": CIFAR10IndexedSubset(
            dataset=training_base,
            indices=split[
                "train_indices"
            ],
        ),
        "validation": (
            CIFAR10IndexedSubset(
                dataset=validation_base,
                indices=split[
                    "validation_indices"
                ],
            )
        ),
        "test": CIFAR10IndexedSubset(
            dataset=test_base,
            indices=range(
                len(test_base)
            ),
        ),
        "class_names": list(
            training_base.classes
        ),
    }


def _seed_worker(
    worker_id,
):
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    np.random.seed(
        worker_seed
    )

    random.seed(
        worker_seed
    )


def build_cifar10_loaders(
    datasets_by_split,
    batch_size=128,
    number_of_workers=2,
    seed=6304,
):
    training_generator = (
        torch.Generator()
    )

    training_generator.manual_seed(
        seed
    )

    use_pin_memory = (
        torch.cuda.is_available()
    )

    return {
        "train": DataLoader(
            datasets_by_split[
                "train"
            ],
            batch_size=batch_size,
            shuffle=True,
            num_workers=(
                number_of_workers
            ),
            pin_memory=(
                use_pin_memory
            ),
            drop_last=False,
            worker_init_fn=(
                _seed_worker
            ),
            generator=(
                training_generator
            ),
        ),
        "validation": DataLoader(
            datasets_by_split[
                "validation"
            ],
            batch_size=batch_size,
            shuffle=False,
            num_workers=(
                number_of_workers
            ),
            pin_memory=(
                use_pin_memory
            ),
            drop_last=False,
            worker_init_fn=(
                _seed_worker
            ),
        ),
        "test": DataLoader(
            datasets_by_split[
                "test"
            ],
            batch_size=batch_size,
            shuffle=False,
            num_workers=(
                number_of_workers
            ),
            pin_memory=(
                use_pin_memory
            ),
            drop_last=False,
            worker_init_fn=(
                _seed_worker
            ),
        ),
    }


def build_cifar10_output_datasets(
    data_root,
    split,
    download=True,
):
    training_base = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=download,
        transform=(
            build_cifar10_evaluation_transform()
        ),
    )

    validation_base = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=download,
        transform=(
            build_cifar10_evaluation_transform()
        ),
    )

    test_base = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=download,
        transform=(
            build_cifar10_evaluation_transform()
        ),
    )

    return {
        "train": CIFAR10IndexedSubset(
            dataset=training_base,
            indices=split[
                "train_indices"
            ],
        ),
        "validation": (
            CIFAR10IndexedSubset(
                dataset=validation_base,
                indices=split[
                    "validation_indices"
                ],
            )
        ),
        "test": CIFAR10IndexedSubset(
            dataset=test_base,
            indices=range(
                len(test_base)
            ),
        ),
        "class_names": list(
            training_base.classes
        ),
    }
