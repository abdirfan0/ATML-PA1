from torch.utils.data import Dataset

from torchvision.datasets import CIFAR100

from task4.data.cifar10 import (
    build_cifar10_evaluation_transform,
)


NEAR_UNKNOWN_CLASSES = (
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel",
)

FAR_UNKNOWN_CLASSES = (
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe",
)


class CIFAR100UnknownSubset(Dataset):
    def __init__(
        self,
        dataset,
        class_names,
        group_name,
    ):
        self.dataset = dataset
        self.class_names = tuple(
            class_names
        )

        self.group_name = str(
            group_name
        )

        missing_classes = [
            class_name
            for class_name
            in self.class_names
            if class_name
            not in dataset.class_to_idx
        ]

        if missing_classes:
            raise ValueError(
                "Missing CIFAR-100 classes: "
                f"{missing_classes}"
            )

        self.class_indices = {
            dataset.class_to_idx[
                class_name
            ]
            for class_name
            in self.class_names
        }

        self.indices = [
            index
            for index, label
            in enumerate(dataset.targets)
            if label
            in self.class_indices
        ]

        if len(self.indices) != 800:
            raise ValueError(
                f"{self.group_name} must "
                "contain exactly 800 images, "
                f"but found {len(self.indices)}."
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(
        self,
        item,
    ):
        dataset_index = (
            self.indices[item]
        )

        image, original_label = (
            self.dataset[
                dataset_index
            ]
        )

        return {
            "image": image,
            "original_label": (
                int(original_label)
            ),
            "original_class": (
                self.dataset.classes[
                    original_label
                ]
            ),
            "group": self.group_name,
            "dataset_index": (
                int(dataset_index)
            ),
        }


def build_cifar100_unknown_datasets(
    data_root,
    download=True,
):
    evaluation_transform = (
        build_cifar10_evaluation_transform()
    )

    cifar100_test = CIFAR100(
        root=str(data_root),
        train=False,
        download=download,
        transform=(
            evaluation_transform
        ),
    )

    near_dataset = (
        CIFAR100UnknownSubset(
            dataset=cifar100_test,
            class_names=(
                NEAR_UNKNOWN_CLASSES
            ),
            group_name="near",
        )
    )

    far_dataset = (
        CIFAR100UnknownSubset(
            dataset=cifar100_test,
            class_names=(
                FAR_UNKNOWN_CLASSES
            ),
            group_name="far",
        )
    )

    return {
        "near": near_dataset,
        "far": far_dataset,
    }
