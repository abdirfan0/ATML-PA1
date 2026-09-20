
import torch

from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STANDARD_DEVIATION = (
    0.229,
    0.224,
    0.225,
)


def build_pacs_train_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (256, 256)
            ),
            transforms.RandomCrop(
                224
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=(
                    IMAGENET_STANDARD_DEVIATION
                ),
            ),
        ]
    )


def build_pacs_evaluation_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (256, 256)
            ),
            transforms.CenterCrop(
                224
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=(
                    IMAGENET_STANDARD_DEVIATION
                ),
            ),
        ]
    )


class PACSLabeledDataset(Dataset):
    def __init__(
        self,
        dataset,
        indices,
        transform,
        domain_name,
    ):
        self.dataset = dataset

        self.indices = [
            int(index)
            for index in indices
        ]

        self.transform = transform
        self.domain_name = domain_name

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        original_index = self.indices[
            position
        ]

        sample = self.dataset[
            original_index
        ]

        image = sample["image"].convert(
            "RGB"
        )

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": int(
                sample["label"]
            ),
            "index": original_index,
            "domain": self.domain_name,
        }


class PACSUnlabeledTargetDataset(Dataset):
    def __init__(
        self,
        dataset,
        indices,
        transform,
        domain_name="sketch",
    ):
        self.original_indices = [
            int(index)
            for index in indices
        ]

        self.dataset = (
            dataset.select(
                self.original_indices
            )
            .remove_columns(
                ["label"]
            )
        )

        self.transform = transform
        self.domain_name = domain_name

    def __len__(self):
        return len(
            self.original_indices
        )

    def __getitem__(self, position):
        sample = self.dataset[position]

        image = sample["image"].convert(
            "RGB"
        )

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "index": (
                self.original_indices[
                    position
                ]
            ),
            "domain": self.domain_name,
        }
