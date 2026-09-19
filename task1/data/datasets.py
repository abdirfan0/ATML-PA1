
from torch.utils.data import Dataset

from task1.data.transforms import (
    apply_intervention,
    build_model_transform,
)


class IndexedDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        indices=None,
        transform=None,
    ):
        self.base_dataset = base_dataset

        if indices is None:
            self.indices = list(
                range(len(base_dataset))
            )
        else:
            self.indices = [
                int(index)
                for index in indices
            ]

        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        original_index = self.indices[position]

        image, label = self.base_dataset[
            original_index
        ]

        if self.transform is not None:
            image = self.transform(image)

        return image, int(label), original_index


class InterventionDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        indices,
        model_name,
        intervention="clean",
        image_size=224,
        seed=6304,
        intervention_parameters=None,
    ):
        self.base_dataset = base_dataset

        self.indices = [
            int(index)
            for index in indices
        ]

        self.model_name = model_name
        self.intervention = intervention
        self.image_size = image_size
        self.seed = seed

        if intervention_parameters is None:
            intervention_parameters = {}

        self.intervention_parameters = (
            intervention_parameters.copy()
        )

        self.model_transform = (
            build_model_transform(
                model_name=model_name,
                image_size=image_size,
            )
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        original_index = self.indices[position]

        image, label = self.base_dataset[
            original_index
        ]

        transformed_image = apply_intervention(
            image=image,
            intervention=self.intervention,
            image_index=original_index,
            image_size=self.image_size,
            seed=self.seed,
            **self.intervention_parameters,
        )

        model_input = self.model_transform(
            transformed_image
        )

        return (
            model_input,
            int(label),
            original_index,
        )
