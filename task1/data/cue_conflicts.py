
from pathlib import Path

import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

from task1.data.transforms import (
    build_model_transform,
)


def load_cue_conflict_metadata(
    metadata_path,
):
    metadata_path = Path(metadata_path)

    metadata = pd.read_csv(
        metadata_path
    )

    required_columns = {
        "candidate_id",
        "direction",
        "content_label",
        "style_label",
        "content_class",
        "style_class",
        "file_name",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing metadata columns: "
            f"{sorted(missing_columns)}"
        )

    return metadata


def validate_selected_conflicts(
    metadata,
    expected_total=200,
    expected_per_direction=20,
):
    if len(metadata) != expected_total:
        raise ValueError(
            f"Expected {expected_total} images, "
            f"but found {len(metadata)}."
        )

    if metadata["candidate_id"].duplicated().any():
        raise ValueError(
            "Duplicate candidate IDs were found."
        )

    if (
        metadata["content_label"]
        == metadata["style_label"]
    ).any():
        raise ValueError(
            "A cue-conflict image has matching "
            "content and style labels."
        )

    if "decision" in metadata.columns:
        decisions = (
            metadata["decision"]
            .astype(str)
            .str.lower()
        )

        if not decisions.eq("accepted").all():
            raise ValueError(
                "The selected metadata contains "
                "a non-accepted image."
            )

    direction_counts = (
        metadata["direction"]
        .value_counts()
        .sort_index()
    )

    if not direction_counts.eq(
        expected_per_direction
    ).all():
        raise ValueError(
            "Each direction must contain exactly "
            f"{expected_per_direction} images."
        )

    return direction_counts


def check_candidate_files(
    metadata,
    candidate_directory,
):
    candidate_directory = Path(
        candidate_directory
    )

    missing_files = []

    for file_name in metadata["file_name"]:
        image_path = (
            candidate_directory
            / file_name
        )

        if not image_path.exists():
            missing_files.append(file_name)

    return missing_files


class CueConflictDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        candidate_directory,
        model_name,
        image_size=224,
    ):
        self.metadata = (
            load_cue_conflict_metadata(
                metadata_path
            )
        )

        self.candidate_directory = Path(
            candidate_directory
        )

        self.model_transform = (
            build_model_transform(
                model_name=model_name,
                image_size=image_size,
            )
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]

        image_path = (
            self.candidate_directory
            / row["file_name"]
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            model_input = (
                self.model_transform(image)
            )

        return {
            "image": model_input,
            "candidate_id": row[
                "candidate_id"
            ],
            "shape_label": int(
                row["content_label"]
            ),
            "texture_label": int(
                row["style_label"]
            ),
            "shape_class": row[
                "content_class"
            ],
            "texture_class": row[
                "style_class"
            ],
        }
