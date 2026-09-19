import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip

from torchvision import transforms
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet50,
    vit_b_16,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (
    0.48145466,
    0.45782750,
    0.40821073,
)

CLIP_STD = (
    0.26862954,
    0.26130258,
    0.27577711,
)

FEATURE_DIMENSIONS = {
    "resnet50": 2048,
    "vit_b_16": 768,
    "clip_vit_b_32": 512,
}


def get_feature_dimension(
    model_name: str,
) -> int:
    """Return a backbone's representation dimension."""

    if model_name not in FEATURE_DIMENSIONS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return FEATURE_DIMENSIONS[model_name]


def get_normalization(
    model_name: str,
) -> transforms.Normalize:
    """Return the required input normalization."""

    if model_name in [
        "resnet50",
        "vit_b_16",
    ]:
        return transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

    if model_name == "clip_vit_b_32":
        return transforms.Normalize(
            mean=CLIP_MEAN,
            std=CLIP_STD,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


class FrozenBackbone(nn.Module):
    """Frozen pretrained feature extractor."""

    def __init__(self, model_name: str):
        super().__init__()

        self.model_name = model_name

        if model_name == "resnet50":
            model = resnet50(
                weights=(
                    ResNet50_Weights.IMAGENET1K_V2
                )
            )

            model.fc = nn.Identity()

        elif model_name == "vit_b_16":
            model = vit_b_16(
                weights=(
                    ViT_B_16_Weights.IMAGENET1K_V1
                )
            )

            model.heads = nn.Identity()

        elif model_name == "clip_vit_b_32":
            model, _, _ = (
                open_clip.create_model_and_transforms(
                    model_name=(
                        "ViT-B-32-quickgelu"
                    ),
                    pretrained="openai",
                )
            )

        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        self.model = model
        self.feature_dim = (
            FEATURE_DIMENSIONS[model_name]
        )

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.eval()

    def train(
        self,
        mode: bool = True,
    ):
        """Keep the frozen backbone in evaluation mode."""

        super().train(False)
        self.model.eval()

        return self

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if self.model_name == "clip_vit_b_32":
            features = self.model.encode_image(
                images
            )

            return F.normalize(
                features,
                dim=1,
            )

        return self.model(images)
