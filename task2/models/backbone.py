
import torch
import torch.nn as nn

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


class ResNet18Backbone(nn.Module):
    feature_dimension = 512

    def __init__(self):
        super().__init__()

        pretrained_model = resnet18(
            weights=(
                ResNet18_Weights
                .IMAGENET1K_V1
            )
        )

        self.feature_extractor = (
            nn.Sequential(
                *list(
                    pretrained_model.children()
                )[:-1]
            )
        )

    def forward(self, images):
        features = self.feature_extractor(
            images
        )

        features = torch.flatten(
            features,
            start_dim=1,
        )

        return features


def freeze_batchnorm_statistics(
    model,
):
    for module in model.modules():
        if isinstance(
            module,
            nn.modules.batchnorm._BatchNorm,
        ):
            module.eval()


def count_trainable_parameters(
    model,
):
    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )
