import torch

from torch import nn

from torchvision.models import (
    resnet18,
)


class CIFARResNet18(
    nn.Module
):
    def __init__(
        self,
        number_of_outputs=10,
    ):
        super().__init__()

        base_model = resnet18(
            weights=None
        )

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn1 = base_model.bn1
        self.relu = base_model.relu

        self.layer1 = (
            base_model.layer1
        )

        self.layer2 = (
            base_model.layer2
        )

        self.layer3 = (
            base_model.layer3
        )

        self.layer4 = (
            base_model.layer4
        )

        self.average_pool = (
            nn.AdaptiveAvgPool2d(
                output_size=(
                    1,
                    1,
                )
            )
        )

        self.feature_dimension = 512

        self.classifier = nn.Linear(
            self.feature_dimension,
            number_of_outputs,
        )

        self.number_of_outputs = int(
            number_of_outputs
        )

    def forward_to_layer2(
        self,
        images,
    ):
        features = self.conv1(
            images
        )

        features = self.bn1(
            features
        )

        features = self.relu(
            features
        )

        features = self.layer1(
            features
        )

        features = self.layer2(
            features
        )

        return features

    def forward_from_layer2(
        self,
        layer2_features,
        return_features=False,
    ):
        features = self.layer3(
            layer2_features
        )

        features = self.layer4(
            features
        )

        features = self.average_pool(
            features
        )

        features = torch.flatten(
            features,
            start_dim=1,
        )

        logits = self.classifier(
            features
        )

        if return_features:
            return logits, features

        return logits

    def forward(
        self,
        images,
        return_features=False,
    ):
        layer2_features = (
            self.forward_to_layer2(
                images
            )
        )

        return self.forward_from_layer2(
            layer2_features,
            return_features=(
                return_features
            ),
        )


def count_trainable_parameters(
    model,
):
    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )
