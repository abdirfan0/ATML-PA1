
import torch.nn as nn

from task2.models.backbone import (
    ResNet18Backbone,
)


class LinearClassifierHead(nn.Module):
    def __init__(
        self,
        input_dimension=512,
        number_of_classes=7,
    ):
        super().__init__()

        self.classifier = nn.Linear(
            input_dimension,
            number_of_classes,
        )

    def forward(self, features):
        return self.classifier(
            features
        )


class PACSClassifier(nn.Module):
    def __init__(
        self,
        number_of_classes=7,
    ):
        super().__init__()

        self.backbone = (
            ResNet18Backbone()
        )

        self.classifier_head = (
            LinearClassifierHead(
                input_dimension=(
                    self.backbone
                    .feature_dimension
                ),
                number_of_classes=(
                    number_of_classes
                ),
            )
        )

    def forward(
        self,
        images,
        return_features=False,
    ):
        features = self.backbone(
            images
        )

        logits = self.classifier_head(
            features
        )

        if return_features:
            return logits, features

        return logits
