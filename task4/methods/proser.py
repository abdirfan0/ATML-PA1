import torch
import torch.nn as nn
import torch.nn.functional as F


class PROSERModel(nn.Module):
    def __init__(
        self,
        backbone,
        number_of_dummy_classes=5,
    ):
        super().__init__()

        self.backbone = backbone
        self.number_of_known_classes = 10
        self.number_of_dummy_classes = int(
            number_of_dummy_classes
        )

        self.dummy_classifier = nn.Linear(
            512,
            self.number_of_dummy_classes,
        )

        nn.init.normal_(
            self.dummy_classifier.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(
            self.dummy_classifier.bias
        )

    def classify_features(
        self,
        features,
    ):
        known_logits = (
            self.backbone.classifier(
                features
            )
        )

        dummy_logits = (
            self.dummy_classifier(
                features
            )
        )

        return (
            known_logits,
            dummy_logits,
        )

    def forward(
        self,
        images,
        return_features=False,
    ):
        known_logits, features = (
            self.backbone(
                images,
                return_features=True,
            )
        )

        dummy_logits = (
            self.dummy_classifier(
                features
            )
        )

        if return_features:
            return (
                known_logits,
                dummy_logits,
                features,
            )

        return known_logits, dummy_logits

    def forward_to_layer2(
        self,
        images,
    ):
        return (
            self.backbone.forward_to_layer2(
                images
            )
        )

    def forward_from_layer2(
        self,
        layer2_features,
    ):
        known_logits, features = (
            self.backbone.forward_from_layer2(
                layer2_features,
                return_features=True,
            )
        )

        dummy_logits = (
            self.dummy_classifier(
                features
            )
        )

        return (
            known_logits,
            dummy_logits,
            features,
        )


def build_different_class_partners(
    labels,
):
    partner_indices = []

    for sample_index in range(
        len(labels)
    ):
        valid_indices = torch.nonzero(
            labels != labels[sample_index],
            as_tuple=False,
        ).flatten()

        if len(valid_indices) == 0:
            raise ValueError(
                "Manifold mixup requires at "
                "least two different classes "
                "in the data-placeholder half."
            )

        selected_position = torch.randint(
            low=0,
            high=len(valid_indices),
            size=(1,),
            device=labels.device,
        )

        partner_indices.append(
            valid_indices[
                selected_position
            ]
        )

    return torch.cat(
        partner_indices
    )


def sample_mixup_coefficient(
    alpha=2.0,
    device=None,
):
    beta_distribution = (
        torch.distributions.Beta(
            float(alpha),
            float(alpha),
        )
    )

    coefficient = (
        beta_distribution.sample()
    )

    if device is not None:
        coefficient = coefficient.to(
            device
        )

    return coefficient


def compute_proser_loss(
    model,
    images,
    labels,
    mixup_alpha=2.0,
    classifier_placeholder_weight=1.0,
    data_placeholder_weight=0.1,
):
    batch_size = len(images)

    if batch_size < 4:
        raise ValueError(
            "PROSER requires a batch size "
            "of at least four."
        )

    half_size = batch_size // 2

    data_images = images[:half_size]
    data_labels = labels[:half_size]

    classifier_images = images[
        half_size:
        2 * half_size
    ]
    classifier_labels = labels[
        half_size:
        2 * half_size
    ]

    layer2_features = (
        model.forward_to_layer2(
            data_images
        )
    )

    partner_indices = (
        build_different_class_partners(
            data_labels
        )
    )

    mixup_coefficient = (
        sample_mixup_coefficient(
            alpha=mixup_alpha,
            device=images.device,
        )
    )

    mixed_layer2_features = (
        mixup_coefficient
        * layer2_features
        + (
            1.0
            - mixup_coefficient
        )
        * layer2_features[
            partner_indices
        ]
    )

    (
        mixed_known_logits,
        mixed_dummy_logits,
        _,
    ) = model.forward_from_layer2(
        mixed_layer2_features
    )

    (
        ordinary_known_logits,
        ordinary_dummy_logits,
        _,
    ) = model(
        classifier_images,
        return_features=True,
    )

    ordinary_all_logits = torch.cat(
        [
            ordinary_known_logits,
            ordinary_dummy_logits,
        ],
        dim=1,
    )

    classification_loss = (
        F.cross_entropy(
            ordinary_all_logits,
            classifier_labels,
        )
    )

    maximum_dummy_logit = (
        ordinary_dummy_logits.max(
            dim=1,
            keepdim=True,
        ).values
    )

    classifier_placeholder_logits = (
        torch.cat(
            [
                ordinary_known_logits.clone(),
                maximum_dummy_logit,
            ],
            dim=1,
        )
    )

    classifier_placeholder_logits[
        torch.arange(
            half_size,
            device=labels.device,
        ),
        classifier_labels,
    ] = -1e9

    placeholder_targets = torch.full(
        size=(half_size,),
        fill_value=(
            model.number_of_known_classes
        ),
        dtype=torch.long,
        device=labels.device,
    )

    classifier_placeholder_loss = (
        F.cross_entropy(
            classifier_placeholder_logits,
            placeholder_targets,
        )
    )

    mixed_all_logits = torch.cat(
        [
            mixed_known_logits,
            mixed_dummy_logits,
        ],
        dim=1,
    )

    data_placeholder_targets = (
        torch.full(
            size=(half_size,),
            fill_value=(
                model.number_of_known_classes
            ),
            dtype=torch.long,
            device=labels.device,
        )
    )

    data_placeholder_loss = (
        F.cross_entropy(
            mixed_all_logits,
            data_placeholder_targets,
        )
    )

    total_loss = (
        classification_loss
        + float(
            classifier_placeholder_weight
        )
        * classifier_placeholder_loss
        + float(
            data_placeholder_weight
        )
        * data_placeholder_loss
    )

    predictions = (
        ordinary_known_logits.argmax(
            dim=1
        )
    )

    ordinary_accuracy = (
        (
            predictions
            == classifier_labels
        )
        .float()
        .mean()
    )

    partner_difference_rate = (
        (
            data_labels
            != data_labels[
                partner_indices
            ]
        )
        .float()
        .mean()
    )

    return {
        "total_loss": total_loss,
        "classification_loss": (
            classification_loss
        ),
        "classifier_placeholder_loss": (
            classifier_placeholder_loss
        ),
        "data_placeholder_loss": (
            data_placeholder_loss
        ),
        "ordinary_accuracy": (
            ordinary_accuracy
        ),
        "mixup_coefficient": (
            mixup_coefficient
        ),
        "partner_difference_rate": (
            partner_difference_rate
        ),
    }
