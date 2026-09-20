
import math

import torch
import torch.nn.functional as F


class GradientReversalFunction(
    torch.autograd.Function
):
    @staticmethod
    def forward(
        context,
        inputs,
        strength,
    ):
        context.strength = float(
            strength
        )

        return inputs.view_as(inputs)

    @staticmethod
    def backward(
        context,
        gradient_output,
    ):
        reversed_gradient = (
            -context.strength
            * gradient_output
        )

        return reversed_gradient, None


def gradient_reverse(
    inputs,
    strength,
):
    return (
        GradientReversalFunction
        .apply(
            inputs,
            strength,
        )
    )


def gradient_reversal_strength(
    progress,
    maximum_strength=1.0,
):
    progress = min(
        max(float(progress), 0.0),
        1.0,
    )

    schedule_value = (
        2.0
        / (
            1.0
            + math.exp(
                -10.0 * progress
            )
        )
        - 1.0
    )

    return (
        float(maximum_strength)
        * schedule_value
    )


def compute_dann_loss(
    model,
    domain_discriminator,
    source_images,
    source_labels,
    target_images,
    progress,
    maximum_grl_strength=1.0,
    domain_loss_weight=1.0,
):
    source_logits, source_features = (
        model(
            source_images,
            return_features=True,
        )
    )

    _, target_features = model(
        target_images,
        return_features=True,
    )

    classification_loss = (
        F.cross_entropy(
            source_logits,
            source_labels,
        )
    )

    combined_features = torch.cat(
        [
            source_features,
            target_features,
        ],
        dim=0,
    )

    source_domain_labels = torch.zeros(
        len(source_features),
        dtype=torch.long,
        device=(
            combined_features.device
        ),
    )

    target_domain_labels = torch.ones(
        len(target_features),
        dtype=torch.long,
        device=(
            combined_features.device
        ),
    )

    domain_labels = torch.cat(
        [
            source_domain_labels,
            target_domain_labels,
        ],
        dim=0,
    )

    reversal_strength = (
        gradient_reversal_strength(
            progress=progress,
            maximum_strength=(
                maximum_grl_strength
            ),
        )
    )

    reversed_features = (
        gradient_reverse(
            combined_features,
            reversal_strength,
        )
    )

    domain_logits = (
        domain_discriminator(
            reversed_features
        )
    )

    domain_loss = F.cross_entropy(
        domain_logits,
        domain_labels,
    )

    total_loss = (
        classification_loss
        + float(domain_loss_weight)
        * domain_loss
    )

    return {
        "total_loss": total_loss,
        "classification_loss": (
            classification_loss
        ),
        "alignment_loss": None,
        "domain_loss": domain_loss,
        "source_logits": source_logits,
        "source_features": (
            source_features
        ),
        "target_features": (
            target_features
        ),
        "domain_logits": domain_logits,
        "domain_labels": domain_labels,
        "grl_strength": (
            reversal_strength
        ),
    }
