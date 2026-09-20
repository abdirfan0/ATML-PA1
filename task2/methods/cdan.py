
import torch
import torch.nn.functional as F

from task2.methods.dann import (
    gradient_reverse,
    gradient_reversal_strength,
)


def class_conditioned_features(
    features,
    logits,
):
    probabilities = F.softmax(
        logits,
        dim=1,
    )

    outer_products = torch.bmm(
        probabilities.unsqueeze(2),
        features.unsqueeze(1),
    )

    conditioned_features = (
        outer_products.flatten(
            start_dim=1
        )
    )

    return conditioned_features


def compute_cdan_loss(
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

    target_logits, target_features = (
        model(
            target_images,
            return_features=True,
        )
    )

    classification_loss = (
        F.cross_entropy(
            source_logits,
            source_labels,
        )
    )

    source_conditioned = (
        class_conditioned_features(
            source_features,
            source_logits,
        )
    )

    target_conditioned = (
        class_conditioned_features(
            target_features,
            target_logits,
        )
    )

    combined_conditioned = torch.cat(
        [
            source_conditioned,
            target_conditioned,
        ],
        dim=0,
    )

    source_domain_labels = torch.zeros(
        len(source_features),
        dtype=torch.long,
        device=(
            combined_conditioned.device
        ),
    )

    target_domain_labels = torch.ones(
        len(target_features),
        dtype=torch.long,
        device=(
            combined_conditioned.device
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

    reversed_conditioned = (
        gradient_reverse(
            combined_conditioned,
            reversal_strength,
        )
    )

    domain_logits = (
        domain_discriminator(
            reversed_conditioned
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
