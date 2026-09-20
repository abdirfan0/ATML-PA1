import torch
import torch.nn.functional as F


def compute_erm_loss(
    model,
    images_by_domain,
    labels_by_domain,
):
    domain_names = list(
        images_by_domain.keys()
    )

    if set(domain_names) != set(
        labels_by_domain.keys()
    ):
        raise ValueError(
            "Image and label domains "
            "do not match."
        )

    losses_by_domain = {}
    logits_by_domain = {}
    features_by_domain = {}

    for domain_name in domain_names:
        logits, features = model(
            images_by_domain[
                domain_name
            ],
            return_features=True,
        )

        domain_loss = F.cross_entropy(
            logits,
            labels_by_domain[
                domain_name
            ],
        )

        losses_by_domain[
            domain_name
        ] = domain_loss

        logits_by_domain[
            domain_name
        ] = logits

        features_by_domain[
            domain_name
        ] = features

    classification_loss = (
        torch.stack(
            list(
                losses_by_domain.values()
            )
        ).mean()
    )

    return {
        "total_loss": (
            classification_loss
        ),
        "classification_loss": (
            classification_loss
        ),
        "alignment_loss": None,
        "losses_by_domain": (
            losses_by_domain
        ),
        "logits_by_domain": (
            logits_by_domain
        ),
        "features_by_domain": (
            features_by_domain
        ),
    }
