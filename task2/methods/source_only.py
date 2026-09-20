
import torch.nn.functional as F


def compute_source_only_loss(
    model,
    source_images,
    source_labels,
):
    source_logits, source_features = (
        model(
            source_images,
            return_features=True,
        )
    )

    classification_loss = (
        F.cross_entropy(
            source_logits,
            source_labels,
        )
    )

    return {
        "total_loss": (
            classification_loss
        ),
        "classification_loss": (
            classification_loss
        ),
        "alignment_loss": None,
        "domain_loss": None,
        "source_logits": source_logits,
        "source_features": (
            source_features
        ),
    }
