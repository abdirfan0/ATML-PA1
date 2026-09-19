
import pandas as pd
import torch
import torch.nn.functional as F


def validate_feature_pair(
    clean_features,
    transformed_features,
):
    if not isinstance(
        clean_features,
        torch.Tensor,
    ):
        clean_features = torch.as_tensor(
            clean_features
        )

    if not isinstance(
        transformed_features,
        torch.Tensor,
    ):
        transformed_features = (
            torch.as_tensor(
                transformed_features
            )
        )

    if (
        clean_features.shape
        != transformed_features.shape
    ):
        raise ValueError(
            "Clean and transformed features "
            "must have identical shapes."
        )

    if clean_features.ndim != 2:
        raise ValueError(
            "Feature tensors must have shape "
            "[number_of_images, feature_dimension]."
        )

    return (
        clean_features.float(),
        transformed_features.float(),
    )


def paired_cosine_similarity(
    clean_features,
    transformed_features,
):
    (
        clean_features,
        transformed_features,
    ) = validate_feature_pair(
        clean_features,
        transformed_features,
    )

    similarities = F.cosine_similarity(
        clean_features,
        transformed_features,
        dim=1,
    )

    return similarities


def summarize_feature_stability(
    clean_features,
    transformed_features,
):
    similarities = (
        paired_cosine_similarity(
            clean_features,
            transformed_features,
        )
    )

    return {
        "n_samples": int(
            similarities.numel()
        ),
        "mean_cosine_similarity": float(
            similarities.mean().item()
        ),
        "std_cosine_similarity": float(
            similarities.std(
                unbiased=False
            ).item()
        ),
        "minimum_cosine_similarity": float(
            similarities.min().item()
        ),
        "maximum_cosine_similarity": float(
            similarities.max().item()
        ),
    }


def build_stability_table(
    feature_pairs,
):
    rows = []

    for (
        model_name,
        intervention_name,
    ), (
        clean_features,
        transformed_features,
    ) in feature_pairs.items():
        summary = (
            summarize_feature_stability(
                clean_features,
                transformed_features,
            )
        )

        rows.append(
            {
                "model": model_name,
                "intervention": (
                    intervention_name
                ),
                **summary,
            }
        )

    return pd.DataFrame(rows)
