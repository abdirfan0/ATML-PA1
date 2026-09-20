
import torch
import torch.nn.functional as F


def pairwise_squared_distance(
    first_features,
    second_features,
):
    return torch.cdist(
        first_features,
        second_features,
        p=2,
    ).pow(2)


def median_squared_distance(
    combined_features,
    epsilon=1e-8,
):
    distance_matrix = (
        pairwise_squared_distance(
            combined_features,
            combined_features,
        )
    )

    number_of_samples = (
        distance_matrix.shape[0]
    )

    off_diagonal_mask = (
        ~torch.eye(
            number_of_samples,
            dtype=torch.bool,
            device=(
                distance_matrix.device
            ),
        )
    )

    off_diagonal_distances = (
        distance_matrix[
            off_diagonal_mask
        ]
    )

    median_distance = (
        off_diagonal_distances
        .detach()
        .median()
        .clamp_min(epsilon)
    )

    return median_distance


def multi_kernel_rbf(
    first_features,
    second_features,
    bandwidths,
):
    squared_distances = (
        pairwise_squared_distance(
            first_features,
            second_features,
        )
    )

    kernel_values = torch.zeros_like(
        squared_distances
    )

    for bandwidth in bandwidths:
        kernel_values = (
            kernel_values
            + torch.exp(
                -squared_distances
                / bandwidth
            )
        )

    return kernel_values


def multi_kernel_mmd(
    source_features,
    target_features,
):
    combined_features = torch.cat(
        [
            source_features,
            target_features,
        ],
        dim=0,
    )

    median_distance = (
        median_squared_distance(
            combined_features
        )
    )

    bandwidths = [
        multiplier * median_distance
        for multiplier in (
            0.5,
            1.0,
            2.0,
        )
    ]

    source_source_kernel = (
        multi_kernel_rbf(
            source_features,
            source_features,
            bandwidths,
        )
    )

    target_target_kernel = (
        multi_kernel_rbf(
            target_features,
            target_features,
            bandwidths,
        )
    )

    source_target_kernel = (
        multi_kernel_rbf(
            source_features,
            target_features,
            bandwidths,
        )
    )

    mmd_loss = (
        source_source_kernel.mean()
        + target_target_kernel.mean()
        - 2.0
        * source_target_kernel.mean()
    )

    return mmd_loss, median_distance


def compute_dan_loss(
    model,
    source_images,
    source_labels,
    target_images,
    mmd_weight=1.0,
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

    mmd_loss, median_distance = (
        multi_kernel_mmd(
            source_features,
            target_features,
        )
    )

    total_loss = (
        classification_loss
        + float(mmd_weight)
        * mmd_loss
    )

    return {
        "total_loss": total_loss,
        "classification_loss": (
            classification_loss
        ),
        "alignment_loss": mmd_loss,
        "domain_loss": None,
        "source_logits": source_logits,
        "source_features": (
            source_features
        ),
        "target_features": (
            target_features
        ),
        "median_squared_distance": (
            median_distance
        ),
    }
