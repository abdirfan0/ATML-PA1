import numpy as np


def fit_shared_diagonal_gaussian(
    features,
    labels,
    number_of_classes=10,
    epsilon=1e-6,
):
    features = np.asarray(
        features,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    class_means = []

    residual_blocks = []

    for class_index in range(
        number_of_classes
    ):
        class_features = features[
            labels == class_index
        ]

        if len(class_features) == 0:
            raise ValueError(
                "Every known class must "
                "have training examples."
            )

        class_mean = class_features.mean(
            axis=0
        )

        class_means.append(
            class_mean
        )

        residual_blocks.append(
            class_features
            - class_mean
        )

    class_means = np.stack(
        class_means,
        axis=0,
    )

    residuals = np.concatenate(
        residual_blocks,
        axis=0,
    )

    diagonal_variance = (
        np.mean(
            residuals ** 2,
            axis=0,
        )
        + float(epsilon)
    )

    return {
        "class_means": (
            class_means
        ),
        "diagonal_variance": (
            diagonal_variance
        ),
        "epsilon": float(
            epsilon
        ),
    }


def mahalanobis_unknownness(
    features,
    estimator,
):
    features = np.asarray(
        features,
        dtype=np.float64,
    )

    class_means = estimator[
        "class_means"
    ]

    diagonal_variance = estimator[
        "diagonal_variance"
    ]

    minimum_distances = np.full(
        len(features),
        np.inf,
        dtype=np.float64,
    )

    for class_mean in class_means:
        differences = (
            features
            - class_mean
        )

        distances = np.sum(
            (
                differences ** 2
            )
            / diagonal_variance,
            axis=1,
        )

        minimum_distances = np.minimum(
            minimum_distances,
            distances,
        )

    return minimum_distances
