
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def to_feature_array(features):
    if isinstance(features, torch.Tensor):
        features = (
            features.detach()
            .cpu()
            .numpy()
        )

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if features.ndim != 2:
        raise ValueError(
            "Features must have shape "
            "[number_of_images, feature_dimension]."
        )

    return features


def compute_tsne_projection(
    features,
    pca_components=50,
    perplexity=30,
    seed=6304,
    max_iterations=1000,
):
    features = to_feature_array(
        features
    )

    number_of_samples = features.shape[0]
    feature_dimension = features.shape[1]

    if perplexity >= number_of_samples:
        raise ValueError(
            "perplexity must be smaller than "
            "the number of samples."
        )

    usable_components = min(
        pca_components,
        number_of_samples - 1,
        feature_dimension,
    )

    pca = PCA(
        n_components=usable_components,
        random_state=seed,
    )

    reduced_features = (
        pca.fit_transform(features)
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=max_iterations,
        random_state=seed,
    )

    embedding = tsne.fit_transform(
        reduced_features
    )

    return embedding, pca


def build_joint_embedding(
    clean_features,
    transformed_features,
    labels,
    transformed_name,
    pca_components=50,
    perplexity=30,
    seed=6304,
    max_iterations=1000,
):
    clean_features = to_feature_array(
        clean_features
    )

    transformed_features = (
        to_feature_array(
            transformed_features
        )
    )

    labels = np.asarray(labels)

    if (
        clean_features.shape
        != transformed_features.shape
    ):
        raise ValueError(
            "Clean and transformed features "
            "must have identical shapes."
        )

    if len(labels) != len(clean_features):
        raise ValueError(
            "There must be one label for "
            "each clean image."
        )

    combined_features = np.concatenate(
        [
            clean_features,
            transformed_features,
        ],
        axis=0,
    )

    combined_labels = np.concatenate(
        [labels, labels],
        axis=0,
    )

    conditions = np.concatenate(
        [
            np.full(
                len(labels),
                "clean",
                dtype=object,
            ),
            np.full(
                len(labels),
                transformed_name,
                dtype=object,
            ),
        ],
        axis=0,
    )

    embedding, pca = (
        compute_tsne_projection(
            features=combined_features,
            pca_components=pca_components,
            perplexity=perplexity,
            seed=seed,
            max_iterations=(
                max_iterations
            ),
        )
    )

    embedding_table = pd.DataFrame(
        {
            "tsne_1": embedding[:, 0],
            "tsne_2": embedding[:, 1],
            "label": combined_labels,
            "condition": conditions,
        }
    )

    return embedding_table, pca
