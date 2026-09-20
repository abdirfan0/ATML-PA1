from itertools import combinations

import torch

from task2.methods.dan import (
    multi_kernel_mmd,
)

from task3.methods.erm import (
    compute_erm_loss,
)


def compute_dan_dg_loss(
    model,
    images_by_domain,
    labels_by_domain,
    mmd_weight=1.0,
):
    erm_outputs = compute_erm_loss(
        model=model,
        images_by_domain=(
            images_by_domain
        ),
        labels_by_domain=(
            labels_by_domain
        ),
    )

    features_by_domain = (
        erm_outputs[
            "features_by_domain"
        ]
    )

    domain_names = list(
        features_by_domain.keys()
    )

    pairwise_mmd_losses = {}
    pairwise_median_distances = {}

    for first_domain, second_domain in (
        combinations(
            domain_names,
            2,
        )
    ):
        mmd_loss, median_distance = (
            multi_kernel_mmd(
                features_by_domain[
                    first_domain
                ],
                features_by_domain[
                    second_domain
                ],
            )
        )

        pair_name = (
            f"{first_domain}__"
            f"{second_domain}"
        )

        pairwise_mmd_losses[
            pair_name
        ] = mmd_loss

        pairwise_median_distances[
            pair_name
        ] = median_distance

    alignment_loss = torch.stack(
        list(
            pairwise_mmd_losses.values()
        )
    ).mean()

    total_loss = (
        erm_outputs[
            "classification_loss"
        ]
        + float(mmd_weight)
        * alignment_loss
    )

    return {
        "total_loss": total_loss,
        "classification_loss": (
            erm_outputs[
                "classification_loss"
            ]
        ),
        "alignment_loss": (
            alignment_loss
        ),
        "losses_by_domain": (
            erm_outputs[
                "losses_by_domain"
            ]
        ),
        "logits_by_domain": (
            erm_outputs[
                "logits_by_domain"
            ]
        ),
        "features_by_domain": (
            features_by_domain
        ),
        "pairwise_mmd_losses": (
            pairwise_mmd_losses
        ),
        "pairwise_median_distances": (
            pairwise_median_distances
        ),
        "mmd_weight": float(
            mmd_weight
        ),
    }
