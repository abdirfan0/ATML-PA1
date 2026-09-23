import torch


def collapse_dummy_logits(
    dummy_logits,
):
    return dummy_logits.max(
        dim=1,
        keepdim=True,
    ).values


def proser_placeholder_score(
    known_logits,
    dummy_logits,
    temperature=1024.0,
):
    if temperature <= 0:
        raise ValueError(
            "Temperature must be positive."
        )

    maximum_dummy_logits = (
        collapse_dummy_logits(
            dummy_logits
        )
    )

    combined_logits = torch.cat(
        [
            known_logits,
            maximum_dummy_logits,
        ],
        dim=1,
    )

    probabilities = torch.softmax(
        combined_logits
        / float(temperature),
        dim=1,
    )

    maximum_known_probability = (
        probabilities[
            :,
            :-1,
        ].max(
            dim=1
        ).values
    )

    dummy_probability = (
        probabilities[
            :,
            -1
        ]
    )

    unknownness = (
        dummy_probability
        - maximum_known_probability
    )

    return unknownness


def proser_mls_score(
    known_logits,
):
    return -known_logits.max(
        dim=1
    ).values
