
import numpy as np
import pandas as pd
import torch


def to_numpy(values):
    if isinstance(values, torch.Tensor):
        values = (
            values.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(values)


def classify_cue_decisions(
    predictions,
    shape_targets,
    texture_targets,
):
    predictions = to_numpy(predictions)
    shape_targets = to_numpy(shape_targets)
    texture_targets = to_numpy(
        texture_targets
    )

    if not (
        len(predictions)
        == len(shape_targets)
        == len(texture_targets)
    ):
        raise ValueError(
            "Predictions and targets must "
            "have equal lengths."
        )

    decisions = np.full(
        len(predictions),
        "other",
        dtype=object,
    )

    decisions[
        predictions == texture_targets
    ] = "texture"

    decisions[
        predictions == shape_targets
    ] = "shape"

    return decisions


def summarize_cue_decisions(decisions):
    decisions = to_numpy(decisions)

    number_shape = int(
        np.sum(decisions == "shape")
    )

    number_texture = int(
        np.sum(decisions == "texture")
    )

    number_other = int(
        np.sum(decisions == "other")
    )

    number_total = len(decisions)

    covered_predictions = (
        number_shape + number_texture
    )

    if covered_predictions == 0:
        shape_bias_percent = float("nan")
    else:
        shape_bias_percent = (
            100.0
            * number_shape
            / covered_predictions
        )

    if number_total == 0:
        coverage_percent = float("nan")
    else:
        coverage_percent = (
            100.0
            * covered_predictions
            / number_total
        )

    return {
        "n_shape": number_shape,
        "n_texture": number_texture,
        "n_other": number_other,
        "n_total": number_total,
        "shape_bias_percent": (
            shape_bias_percent
        ),
        "coverage_percent": (
            coverage_percent
        ),
    }


def evaluate_cue_conflicts(
    predictions,
    shape_targets,
    texture_targets,
):
    decisions = classify_cue_decisions(
        predictions=predictions,
        shape_targets=shape_targets,
        texture_targets=texture_targets,
    )

    summary = summarize_cue_decisions(
        decisions
    )

    return decisions, summary


def evaluate_cue_conflict_logits(
    logits,
    shape_targets,
    texture_targets,
):
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)

    predictions = logits.argmax(dim=1)

    return evaluate_cue_conflicts(
        predictions=predictions,
        shape_targets=shape_targets,
        texture_targets=texture_targets,
    )


def build_cue_conflict_summary(
    prediction_table,
):
    required_columns = {
        "model",
        "evaluation",
        "content_class",
        "style_class",
        "predicted_class",
    }

    missing_columns = (
        required_columns
        - set(prediction_table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing prediction columns: "
            f"{sorted(missing_columns)}"
        )

    summary_rows = []

    grouped_predictions = (
        prediction_table.groupby(
            ["model", "evaluation"],
            sort=False,
        )
    )

    for (
        model_name,
        evaluation_name,
    ), group in grouped_predictions:
        decisions, summary = (
            evaluate_cue_conflicts(
                predictions=group[
                    "predicted_class"
                ].to_numpy(),
                shape_targets=group[
                    "content_class"
                ].to_numpy(),
                texture_targets=group[
                    "style_class"
                ].to_numpy(),
            )
        )

        summary_rows.append(
            {
                "model": model_name,
                "evaluation": (
                    evaluation_name
                ),
                **summary,
            }
        )

    return pd.DataFrame(summary_rows)
