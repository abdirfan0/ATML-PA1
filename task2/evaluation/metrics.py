
import numpy as np
import torch

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
)


def to_numpy(values):
    if isinstance(values, torch.Tensor):
        values = (
            values.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(values)


def classification_metrics(
    predictions,
    targets,
    number_of_classes=7,
):
    predictions = to_numpy(
        predictions
    ).astype(int)

    targets = to_numpy(
        targets
    ).astype(int)

    if len(predictions) != len(targets):
        raise ValueError(
            "Predictions and targets must "
            "have equal lengths."
        )

    labels = np.arange(
        number_of_classes
    )

    accuracy = float(
        np.mean(
            predictions == targets
        )
    )

    macro_f1 = float(
        f1_score(
            targets,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=labels,
    )

    class_totals = matrix.sum(
        axis=1
    )

    per_class_accuracy = np.divide(
        np.diag(matrix),
        class_totals,
        out=np.zeros(
            number_of_classes,
            dtype=float,
        ),
        where=class_totals != 0,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_accuracy": (
            per_class_accuracy.tolist()
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "number_of_samples": int(
            len(targets)
        ),
    }


def metrics_from_logits(
    logits,
    targets,
    number_of_classes=7,
):
    if not isinstance(
        logits,
        torch.Tensor,
    ):
        logits = torch.as_tensor(
            logits
        )

    predictions = logits.argmax(
        dim=1
    )

    return classification_metrics(
        predictions=predictions,
        targets=targets,
        number_of_classes=(
            number_of_classes
        ),
    )


def aggregate_source_metrics(
    domain_metrics,
):
    if not domain_metrics:
        raise ValueError(
            "No source-domain metrics "
            "were supplied."
        )

    accuracies = [
        metrics["accuracy"]
        for metrics
        in domain_metrics.values()
    ]

    macro_f1_scores = [
        metrics["macro_f1"]
        for metrics
        in domain_metrics.values()
    ]

    return {
        "mean_accuracy": float(
            np.mean(accuracies)
        ),
        "worst_accuracy": float(
            np.min(accuracies)
        ),
        "mean_macro_f1": float(
            np.mean(
                macro_f1_scores
            )
        ),
        "worst_macro_f1": float(
            np.min(
                macro_f1_scores
            )
        ),
    }
