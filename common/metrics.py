from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.inference_mode()
def summarize_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    """Calculate predictions and classification metrics."""

    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    confidence = probabilities.max(dim=1).values

    labels_cpu = labels.detach().cpu()
    predictions_cpu = predictions.detach().cpu()
    probabilities_cpu = probabilities.detach().cpu()
    logits_cpu = logits.detach().cpu()

    accuracy = accuracy_score(
        labels_cpu.numpy(),
        predictions_cpu.numpy(),
    )

    macro_f1 = f1_score(
        labels_cpu.numpy(),
        predictions_cpu.numpy(),
        average="macro",
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "mean_max_confidence": float(
            confidence.mean().item()
        ),
        "logits": logits_cpu,
        "probabilities": probabilities_cpu,
        "predictions": predictions_cpu,
        "labels": labels_cpu,
    }


def prediction_consistency(
    clean_predictions: torch.Tensor | np.ndarray,
    transformed_predictions: torch.Tensor | np.ndarray,
) -> float:
    """Return the fraction of unchanged predictions."""

    clean_array = np.asarray(
        clean_predictions.detach().cpu()
        if isinstance(clean_predictions, torch.Tensor)
        else clean_predictions
    )

    transformed_array = np.asarray(
        transformed_predictions.detach().cpu()
        if isinstance(transformed_predictions, torch.Tensor)
        else transformed_predictions
    )

    if clean_array.shape != transformed_array.shape:
        raise ValueError(
            "Clean and transformed predictions "
            "must have the same shape."
        )

    return float(
        np.mean(clean_array == transformed_array)
    )
