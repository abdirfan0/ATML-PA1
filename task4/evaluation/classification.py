import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


def evaluate_closed_set(
    model,
    data_loader,
    device,
    number_of_classes=10,
):
    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0.0
    number_of_samples = 0

    with torch.inference_mode():
        for batch in data_loader:
            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            labels = batch[
                "label"
            ].to(
                device,
                non_blocking=True,
            )

            logits = model(
                images
            )

            loss = F.cross_entropy(
                logits,
                labels,
                reduction="sum",
            )

            predictions = logits.argmax(
                dim=1
            )

            total_loss += float(
                loss.item()
            )

            number_of_samples += int(
                len(labels)
            )

            all_labels.append(
                labels.cpu().numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

    labels = np.concatenate(
        all_labels
    )

    predictions = np.concatenate(
        all_predictions
    )

    return {
        "loss": (
            total_loss
            / number_of_samples
        ),
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            labels=list(
                range(
                    number_of_classes
                )
            ),
            zero_division=0,
        ),
        "number_of_samples": (
            number_of_samples
        ),
    }
