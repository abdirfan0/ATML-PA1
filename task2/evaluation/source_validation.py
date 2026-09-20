
import torch
import torch.nn as nn

from task2.evaluation.metrics import (
    aggregate_source_metrics,
    classification_metrics,
)


def evaluate_labeled_loader(
    model,
    loader,
    device,
    number_of_classes=7,
):
    model.eval()

    loss_function = (
        nn.CrossEntropyLoss(
            reduction="sum"
        )
    )

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            targets = batch["label"].to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            total_loss += float(
                loss_function(
                    logits,
                    targets,
                ).item()
            )

            predictions = logits.argmax(
                dim=1
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                targets.cpu()
            )

    all_predictions = torch.cat(
        all_predictions,
        dim=0,
    )

    all_targets = torch.cat(
        all_targets,
        dim=0,
    )

    metrics = classification_metrics(
        predictions=all_predictions,
        targets=all_targets,
        number_of_classes=(
            number_of_classes
        ),
    )

    metrics["loss"] = (
        total_loss
        / metrics["number_of_samples"]
    )

    return metrics


def evaluate_source_domains(
    model,
    validation_loaders,
    device,
    number_of_classes=7,
):
    domain_metrics = {}

    for (
        domain_name,
        validation_loader,
    ) in validation_loaders.items():
        domain_metrics[domain_name] = (
            evaluate_labeled_loader(
                model=model,
                loader=validation_loader,
                device=device,
                number_of_classes=(
                    number_of_classes
                ),
            )
        )

    aggregate_metrics = (
        aggregate_source_metrics(
            domain_metrics
        )
    )

    return {
        "domains": domain_metrics,
        "aggregate": aggregate_metrics,
    }
