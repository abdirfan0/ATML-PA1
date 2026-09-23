from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

from task4.methods.proser import (
    compute_proser_loss,
)


def evaluate_proser_validation(
    model,
    validation_loader,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for batch in validation_loader:
            images = batch["image"].to(
                device
            )

            labels = batch["label"].to(
                device
            )

            known_logits, _ = model(
                images
            )

            loss = F.cross_entropy(
                known_logits,
                labels,
                reduction="sum",
            )

            predictions = (
                known_logits.argmax(
                    dim=1
                )
            )

            total_loss += loss.item()

            total_correct += (
                predictions
                .eq(labels)
                .sum()
                .item()
            )

            total_samples += len(labels)

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "accuracy": (
            total_correct
            / total_samples
        ),
        "number_of_samples": (
            total_samples
        ),
    }


def train_proser(
    model,
    train_loader,
    validation_loader,
    device,
    checkpoint_path,
    maximum_epochs=50,
    learning_rate=1e-3,
    momentum=0.9,
    weight_decay=5e-4,
    mixup_alpha=2.0,
    classifier_placeholder_weight=1.0,
    data_placeholder_weight=0.1,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(learning_rate),
        momentum=float(momentum),
        weight_decay=float(
            weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=int(maximum_epochs),
        )
    )

    use_mixed_precision = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_mixed_precision,
    )

    best_validation_accuracy = (
        float("-inf")
    )

    best_epoch = None
    history_rows = []

    for epoch in range(
        1,
        int(maximum_epochs) + 1,
    ):
        model.train()

        running_total_loss = 0.0
        running_classification_loss = 0.0
        running_classifier_placeholder = 0.0
        running_data_placeholder = 0.0
        running_accuracy = 0.0
        running_mixup = 0.0
        running_partner_rate = 0.0
        number_of_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=(
                f"proser epoch {epoch}"
            ),
        )

        for batch in progress_bar:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            if len(images) % 2 != 0:
                images = images[:-1]
                labels = labels[:-1]

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                enabled=(
                    use_mixed_precision
                ),
            ):
                loss_values = (
                    compute_proser_loss(
                        model=model,
                        images=images,
                        labels=labels,
                        mixup_alpha=(
                            mixup_alpha
                        ),
                        classifier_placeholder_weight=(
                            classifier_placeholder_weight
                        ),
                        data_placeholder_weight=(
                            data_placeholder_weight
                        ),
                    )
                )

                total_loss = loss_values[
                    "total_loss"
                ]

            scaler.scale(
                total_loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            running_total_loss += (
                total_loss.item()
            )

            running_classification_loss += (
                loss_values[
                    "classification_loss"
                ].item()
            )

            running_classifier_placeholder += (
                loss_values[
                    "classifier_placeholder_loss"
                ].item()
            )

            running_data_placeholder += (
                loss_values[
                    "data_placeholder_loss"
                ].item()
            )

            running_accuracy += (
                loss_values[
                    "ordinary_accuracy"
                ].item()
            )

            running_mixup += (
                loss_values[
                    "mixup_coefficient"
                ].item()
            )

            running_partner_rate += (
                loss_values[
                    "partner_difference_rate"
                ].item()
            )

            number_of_batches += 1

            progress_bar.set_postfix(
                loss=(
                    running_total_loss
                    / number_of_batches
                )
            )

        validation_metrics = (
            evaluate_proser_validation(
                model=model,
                validation_loader=(
                    validation_loader
                ),
                device=device,
            )
        )

        current_learning_rate = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        history_rows.append(
            {
                "epoch": epoch,
                "learning_rate": (
                    current_learning_rate
                ),
                "training_total_loss": (
                    running_total_loss
                    / number_of_batches
                ),
                "training_classification_loss": (
                    running_classification_loss
                    / number_of_batches
                ),
                "training_classifier_placeholder_loss": (
                    running_classifier_placeholder
                    / number_of_batches
                ),
                "training_data_placeholder_loss": (
                    running_data_placeholder
                    / number_of_batches
                ),
                "training_ordinary_accuracy": (
                    running_accuracy
                    / number_of_batches
                ),
                "mean_mixup_coefficient": (
                    running_mixup
                    / number_of_batches
                ),
                "different_class_pair_rate": (
                    running_partner_rate
                    / number_of_batches
                ),
                "validation_loss": (
                    validation_metrics[
                        "loss"
                    ]
                ),
                "validation_accuracy": (
                    validation_metrics[
                        "accuracy"
                    ]
                ),
            }
        )

        if (
            validation_metrics[
                "accuracy"
            ]
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_metrics[
                    "accuracy"
                ]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "selection_metric": (
                        "validation_accuracy"
                    ),
                    "selection_score": (
                        best_validation_accuracy
                    ),
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "number_of_known_classes": (
                        model.number_of_known_classes
                    ),
                    "number_of_dummy_classes": (
                        model.number_of_dummy_classes
                    ),
                    "extra_state": {
                        "mixup_alpha": float(
                            mixup_alpha
                        ),
                        "classifier_placeholder_weight": float(
                            classifier_placeholder_weight
                        ),
                        "data_placeholder_weight": float(
                            data_placeholder_weight
                        ),
                    },
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch}: "
            f"train loss="
            f"{running_total_loss / number_of_batches:.4f}, "
            f"validation accuracy="
            f"{validation_metrics['accuracy']:.4f}, "
            f"best="
            f"{best_validation_accuracy:.4f}"
        )

        scheduler.step()

    selected_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        selected_checkpoint[
            "model_state_dict"
        ]
    )

    history = pd.DataFrame(
        history_rows
    )

    return {
        "model": model,
        "history": history,
        "selected_checkpoint": (
            selected_checkpoint
        ),
        "best_epoch": best_epoch,
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
    }
