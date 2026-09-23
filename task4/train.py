import pandas as pd
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

from task4.evaluation.classification import (
    evaluate_closed_set,
)


def train_closed_set_model(
    method_name,
    model,
    loaders,
    checkpoint_path,
    device,
    maximum_epochs=100,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    number_of_classes=10,
    use_mixed_precision=True,
):
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=maximum_epochs,
        )
    )

    mixed_precision_enabled = (
        bool(use_mixed_precision)
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            mixed_precision_enabled
        ),
    )

    history_rows = []

    best_validation_accuracy = float(
        "-inf"
    )

    best_epoch = None

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        model.train()

        running_loss = 0.0
        number_correct = 0
        number_seen = 0

        learning_rate_this_epoch = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        progress_bar = tqdm(
            loaders["train"],
            desc=(
                f"{method_name} "
                f"epoch {epoch}"
            ),
        )

        for batch in progress_bar:
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

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(
                    mixed_precision_enabled
                ),
            ):
                logits = model(
                    images
                )

                loss = F.cross_entropy(
                    logits,
                    labels,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            batch_size = len(
                labels
            )

            running_loss += (
                float(
                    loss.detach().item()
                )
                * batch_size
            )

            number_correct += int(
                (
                    logits.detach().argmax(
                        dim=1
                    )
                    == labels
                ).sum().item()
            )

            number_seen += batch_size

            progress_bar.set_postfix(
                loss=(
                    running_loss
                    / number_seen
                )
            )

        validation_metrics = (
            evaluate_closed_set(
                model=model,
                data_loader=loaders[
                    "validation"
                ],
                device=device,
                number_of_classes=(
                    number_of_classes
                ),
            )
        )

        training_loss = (
            running_loss
            / number_seen
        )

        training_accuracy = (
            number_correct
            / number_seen
        )

        history_rows.append(
            {
                "epoch": epoch,
                "learning_rate": (
                    learning_rate_this_epoch
                ),
                "training_loss": (
                    training_loss
                ),
                "training_accuracy": (
                    training_accuracy
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
                "validation_macro_f1": (
                    validation_metrics[
                        "macro_f1"
                    ]
                ),
            }
        )

        improved = (
            validation_metrics[
                "accuracy"
            ]
            > best_validation_accuracy
        )

        if improved:
            best_validation_accuracy = (
                validation_metrics[
                    "accuracy"
                ]
            )

            best_epoch = epoch

            checkpoint = {
                "method": method_name,
                "epoch": epoch,
                "selection_metric": (
                    "cifar10_validation_accuracy"
                ),
                "selection_score": float(
                    best_validation_accuracy
                ),
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "scheduler_state_dict": (
                    scheduler.state_dict()
                ),
                "number_of_classes": int(
                    number_of_classes
                ),
                "mixed_precision": bool(
                    mixed_precision_enabled
                ),
            }

            torch.save(
                checkpoint,
                checkpoint_path,
            )

        print(
            f"Epoch {epoch}: "
            f"train loss="
            f"{training_loss:.4f}, "
            f"train accuracy="
            f"{training_accuracy:.4f}, "
            f"validation accuracy="
            f"{validation_metrics['accuracy']:.4f}, "
            f"best="
            f"{best_validation_accuracy:.4f}"
        )

        scheduler.step()

    selected_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        selected_checkpoint[
            "model_state_dict"
        ]
    )

    return {
        "model": model,
        "history": pd.DataFrame(
            history_rows
        ),
        "selected_checkpoint": (
            selected_checkpoint
        ),
        "best_epoch": best_epoch,
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
    }
