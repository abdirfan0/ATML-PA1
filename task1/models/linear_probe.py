from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from common.metrics import summarize_logits
from common.seed import make_generator, set_seed


@torch.inference_mode()
def evaluate_linear_head(
    head: nn.Linear,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a linear classifier on cached features."""

    head.eval()

    logits = head(
        features.float().to(device)
    )

    return summarize_logits(
        logits,
        labels.long().to(device),
    )


def train_linear_head(
    model_name: str,
    train_data: dict,
    validation_data: dict,
    device: torch.device,
    number_of_classes: int = 10,
    seed: int = 6304,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    maximum_epochs: int = 50,
    patience: int = 5,
    batch_size: int = 256,
    print_progress: bool = True,
) -> tuple[nn.Linear, list[dict]]:
    """Train a linear head over frozen features."""

    set_seed(seed)

    feature_dimension = (
        train_data["features"].shape[1]
    )

    head = nn.Linear(
        feature_dimension,
        number_of_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    training_dataset = TensorDataset(
        train_data["features"].float(),
        train_data["labels"].long(),
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=make_generator(seed),
    )

    best_validation_accuracy = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        head.train()

        running_loss = 0.0
        number_correct = 0
        number_seen = 0

        for features, labels in training_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = head(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size_actual = labels.size(0)

            running_loss += (
                loss.item()
                * batch_size_actual
            )

            predictions = logits.argmax(dim=1)

            number_correct += (
                predictions.eq(labels)
                .sum()
                .item()
            )

            number_seen += batch_size_actual

        training_loss = (
            running_loss / number_seen
        )

        training_accuracy = (
            number_correct / number_seen
        )

        validation_output = (
            evaluate_linear_head(
                head,
                validation_data["features"],
                validation_data["labels"],
                device,
            )
        )

        validation_accuracy = (
            validation_output["accuracy"]
        )

        history.append({
            "epoch": epoch,
            "train_loss": training_loss,
            "train_accuracy": (
                training_accuracy
            ),
            "validation_accuracy": (
                validation_accuracy
            ),
        })

        if print_progress:
            print(
                f"{model_name} | "
                f"Epoch {epoch:02d} | "
                f"Loss {training_loss:.4f} | "
                f"Train Acc "
                f"{training_accuracy:.4f} | "
                f"Val Acc "
                f"{validation_accuracy:.4f}"
            )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_accuracy
            )

            best_state = deepcopy(
                head.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            if print_progress:
                print(
                    "Early stopping after "
                    f"{patience} epochs "
                    "without improvement."
                )

            break

    if best_state is None:
        raise RuntimeError(
            "No valid classifier checkpoint "
            "was produced."
        )

    head.load_state_dict(best_state)

    return head, history


def save_linear_head(
    head: nn.Linear,
    checkpoint_path: str | Path,
    model_name: str,
    class_names: list[str],
    seed: int = 6304,
) -> None:
    """Save a trained linear classifier."""

    checkpoint_path = Path(
        checkpoint_path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_dict_cpu = {
        name: tensor.detach().cpu()
        for name, tensor
        in head.state_dict().items()
    }

    checkpoint = {
        "model_name": model_name,
        "feature_dim": head.in_features,
        "number_of_classes": (
            head.out_features
        ),
        "class_names": class_names,
        "seed": seed,
        "state_dict": state_dict_cpu,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def load_linear_head(
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Linear:
    """Load a trained linear classifier."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    head = nn.Linear(
        checkpoint["feature_dim"],
        checkpoint["number_of_classes"],
    ).to(device)

    head.load_state_dict(
        checkpoint["state_dict"]
    )

    head.eval()

    return head
