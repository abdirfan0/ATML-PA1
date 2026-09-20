
from pathlib import Path

import torch


class SourceValidationSelector:
    def __init__(
        self,
        checkpoint_path,
        patience=5,
        minimum_improvement=0.0,
    ):
        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.patience = int(patience)

        self.minimum_improvement = float(
            minimum_improvement
        )

        self.best_score = float("-inf")
        self.best_epoch = None
        self.epochs_without_improvement = 0

    def update(
        self,
        score,
        epoch,
        model,
        optimizer,
        additional_models=None,
        extra_state=None,
    ):
        score = float(score)
        epoch = int(epoch)

        improved = (
            score
            > self.best_score
            + self.minimum_improvement
        )

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0

            if additional_models is None:
                additional_models = {}

            additional_model_states = {
                name: additional_model.state_dict()
                for name, additional_model
                in additional_models.items()
            }

            self.checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            checkpoint = {
                "epoch": epoch,
                "selection_metric": (
                    "mean_source_validation_macro_f1"
                ),
                "selection_score": score,
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "additional_model_state_dicts": (
                    additional_model_states
                ),
                "extra_state": (
                    {}
                    if extra_state is None
                    else extra_state
                ),
            }

            torch.save(
                checkpoint,
                self.checkpoint_path,
            )

        else:
            self.epochs_without_improvement += 1

        return improved

    @property
    def should_stop(self):
        return (
            self.epochs_without_improvement
            >= self.patience
        )


def load_selected_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    additional_models=None,
    device="cpu",
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    if additional_models is not None:
        saved_additional_states = (
            checkpoint[
                "additional_model_state_dicts"
            ]
        )

        for name, additional_model in (
            additional_models.items()
        ):
            additional_model.load_state_dict(
                saved_additional_states[
                    name
                ]
            )

    return checkpoint
