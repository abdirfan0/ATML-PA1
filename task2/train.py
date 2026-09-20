
import math

import pandas as pd
import torch

from tqdm.auto import tqdm

from shared.pacs_loaders import (
    cycle_loader,
)

from shared.pacs_protocol import (
    SOURCE_DOMAINS,
)

from task2.evaluation.source_validation import (
    evaluate_source_domains,
)

from task2.methods.cdan import (
    compute_cdan_loss,
)

from task2.methods.dan import (
    compute_dan_loss,
)

from task2.methods.dann import (
    compute_dann_loss,
)

from task2.methods.source_only import (
    compute_source_only_loss,
)

from task2.models.backbone import (
    freeze_batchnorm_statistics,
)

from task2.selection.checkpointing import (
    load_selected_checkpoint,
)


def compute_method_loss(
    method_name,
    model,
    source_images,
    source_labels,
    target_images=None,
    domain_discriminator=None,
    progress=0.0,
    method_parameters=None,
):
    if method_parameters is None:
        method_parameters = {}

    if method_name == "source_only":
        return compute_source_only_loss(
            model=model,
            source_images=source_images,
            source_labels=source_labels,
        )

    if target_images is None:
        raise ValueError(
            f"{method_name} requires "
            "unlabeled target images."
        )

    if method_name == "dan":
        return compute_dan_loss(
            model=model,
            source_images=source_images,
            source_labels=source_labels,
            target_images=target_images,
            mmd_weight=(
                method_parameters.get(
                    "mmd_weight",
                    1.0,
                )
            ),
        )

    if domain_discriminator is None:
        raise ValueError(
            f"{method_name} requires a "
            "domain discriminator."
        )

    if method_name == "dann":
        return compute_dann_loss(
            model=model,
            domain_discriminator=(
                domain_discriminator
            ),
            source_images=source_images,
            source_labels=source_labels,
            target_images=target_images,
            progress=progress,
            maximum_grl_strength=(
                method_parameters.get(
                    "maximum_grl_strength",
                    1.0,
                )
            ),
            domain_loss_weight=(
                method_parameters.get(
                    "domain_loss_weight",
                    1.0,
                )
            ),
        )

    if method_name == "cdan":
        return compute_cdan_loss(
            model=model,
            domain_discriminator=(
                domain_discriminator
            ),
            source_images=source_images,
            source_labels=source_labels,
            target_images=target_images,
            progress=progress,
            maximum_grl_strength=(
                method_parameters.get(
                    "maximum_grl_strength",
                    1.0,
                )
            ),
            domain_loss_weight=(
                method_parameters.get(
                    "domain_loss_weight",
                    1.0,
                )
            ),
        )

    raise ValueError(
        f"Unknown method: {method_name}"
    )


def train_task2_method(
    method_name,
    model,
    loaders,
    optimizer,
    selector,
    device,
    maximum_epochs=30,
    domain_discriminator=None,
    method_parameters=None,
    number_of_classes=7,
):
    model = model.to(device)

    if domain_discriminator is not None:
        domain_discriminator = (
            domain_discriminator.to(
                device
            )
        )

    steps_per_epoch = int(
        loaders["steps_per_epoch"]
    )

    total_planned_steps = (
        maximum_epochs
        * steps_per_epoch
    )

    history_rows = []

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        model.train()

        freeze_batchnorm_statistics(
            model
        )

        if domain_discriminator is not None:
            domain_discriminator.train()

        source_iterators = {
            domain_name: cycle_loader(
                loaders[
                    "source_train"
                ][domain_name]
            )
            for domain_name
            in SOURCE_DOMAINS
        }

        if method_name == "source_only":
            target_iterator = None
        else:
            target_iterator = cycle_loader(
                loaders[
                    "target_adaptation"
                ]
            )

        total_loss_sum = 0.0
        classification_loss_sum = 0.0
        alignment_loss_sum = 0.0
        domain_loss_sum = 0.0
        grl_strength_sum = 0.0

        alignment_steps = 0
        domain_steps = 0
        grl_steps = 0

        correct_source_predictions = 0
        total_source_predictions = 0

        progress_bar = tqdm(
            range(steps_per_epoch),
            desc=(
                f"{method_name} "
                f"epoch {epoch}"
            ),
        )

        for step_index in progress_bar:
            source_batches = [
                next(
                    source_iterators[
                        domain_name
                    ]
                )
                for domain_name
                in SOURCE_DOMAINS
            ]

            source_images = torch.cat(
                [
                    batch["image"]
                    for batch
                    in source_batches
                ],
                dim=0,
            ).to(
                device,
                non_blocking=True,
            )

            source_labels = torch.cat(
                [
                    batch["label"]
                    for batch
                    in source_batches
                ],
                dim=0,
            ).to(
                device,
                non_blocking=True,
            )

            if target_iterator is None:
                target_images = None
            else:
                target_batch = next(
                    target_iterator
                )

                target_images = (
                    target_batch["image"]
                    .to(
                        device,
                        non_blocking=True,
                    )
                )

            global_step = (
                (epoch - 1)
                * steps_per_epoch
                + step_index
            )

            progress_denominator = max(
                total_planned_steps - 1,
                1,
            )

            progress = (
                global_step
                / progress_denominator
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss_output = (
                compute_method_loss(
                    method_name=(
                        method_name
                    ),
                    model=model,
                    source_images=(
                        source_images
                    ),
                    source_labels=(
                        source_labels
                    ),
                    target_images=(
                        target_images
                    ),
                    domain_discriminator=(
                        domain_discriminator
                    ),
                    progress=progress,
                    method_parameters=(
                        method_parameters
                    ),
                )
            )

            total_loss = loss_output[
                "total_loss"
            ]

            if not torch.isfinite(
                total_loss
            ):
                raise RuntimeError(
                    "A non-finite loss "
                    "was encountered."
                )

            total_loss.backward()
            optimizer.step()

            total_loss_sum += float(
                total_loss.item()
            )

            classification_loss_sum += (
                float(
                    loss_output[
                        "classification_loss"
                    ].item()
                )
            )

            if (
                loss_output[
                    "alignment_loss"
                ]
                is not None
            ):
                alignment_loss_sum += float(
                    loss_output[
                        "alignment_loss"
                    ].item()
                )

                alignment_steps += 1

            if (
                loss_output[
                    "domain_loss"
                ]
                is not None
            ):
                domain_loss_sum += float(
                    loss_output[
                        "domain_loss"
                    ].item()
                )

                domain_steps += 1

            if "grl_strength" in loss_output:
                grl_strength_sum += float(
                    loss_output[
                        "grl_strength"
                    ]
                )

                grl_steps += 1

            source_predictions = (
                loss_output[
                    "source_logits"
                ].argmax(dim=1)
            )

            correct_source_predictions += int(
                (
                    source_predictions
                    == source_labels
                ).sum().item()
            )

            total_source_predictions += int(
                len(source_labels)
            )

            progress_bar.set_postfix(
                loss=(
                    total_loss_sum
                    / (step_index + 1)
                )
            )

        validation_results = (
            evaluate_source_domains(
                model=model,
                validation_loaders=(
                    loaders[
                        "source_validation"
                    ]
                ),
                device=device,
                number_of_classes=(
                    number_of_classes
                ),
            )
        )

        mean_macro_f1 = (
            validation_results[
                "aggregate"
            ]["mean_macro_f1"]
        )

        additional_models = {}

        if domain_discriminator is not None:
            additional_models[
                "domain_discriminator"
            ] = domain_discriminator

        epoch_state = {
            "method": method_name,
            "mean_source_macro_f1": (
                mean_macro_f1
            ),
        }

        improved = selector.update(
            score=mean_macro_f1,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            additional_models=(
                additional_models
            ),
            extra_state=epoch_state,
        )

        history_row = {
            "epoch": epoch,
            "training_total_loss": (
                total_loss_sum
                / steps_per_epoch
            ),
            "training_classification_loss": (
                classification_loss_sum
                / steps_per_epoch
            ),
            "training_alignment_loss": (
                alignment_loss_sum
                / alignment_steps
                if alignment_steps > 0
                else math.nan
            ),
            "training_domain_loss": (
                domain_loss_sum
                / domain_steps
                if domain_steps > 0
                else math.nan
            ),
            "training_source_accuracy": (
                correct_source_predictions
                / total_source_predictions
            ),
            "mean_grl_strength": (
                grl_strength_sum
                / grl_steps
                if grl_steps > 0
                else math.nan
            ),
            "mean_source_validation_accuracy": (
                validation_results[
                    "aggregate"
                ]["mean_accuracy"]
            ),
            "worst_source_validation_accuracy": (
                validation_results[
                    "aggregate"
                ]["worst_accuracy"]
            ),
            "mean_source_validation_macro_f1": (
                mean_macro_f1
            ),
            "worst_source_validation_macro_f1": (
                validation_results[
                    "aggregate"
                ]["worst_macro_f1"]
            ),
            "checkpoint_improved": (
                improved
            ),
        }

        for domain_name in SOURCE_DOMAINS:
            domain_metrics = (
                validation_results[
                    "domains"
                ][domain_name]
            )

            history_row[
                f"{domain_name}_validation_accuracy"
            ] = domain_metrics[
                "accuracy"
            ]

            history_row[
                f"{domain_name}_validation_macro_f1"
            ] = domain_metrics[
                "macro_f1"
            ]

        history_rows.append(
            history_row
        )

        print(
            f"Epoch {epoch}: "
            f"train loss="
            f"{history_row['training_total_loss']:.4f}, "
            f"mean validation macro-F1="
            f"{mean_macro_f1:.4f}, "
            f"best={selector.best_score:.4f}"
        )

        if selector.should_stop:
            print(
                "Early stopping triggered."
            )

            break

    selected_checkpoint = (
        load_selected_checkpoint(
            checkpoint_path=(
                selector.checkpoint_path
            ),
            model=model,
            additional_models=(
                additional_models
            ),
            device=device,
        )
    )

    history_table = pd.DataFrame(
        history_rows
    )

    return {
        "model": model,
        "domain_discriminator": (
            domain_discriminator
        ),
        "history": history_table,
        "selected_checkpoint": (
            selected_checkpoint
        ),
    }
