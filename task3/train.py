import pandas as pd
import torch

from tqdm.auto import tqdm

from task2.evaluation.source_validation import (
    evaluate_source_domains,
)

from task2.models.backbone import (
    freeze_batchnorm_statistics,
)

from task3.data.source_loaders import (
    cycle_loader,
)

from task3.methods.dan_dg import (
    compute_dan_dg_loss,
)

from task3.methods.erm import (
    compute_erm_loss,
)


SUPPORTED_METHODS = {
    "dan_dg",
    "sam",
}


def _prepare_source_batches(
    source_iterators,
    device,
):
    images_by_domain = {}
    labels_by_domain = {}

    for domain_name, iterator in (
        source_iterators.items()
    ):
        batch = next(iterator)

        images_by_domain[
            domain_name
        ] = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels_by_domain[
            domain_name
        ] = batch["label"].to(
            device,
            non_blocking=True,
        )

    return (
        images_by_domain,
        labels_by_domain,
    )


def _classification_accuracy(
    logits_by_domain,
    labels_by_domain,
):
    number_correct = 0
    number_total = 0

    for domain_name, logits in (
        logits_by_domain.items()
    ):
        predictions = logits.argmax(
            dim=1
        )

        labels = labels_by_domain[
            domain_name
        ]

        number_correct += int(
            (
                predictions == labels
            ).sum().item()
        )

        number_total += int(
            labels.numel()
        )

    return (
        number_correct
        / max(number_total, 1)
    )


def train_task3_method(
    method_name,
    model,
    loaders,
    optimizer,
    selector,
    device,
    maximum_epochs=30,
    mmd_weight=1.0,
    gradient_clip_norm=1.0,
    number_of_classes=7,
):
    if method_name not in (
        SUPPORTED_METHODS
    ):
        raise ValueError(
            f"Unsupported Task 3 method: "
            f"{method_name}"
        )

    source_train_loaders = loaders[
        "source_train"
    ]

    source_validation_loaders = (
        loaders[
            "source_validation"
        ]
    )

    steps_per_epoch = int(
        loaders["steps_per_epoch"]
    )

    training_rows = []

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        model.train()

        freeze_batchnorm_statistics(
            model
        )

        source_iterators = {
            domain_name: cycle_loader(
                data_loader
            )
            for domain_name, data_loader
            in source_train_loaders.items()
        }

        epoch_total_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_alignment_loss = 0.0
        epoch_unperturbed_loss = 0.0
        epoch_gradient_norm = 0.0
        epoch_source_accuracy = 0.0

        progress_bar = tqdm(
            range(steps_per_epoch),
            desc=(
                f"{method_name} "
                f"epoch {epoch}"
            ),
        )

        for _ in progress_bar:
            (
                images_by_domain,
                labels_by_domain,
            ) = _prepare_source_batches(
                source_iterators=(
                    source_iterators
                ),
                device=device,
            )

            if method_name == "dan_dg":
                optimizer.zero_grad(
                    set_to_none=True
                )

                outputs = (
                    compute_dan_dg_loss(
                        model=model,
                        images_by_domain=(
                            images_by_domain
                        ),
                        labels_by_domain=(
                            labels_by_domain
                        ),
                        mmd_weight=(
                            mmd_weight
                        ),
                    )
                )

                outputs[
                    "total_loss"
                ].backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(
                        gradient_clip_norm
                    ),
                )

                optimizer.step()

                total_loss = outputs[
                    "total_loss"
                ]

                classification_loss = (
                    outputs[
                        "classification_loss"
                    ]
                )

                alignment_loss = outputs[
                    "alignment_loss"
                ]

                unperturbed_loss = (
                    classification_loss
                )

                gradient_norm_value = (
                    float("nan")
                )

                accuracy_logits = outputs[
                    "logits_by_domain"
                ]

            else:
                optimizer.zero_grad(
                    set_to_none=True
                )

                first_outputs = (
                    compute_erm_loss(
                        model=model,
                        images_by_domain=(
                            images_by_domain
                        ),
                        labels_by_domain=(
                            labels_by_domain
                        ),
                    )
                )

                first_outputs[
                    "total_loss"
                ].backward()

                sam_gradient_norm = (
                    optimizer.first_step()
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                try:
                    freeze_batchnorm_statistics(
                        model
                    )

                    second_outputs = (
                        compute_erm_loss(
                            model=model,
                            images_by_domain=(
                                images_by_domain
                            ),
                            labels_by_domain=(
                                labels_by_domain
                            ),
                        )
                    )

                    second_outputs[
                        "total_loss"
                    ].backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=(
                            gradient_clip_norm
                        ),
                    )

                    optimizer.second_step()

                except Exception:
                    optimizer.restore_parameters()
                    raise

                total_loss = second_outputs[
                    "total_loss"
                ]

                classification_loss = (
                    second_outputs[
                        "classification_loss"
                    ]
                )

                alignment_loss = None

                unperturbed_loss = (
                    first_outputs[
                        "classification_loss"
                    ]
                )

                gradient_norm_value = float(
                    sam_gradient_norm.item()
                )

                accuracy_logits = (
                    first_outputs[
                        "logits_by_domain"
                    ]
                )

            batch_accuracy = (
                _classification_accuracy(
                    logits_by_domain=(
                        accuracy_logits
                    ),
                    labels_by_domain=(
                        labels_by_domain
                    ),
                )
            )

            epoch_total_loss += float(
                total_loss.detach().item()
            )

            epoch_classification_loss += (
                float(
                    classification_loss
                    .detach()
                    .item()
                )
            )

            if alignment_loss is not None:
                epoch_alignment_loss += (
                    float(
                        alignment_loss
                        .detach()
                        .item()
                    )
                )

            epoch_unperturbed_loss += float(
                unperturbed_loss
                .detach()
                .item()
            )

            if method_name == "sam":
                epoch_gradient_norm += (
                    gradient_norm_value
                )

            epoch_source_accuracy += (
                batch_accuracy
            )

            progress_bar.set_postfix(
                loss=(
                    epoch_total_loss
                    / (
                        progress_bar.n + 1
                    )
                )
            )

        validation_results = (
            evaluate_source_domains(
                model=model,
                validation_loaders=(
                    source_validation_loaders
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

        history_row = {
            "epoch": epoch,
            "training_total_loss": (
                epoch_total_loss
                / steps_per_epoch
            ),
            "training_classification_loss": (
                epoch_classification_loss
                / steps_per_epoch
            ),
            "training_alignment_loss": (
                epoch_alignment_loss
                / steps_per_epoch
                if method_name
                == "dan_dg"
                else float("nan")
            ),
            "training_unperturbed_loss": (
                epoch_unperturbed_loss
                / steps_per_epoch
            ),
            "training_source_accuracy": (
                epoch_source_accuracy
                / steps_per_epoch
            ),
            "mean_sam_gradient_norm": (
                epoch_gradient_norm
                / steps_per_epoch
                if method_name == "sam"
                else float("nan")
            ),
            "mean_source_validation_accuracy": (
                validation_results[
                    "aggregate"
                ]["mean_accuracy"]
            ),
            "mean_source_validation_macro_f1": (
                mean_macro_f1
            ),
            "worst_source_validation_accuracy": (
                validation_results[
                    "aggregate"
                ]["worst_accuracy"]
            ),
            "worst_source_validation_macro_f1": (
                validation_results[
                    "aggregate"
                ]["worst_macro_f1"]
            ),
        }

        for domain_name, metrics in (
            validation_results[
                "domains"
            ].items()
        ):
            history_row[
                f"{domain_name}_"
                "validation_accuracy"
            ] = metrics["accuracy"]

            history_row[
                f"{domain_name}_"
                "validation_macro_f1"
            ] = metrics["macro_f1"]

        training_rows.append(
            history_row
        )

        selector.update(
            score=mean_macro_f1,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            extra_state={
                "method": method_name,
                "mmd_weight": (
                    float(mmd_weight)
                    if method_name
                    == "dan_dg"
                    else None
                ),
            },
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

    selected_checkpoint = torch.load(
        selector.checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        selected_checkpoint[
            "model_state_dict"
        ]
    )

    training_history = pd.DataFrame(
        training_rows
    )

    return {
        "model": model,
        "history": training_history,
        "selected_checkpoint": (
            selected_checkpoint
        ),
    }
