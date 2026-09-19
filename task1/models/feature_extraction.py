
from pathlib import Path

import torch

from tqdm.auto import tqdm


def extract_features(
    model,
    dataloader,
    device,
    description="Extracting features",
):
    model = model.to(device)
    model.eval()

    feature_batches = []
    label_batches = []
    index_batches = []

    running_index = 0

    with torch.inference_mode():
        for batch in tqdm(
            dataloader,
            desc=description,
        ):
            if len(batch) == 3:
                images, labels, indices = batch

            elif len(batch) == 2:
                images, labels = batch

                batch_size = len(labels)

                indices = torch.arange(
                    running_index,
                    running_index + batch_size,
                )

                running_index += batch_size

            else:
                raise ValueError(
                    "Each batch must contain either "
                    "(images, labels) or "
                    "(images, labels, indices)."
                )

            images = images.to(
                device,
                non_blocking=True,
            )

            features = model(images)

            if isinstance(features, tuple):
                features = features[0]

            feature_batches.append(
                features.detach().cpu()
            )

            label_batches.append(
                torch.as_tensor(
                    labels
                ).detach().cpu()
            )

            index_batches.append(
                torch.as_tensor(
                    indices
                ).detach().cpu()
            )

    return {
        "features": torch.cat(
            feature_batches,
            dim=0,
        ),
        "labels": torch.cat(
            label_batches,
            dim=0,
        ).long(),
        "indices": torch.cat(
            index_batches,
            dim=0,
        ).long(),
    }


def save_feature_cache(
    output_path,
    features,
    labels,
    indices,
    metadata=None,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = {
        "features": (
            torch.as_tensor(features)
            .detach()
            .cpu()
        ),
        "labels": (
            torch.as_tensor(labels)
            .detach()
            .cpu()
            .long()
        ),
        "indices": (
            torch.as_tensor(indices)
            .detach()
            .cpu()
            .long()
        ),
        "metadata": (
            {} if metadata is None
            else dict(metadata)
        ),
    }

    torch.save(
        cache,
        output_path,
    )


def load_feature_cache(
    cache_path,
):
    cache_path = Path(cache_path)

    cache = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=True,
    )

    required_keys = {
        "features",
        "labels",
        "indices",
        "metadata",
    }

    missing_keys = (
        required_keys - set(cache)
    )

    if missing_keys:
        raise ValueError(
            "Feature cache is missing keys: "
            f"{sorted(missing_keys)}"
        )

    return cache
