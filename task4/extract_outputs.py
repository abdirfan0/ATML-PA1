import numpy as np
import torch

from tqdm.auto import tqdm


def extract_model_outputs(
    model,
    data_loader,
    device,
    description,
):
    model.eval()

    logits_blocks = []
    feature_blocks = []
    label_blocks = []
    index_blocks = []

    with torch.inference_mode():
        for batch in tqdm(
            data_loader,
            desc=description,
        ):
            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            logits, features = model(
                images,
                return_features=True,
            )

            logits_blocks.append(
                logits.cpu().numpy()
            )

            feature_blocks.append(
                features.cpu().numpy()
            )

            label_blocks.append(
                batch[
                    "label"
                ].cpu().numpy()
            )

            index_blocks.append(
                batch[
                    "dataset_index"
                ].cpu().numpy()
            )

    return {
        "logits": np.concatenate(
            logits_blocks,
            axis=0,
        ),
        "features": np.concatenate(
            feature_blocks,
            axis=0,
        ),
        "labels": np.concatenate(
            label_blocks,
            axis=0,
        ),
        "dataset_indices": (
            np.concatenate(
                index_blocks,
                axis=0,
            )
        ),
    }
