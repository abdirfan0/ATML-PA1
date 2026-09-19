
import importlib.util

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torchvision.transforms import functional as TF
from tqdm.auto import tqdm


def calculate_channel_statistics(
    features,
    epsilon=1e-5,
):
    if features.ndim != 4:
        raise ValueError(
            "Features must have shape "
            "[batch, channels, height, width]."
        )

    batch_size, channels = (
        features.shape[:2]
    )

    flattened = features.reshape(
        batch_size,
        channels,
        -1,
    )

    channel_mean = (
        flattened.mean(dim=2)
        .reshape(
            batch_size,
            channels,
            1,
            1,
        )
    )

    channel_variance = (
        flattened.var(dim=2)
        + epsilon
    )

    channel_standard_deviation = (
        channel_variance.sqrt()
        .reshape(
            batch_size,
            channels,
            1,
            1,
        )
    )

    return (
        channel_mean,
        channel_standard_deviation,
    )


def adaptive_instance_normalization(
    content_features,
    style_features,
):
    if (
        content_features.shape[:2]
        != style_features.shape[:2]
    ):
        raise ValueError(
            "Content and style features must "
            "have the same batch and channel sizes."
        )

    (
        content_mean,
        content_standard_deviation,
    ) = calculate_channel_statistics(
        content_features
    )

    (
        style_mean,
        style_standard_deviation,
    ) = calculate_channel_statistics(
        style_features
    )

    normalized_content = (
        content_features - content_mean
    ) / content_standard_deviation

    return (
        normalized_content
        * style_standard_deviation
        + style_mean
    )


def load_adain_networks(
    adain_repository,
    decoder_weights,
    encoder_weights,
    device,
):
    adain_repository = Path(
        adain_repository
    )

    network_file = (
        adain_repository / "net.py"
    )

    if not network_file.exists():
        raise FileNotFoundError(
            f"AdaIN network file not found: "
            f"{network_file}"
        )

    module_specification = (
        importlib.util.spec_from_file_location(
            "external_adain_network",
            network_file,
        )
    )

    adain_network = (
        importlib.util.module_from_spec(
            module_specification
        )
    )

    module_specification.loader.exec_module(
        adain_network
    )

    decoder = adain_network.decoder
    encoder = adain_network.vgg

    decoder_state = torch.load(
        decoder_weights,
        map_location="cpu",
        weights_only=True,
    )

    encoder_state = torch.load(
        encoder_weights,
        map_location="cpu",
        weights_only=True,
    )

    decoder.load_state_dict(
        decoder_state
    )

    encoder.load_state_dict(
        encoder_state
    )

    encoder = nn.Sequential(
        *list(
            encoder.children()
        )[:31]
    )

    decoder = decoder.to(device).eval()
    encoder = encoder.to(device).eval()

    for parameter in decoder.parameters():
        parameter.requires_grad = False

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    return encoder, decoder


def prepare_adain_image(
    image,
    image_size=224,
):
    image = image.convert("RGB")

    image = TF.resize(
        image,
        [image_size, image_size],
        antialias=True,
    )

    image_tensor = TF.to_tensor(
        image
    )

    return image_tensor.unsqueeze(0)


def stylize_with_adain(
    content_image,
    style_image,
    encoder,
    decoder,
    device,
    alpha=0.8,
    image_size=224,
):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    content_tensor = (
        prepare_adain_image(
            content_image,
            image_size=image_size,
        ).to(device)
    )

    style_tensor = (
        prepare_adain_image(
            style_image,
            image_size=image_size,
        ).to(device)
    )

    with torch.inference_mode():
        content_features = encoder(
            content_tensor
        )

        style_features = encoder(
            style_tensor
        )

        stylized_features = (
            adaptive_instance_normalization(
                content_features,
                style_features,
            )
        )

        blended_features = (
            alpha * stylized_features
            + (1.0 - alpha)
            * content_features
        )

        output_tensor = decoder(
            blended_features
        )

    output_tensor = (
        output_tensor.squeeze(0)
        .detach()
        .cpu()
        .clamp(0.0, 1.0)
    )

    return TF.to_pil_image(
        output_tensor
    )


def generate_candidates_from_metadata(
    base_dataset,
    metadata_path,
    output_directory,
    encoder,
    decoder,
    device,
    image_size=224,
    overwrite=False,
):
    metadata = pd.read_csv(
        metadata_path
    )

    required_columns = {
        "content_index",
        "style_index",
        "content_label",
        "style_label",
        "style_strength",
        "file_name",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing candidate columns: "
            f"{sorted(missing_columns)}"
        )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for row in tqdm(
        metadata.itertuples(index=False),
        total=len(metadata),
        desc="Generating cue conflicts",
    ):
        output_path = (
            output_directory
            / row.file_name
        )

        if (
            output_path.exists()
            and not overwrite
        ):
            continue

        content_image, content_label = (
            base_dataset[
                int(row.content_index)
            ]
        )

        style_image, style_label = (
            base_dataset[
                int(row.style_index)
            ]
        )

        if int(content_label) != int(
            row.content_label
        ):
            raise ValueError(
                f"Content label mismatch for "
                f"{row.file_name}."
            )

        if int(style_label) != int(
            row.style_label
        ):
            raise ValueError(
                f"Style label mismatch for "
                f"{row.file_name}."
            )

        output_image = stylize_with_adain(
            content_image=content_image,
            style_image=style_image,
            encoder=encoder,
            decoder=decoder,
            device=device,
            alpha=float(
                row.style_strength
            ),
            image_size=image_size,
        )

        output_image.save(
            output_path
        )
