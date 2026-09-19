
import random

import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from task1.models.backbones import get_normalization


VALID_DIRECTIONS = {
    "left",
    "right",
    "up",
    "down",
}


def resize_image(image, image_size=224):
    return TF.resize(
        image,
        [image_size, image_size],
        antialias=True,
    )


def convert_to_grayscale(image, image_size=224):
    image = resize_image(image, image_size)

    return TF.rgb_to_grayscale(
        image,
        num_output_channels=3,
    )


def rotate_hue(
    image,
    hue_factor=0.25,
    image_size=224,
):
    image = resize_image(image, image_size)

    return TF.adjust_hue(
        image,
        hue_factor=hue_factor,
    )


def translate_with_reflection(
    image,
    displacement,
    direction,
    image_size=224,
):
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"Unknown translation direction: {direction}"
        )

    if displacement < 0:
        raise ValueError(
            "displacement must be non-negative"
        )

    image = resize_image(image, image_size)

    if displacement == 0:
        return image

    if displacement >= image_size:
        raise ValueError(
            "displacement must be smaller than image_size"
        )

    image_tensor = TF.pil_to_tensor(image)

    padded = TF.pad(
        image_tensor,
        padding=[
            displacement,
            displacement,
            displacement,
            displacement,
        ],
        padding_mode="reflect",
    )

    horizontal_shift = 0
    vertical_shift = 0

    if direction == "left":
        horizontal_shift = -displacement
    elif direction == "right":
        horizontal_shift = displacement
    elif direction == "up":
        vertical_shift = -displacement
    elif direction == "down":
        vertical_shift = displacement

    crop_left = displacement - horizontal_shift
    crop_top = displacement - vertical_shift

    translated = TF.crop(
        padded,
        top=crop_top,
        left=crop_left,
        height=image_size,
        width=image_size,
    )

    return TF.to_pil_image(translated)


def make_patch_permutation(
    image_index,
    grid_size=4,
    seed=6304,
):
    number_of_patches = grid_size * grid_size

    original_order = list(
        range(number_of_patches)
    )

    generator = random.Random(
        seed + int(image_index)
    )

    shuffled_order = original_order.copy()

    while shuffled_order == original_order:
        generator.shuffle(shuffled_order)

    return shuffled_order


def shuffle_patches(
    image,
    image_index,
    grid_size=4,
    image_size=224,
    seed=6304,
):
    if image_size % grid_size != 0:
        raise ValueError(
            "image_size must be divisible by grid_size"
        )

    image = resize_image(image, image_size)
    image_tensor = TF.pil_to_tensor(image)

    patch_size = image_size // grid_size
    patches = []

    for row in range(grid_size):
        for column in range(grid_size):
            top = row * patch_size
            left = column * patch_size

            patch = image_tensor[
                :,
                top : top + patch_size,
                left : left + patch_size,
            ]

            patches.append(patch)

    permutation = make_patch_permutation(
        image_index=image_index,
        grid_size=grid_size,
        seed=seed,
    )

    shuffled = torch.empty_like(image_tensor)

    for destination, source in enumerate(permutation):
        destination_row = destination // grid_size
        destination_column = destination % grid_size

        top = destination_row * patch_size
        left = destination_column * patch_size

        shuffled[
            :,
            top : top + patch_size,
            left : left + patch_size,
        ] = patches[source]

    return TF.to_pil_image(shuffled)


def apply_intervention(
    image,
    intervention,
    image_index=0,
    image_size=224,
    seed=6304,
    hue_factor=0.25,
    displacement=0,
    direction="right",
    grid_size=4,
):
    if intervention == "clean":
        return resize_image(image, image_size)

    if intervention == "grayscale":
        return convert_to_grayscale(
            image,
            image_size=image_size,
        )

    if intervention == "hue":
        return rotate_hue(
            image,
            hue_factor=hue_factor,
            image_size=image_size,
        )

    if intervention == "translation":
        return translate_with_reflection(
            image,
            displacement=displacement,
            direction=direction,
            image_size=image_size,
        )

    if intervention == "patch_shuffle":
        return shuffle_patches(
            image,
            image_index=image_index,
            grid_size=grid_size,
            image_size=image_size,
            seed=seed,
        )

    raise ValueError(
        f"Unknown intervention: {intervention}"
    )


def build_model_transform(
    model_name,
    image_size=224,
):
    normalization = get_normalization(
        model_name
    )

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )
