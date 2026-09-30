
from pathlib import Path

import numpy as np
import rasterio
import torch


def get_patch_starts(image_size, patch_size):
    """
    Generate patch start positions while ensuring the final
    patch reaches the image boundary.
    """

    if image_size <= patch_size:
        return [0]

    starts = list(
        range(
            0,
            image_size - patch_size + 1,
            patch_size,
        )
    )

    final_start = image_size - patch_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


@torch.no_grad()
def predict_full_tile(
    model,
    image_path,
    mean,
    std,
    patch_size=256,
    threshold=0.5,
    device="cuda",
):
    """
    Predict a complete tile using overlapping patches.

    Returns
    -------
    image : np.ndarray
        Original image in (C, H, W) format.

    full_probability : np.ndarray
        Averaged building probability map.

    full_prediction : np.ndarray
        Binary building prediction.
    """

    image_path = Path(image_path)

    with rasterio.open(image_path) as src:
        image = src.read()

    _, height, width = image.shape

    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    std = np.asarray(
        std,
        dtype=np.float32,
    )

    y_starts = get_patch_starts(
        height,
        patch_size,
    )

    x_starts = get_patch_starts(
        width,
        patch_size,
    )

    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    probability_count = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    model.eval()

    for y in y_starts:

        for x in x_starts:

            patch = image[
                :,
                y:y + patch_size,
                x:x + patch_size,
            ]

            patch = np.transpose(
                patch,
                (1, 2, 0),
            )

            patch = patch.astype(
                np.float32
            )

            patch = (
                patch - mean
            ) / std

            patch_tensor = torch.from_numpy(
                np.transpose(
                    patch,
                    (2, 0, 1),
                )
            ).float()

            patch_tensor = (
                patch_tensor
                .unsqueeze(0)
                .to(device)
            )

            logits = model(
                patch_tensor
            )

            probability = (
                torch.sigmoid(logits)
                [0, 0]
                .cpu()
                .numpy()
            )

            probability_sum[
                y:y + patch_size,
                x:x + patch_size,
            ] += probability

            probability_count[
                y:y + patch_size,
                x:x + patch_size,
            ] += 1

    full_probability = (
        probability_sum
        / np.maximum(
            probability_count,
            1,
        )
    )

    full_prediction = (
        full_probability >= threshold
    ).astype(np.uint8)

    return (
        image,
        full_probability,
        full_prediction,
    )
