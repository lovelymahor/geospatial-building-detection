
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class CachedBuildingPatchDataset(Dataset):
    """
    PyTorch dataset for extracting building-detection patches
    from cached full-tile masks.
    """

    def __init__(
        self,
        dataframe,
        mask_cache_dir,
        mean,
        std,
        patch_size=256,
        transform=None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.mask_cache_dir = Path(mask_cache_dir)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        image_id = row["image_id"]

        x = int(row["x"])
        y = int(row["y"])

        with rasterio.open(image_path) as src:
            image = src.read()

        mask_path = self.mask_cache_dir / f"{image_id}.npy"
        mask = np.load(mask_path)

        size = self.patch_size

        image = image[:, y:y + size, x:x + size]
        mask = mask[y:y + size, x:x + size]

        # Convert image from (C, H, W) to (H, W, C)
        image = np.transpose(image, (1, 2, 0))

        # Convert uint16 imagery to float32 and normalize.
        image = image.astype(np.float32)
        image = (image - self.mean) / self.std

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask,
            )
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert image back to PyTorch (C, H, W).
        image = torch.from_numpy(
            np.transpose(image, (2, 0, 1))
        ).float()

        # Binary mask: (H, W) -> (1, H, W)
        mask = torch.from_numpy(
            mask.astype(np.float32)
        ).unsqueeze(0)

        return image, mask
