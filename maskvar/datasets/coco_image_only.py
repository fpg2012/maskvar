import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Callable
import torch.utils.data


class CocoImageOnlyDataset(torch.utils.data.Dataset):
    """
    COCO image-only dataset for generating image feature cache.

    Loads COCO images without annotations, sorted by image_id.
    This serves as the "universe" dataset - cocolvis and coconut are subsets.

    Args:
        image_root: Path to COCO images (e.g., data/coco/train2017)
        transform: Optional transform to apply
    """

    DEFAULT_NUM_MASKS_SPLITS = [118287, 118287, 118287, 118287]

    def __init__(
        self,
        image_root: Union[str, Path],
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.image_root = Path(image_root)
        self.transform = transform

        # Load all images and sort by image_id
        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root not found: {self.image_root}")
        self.image_files = sorted(self.image_root.glob("*.jpg"))
        self.image_ids = [int(f.stem) for f in self.image_files]

        # Create mapping: image_id -> index
        self.image_id_to_index = {img_id: idx for idx, img_id in enumerate(self.image_ids)}

        print(f"Loaded {len(self)} COCO images from {image_root}")
        print(f"Image ID range: {min(self.image_ids)} - {max(self.image_ids)}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        """
        Returns:
            image: numpy.ndarray (H, W, 3) RGB format
            None: placeholder for layers (no annotations)
            None: placeholder for instances_info (no annotations)
        """
        image_path = self.image_files[index]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, None, None

    def get_index_by_image_id(self, image_id: int) -> int:
        """Get dataset index by image_id."""
        return self.image_id_to_index[image_id]

    def get_image_id_by_index(self, index: int) -> int:
        """Get image_id by dataset index."""
        return self.image_ids[index]

    def count_masks(self, world_size: int = 4, rank: int = 0) -> int:
        """For image-only dataset, return image count instead of mask count."""
        total = len(self)
        per_rank = total // world_size
        return per_rank

    def max_count_masks(self, world_size: int = 4) -> int:
        """For image-only dataset, return image count per rank."""
        return self.count_masks(world_size, 0)
