import torch
import numpy as np
from typing import Tuple, Dict

from .instance_info import InstanceInfo


class MySegDataset(torch.utils.data.Dataset):
    """
    Base class for all segmentation datasets in the project.

    All subclasses should return data in the format:
    - image: (H, W, 3) numpy array (RGB)
    - layers: (H, W, L) numpy array with instance masks
    - instances_info: dict[int, InstanceInfo]

    Subclasses should define:
    - DEFAULT_NUM_MASKS_SPLITS: for distributed training
    - num_masks_splits: instance attribute with actual splits
    """

    # Default mask splits for distributed training
    # Subclasses should override this
    DEFAULT_NUM_MASKS_SPLITS = [100000, 100000, 100000, 100000]

    def count_masks(self, world_size: int = 4, rank: int = 0) -> int:
        """
        Count masks for distributed training.

        Args:
            world_size: Number of distributed processes
            rank: Current process rank

        Returns:
            Number of masks for this rank
        """
        assert world_size in [1, 2, 4], f"Unsupported world_size: {world_size}"

        if not hasattr(self, 'num_masks_splits'):
            self.num_masks_splits = self.DEFAULT_NUM_MASKS_SPLITS

        if world_size == 4:
            return self.num_masks_splits[rank]
        elif world_size == 2:
            return self.num_masks_splits[rank] + self.num_masks_splits[rank + 2]
        else:
            return sum(self.num_masks_splits)

    def max_count_masks(self, world_size: int = 4) -> int:
        """
        Get max mask count across distributed ranks.

        Args:
            world_size: Number of distributed processes

        Returns:
            Maximum number of masks any rank will handle
        """
        assert world_size in [1, 2, 4], f"Unsupported world_size: {world_size}"

        if not hasattr(self, 'num_masks_splits'):
            self.num_masks_splits = self.DEFAULT_NUM_MASKS_SPLITS

        if world_size == 4:
            return min(self.num_masks_splits)
        elif world_size == 2:
            return min(
                self.num_masks_splits[0] + self.num_masks_splits[2],
                self.num_masks_splits[1] + self.num_masks_splits[3]
            )
        else:
            return sum(self.num_masks_splits)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        raise NotImplementedError("Subclasses must implement __len__")

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, Dict[int, InstanceInfo]]:
        """
        Get a sample from the dataset.

        Args:
            index: Sample index

        Returns:
            tuple: (image, layers, instances_info) where:
                - image: (H, W, 3) numpy array (RGB)
                - layers: (H, W, L) numpy array with instance masks
                - instances_info: dict mapping instance IDs to InstanceInfo
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
