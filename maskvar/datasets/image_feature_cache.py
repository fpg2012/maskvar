from pathlib import Path
import json
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import Dataset

class ShardCacheLine:

    def __init__(self, shard_index: int, shard: np.ndarray):
        self.shard_index = shard_index
        self.shard = shard

class ImageFeatureCache(Dataset):

    def __init__(self, cache_dir: Path, dataset: str, model_name: str, device='cpu', original_batch_mode=False, max_cache_shard=1):
        self.cache_dir = cache_dir
        self.dataset = dataset
        self.model_name = model_name
        # self.device = device
        self.metadata = {
            "count": 0,
            "resolution": 1024,
            "feature_shape": (0, 0, 0, 0),
            "feature_dim": "BCHW",
            "feature_size_in_bytes": 0,
            "avg_encode_time": 0.0,
            "batch_size": 0,
            "dtype": "float32"
        } # example

        with open(self.cache_dir / self.model_name / f'{self.dataset}_metadata.json', 'r') as f:
            self.metadata = json.load(f)

        self.internal_batch_size = self.metadata.get("batch_size", 1)
        dtype_name = self.metadata.get("dtype", "float32").replace("torch.", "")
        self.internal_dtype = getattr(torch, dtype_name)
        self.source_index_to_cache_index = None
        source_indices = self.metadata.get("source_indices")
        if source_indices is not None:
            self.source_index_to_cache_index = {
                int(source_index): cache_index
                for cache_index, source_index in enumerate(source_indices)
            }

        self.original_batch_mode = original_batch_mode

        print("ImageFeatureCache loaded with metadata:\n", json.dumps(self.metadata, ensure_ascii=False, indent=2))

        self.current_shard = OrderedDict()
        self.max_cache_shard = max_cache_shard

    def __len__(self):
        if self.original_batch_mode:
            return self.metadata["count"]
        if "num_images" in self.metadata:
            return self.metadata["num_images"]
        return self.metadata["count"] * self.internal_batch_size

    def _get_shard(self, shard_index: int) -> np.ndarray:
        if shard_index not in self.current_shard.keys():
            if len(self.current_shard) >= self.max_cache_shard:
                self.current_shard.popitem(last=False)
            self.current_shard[shard_index] = np.load(self.cache_dir / self.model_name / f'{self.dataset}/batch_{shard_index:06d}.npy', mmap_mode='r')
            # update access count
        self.current_shard.move_to_end(shard_index)
        return self.current_shard[shard_index]

    def __getitem__(self, index) -> np.ndarray:
        """
        return CHW image feature if original_batch_mode is disabled
        otherwhise, return BCHW image feature
        """
        if self.original_batch_mode:
            image_feature = np.load(self.cache_dir / self.model_name / f'{self.dataset}/batch_{index:06d}.npy', mmap_mode='r')
            return image_feature

        if self.source_index_to_cache_index is not None:
            try:
                index = self.source_index_to_cache_index[int(index)]
            except KeyError as exc:
                raise KeyError(f"Image index {index} is not available in cache {self.cache_dir / self.model_name / self.dataset}") from exc

        batch_index = index // self.internal_batch_size
        item_index_in_batch = index % self.internal_batch_size
        image_feature = self._get_shard(batch_index)
        return image_feature[item_index_in_batch]
