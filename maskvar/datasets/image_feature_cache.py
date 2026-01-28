from pathlib import Path
import json

import torch
from torch.utils.data import Dataset

class ImageFeatureCache(Dataset):

    def __init__(self, cache_dir: Path, dataset: str, model_name: str, device='cuda', original_batch_mode=False):
        self.cache_dir = cache_dir
        self.dataset = dataset
        self.model_name = model_name
        self.device = device
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
        self.internal_dtype = getattr(torch, self.metadata.get("dtype", "float32"))

        self.original_batch_mode = original_batch_mode

        print("ImageFeatureCache loaded with metadata:\n", json.dumps(self.metadata, ensure_ascii=False, indent=2))

    def __len__(self):
        if self.original_batch_mode:
            return self.metadata["count"]
        return self.metadata["count"] * self.internal_batch_size

    def __getitem__(self, index) -> torch.Tensor:
        """
        return CHW image feature if original_batch_mode is disabled
        otherwhise, return BCHW image feature
        """
        if self.original_batch_mode:
            image_feature = torch.load(self.cache_dir / self.model_name / f'{self.dataset}/batch_{index:06d}.pt', map_location=self.device)
            return image_feature
        batch_index = index // self.internal_batch_size
        item_index_in_batch = index % self.internal_batch_size
        image_feature = torch.load(self.cache_dir / self.model_name / f'{self.dataset}/batch_{batch_index:06d}.pt', map_location=self.device)
        return image_feature[item_index_in_batch]