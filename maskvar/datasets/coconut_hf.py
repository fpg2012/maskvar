import io
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Union, List, Dict, Any
from PIL import Image
import requests
import cv2

from .instance_info import InstanceInfo
from .my_seg_dataset import MySegDataset


class CoconutHFDataset(MySegDataset):
    """
    COCONut dataset loader from HuggingFace parquet files.

    This dataset loads COCONut from local parquet files (downloaded from HF),
    and converts the panoptic format on-the-fly to match the project's format.

    Parquet format (from xdeng77/coconut_s):
    - mask: PNG bytes (panoptic mask with instance IDs)
    - segments_info: dict with segments_info list (category_id, id, bbox, area)
    - image_info: dict with COCO image metadata (coco_url, file_name, height, width)

    Args:
        parquet_path: Path to parquet file or directory containing parquet files
        image_root: Root directory containing COCO images (train2017/val2017)
        stuff_prob: Probability of keeping background (stuff) objects
        transform: Optional transform to apply
        cache_images: Whether to cache decoded images in memory
    """

    DEFAULT_NUM_MASKS_SPLITS = [450000, 445000, 442000, 455000]

    def __init__(
        self,
        parquet_path: Union[str, Path],
        image_root: Union[str, Path],
        stuff_prob: float = 1.0,
        transform: Optional[Callable] = None,
        cache_images: bool = False,
    ):
        super().__init__()
        self.parquet_path = Path(parquet_path)
        self.image_root = Path(image_root)
        self.stuff_prob = stuff_prob
        self.transform = transform
        self.cache_images = cache_images
        self._image_cache = {} if cache_images else None

        # Load parquet data
        self.df = self._load_parquet(self.parquet_path)
        self.dataset_samples = list(range(len(self.df)))

        print(f"Loaded {len(self)} samples from {parquet_path}")

    def _load_parquet(self, parquet_path: Path):
        """Load parquet file(s)."""
        import pandas as pd

        if parquet_path.is_file():
            return pd.read_parquet(parquet_path)
        elif parquet_path.is_dir():
            files = sorted(parquet_path.glob("*.parquet"))
            if not files:
                raise ValueError(f"No parquet files found in {parquet_path}")
            dfs = [pd.read_parquet(f) for f in files]
            import pandas as pd
            return pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError(f"Invalid parquet path: {parquet_path}")

    def _is_thing(self, segment: dict) -> bool:
        """
        Check if segment is a thing (foreground) or stuff (background).

        COCONut parquet includes 'isthing' field in segments_info:
        - isthing=1: thing (countable objects like person, car)
        - isthing=0: stuff (amorphous regions like sky, grass)
        """
        return segment.get('isthing', 1) == 1

    def count_masks(self, world_size: int = 4, rank: int = 0) -> int:
        """Count masks for distributed training."""
        total = sum(len(self._get_segments_info(i)) for i in range(len(self)))
        return total // world_size

    def max_count_masks(self, world_size: int = 4) -> int:
        """Get max mask count across distributed ranks."""
        return self.count_masks(world_size, 0)

    def __len__(self) -> int:
        return len(self.df)

    def _get_segments_info(self, idx: int) -> List[Dict[str, Any]]:
        """Get segments info for a sample."""
        row = self.df.iloc[idx]
        segments = row['segments_info']

        if isinstance(segments, dict):
            return segments.get('segments_info', [])
        elif isinstance(segments, list):
            return segments
        elif isinstance(segments, np.ndarray):
            return segments.tolist()
        return []

    def _load_image(self, idx: int) -> np.ndarray:
        """Load image from local path."""
        if self.cache_images and idx in self._image_cache:
            return self._image_cache[idx]

        row = self.df.iloc[idx]
        image_info = row['image_info']

        if isinstance(image_info, dict):
            file_name = image_info.get('file_name', '')
            # Extract image ID from file_name (e.g., "000000000009.jpg" -> "000000000009")
            image_id = Path(file_name).stem
        else:
            image_id = str(idx).zfill(12)

        # Try to find image in image_root
        # COCONut uses COCO train2017/val2017 images
        for subdir in ['train2017', 'val2017', '']:
            img_path = self.image_root / subdir / f"{image_id}.jpg"
            if img_path.exists():
                break
        else:
            raise FileNotFoundError(f"Image not found for ID {image_id} in {self.image_root}")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.cache_images:
            self._image_cache[idx] = image

        return image

    def _load_mask(self, idx: int) -> np.ndarray:
        """
        Load and decode panoptic mask from PNG bytes.

        COCONut uses RGB encoding for panoptic masks:
        - For png format: R + G * 256 = segment_id (16-bit instance ID)
        - B channel is typically 0 or used for additional info
        """
        row = self.df.iloc[idx]
        mask_data = row['mask']

        if isinstance(mask_data, dict) and 'bytes' in mask_data:
            png_bytes = mask_data['bytes']
        elif isinstance(mask_data, bytes):
            png_bytes = mask_data
        else:
            raise ValueError(f"Unknown mask format: {type(mask_data)}")

        # Decode PNG
        mask = Image.open(io.BytesIO(png_bytes))
        mask_array = np.array(mask)

        # Handle RGB encoding (R + G * 256)
        if len(mask_array.shape) == 3 and mask_array.shape[2] >= 2:
            # RGB image: combine R and G channels for 16-bit segment ID
            segment_ids = mask_array[:, :, 0].astype(np.uint16) + mask_array[:, :, 1].astype(np.uint16) * 256
            return segment_ids
        elif len(mask_array.shape) == 3:
            # Single channel or other format - take first channel
            return mask_array[:, :, 0]
        else:
            # Already single channel
            return mask_array

    def _split_panoptic_mask(self, panoptic_mask: np.ndarray, segments_info: List[Dict]) -> tuple:
        """
        Split panoptic mask into individual instance masks.

        Returns:
            layers: (H, W, L) array with each layer containing one instance
            objs_mapping: dict mapping instance_id -> (layer_idx, mask_id)
            is_thing: list indicating if each layer is a thing (foreground)
        """
        H, W = panoptic_mask.shape

        # Create mapping from segment id to segment info
        segment_map = {}
        for seg in segments_info:
            seg_id = seg['id']
            segment_map[seg_id] = seg

        # Get unique instance IDs (excluding 0 which is background)
        unique_ids = np.unique(panoptic_mask)
        unique_ids = unique_ids[unique_ids != 0]

        if len(unique_ids) == 0:
            # No instances
            return np.zeros((H, W, 1), dtype=np.uint8), {0: (0, 0)}, [False]

        # Create layers
        layers = []
        objs_mapping = {}
        is_thing = []

        for i, inst_id in enumerate(unique_ids):
            # Extract binary mask for this instance
            binary_mask = (panoptic_mask == inst_id).astype(np.uint8)

            # Check if it's a thing (using isthing field from COCONut)
            seg_info = segment_map.get(int(inst_id), {})
            is_thing_flag = self._is_thing(seg_info)
            is_thing.append(is_thing_flag)

            # Assign mask ID (i+1, reserving 0 for background)
            mask_id = i + 1
            layers.append(binary_mask * mask_id)
            objs_mapping[i] = (i, mask_id)

        layers = np.stack(layers, axis=2) if layers else np.zeros((H, W, 1), dtype=np.uint8)
        return layers, objs_mapping, is_thing

    def __getitem__(self, index: int):
        """
        Get a sample and convert to project format.

        Returns:
            image: (H, W, 3) numpy array (RGB)
            layers: (H, W, L) numpy array with instance masks
            instances_info: dict[int, InstanceInfo]
        """
        # Load image and panoptic mask
        image = self._load_image(index)
        panoptic_mask = self._load_mask(index)
        segments_info = self._get_segments_info(index)

        # Split into layers
        layers, objs_mapping, is_thing = self._split_panoptic_mask(panoptic_mask, segments_info)

        # Build instance info
        instances_info = {}
        num_instance_masks = 0

        for i, is_thing_flag in enumerate(is_thing):
            instances_info[i] = InstanceInfo(
                mapping=objs_mapping[i],
                node_level=0
            )
            if is_thing_flag:
                num_instance_masks += 1

        # Handle stuff (background) based on probability
        if self.stuff_prob > 0:
            # Keep stuff with given probability
            for i, is_thing_flag in enumerate(is_thing):
                if not is_thing_flag and np.random.random() >= self.stuff_prob:
                    # Remove this stuff layer
                    layer_idx, mask_id = objs_mapping[i]
                    layers[:, :, layer_idx] = 0
                    del instances_info[i]
        else:
            # Remove all stuff by default (keep only things)
            for i, is_thing_flag in enumerate(is_thing):
                if not is_thing_flag and i in instances_info:
                    layer_idx, mask_id = objs_mapping[i]
                    layers[:, :, layer_idx][layers[:, :, layer_idx] == mask_id] = 0
                    del instances_info[i]

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            layers = self.transform(layers)

        return image, layers, instances_info


class CoconutHFConverter:
    """
    Convert HuggingFace parquet format to project's pickle format.

    This is useful for pre-converting the dataset for faster training.
    """

    def __init__(
        self,
        parquet_path: Union[str, Path],
        image_root: Union[str, Path],
        output_dir: Union[str, Path],
    ):
        self.dataset = CoconutHFDataset(parquet_path, image_root)
        self.output_dir = Path(output_dir)

    def convert(self):
        """Convert all samples to pickle format."""
        import pickle

        output_images = self.output_dir / 'images'
        output_masks = self.output_dir / 'masks'
        output_images.mkdir(parents=True, exist_ok=True)
        output_masks.mkdir(parents=True, exist_ok=True)

        hannotation = {}

        for idx in range(len(self.dataset)):
            image, layers, instances_info = self.dataset[idx]

            # Get image ID from parquet
            row = self.dataset.df.iloc[idx]
            image_info = row['image_info']
            if isinstance(image_info, dict):
                file_name = image_info.get('file_name', '')
                image_id = Path(file_name).stem
            else:
                image_id = str(idx).zfill(12)

            # Save image
            img_path = output_images / f'{image_id}.jpg'
            cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Encode and save masks
            encoded_layers = []
            for i in range(layers.shape[2]):
                layer = layers[:, :, i]
                _, encoded = cv2.imencode('.png', layer)
                encoded_layers.append(encoded)

            # Build objs_mapping
            objs_mapping = {}
            for inst_id, info in instances_info.items():
                objs_mapping[inst_id] = info.mapping

            mask_path = output_masks / f'{image_id}.pickle'
            with open(mask_path, 'wb') as f:
                pickle.dump((encoded_layers, objs_mapping), f)

            # Build hierarchy
            hierarchy = {}
            for inst_id, info in instances_info.items():
                hierarchy[inst_id] = {
                    'parent': info.parent,
                    'children': info.children,
                    'node_level': info.node_level
                } if info.parent is not None else None

            hannotation[image_id] = {
                'hierarchy': hierarchy,
                'num_instance_masks': len(instances_info)
            }

            if (idx + 1) % 1000 == 0:
                print(f"Converted {idx + 1}/{len(self.dataset)} samples")

        # Save hannotation
        with open(self.output_dir / 'hannotation.pickle', 'wb') as f:
            pickle.dump(hannotation, f)

        print(f"Conversion complete! Saved to {self.output_dir}")
