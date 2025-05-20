import json
import torch
from torchvision import transforms
import numpy as np
from pathlib import Path
import pickle
import cv2
from copy import deepcopy

from datasets.instance_info import InstanceInfo

class LvisDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split='train', img_split='train2017', stuff_prob=0.0,
                 allow_list_name=None, anno_file='hannotation.pickle', transform=None):
        """
        Initialize the LVIS dataset loader.
        
        Args:
            dataset_path (str): Path to the dataset directory
            split (str): Dataset split (e.g., 'train', 'val')
            img_split (str): Image set split (e.g., 'train2017')
            stuff_prob (float): Probability of keeping background objects (0.0-1.0)
            allow_list_name (str): Optional file containing allowed image IDs
            anno_file (str): Annotation file name
            transform: Optional transform to be applied to images and masks
        """
        super(LvisDataset, self).__init__()
        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self.img_split_path = self.dataset_path / img_split
        self.split = split
        self.img_split = img_split
        self._images_path = self._split_path / 'images'
        self._masks_path = self._split_path / 'masks'
        self.stuff_prob = stuff_prob
        self.transform = transform

        with open(self._split_path / anno_file, 'rb') as f:
            self.dataset_samples = sorted(pickle.load(f).items())

        if allow_list_name is not None:
            allow_list_path = self._split_path / allow_list_name
            with open(allow_list_path, 'r') as f:
                allow_images_ids = json.load(f)
            allow_images_ids = set(allow_images_ids)

            self.dataset_samples = [sample for sample in self.dataset_samples
                                   if sample[0] in allow_images_ids]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataset_samples)

    def __getitem__(self, index):
        """
        Loads and processes a single sample from the dataset.
        
        Args:
            index (int): Index of the sample to load
            
        Returns:
            tuple: (image, layers, instances_info) where:
                - image: numpy.ndarray or torch.Tensor (if transformed)
                    The loaded image in RGB format (H, W, 3)
                - layers: numpy.ndarray or torch.Tensor (if transformed)
                    Stack of instance masks (H, W, L) where L is number of layers
                - instances_info: dict[int, DatasetInstanceInfo]
                    Dictionary mapping instance IDs to their hierarchical information
        """
        # Get sample metadata
        image_id, sample = self.dataset_samples[index]
        
        # Load and convert image to RGB
        image_path = self._images_path / f'{image_id}.jpg'
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load and decode packed masks
        packed_masks_path = self._masks_path / f'{image_id}.pickle'
        with open(packed_masks_path, 'rb') as f:
            encoded_layers, objs_mapping = pickle.load(f)
        layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]
        layers = np.stack(layers, axis=2)

        # Process instance hierarchy information
        instances_info = {}
        for inst_id, inst_info in deepcopy(sample['hierarchy']).items():
            if inst_info is None:
                instances_info[inst_id] = InstanceInfo(
                    mapping=objs_mapping[inst_id],
                    node_level=0
                )
            else:
                instances_info[inst_id] = InstanceInfo(
                    mapping=objs_mapping[inst_id],
                    parent=inst_info.get('parent'),
                    children=inst_info.get('children', []),
                    node_level=inst_info.get('node_level', 0)
                )

        # Handle background objects (stuff) based on probability
        if self.stuff_prob > 0 and np.random.random() < self.stuff_prob:
            # Keep background objects
            for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
                instances_info[inst_id] = InstanceInfo(
                    mapping=objs_mapping[inst_id]
                )
        else:
            # Remove background objects by zeroing their masks
            for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
                layer_indx, mask_id = objs_mapping[inst_id]
                layers[:, :, layer_indx][layers[:, :, layer_indx] == mask_id] = 0

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
            layers = self.transform(layers)

        return image, layers, instances_info
