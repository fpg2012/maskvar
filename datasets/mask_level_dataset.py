from json.encoder import py_encode_basestring
import torch
from torch.utils.data import IterableDataset
import torch.distributed as dist
import torch.nn.functional as F

from typing import Optional, Iterator, Tuple

from .coco_lvis import LvisDataset
from .hqseg44k import HQSeg44KTrainDataset
from utils import resize_longest_side
from models.sam_image_encoder import ImageEncoderViT as SamImageEncoder
from tqdm import tqdm

def count_masks(dataset: LvisDataset | HQSeg44KTrainDataset, world_size=1, rank=0):
    count = 0
    if rank == 0:
        iters = tqdm(range(rank, len(dataset), world_size))
    else:
        iters = range(rank, len(dataset), world_size)
    # for i in iters:
    #     _, _, instance_infos = dataset[i]
    #     count += len(instance_infos)
    count = sum(len(dataset[i][2]) for i in iters)
    return count

class MaskLevelDataset(IterableDataset):

    def __init__(self, dataset: Optional[LvisDataset | HQSeg44KTrainDataset], sam_encoder: SamImageEncoder, device: str):
        self.dataset = dataset
        self.sam_encoder = sam_encoder
        self.device = device

        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).to(device) # copied from sam
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).to(device) # copied from sam

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterate through the dataset

        Returns:
            image, image_embed_sam, single_mask_normalized, single_mask
        """
        if not dist.is_initialized():
            rank = 0
            world_size = 1
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        for i in range(len(self.dataset)):
            if i % world_size != rank:
                continue

            image, mask, instance_info = self.dataset[i]
            image, image_embed_sam = self.preprocess_image(image)
            for instance_idx in instance_info.keys():
                single_mask_normalized, single_mask = self.preprocess_mask(mask, instance_info, instance_idx)
                # print(f'image.dtype, single_mask_normalized.dtype, single_mask.dtype: {image.dtype, single_mask_normalized.dtype, single_mask.dtype}')
                yield image, image_embed_sam, single_mask_normalized, single_mask

    def preprocess_image(self, image):
        """
        preprocess image for image encoder

        image: (H, W, 3)
        """
        image = torch.from_numpy(image).to(self.device) / 255.0
        image = image.permute(2, 0, 1) # (H, W, 3) -> (3, H, W)
        image = resize_longest_side(image.unsqueeze(0), 1024).squeeze(0)

        # normalize image
        image = image.permute(1, 2, 0) # (3, H, W) -> (H, W, 3)
        image = (image - self.pixel_mean) / self.pixel_std
        image = image.permute(2, 0, 1) # (H, W, 3) -> (3, H, W)

        # pad image to 1024
        h, w = image.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        image = F.pad(image, (0, padw, 0, padh), value=0)

        # print(f'image shape: {image.shape}')

        # image_embed = self.image_encoder(image.unsqueeze(0)).squeeze(0)
        with torch.no_grad():
            image_embed_sam = self.sam_encoder(image.unsqueeze(0)).squeeze(0)

        return image, image_embed_sam

    def preprocess_mask(self, gt_mask, instance_info, instance_idx):
        mask = gt_mask[:, :, instance_info[instance_idx].mapping[0]] == instance_info[instance_idx].mapping[1]

        # to tensor
        mask = torch.from_numpy(mask).to(self.device, dtype=torch.float32).unsqueeze(0)

        mask = resize_longest_side(mask.unsqueeze(0), 256, 'nearest').squeeze(0)
        # mask = mask.long()

        # pad mask to 256
        h, w = mask.shape[-2:]
        padh = 256 - h
        padw = 256 - w
        mask = F.pad(mask, (0, padw, 0, padh), value=0)

        # normalize mask
        mask_normalized = mask * 2 - 1

        return mask_normalized, mask