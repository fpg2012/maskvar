from json.encoder import py_encode_basestring
import torch
from torch.utils.data import IterableDataset
import torch.distributed as dist
import torch.nn.functional as F

from typing import Optional, Iterator, Tuple

from .coco_lvis import LvisDataset
from .hqseg44k import HQSeg44KTrainDataset
from ..utils import resize_longest_side
from ..models.sam_image_encoder import ImageEncoderViT as SamImageEncoder
from tqdm import tqdm
import numpy as np

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

    def __init__(
        self, 
        dataset: Optional[LvisDataset | HQSeg44KTrainDataset], 
        sam_encoder: Optional[SamImageEncoder], 
        device: str, 
        with_image_embed=True,
        mask_filter_thresh=0.1
    ):
        self.dataset = dataset
        self.sam_encoder = sam_encoder
        self.device = device
        self.with_image_embed = with_image_embed
        self.mask_filter_thresh = mask_filter_thresh
        
        if self.with_image_embed:
            assert self.sam_encoder is not None

        # 使用register_buffer避免每次都创建新tensor
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device) # copied from sam
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device) # copied from sam

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
                if not self.filter_mask(single_mask, self.mask_filter_thresh):
                    continue
                yield image.detach(), image_embed_sam.detach() if isinstance(image_embed_sam, torch.Tensor) else image_embed_sam, single_mask_normalized.detach(), single_mask.detach()

    def preprocess_image(self, image):
        """
        preprocess image for image encoder

        image: (H, W, 3)
        """
        with torch.no_grad():  # 整个预处理过程都不需要梯度
            image = torch.from_numpy(image).to(self.device, dtype=torch.float32, non_blocking=True) / 255.0
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
            if self.with_image_embed:
                # 使用clone()创建新tensor，避免保留对encoder输出的引用
                image_embed_sam = self.sam_encoder(image.unsqueeze(0)).squeeze(0).clone()
            else:
                image_embed_sam = 0
        return image, image_embed_sam

    def preprocess_mask(self, gt_mask, instance_info, instance_idx):
        with torch.no_grad():  # mask预处理不需要梯度
            mask = gt_mask[:, :, instance_info[instance_idx].mapping[0]] == instance_info[instance_idx].mapping[1]

            # to tensor
            mask = torch.from_numpy(mask).to(self.device, dtype=torch.float32, non_blocking=True).unsqueeze(0)

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
    
    def filter_mask(self, mask, thresh=0.1):
        """
        Drop mask if it is too small

        mask: (1, H, W) torch.tensor
        """
        _, H, W = mask.shape
        # count number of pixels > 0 in mask
        num_pixels = torch.sum(mask > 0)
        if num_pixels / (H * W) < thresh:
            return False
        return True

class MaskLevelDatasetRandom(MaskLevelDataset):

    def __init__(
        self, 
        dataset: Optional[LvisDataset | HQSeg44KTrainDataset], 
        sam_encoder: Optional[SamImageEncoder], 
        device: str, 
        with_image_embed=True,
        mask_filter_thresh=0.1,
        seed=42,
        infinite=False,
    ):
        super().__init__(dataset, sam_encoder, device, with_image_embed, mask_filter_thresh)
        self.rng = np.random.default_rng(seed)
        self.infinite = infinite
    
    def __sample_image(self):
        # sample a data point from dataset
        index = self.rng.integers(0, len(self.dataset) - 1)
        image, mask, instance_info = self.dataset[index]
        image, image_embed_sam = self.preprocess_image(image)
        for instance_idx in instance_info.keys():
            single_mask_normalized, single_mask = self.preprocess_mask(mask, instance_info, instance_idx)
            if not self.filter_mask(single_mask, self.mask_filter_thresh):
                continue
            yield image.detach(), image_embed_sam.detach() if isinstance(image_embed_sam, torch.Tensor) else image_embed_sam, single_mask_normalized.detach(), single_mask.detach()

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterate through the dataset

        Returns:
            image, image_embed_sam, single_mask_normalized, single_mask
        """

        yield from self.__sample_image()

        while self.infinite:
            yield from self.__sample_image()