import torch
from torch.utils.data import IterableDataset
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from typing import Optional, Iterator, Tuple

from .coco_lvis import LvisDataset
from .hqseg44k import HQSeg44KTrainDataset
from ..utils import resize_longest_side
from ..models.sam import ImageEncoderViT as SamImageEncoder

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
        mask_filter_thresh=0.1,
        image_size_encoder=1024,
        image_size_mask=256,
    ):
        self.dataset = dataset
        self.sam_encoder = sam_encoder
        self.device = device
        self.with_image_embed = with_image_embed
        self.mask_filter_thresh = mask_filter_thresh

        self.image_size_encoder = image_size_encoder
        self.image_size_mask = image_size_mask
        
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
            # !MUST NOT DIVIDE 255 HERE
            image = torch.from_numpy(image).to(self.device, dtype=torch.float32, non_blocking=True)
            image = image.permute(2, 0, 1) # (H, W, 3) -> (3, H, W)
            image = resize_longest_side(image.unsqueeze(0), self.image_size_encoder).squeeze(0)

            # normalize image
            image = (image - self.pixel_mean.view(-1, 1, 1)) / self.pixel_std.view(-1, 1, 1)

            # pad image to image_size_encoder (default 1024)
            h, w = image.shape[-2:]
            padh = self.image_size_encoder - h
            padw = self.image_size_encoder - w
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

            mask = resize_longest_side(mask.unsqueeze(0), self.image_size_mask, 'nearest').squeeze(0)
            # mask = mask.long()

            # pad mask to image_size_mask (default 256)
            h, w = mask.shape[-2:]
            padh = self.image_size_mask - h
            padw = self.image_size_mask - w
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
        image_size_encoder=1024,
        image_size_mask=256
    ):
        super().__init__(dataset, sam_encoder, device, with_image_embed, mask_filter_thresh, image_size_encoder, image_size_mask)
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