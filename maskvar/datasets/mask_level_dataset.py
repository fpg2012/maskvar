import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from itertools import islice
from pathlib import Path

from typing import Optional, Iterator, Tuple

from .coco_lvis import LvisDataset
from .hqseg44k import HQSeg44KTrainDataset
from .image_feature_cache import ImageFeatureCache
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
        device: str, 
        with_image_embed=True,
        mask_filter_thresh=0.1,
        image_size_encoder=1024,
        image_size_mask=256,
        dtype=torch.float32,
        image_feature_cache: Optional[ImageFeatureCache] = None
    ):
        self.dataset = dataset
        self.device = device
        self.dtype=dtype
        self.with_image_embed = with_image_embed
        self.mask_filter_thresh = mask_filter_thresh

        self.image_size_encoder = image_size_encoder
        self.image_size_mask = image_size_mask
        
        if self.with_image_embed:
            assert (image_feature_cache is not None)

        # 使用register_buffer避免每次都创建新tensor
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53], ) # copied from sam
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375], ) # copied from sam

        self.image_feature_cache = image_feature_cache

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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
            image, image_embed_sam = self.preprocess_image(image, index=i)
            for instance_idx in instance_info.keys():
                single_mask_normalized, single_mask = self.preprocess_mask(mask, instance_info, instance_idx)
                if not self.filter_mask(single_mask, self.mask_filter_thresh):
                    continue
                yield image, image_embed_sam, single_mask_normalized, single_mask

    @torch.no_grad()
    def preprocess_image(self, image, index=None):
        """
        preprocess image for image encoder

        image: (H, W, 3)
        """
        # !MUST NOT DIVIDE 255 HERE
        image = torch.from_numpy(image).to(dtype=self.dtype, non_blocking=True)
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
            if self.image_feature_cache is not None:
                assert index is not None
                image_embed_sam = torch.from_numpy(self.image_feature_cache[index])
            else:
                raise NotImplementedError("sam embedding in dataset is now disabled! Please use image feature cache instead.")
                # with torch.autocast(self.device, dtype=self.dtype):
                #     image_embed_sam = self.sam_encoder(image.unsqueeze(0)).squeeze(0).clone()
                # image_embed_sam = image_embed_sam.to('cpu')
        else:
            image_embed_sam = None
        return image.detach(), image_embed_sam

    @torch.no_grad()
    def preprocess_mask(self, gt_mask, instance_info, instance_idx):
        mask = gt_mask[:, :, instance_info[instance_idx].mapping[0]] == instance_info[instance_idx].mapping[1]

        # to tensor
        mask = torch.from_numpy(mask).to(dtype=self.dtype, non_blocking=True).unsqueeze(0)

        mask = resize_longest_side(mask.unsqueeze(0), self.image_size_mask, 'nearest').squeeze(0)
        # mask = mask.long()

        # pad mask to image_size_mask (default 256)
        h, w = mask.shape[-2:]
        padh = self.image_size_mask - h
        padw = self.image_size_mask - w
        mask = F.pad(mask, (0, padw, 0, padh), value=0)

        # normalize mask
        mask_normalized = mask * 2 - 1

        return mask_normalized.detach(), mask.detach()
    
    def filter_mask(self, mask, thresh=0.1):
        """
        Drop mask if it is too small. Return False if dropped.

        mask: (1, H, W) torch.tensor
        """
        _, H, W = mask.shape
        # count number of pixels > 0 in mask
        num_pixels = torch.sum(mask > 0)
        if num_pixels / (H * W) < thresh:
            return False
        return True

class MaskLevelDatasetDummy(MaskLevelDataset):

    def __init__(
        self, 
        dataset: Optional[LvisDataset | HQSeg44KTrainDataset], 
        device: str, 
        with_image_embed=True,
        mask_filter_thresh=0.1,
        seed=42,
        image_size_encoder=1024,
        image_size_mask=256,
        count=1,
        dtype=torch.float32,
        image_feature_cache: Optional[ImageFeatureCache] = None
    ):
        super().__init__(
            dataset=dataset,
            device=device,
            with_image_embed=with_image_embed,
            mask_filter_thresh=mask_filter_thresh,
            image_size_encoder=image_size_encoder,
            image_size_mask=image_size_mask,
            dtype=dtype,
            image_feature_cache=image_feature_cache
        )
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.count = count

        self.results = []

        _count = 0

        for _ in range(len(self.dataset)):
            # sample a data point from dataset
            index = self.rng.integers(0, len(self.dataset))
            image, mask, instance_info = self.dataset[index]
            image, image_embed_sam = self.preprocess_image(image, index=index)

            for instance_idx in instance_info.keys():  # Take first count instances only
                single_mask_normalized, single_mask = self.preprocess_mask(mask, instance_info, instance_idx)
                if not self.filter_mask(single_mask, self.mask_filter_thresh):
                    continue
                result = (image, image_embed_sam, single_mask_normalized, single_mask)
                self.results.append(result)

                _count += 1
                if _count >= self.count:
                    break
            
            if _count >= self.count:
                break
        
        # print(len(self.results))

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterate through the dataset

        Returns:
            image, image_embed_sam, single_mask_normalized, single_mask
        """
        while True:
            for result in self.results:
                yield result

class MaskLevelDatasetRandom(MaskLevelDataset):

    def __init__(
        self, 
        dataset: Optional[LvisDataset | HQSeg44KTrainDataset], 
        device: str, 
        with_image_embed=True,
        mask_filter_thresh=0.1,
        seed=42,
        num_masks=-1, # -1表示无限制
        infinite=False,
        image_size_encoder=1024,
        image_size_mask=256,
        shuffle=True, # 是否每次顺序都相同
        dtype=torch.float32,
        image_feature_cache: Optional[ImageFeatureCache] = None
    ):
        super().__init__(
            dataset=dataset,
            device=device,
            with_image_embed=with_image_embed,
            mask_filter_thresh=mask_filter_thresh,
            image_size_encoder=image_size_encoder,
            image_size_mask=image_size_mask,
            dtype=dtype,
            image_feature_cache=image_feature_cache
        )
        self.rng = np.random.default_rng(seed)
        self.num_masks = num_masks
        self.infinite = infinite
        self.seed = seed
        self.counter = 0
        self.shuffle = shuffle

    def __reset_rng_and_counter(self):
        self.rng = np.random.default_rng(self.seed)
        self.counter = 0
    
    def __sample_image(self):
        if not self.shuffle:
            self.__reset_rng_and_counter()

        # sample a data point from dataset
        index_range = (0, len(self.dataset))
        worker_info = get_worker_info()
        if worker_info is not None:
            shard_size = len(self.dataset) // worker_info.num_workers
            index_range = (worker_info.id * shard_size, (worker_info.id + 1) * shard_size)

        index = self.rng.integers(index_range[0], index_range[1])
        
        image, mask, instance_info = self.dataset[index]
        image, image_embed_sam = self.preprocess_image(image, index=index)

        for instance_idx in instance_info.keys():
            if self.num_masks > 0 and self.counter >= self.num_masks:
                break

            single_mask_normalized, single_mask = self.preprocess_mask(mask, instance_info, instance_idx)
            if not self.filter_mask(single_mask, self.mask_filter_thresh):
                continue
            yield image, image_embed_sam, single_mask_normalized, single_mask
            
            self.counter += 1

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterate through the dataset

        Returns:
            image, image_embed_sam, single_mask_normalized, single_mask
        """

        yield from self.__sample_image()

        while self.infinite:
            yield from self.__sample_image()


class MaskLevelFlatDataset(Dataset):

    def __init__(
        self, 
        index_mapping_path: Path,
        dataset: Optional[LvisDataset | HQSeg44KTrainDataset], 
        with_image_embed=True,
        mask_filter_thresh=0.1,
        dtype=torch.float32,
        image_size_encoder=1024,
        image_size_mask=256,
        image_feature_cache: Optional[ImageFeatureCache] = None,
        skip_preprocess=False,
    ):
        self.index_mapping_path = index_mapping_path
        self.dataset = dataset
        self.dtype=dtype
        self.with_image_embed = with_image_embed
        self.mask_filter_thresh = mask_filter_thresh
        self.skip_preprocess=skip_preprocess
        
        if self.with_image_embed:
            assert (image_feature_cache is not None)

        # 使用register_buffer避免每次都创建新tensor
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53], ) # copied from sam
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375], ) # copied from sam

        self.image_size_encoder = image_size_encoder
        self.image_size_mask = image_size_mask
        self.image_feature_cache = image_feature_cache

        # load index mapping (a array)
        self.index_mapping = np.load(index_mapping_path)
        ## check index mapping shape (should be (N,2))
        assert self.index_mapping.ndim == 2 and self.index_mapping.shape[1] == 2
    
    def __len__(self):
        return self.index_mapping.shape[0]

    def __getitem__(self, index):
        image_index = self.index_mapping[index][0]
        mask_index = self.index_mapping[index][1]

        image, mask, instance_info = self.dataset[image_index]
        if self.skip_preprocess:
            image = torch.from_numpy(image.astype(np.float32))
        else:
            image = self.preprocess_image(image, index=image_index)
        
        if self.with_image_embed:
            image_embed_sam = torch.from_numpy(self.image_feature_cache[image_index])
        else:
            image_embed_sam = None

        single_mask_normalized, single_mask = self.preprocess_mask(mask, instance_info, mask_index)
        return image, image_embed_sam, single_mask_normalized, single_mask

    @torch.no_grad()
    def preprocess_image(self, image, index=None):
        """
        preprocess image for image encoder

        image: (H, W, 3)
        """
        # !MUST NOT DIVIDE 255 HERE
        image = torch.from_numpy(image.astype(np.float32))
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
        return image.detach()

    @torch.no_grad()
    def preprocess_mask(self, gt_mask, instance_info, instance_idx):
        mask = gt_mask[:, :, instance_info[instance_idx].mapping[0]] == instance_info[instance_idx].mapping[1]

        # to tensor
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        mask = resize_longest_side(mask.unsqueeze(0), self.image_size_mask, 'nearest').squeeze(0)
        # mask = mask.long()

        # pad mask to image_size_mask (default 256)
        h, w = mask.shape[-2:]
        padh = self.image_size_mask - h
        padw = self.image_size_mask - w
        mask = F.pad(mask, (0, padw, 0, padh), value=0)

        # normalize mask
        mask_normalized = mask * 2 - 1

        return mask_normalized.detach(), mask.detach()
    
    def filter_mask(self, mask, thresh=0.1):
        """
        Drop mask if it is too small. Return False if dropped.

        mask: (1, H, W) torch.tensor
        """
        _, H, W = mask.shape
        # count number of pixels > 0 in mask
        num_pixels = torch.sum(mask > 0)
        if num_pixels / (H * W) < thresh:
            return False
        return True