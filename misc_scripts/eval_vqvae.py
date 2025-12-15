# Standard library
import argparse
import json
import os
from typing import List

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Local application
from maskvar.datasets.coco_lvis import LvisDataset
from maskvar.datasets.hqseg44k import HQSeg44KTestDataset, HQSeg44KTrainDataset
from maskvar.datasets.mask_level_dataset import MaskLevelDataset
from maskvar.maskseg_build_everything import (
    build_vqvae_single,
    build_vqvae_single_4_stages,
    build_vqvae_single_4_stages_4_slices,
    build_vqvae_single_4_stages_4_slices_v2,
    build_vqvae_single_4_stages_v2,
    build_vqvae_single_5_stages_v1,
    build_vqvae_single_fewer_stages,
)
from maskvar.models.vqvae_single import VQVAE_Single
from maskvar.utils import divide_image, merge_image
from maskvar.utils.metrics import calc_iou

class BadCase:

    def __init__(self, batch_ind, ind, gt_normalized, recons_normalized):
        self.batch_ind = batch_ind
        self.ind = ind
        self.gt_normalized = gt_normalized
        self.recons_normalized = recons_normalized

class VQVAE_Evaluator:

    def __init__(self, vqvae: VQVAE_Single, batch_size=4, low_iou_thresh=0.9, device='cpu', division=1):
        self.vqvae = vqvae
        self.device = device

        self.batch_size = batch_size
        self.low_iou_thresh = low_iou_thresh
        self.division = division
    
    def eval_dataset(self, dataset: LvisDataset | HQSeg44KTestDataset | HQSeg44KTrainDataset):
        self.vqvae.to(self.device)
        self.vqvae.eval()

        bad_cases: List[BadCase] = []
        iou_list = []
        min_iou, max_iou = 1, 0
        nan_count = 0

        mask_level_ds = MaskLevelDataset(
            dataset=dataset,
            sam_encoder=None,
            device=self.device,
            with_image_embed=False
        )
        mask_level_dl = DataLoader(mask_level_ds, self.batch_size, shuffle=False)
        # mask_level_dl = islice(mask_level_dl, 44, 45) # for debugging
        for ind, (image, image_embed_sam, single_mask_normalized, single_mask) in tqdm(enumerate(mask_level_dl)):
            single_mask_normalized = single_mask_normalized.to(self.device)
            with torch.no_grad():
                if self.division > 1:
                    single_mask_normalized_divided = divide_image(single_mask_normalized, self.division)
                    results = self.vqvae.img_to_reconstructed_img(single_mask_normalized_divided)
                    real_result = merge_image(results[-1], self.division)
                else:
                    results = self.vqvae.img_to_reconstructed_img(single_mask_normalized)
                    real_result = results[-1]
            B = real_result.shape[0]
            ious, nan_count = calc_iou(real_result, single_mask_normalized, return_nan_count=True)
            nan_count += nan_count.item()
            for i in range(B):
                if ious[i] <= self.low_iou_thresh:
                    bad_cases.append(BadCase(
                        batch_ind=ind,
                        ind=i,
                        gt_normalized=single_mask_normalized[i].to('cpu'),
                        recons_normalized=real_result[i].to('cpu')
                    ))
                min_iou = min(ious[i].item(), min_iou)
                max_iou = max(ious[i].item(), max_iou)
            m_iou = torch.mean(ious).item()
            assert not np.isnan(m_iou)
            iou_list.append(m_iou)
            if ind % 50 == 0:
                torch.cuda.empty_cache()
        mean_iou = np.mean(iou_list)
        return mean_iou, min_iou, max_iou, bad_cases, nan_count

    def analyze_bad_cases(self, bad_cases: List[BadCase], visualize_dir: str):
        # check if the directory exists
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)

        for bad_case in bad_cases:
            chw = self.visualize(bad_case)
            torchvision.utils.save_image(chw, f'{visualize_dir}/{bad_case.batch_ind}_{bad_case.ind}.png')

    def visualize(self, bad_case: BadCase):
        gt = (bad_case.gt_normalized > 0).to(torch.float)
        recons = (bad_case.recons_normalized > 0).to(torch.float)
        chw = torchvision.utils.make_grid(torch.stack([gt, recons], dim=0), nrow=2, padding=1, pad_value=1.0)
        return chw
        

if __name__ == '__main__':
    # example of usage
    # python eval_vqvae.py --ckpt ckpt/vqvae_single.pth --dataset lvis_val --batch_size 4 --low_iou_thresh 0.9 --device cpu --visualize_dir visualize_dir --vqvae_config single

    args = argparse.ArgumentParser()
    args.add_argument('--ckpt', type=str, required=True)
    args.add_argument('--dataset', type=str, required=True) 
    args.add_argument('--batch_size', type=int, default=4)
    args.add_argument('--low_iou_thresh', type=float, default=0.9)
    args.add_argument('--device', type=str, default='cpu')
    args.add_argument('--visualize_dir', type=str, default=None)
    args.add_argument('--vqvae_config', type=str, default='single')
    args.add_argument('--division', type=int, default=1)
    args = args.parse_args()

    if args.vqvae_config == 'single':
        build_vqvae = build_vqvae_single
    elif args.vqvae_config == 'single_fewer_stages':
        build_vqvae = build_vqvae_single_fewer_stages
    elif args.vqvae_config == 'single_4_stages':
        build_vqvae = build_vqvae_single_4_stages
    elif args.vqvae_config == 'single_4_stages_v2':
        build_vqvae = build_vqvae_single_4_stages_v2
    elif args.vqvae_config == 'single_4_stages_4_slices':
        build_vqvae = build_vqvae_single_4_stages_4_slices
    elif args.vqvae_config == 'single_4_stages_4_slices_v2':
        build_vqvae = build_vqvae_single_4_stages_4_slices_v2
    elif args.vqvae_config == 'single_5_stages_v1':
        build_vqvae = build_vqvae_single_5_stages_v1
    else:
        raise ValueError(f'Unknown vqvae config: {args.vqvae_config}')

    vqvae = build_vqvae(args.ckpt)
    vqvae = torch.compile(vqvae)
    evaluator = VQVAE_Evaluator(vqvae, args.batch_size, args.low_iou_thresh, args.device, args.division)
    if args.dataset == 'lvis_val':
        dataset = LvisDataset(dataset_path='../data/coco_lvis', split='val', img_split='val')
    elif args.dataset == 'hqseg44k_val':
        dataset = HQSeg44KTestDataset(data_root='../data/sam-hq')
    elif args.dataset == 'hqseg44k_train':
        dataset = HQSeg44KTrainDataset(data_root='../data/sam-hq')
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    mean_iou, min_iou, max_iou, bad_cases, nan_count = evaluator.eval_dataset(dataset)
    print(f'mean iou: {mean_iou}, min iou: {min_iou}, max iou: {max_iou}')
    print(f'nan count: {nan_count}')
    with open(f'result-{args.dataset}-{args.vqvae_config}.json', 'w') as f:
        json.dump({
            'mean_iou': mean_iou,
            'min_iou': min_iou,
            'max_iou': max_iou,
            'nan_count': nan_count,
        }, f, indent=4)
    if args.visualize_dir is not None:
        evaluator.analyze_bad_cases(bad_cases, args.visualize_dir)