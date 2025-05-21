from build_everything import build_maskseg
from trainer import MaskSegTrainer, MaskLevelDataset
from models.maskseg import MaskSeg
from datasets.coco_lvis import LvisDataset
from utils.transforms import ResizeLongestSide

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    maskseg = build_maskseg(
        vqvae_checkpoint_path="ckpt/vqvae_single.pth",
        sam_checkpoint_path="ckpt/sam_vit_b_01ec64.pth",
    )

    optimizer = optim.Adam(maskseg.parameters(), lr=0.0001)
    device = 'cuda'

    trainer = MaskSegTrainer(
        maskseg=maskseg,
        optimizer=optimizer,
        device=device,
        batch_size=2,
        num_epoch=1,
    )

    dataset = LvisDataset(
        dataset_path='data/coco_lvis',
        split='train',
        img_split='train',
        stuff_prob=0.0,
    )

    mask_level_dataset = MaskLevelDataset(dataset, maskseg.image_encoder, trainer.device)

    trainer.train(mask_level_dataset)

