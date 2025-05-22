from build_everything import build_maskseg, build_sam_image_encoder
from trainer import MaskSegTrainer, MaskLevelDataset
from ddp_trainer import DDPMaskSegTrainer
from models.maskseg import MaskSeg
from datasets.coco_lvis import LvisDataset
from datasets.hqseg44k import HQSeg44KTrainDataset
from utils.transforms import ResizeLongestSide

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse

def setup(rank, world_size, master_addr='localhost', master_port='12355'):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def build_dataset(dataset_name):
    if dataset_name == 'coco_lvis':
        return LvisDataset(
            dataset_path='data/coco_lvis',
            split='train',
            img_split='train',
            stuff_prob=0.0,
        )
    elif dataset_name == 'hqseg44k':
        return HQSeg44KTrainDataset(data_root='data/sam-hq')
    else:
        raise ValueError(f'Dataset {dataset_name} not found')

def train(rank, world_size, args):
    setup(rank, world_size, args.master_addr, args.master_port)

    maskseg = build_maskseg(
        vqvae_checkpoint_path="ckpt/vqvae_single.pth",
        sam_checkpoint_path="ckpt/sam_vit_b_01ec64.pth",
    )

    optimizer = optim.Adam(maskseg.parameters(), lr=0.0001)

    device = f'cuda:{rank}'

    # sam_image_encoder = build_sam_image_encoder(
    #     sam_checkpoint_path="ckpt/sam_vit_b_01ec64.pth",
    # )
    # sam_image_encoder.to(device)
    # sam_image_encoder.eval()

    trainer = DDPMaskSegTrainer(
        maskseg=maskseg,
        optimizer=optimizer,
        device=device,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        rank=rank,
        world_size=world_size,
    )

    dataset = LvisDataset(
        dataset_path='data/coco_lvis',
        split='train',
        img_split='train',
        stuff_prob=0.0,
    )

    mask_level_dataset = MaskLevelDataset(
        dataset=dataset,
        # sam_encoder=sam_image_encoder,
        sam_encoder=trainer.maskseg.module.image_encoder.sam_encoder,
        device=trainer.device,
    )

    trainer.train(mask_level_dataset)
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12355', help='Master port for distributed training')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_epoch', type=int, default=1, help='Number of training epochs')
    
    args = parser.parse_args()

    mp.spawn(train, args=(args.world_size, args), nprocs=args.world_size, join=True)