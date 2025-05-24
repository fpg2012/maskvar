import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from tqdm import tqdm
import os
import argparse
from torch.cuda.amp import GradScaler
import numpy as np

from models.vqvae_single import VQVAE_Single
from datasets.hqseg44k import HQSeg44KTrainDataset
from utils.loss import NormalizedFocalLoss, FocalLoss, NormalizedFocalLoss2
from build_everything import build_vqvae_single_monoscale_v2

def resize_longest_side(image, target_length, mode='bilinear'):
    scale = target_length * 1.0 / max(image.shape[-2], image.shape[-1])
    newh, neww = image.shape[-2] * scale, image.shape[-1] * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)

    if mode == 'bilinear':
        return F.interpolate(
            image, (newh, neww), mode=mode, align_corners=False, antialias=True
        )
    else:
        return F.interpolate(
            image, (newh, neww), mode=mode,
        )

class MaskOnlyHQSeg44K(Dataset):

    def __init__(self, inside_dataset: HQSeg44KTrainDataset):
        self.inside_dataset = inside_dataset
    
    def __getitem__(self, index):
        img, mask, instance_info = self.inside_dataset[index]
        real_mask_torch = mask
        mask_normalized, _ = self.preprocess_mask(real_mask_torch, instance_info, 0)
        return mask_normalized
    
    def __len__(self):
        return len(self.inside_dataset)

    def preprocess_mask(self, gt_mask, instance_info, instance_idx):
        mask = gt_mask[:, :, instance_info[instance_idx].mapping[0]] == instance_info[instance_idx].mapping[1]

        # to tensor
        mask = torch.from_numpy(mask).to(dtype=torch.float32).unsqueeze(0)

        mask = resize_longest_side(mask.unsqueeze(0), 256, 'nearest').squeeze(0)
        mask = mask.long()

        # pad mask to 256
        h, w = mask.shape[-2:]
        padh = 256 - h
        padw = 256 - w
        mask = F.pad(mask, (0, padw, 0, padh), value=0)

        # normalize mask
        mask_normalized = mask * 2 - 1

        return mask_normalized.float(), mask    

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_checkpoint(model, optimizer: torch.optim.Adam, checkpoint_path, device, rank):
    """
    加载checkpoint
    
    Args:
        model: 模型实例
        optimizer: 优化器实例
        checkpoint_path: checkpoint文件路径
        device: 设备
    
    Returns:
        start_epoch: 开始训练的epoch
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到checkpoint文件: {checkpoint_path}")
    
    map_location = 'cpu'

    dist.barrier()

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 由于没有优化器状态，我们从头开始
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    
    if rank == 0:
        print(f"成功加载checkpoint，从epoch {start_epoch} 继续训练")
    
    return start_epoch

def train_epoch(model, dataloader, optimizer, device, epoch, rank, use_focal_loss=False):
    model.train()
    
    # total_loss = 0
    if not use_focal_loss:
        focal_loss = NormalizedFocalLoss(alpha=0.5, gamma=2.0).to(device)
    else:
        # focal_loss = FocalLoss(alpha=0.5, gamma=2.0).to(device)
        focal_loss = NormalizedFocalLoss2(alpha=0.5, gamma=2.0).to(device)

    dataloader_iter = tqdm(dataloader) if rank == 0 else dataloader
    
    for batch in dataloader_iter:
        x = batch.to(device)
        # with torch.autocast(device_type='cuda'):
        x_recon, usage, vq_loss = model(x, ret_usages=True)
        if use_focal_loss:
            recons_loss, scale = focal_loss(x_recon, x)
            loss = scale * (recons_loss + vq_loss)
        else:
            recon_loss = focal_loss(x_recon, x)
            loss = recon_loss + vq_loss
        
        optimizer.zero_grad()
        # scaler.scale(loss).backward()
        loss.backward()
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()
        if rank == 0:
            dataloader_iter.set_description(f'loss: {loss.item():.3f}={recon_loss.item():.3f}+{vq_loss.item():.3f}, usage: {usage}')
        # total_loss += loss.item()
    
    # dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    # avg_loss = total_loss / (len(dataloader) * dist.get_world_size())
    # return avg_loss

def main(rank, world_size, args):
    setup(rank, world_size)

    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    DEVICE = torch.device(f'cuda:{rank}')
    # DEVICE = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型参数

    # 创建数据加载器
    inside_dataset = HQSeg44KTrainDataset(data_root='data/sam-hq')
    dataset = MaskOnlyHQSeg44K(inside_dataset=inside_dataset)
    # sampler = None
    sampler = DistributedSampler(dataset) if world_size > 1 else None

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=(sampler is None),
        num_workers=4,
        sampler=sampler,
        pin_memory=True
    )

    # model = VQVAE_Single(
    #     vocab_size=VOCAB_SIZE,
    #     z_channels=Z_CHANNELS,
    #     ch=BASE_CHANNELS,
    #     beta=BETA,
    #     # v_patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32),
    #     v_patch_nums=[32],
    #     test_mode=False,
    #     ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
    #                 using_sa=True, using_mid_sa=True,)
    # ).to(DEVICE)

    model = build_vqvae_single_monoscale_v2(require_grad=True)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 如果指定了checkpoint，则加载它
    if args.checkpoint is not None:
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint, DEVICE, rank)
    elif args.start_epoch is not None:
        start_epoch = args.start_epoch
    else:
        start_epoch = 0

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # 训练循环
    for epoch in range(start_epoch, NUM_EPOCHS):
        sampler.set_epoch(epoch)
        train_epoch(model, dataloader, optimizer, DEVICE, epoch, rank, use_focal_loss=args.focal_loss)
        
        if rank == 0:
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
            if (epoch + 1) % 1 == 0:
                # 保存checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, f'vqvae_single_monoscale_v2_epoch_{epoch+1}.pth')
    
    cleanup()

if __name__ == '__main__':
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='训练VQVAE模型')
    parser.add_argument('--checkpoint', type=str, default=None, help='要加载的checkpoint路径')
    parser.add_argument('--start_epoch', type=int, default=0, help='开始训练的epoch')
    parser.add_argument('--num_epochs', type=int, default=8, help='总训练epoch数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--focal_loss', action='store_true', help='是否使用FocalLoss，不开启默认使用NormalizedFocalLoss')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

    