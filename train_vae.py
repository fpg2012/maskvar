import os
import time
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from models.vqvae_single import VQVAE_Single
from dist import init_distributed, cleanup, get_device, is_master


class SingleChannelDataset(Dataset):
    def __init__(self, data_root, subdirs, transform=None, is_train=True):
        self.data_root = Path(data_root)
        self.transform = transform
        self.is_train = is_train
        self.data_list = []
        
        # 定义子目录
        self.subdirs = subdirs
        
        # 加载数据列表
        self._load_data_list()
        
    def _load_data_list(self):
        for subdir in self.subdirs:
            img_files = {
                p.stem: p for p in
                (self.data_root / subdir).rglob('*.jpg')
                if 'images_test' not in str(p) and
                   'masks_test' not in str(p)}
            ann_files = {
                p.stem: p for p in
                (self.data_root / subdir).rglob('*.png')
                if 'images_test' not in str(p) and
                   'masks_test' not in str(p)}
            
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            for prefix in sorted(img_files.keys()):
                if prefix in prefixes:
                    self.data_list.append({
                        'img_path': str(img_files[prefix]),
                        'mask_path': str(ann_files[prefix])
                    })
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 加载掩码
        with Image.open(item['mask_path']) as mask:
            mask = np.array(mask)
        
        # 确保掩码是二值的
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)
        
        # 转换为PIL图像
        mask = Image.fromarray(mask)
        
        # 转换为张量
        mask = torch.from_numpy(np.array(mask)).float()
        mask = mask.unsqueeze(0)  # 添加通道维度
        
        # 归一化
        mask = (mask - 0.5) / 0.5  # 归一化到[-1, 1]范围
        
        return mask


def train_vae(rank, world_size, args):
    """训练VQVAE模型
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        args: 训练参数
    """
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    
    # 初始化分布式环境
    init_distributed()
    
    # 创建模型
    model = VQVAE_Single(
        vocab_size=args.vocab_size,
        z_channels=args.z_channels,
        ch=args.ch,
        dropout=args.dropout,
        beta=args.beta,
        using_znorm=args.using_znorm,
        quant_conv_ks=args.quant_conv_ks,
        quant_resi=args.quant_resi,
        share_quant_resi=args.share_quant_resi,
        default_qresi_counts=args.default_qresi_counts,
        v_patch_nums=args.v_patch_nums,
        test_mode=False
    )
    
    # 将模型移动到GPU
    device = get_device()
    model = model.to(device)
    
    # 包装模型用于分布式训练
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device
    )
    
    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1
    )
    
    # 创建梯度缩放器（用于混合精度训练）
    scaler = GradScaler()
    
    # 创建数据加载器
    train_subdirs = [
        'DIS5K/DIS-TR',
        'thin_object_detection/ThinObject5K',
        'cascade_psp/fss_all',
        'cascade_psp/DUTS-TR',
        'cascade_psp/DUTS-TE',
        'cascade_psp/ecssd',
        'cascade_psp/MSRA_10K'
    ]
    
    val_subdirs = [
        'thin_object_detection/COIFT',
        'thin_object_detection/HRSOD',
        'thin_object_detection/ThinObject5K',
        'DIS5K/DIS-VD'
    ]
    
    dataset_train = SingleChannelDataset(
        data_root=args.data_path,
        subdirs=train_subdirs,
        is_train=True
    )
    
    dataset_val = SingleChannelDataset(
        data_root=args.data_path,
        subdirs=val_subdirs,
        is_train=False
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_val,
        num_replicas=world_size,
        rank=rank
    )
    
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # 训练一个epoch
        model.train()
        train_loss = 0
        start_time = time.time()
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                # 前向传播
                reconstructed, _, vq_loss = model(images)
                
                # 计算重建损失
                recon_loss = nn.functional.mse_loss(reconstructed, images)
                
                # 总损失
                loss = recon_loss + vq_loss
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if batch_idx % args.log_interval == 0 and is_master():
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}')
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                reconstructed, _, vq_loss = model(images)
                recon_loss = nn.functional.mse_loss(reconstructed, images)
                loss = recon_loss + vq_loss
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        
        if is_master():
            print(f'Epoch: {epoch}')
            print(f'Time taken: {time.time() - start_time:.2f} seconds')
            print(f'Average training loss: {train_loss:.6f}')
            print(f'Average validation loss: {val_loss:.6f}')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(args.save_dir, 'best_model.pth'))
            
            # 保存最新模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, 'latest_model.pth'))
    
    # 清理分布式环境
    cleanup()


def main():
    """主函数，启动分布式训练"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练VQVAE模型')
    
    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=4096, help='码本大小')
    parser.add_argument('--z_channels', type=int, default=32, help='潜在空间通道数')
    parser.add_argument('--ch', type=int, default=128, help='基础通道数')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout比率')
    parser.add_argument('--beta', type=float, default=0.25, help='commitment loss权重')
    parser.add_argument('--using_znorm', action='store_true', help='是否使用归一化')
    parser.add_argument('--quant_conv_ks', type=int, default=3, help='量化卷积核大小')
    parser.add_argument('--quant_resi', type=float, default=0.5, help='残差连接比例')
    parser.add_argument('--share_quant_resi', type=int, default=4, help='共享残差层数量')
    parser.add_argument('--default_qresi_counts', type=int, default=0, help='默认残差层数量')
    
    # 训练参数
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--image_size', type=int, default=256, help='图像大小')
    parser.add_argument('--batch_size', type=int, default=32, help='每个GPU的批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作进程数')
    parser.add_argument('--log_interval', type=int, default=100, help='日志打印间隔')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    
    args = parser.parse_args()
    
    # 创建保存目录
    if is_master():
        os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置patch数量
    args.v_patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    
    # 启动分布式训练
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("警告：没有足够的GPU用于分布式训练，将使用单GPU模式")
        train_vae(0, 1, args)
    else:
        mp.spawn(
            train_vae,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main() 