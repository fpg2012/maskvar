import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os
import argparse

from models.vqvae_single import VQVAE_Single
from hqseg44k import HQSeg44KTrainDataset

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='训练VQVAE模型')
parser.add_argument('--checkpoint', type=str, default=None, help='要加载的checkpoint路径')
parser.add_argument('--start_epoch', type=int, default=0, help='开始训练的epoch')
parser.add_argument('--num_epochs', type=int, default=8, help='总训练epoch数')
parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
args = parser.parse_args()

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        return rank, world_size, gpu
    else:
        return 0, 1, 0

# 初始化分布式环境
rank, world_size, gpu = setup_distributed()

BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
DEVICE = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
VOCAB_SIZE = 4096  # 码本大小
Z_CHANNELS = 32   # 潜在空间通道数
BASE_CHANNELS = 128  # 基础通道数
BETA = 0.25  # commitment loss权重

class NormalizedFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, eps=torch.finfo(torch.float).eps):
        """
        归一化的Focal Loss，用于解决梯度消失问题
        
        Args:
            alpha (float): 平衡正负样本的权重
            gamma (float): 聚焦参数，用于降低易分类样本的权重
            eps (float): 数值稳定性参数
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值，范围在[-1,1]之间
            target: 目标值，范围在[-1,1]之间
        """
        # 对预测值应用sigmoid，将范围转换到[0,1]
        pred = torch.sigmoid(pred)
        
        # 将目标值二值化（0为阈值）
        target = (target > 0).float()
        
        # 计算alpha权重
        alpha = torch.where(target > 0, self.alpha, (1 - self.alpha))
        
        # 计算pt（预测正确的概率）
        pt = 1.0 - (pred - target).abs()
        
        # 计算beta（难易样本权重）
        beta = (1.0 - pt) ** self.gamma
        
        # 计算归一化因子
        scale = target.numel() / (beta.sum() + self.eps)
        scale = scale.detach()  # 阻止梯度传播
        
        # 计算最终的loss
        beta = scale * beta
        loss = -alpha * beta * (pt + self.eps).log()
        
        return loss.mean()

# 数据预处理
transform = None  # 我们已经在__getitem__中处理了所有转换

def load_checkpoint(model, optimizer, checkpoint_path, device):
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 如果是DDP模型，需要特殊处理
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 由于没有优化器状态，我们从头开始
    start_epoch = args.start_epoch
    
    if rank == 0:
        print(f"成功加载checkpoint，从epoch {start_epoch} 继续训练")
    
    return start_epoch

# 创建数据加载器
dataset = HQSeg44KTrainDataset(data_root='datasets/sam-hq', transform=transform)
sampler = DistributedSampler(dataset) if world_size > 1 else None
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=(sampler is None),
    num_workers=4,
    sampler=sampler,
    pin_memory=True
)

model = VQVAE_Single(
    vocab_size=VOCAB_SIZE,
    z_channels=Z_CHANNELS,
    ch=BASE_CHANNELS,
    beta=BETA,
    v_patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32),
    test_mode=False,
    ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                using_sa=True, using_mid_sa=True,)
).to(DEVICE)

# 将模型包装为DDP模型
if world_size > 1:
    model = DDP(model, device_ids=[gpu])

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 如果指定了checkpoint，则加载它
start_epoch = args.start_epoch
if args.checkpoint is not None:
    start_epoch = load_checkpoint(model, optimizer, args.checkpoint, DEVICE)

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
    
    total_loss = 0
    focal_loss = NormalizedFocalLoss(alpha=0.5, gamma=2.0).to(device)
    
    for batch in tqdm(dataloader, disable=rank != 0):
        x = batch.to(device)
        x_recon, _, vq_loss = model(x)
        recon_loss = focal_loss(x_recon, x)
        loss = recon_loss + vq_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 同步所有进程的损失
    if world_size > 1:
        dist.all_reduce(torch.tensor(total_loss).to(device))
        total_loss /= world_size
    
    return total_loss / len(dataloader)

# 训练循环
train_losses = []
for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_loss = train_epoch(model, dataloader, optimizer, DEVICE, epoch)
    train_losses.append(epoch_loss)
    
    if rank == 0:  # 只在主进程打印和保存
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}')
        
        if (epoch + 1) % 2 == 0:
            # 保存checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'vqvae_single_epoch_{epoch+1}.pth')

    if rank == 0:  # 只在主进程绘图
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

def visualize_reconstruction(model, dataloader, device, num_samples=5):
    if rank != 0:  # 只在主进程可视化
        return
        
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x = batch[:num_samples].to(device)
            x_recon, _, _ = model(x)
            
            fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
            for i in range(num_samples):
                axes[0, i].imshow(x[i, 0].cpu().numpy(), cmap='gray')
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                axes[1, i].imshow(x_recon[i, 0].cpu().numpy(), cmap='gray')
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.show()
            break

# 可视化重建结果
visualize_reconstruction(model, dataloader, DEVICE)

# 清理分布式环境
if world_size > 1:
    dist.destroy_process_group()

