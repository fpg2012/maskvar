from hqseg44k import HQSeg44KTestDataset
from models.vqvae_single import VQVAE_Single
import torch
from myutils import calc_iou
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='评估VQVAE模型')
parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
parser.add_argument('--device', type=str, default='cpu', help='设备类型 (cuda 或 cpu)')
args = parser.parse_args()

# 使用命令行参数
BATCH_SIZE = args.batch_size
DEVICE = args.device

dataset = HQSeg44KTestDataset(data_root='datasets/sam-hq')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

checkpoint_path = 'vqvae_single_epoch_32.pth'

VOCAB_SIZE = 4096  # 码本大小
Z_CHANNELS = 32   # 潜在空间通道数
BASE_CHANNELS = 128  # 基础通道数
BETA = 0.25  # commitment loss权重

vqvae = VQVAE_Single(
    vocab_size=VOCAB_SIZE,
    z_channels=Z_CHANNELS,
    ch=BASE_CHANNELS,
    beta=BETA,
    v_patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32),
    test_mode=False,
    ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                using_sa=True, using_mid_sa=True,)
).to(DEVICE)

vqvae.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

ious = []

for i, item in enumerate(dataloader):
    with torch.no_grad():
        result = vqvae.img_to_reconstructed_img(item.to(DEVICE))
        iou = calc_iou(result[-1], item.to(DEVICE))
        iou = torch.mean(iou).item()
        print(f'iou: {iou} {i}/{len(dataloader)} \r', end='')
        ious.append(iou)

print(f'mean iou: {np.mean(ious)}')