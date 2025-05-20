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

maskseg = build_maskseg(
    vqvae_checkpoint_path="ckpt/vqvae_single.pth",
    sam_checkpoint_path="ckpt/sam_vit_b_01ec64.pth",
)

optimizer = optim.Adam(maskseg.parameters(), lr=0.0001)
device = 'cuda'

trainer = MaskSegTrainer(maskseg, optimizer, device)

dataset = LvisDataset(
    dataset_path='data/coco_lvis',
    split='train',
    img_split='train',
    stuff_prob=0.0,
)

dataset = MaskLevelDataset(dataset, maskseg.image_encoder, trainer.device)

dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

for image, image_embed, mask in dataloader:
    loss, logits = trainer.forward_pass(image_embed, mask)

    print(f'loss: {loss}')
    print(f'logits: {logits.shape}')

    loss.backward()
    break

# 检查 prompt_encoder 是否被冻结
print("Prompt Encoder parameters frozen:", end=' ')
for name, param in maskseg.prompt_encoder.named_parameters():
    assert param.requires_grad == False
print('yes')

# 检查 image_encoder 中的 SAM 编码器是否被冻结
print("Image Encoder (SAM) parameters frozen:", end=' ')
for name, param in maskseg.image_encoder.sam_encoder.named_parameters():
    assert param.requires_grad == False
print('yes')

# 检查 maskgit 中的 VQVAE 是否被冻结
print("MaskGIT VQVAE parameters frozen:", end=' ')
for name, param in maskseg.maskgit.vqvae.named_parameters():
    assert param.requires_grad == False
print('yes')
