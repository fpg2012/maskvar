from build_everything import build_maskseg
from trainer import MaskSegTrainer
from models.maskseg import MaskSeg
from datasets.coco_lvis import LvisDataset
from utils.transforms import ResizeLongestSide

import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

def preprocess_input(image, gt_mask, instance_info, instance_idx, device='cpu'):
    mask = gt_mask[:, :, instance_info[instance_idx].mapping[0]] == instance_info[instance_idx].mapping[1]

    # to tensor
    image = torch.from_numpy(image).to(device) / 255.0
    mask = torch.from_numpy(mask).to(device, dtype=torch.float32).unsqueeze(0)


    image = image.permute(2, 0, 1) # (H, W, 3) -> (3, H, W)
    image = resize_longest_side(image.unsqueeze(0), 1024).squeeze(0)
    mask = resize_longest_side(mask.unsqueeze(0), 256, 'nearest').squeeze(0)
    mask = mask.long()
    
    # normalize image
    image = image.permute(1, 2, 0) # (3, H, W) -> (H, W, 3)
    image = (image - trainer.pixel_mean) / trainer.pixel_std
    image = image.permute(2, 0, 1) # (H, W, 3) -> (3, H, W)

    # pad image to 1024
    h, w = image.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    image = F.pad(image, (0, padw, 0, padh), value=0)

    # pad mask to 256
    h, w = mask.shape[-2:]
    padh = 256 - h
    padw = 256 - w
    mask = F.pad(mask, (0, padw, 0, padh), value=0)

    # normalize mask
    mask = mask * 2 - 1

    return image, mask


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

image, gt_mask, instance_info = dataset[10]

image, mask = preprocess_input(image, gt_mask, instance_info, 0, trainer.device)

image = image.unsqueeze(0)
mask = mask.unsqueeze(0)

loss, logits = trainer.forward_pass(image, mask)

print(f'loss: {loss}')
print(f'logits: {logits.shape}')

