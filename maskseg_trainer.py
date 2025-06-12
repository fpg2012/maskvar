from typing import List, Tuple, Optional, Iterator
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.clicker import Clicker
from torch.utils.data import IterableDataset, Dataset, DataLoader
from datasets.coco_lvis import LvisDataset
from datasets.hqseg44k import HQSeg44KTrainDataset
from tqdm import tqdm
import numpy as np
import time
from models.maskseg import MaskSeg
from models.image_encoder import ImageEncoder
from models.sam_image_encoder import ImageEncoderViT as SamImageEncoder
import torch.distributed as dist

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

class MaskLevelDataset(IterableDataset):

    def __init__(self, dataset: Optional[LvisDataset | HQSeg44KTrainDataset], sam_encoder: SamImageEncoder, device: str):
        self.dataset = dataset
        self.sam_encoder = sam_encoder
        self.device = device

        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).to(device) # copied from sam
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).to(device) # copied from sam

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
            image, image_embed_sam = self.preprocess_image(image)
            for instance_idx in instance_info.keys():
                single_mask_normalized, single_mask = self.preprocess_mask(mask, instance_info, instance_idx)
                yield image, image_embed_sam, single_mask_normalized, single_mask

    def preprocess_image(self, image):
        """
        preprocess image for image encoder

        image: (H, W, 3)
        """
        image = torch.from_numpy(image).to(self.device) / 255.0
        image = image.permute(2, 0, 1) # (H, W, 3) -> (3, H, W)
        image = resize_longest_side(image.unsqueeze(0), 1024).squeeze(0)

        # normalize image
        image = image.permute(1, 2, 0) # (3, H, W) -> (H, W, 3)
        image = (image - self.pixel_mean) / self.pixel_std
        image = image.permute(2, 0, 1) # (H, W, 3) -> (3, H, W)

        # pad image to 1024
        h, w = image.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        image = F.pad(image, (0, padw, 0, padh), value=0)

        # print(f'image shape: {image.shape}')

        # image_embed = self.image_encoder(image.unsqueeze(0)).squeeze(0)
        with torch.no_grad():
            image_embed_sam = self.sam_encoder(image.unsqueeze(0)).squeeze(0)

        return image, image_embed_sam

    def preprocess_mask(self, gt_mask, instance_info, instance_idx):
        mask = gt_mask[:, :, instance_info[instance_idx].mapping[0]] == instance_info[instance_idx].mapping[1]

        # to tensor
        mask = torch.from_numpy(mask).to(self.device, dtype=torch.float32).unsqueeze(0)

        mask = resize_longest_side(mask.unsqueeze(0), 256, 'nearest').squeeze(0)
        mask = mask.long()

        # pad mask to 256
        h, w = mask.shape[-2:]
        padh = 256 - h
        padw = 256 - w
        mask = F.pad(mask, (0, padw, 0, padh), value=0)

        # normalize mask
        mask_normalized = mask * 2 - 1

        return mask_normalized, mask

class MaskSegTrainer:
    
    def __init__(self, maskseg: MaskSeg, optimizer: optim.Optimizer, device: str,
                  batch_size: int = 2, num_epoch: int = 1):
        self.maskseg = maskseg
        self.optimizer = optimizer
        self.gen_tokens = [1, 4, 16, 64, 256, 1024]
        self.max_num_clicks = 10
        self.num_init_clicks = 2
        self.max_num_iter_clicks = self.max_num_clicks - self.num_init_clicks
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        
        self.sequence_size = (256 // 8, 256 // 8)
        self.dim = maskseg.maskgit.dim
        self.device = device

        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).to(device) # copied from sam
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).to(device) # copied from sam
        self.maskseg.to(device)
    
    def sample_from_logits(self, logits: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        sample from logits

        logits: (B, L)
        """
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=num_samples, replacement=True)

    def train(self, dataset: MaskLevelDataset):
        """
        train the model

        dataset: MaskLevelDataset
        """
        for i in range(self.num_epoch):
            epoch_start_time = time.time()
            print(f'Epoch {i+1} / {self.num_epoch}')

            dataloader = DataLoader(dataset, batch_size=self.batch_size)
            losses = []
            progress_bar = tqdm(dataloader, desc=f'Loss: {0:.4f}')
            last_checkpoint_time = time.time()  # Initialize the timer
            for image, image_embed, gt_mask_normalized, gt_mask in progress_bar:
                loss, logits = self.forward_pass(image_embed, gt_mask_normalized, gt_mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                progress_bar.set_description(f'Loss: {loss.item():.4f}')

                # Check if 8 hours have passed since the last checkpoint
                current_time = time.time()
                if current_time - last_checkpoint_time >= 8 * 3600:  # 8 hours in seconds
                    torch.save(self.maskseg.state_dict(), f'ckpt/maskseg_wallclock_{int(current_time)}.pth')
                    last_checkpoint_time = current_time  # Reset the timer
            
            avg_loss = np.mean(losses)
            epoch_time_elapsed = time.time() - epoch_start_time
            print(f'Average loss: {avg_loss:.4f}, Time elapsed: {epoch_time_elapsed:.2f}s')

            # save checkpoint for each epoch
            torch.save(self.maskseg.state_dict(), f'ckpt/maskseg_epoch_{i+1}.pth')

    def forward_pass(self, image_embed_sam, gt_mask_normalized, gt_mask):
        """
        second stage non-interactive training

        image_embed_sam: (B, C_enc, H, W)
        gt_mask_normalized: (B, 1, H, W) -1 ~ 1 mask
        """
        image_embed = self.maskseg.image_encoder.adapt_conv(image_embed_sam).permute(0, 2, 3, 1) # (B, H/2, W/2, C)

        B, _, H, W = gt_mask_normalized.shape
        L = 1024
        C = self.dim
        N_iter = len(self.gen_tokens) # num of iterations for generating

        clickers = [Clicker(num_random_clicks=self.num_init_clicks) for _ in range(B)]
        for i, clicker in enumerate(clickers):
            clicker.set_gt_mask(gt_mask[i, 0].cpu().numpy())
            clicks = clicker.init_clicks()

        gt_idx = self.maskseg.maskgit.vqvae.img_to_idxBl(gt_mask_normalized.float())[-1] # (B, L)

        blank_tokens = torch.zeros_like(gt_idx) + self.maskseg.maskgit.vocab_size # (B, L)
        
        # shuffle positions
        positions = torch.randperm(L) # (L,)

        masks = torch.zeros_like(gt_idx) # (B, L)
        masks = masks.view(B, 1, L).repeat(1, N_iter, 1) # (B, N_iter, L)

        for i, num_tokens in enumerate(self.gen_tokens):
            masks[:, i, positions[:num_tokens]] = 1

        x = torch.where(
            masks == 1, 
            gt_idx.unsqueeze(1).expand(-1, N_iter, -1), 
            blank_tokens.unsqueeze(1).expand(-1, N_iter, -1)
        ) # (B, N_iter, L)
        x_pos = torch.arange(L).to(self.device).unsqueeze(0).unsqueeze(0).repeat(B, N_iter, 1) # (B, N_iter, L)

        dense_pe = self.maskseg.prompt_encoder.get_dense_pe() # (1, C_enc, H, W)
        dense_pe = dense_pe.permute(0, 2, 3, 1) # (1, H, W, C_enc)
        # dense_pe_transformed = self.maskseg.prompt_enc_adapter(dense_pe) # (1, H, W, C)

        image_embed = image_embed + dense_pe

        click_sam_format = [clicker.to_sam_format(pad_size=2) for clicker in clickers]
        click_pos_sam_format = torch.stack([click_tuple[0] for click_tuple in click_sam_format], dim=0) # (B * num_iter_clicks, 2)
        click_label_sam_format = torch.stack([click_tuple[1] for click_tuple in click_sam_format], dim=0) # (B * num_iter_clicks,)

        prompt_embed, _ = self.maskseg.prompt_encoder(
            points=(click_pos_sam_format.to(self.device), click_label_sam_format.to(self.device)),
            boxes=None,
            masks=None
        ) # (B, L_click, C)

        prompt_embed = prompt_embed.view(B, 1, -1, C).repeat(1, N_iter, 1, 1) # (B, N_iter, L_click, C)
        prompt_embed = prompt_embed.view(B * N_iter, -1, C) # (B*N_iter, L_click, C)

        image_embed = image_embed.view(B, -1, C)
        image_embed = image_embed.unsqueeze(1).expand(-1, N_iter, -1, -1) # (B, N_iter, L, C)
        image_embed = image_embed.reshape(B * N_iter, -1, C) # (B*N_iter, L, C)

        logits = self.maskseg.maskgit(x.view(-1, L), x_pos.view(-1, L), image_embed, prompt_embed, dense_pe.view(1, L, C)) # (B*N_iter, L, vocal_size)
        
        gt_idx_for_loss = gt_idx.unsqueeze(1).expand(-1, N_iter, -1).reshape(B*N_iter*L) # (B*N_iter*L,)
        logits_for_loss = logits.reshape(B*N_iter*L, -1) # (B*N_iter*L, vocal_size)

        loss = F.cross_entropy(logits_for_loss, gt_idx_for_loss)

        return loss, logits