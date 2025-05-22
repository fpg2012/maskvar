from trainer import MaskLevelDataset, MaskSegTrainer
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.clicker import Clicker
import torch.nn.functional as F

class DDPMaskSegTrainer(MaskSegTrainer):

    def __init__(self, maskseg: nn.Module, optimizer: optim.Optimizer, device: str,
                 batch_size: int = 2, num_epoch: int = 1, rank: int = 0, world_size: int = 1):
        super().__init__(maskseg, optimizer, device, batch_size, num_epoch)
        self.rank = rank
        self.world_size = world_size
        self.maskseg = DDP(maskseg.to(device), device_ids=[rank])

    def train(self, dataset: MaskLevelDataset):
        """
        Train the model with DDP support.
        """
        # sampler = torch.utils.data.distributed.DistributedSampler(
        #     dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        # )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            num_workers=4,
        )

        for i in range(self.num_epoch):
            # sampler.set_epoch(i)
            epoch_start_time = time.time()
            print(f'Epoch {i+1} / {self.num_epoch}')

            losses = []
            progress_bar = tqdm(dataloader, desc=f'loss={0:.4f}|rank={self.rank}')
            last_checkpoint_time = time.time()  # Initialize the timer
            for image, image_embed, gt_mask_normalized, gt_mask in progress_bar:
                loss, logits = self.forward_pass(image_embed, gt_mask_normalized, gt_mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                progress_bar.set_description(f'loss={loss.item():.4f}|rank={self.rank}')

                # Check if 8 hours have passed since the last checkpoint
                current_time = time.time()
                if current_time - last_checkpoint_time >= 8 * 3600 and self.rank == 0:  # 8 hours in seconds
                    print(f'Saving checkpoint for rank {self.rank} to ckpt/maskseg_wallclock_{int(current_time)}.pth')
                    torch.save(self.maskseg.module.state_dict(), f'ckpt/maskseg_wallclock_{int(current_time)}.pth')
                    last_checkpoint_time = current_time  # Reset the timer

            avg_loss = np.mean(losses)
            epoch_time_elapsed = time.time() - epoch_start_time
            print(f'Average loss: {avg_loss:.4f}, Time elapsed: {epoch_time_elapsed:.2f}s')

            if self.rank == 0:
                torch.save(self.maskseg.module.state_dict(), f'ckpt/maskseg_epoch_{i+1}.pth')
    
    def forward_pass(self, image_embed_sam, gt_mask_normalized, gt_mask):
        """
        second stage non-interactive training

        image_embed_sam: (B, C_enc, H, W)
        gt_mask_normalized: (B, 1, H, W) -1 ~ 1 mask
        """
        image_embed = self.maskseg.module.image_encoder.adapt_conv(image_embed_sam).permute(0, 2, 3, 1) # (B, H/2, W/2, C)

        B, _, H, W = gt_mask_normalized.shape
        L = 1024
        C = self.dim
        N_iter = len(self.gen_tokens) # num of iterations for generating

        clickers = [Clicker(num_random_clicks=self.num_init_clicks) for _ in range(B)]
        for i, clicker in enumerate(clickers):
            clicker.set_gt_mask(gt_mask[i, 0].cpu().numpy())
            clicks = clicker.init_clicks()

        gt_idx = self.maskseg.module.maskgit.vqvae.img_to_idxBl(gt_mask_normalized.float())[-1] # (B, L)

        blank_tokens = torch.zeros_like(gt_idx) + self.maskseg.module.maskgit.vocab_size # (B, L)
        
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

        dense_pe = self.maskseg.module.prompt_encoder.get_dense_pe() # (1, C_enc, H, W)
        dense_pe = dense_pe.permute(0, 2, 3, 1) # (1, H, W, C_enc)
        # dense_pe_transformed = self.maskseg.prompt_enc_adapter(dense_pe) # (1, H, W, C)

        image_embed = image_embed + dense_pe

        click_sam_format = [clicker.to_sam_format(pad_size=2) for clicker in clickers]
        click_pos_sam_format = torch.stack([click_tuple[0] for click_tuple in click_sam_format], dim=0) # (B * num_iter_clicks, 2)
        click_label_sam_format = torch.stack([click_tuple[1] for click_tuple in click_sam_format], dim=0) # (B * num_iter_clicks,)

        prompt_embed, _ = self.maskseg.module.prompt_encoder(
            points=(click_pos_sam_format.to(self.device), click_label_sam_format.to(self.device)),
            boxes=None,
            masks=None
        ) # (B, L_click, C)

        prompt_embed = prompt_embed.view(B, 1, -1, C).repeat(1, N_iter, 1, 1) # (B, N_iter, L_click, C)
        prompt_embed = prompt_embed.view(B * N_iter, -1, C) # (B*N_iter, L_click, C)

        image_embed = image_embed.view(B, -1, C)
        image_embed = image_embed.unsqueeze(1).expand(-1, N_iter, -1, -1) # (B, N_iter, L, C)
        image_embed = image_embed.reshape(B * N_iter, -1, C) # (B*N_iter, L, C)

        logits = self.maskseg.module.maskgit(x.view(-1, L), x_pos.view(-1, L), image_embed, prompt_embed, dense_pe.view(1, L, C)) # (B*N_iter, L, vocal_size)
        
        gt_idx_for_loss = gt_idx.unsqueeze(1).expand(-1, N_iter, -1).reshape(B*N_iter*L) # (B*N_iter*L,)
        logits_for_loss = logits.reshape(B*N_iter*L, -1) # (B*N_iter*L, vocal_size)

        loss = F.cross_entropy(logits_for_loss, gt_idx_for_loss)

        return loss, logits