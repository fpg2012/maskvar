from typing import List, Tuple
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2

from models.maskseg import MaskSeg

class Clicker:

    def __init__(self, num_random_clicks: int = 2):
        self.click_list: List[Tuple[int, int, int]] = []
        # gt_mask: (H, W)
        self.gt_mask = None
        self.not_ignore_mask = None # ignore pixels that are -1
        self.not_clicked_map = None # mask out clicked pixels
        self.num_random_clicks = num_random_clicks

    def init_clicks(self) -> List[Tuple[int, int, int]]:
        """
        random sample some clickes predict initial clicks
        """
        for _ in range(self.num_random_clicks):
            # Erode the mask to get points away from edges
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(self.gt_mask.astype(np.uint8), kernel, iterations=1)
            
            # pad eroded_mask with 1 pixel
            eroded_mask = np.pad(eroded_mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)

            # Compute distance transform - points closer to center have higher values
            dt = cv2.distanceTransform(eroded_mask, cv2.DIST_L2, 3)

            # unpad dt
            dt = dt[1:-1, 1:-1]

            # Sample a point based on the probability map
            flat_probs = ((dt*self.not_clicked_map)**2).flatten()
            flat_probs = flat_probs / flat_probs.sum()  # Normalize to probabilities
            idx = np.random.choice(len(flat_probs), p=flat_probs)
            y, x = np.unravel_index(idx, dt.shape)
            
            # Add random click (1 for positive since sampling from gt_mask)
            self.click_list.append((y, x, 1))
            self.not_clicked_map[y, x] = False

        return self.click_list, eroded_mask, dt
    
    def set_gt_mask(self, gt_mask):
        """
        gt_mask: (H, W)
        """
        assert gt_mask.ndim == 2
        self.gt_mask = gt_mask == 1
        self.not_ignore_mask = gt_mask != -1
        self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)

    def predict_next_click(self, pred_mask) -> Tuple[int, int, int]:
        """
        predict next click and update click list

        pred_mask: (H, W)
        
        Returns:
            Tuple[int, int, int]: (y, x, is_positive) coordinates of next click
        """
        if self.gt_mask is None:
            raise ValueError("Ground truth mask not set. Call set_gt_mask first.")
        
        assert pred_mask.ndim == 2
        
        # Calculate false negative mask (ground truth is 1 but prediction is 0)
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        # Calculate false positive mask (ground truth is 0 but prediction is 1)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        # pad fn_mask and fp_mask with 1 pixel
        fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)

        # Compute distance transforms to find farthest points from boundaries
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        # unpad fn_mask_dt and fp_mask_dt
        fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        # Mask out already clicked points
        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        # Find maximum distances in each mask
        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        # Determine if next click should be positive (add) or negative (remove)
        is_positive = fn_max_dist > fp_max_dist
        
        # Get coordinates of point with maximum distance
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)

        # Store click and update state
        click = (coords_y[0], coords_x[0], 1 if is_positive else 0)
        self.click_list.append(click)
        self.not_clicked_map[coords_y[0], coords_x[0]] = False
        
        return click
    
    def to_sam_format(self, pad_size: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = torch.tensor([(click[1], click[0]) for click in self.click_list])
        # label: 1 for positive, 0 for negative, -1 for padding
        label = torch.tensor([click[2] for click in self.click_list])
        L_clicks = len(self.click_list)
        if pad_size > 0 and pad_size > L_clicks:
            coords = torch.cat([coords, torch.zeros(pad_size - L_clicks, 2)], dim=0)
            label = torch.cat([label, torch.zeros(pad_size - L_clicks, dtype=torch.long) - 1], dim=0)
        return coords, label

class MaskSegTrainer:
    
    def __init__(self, maskseg: MaskSeg, optimizer: optim.Optimizer, device: str):
        self.maskseg = maskseg
        self.optimizer = optimizer
        self.gen_tokens = [1, 4, 16, 64, 256, 1024]
        self.max_num_clicks = 10
        self.num_init_clicks = 2
        self.max_num_iter_clicks = self.max_num_clicks - self.num_init_clicks
        
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

    def preprocess_input(self, image: torch.Tensor, gt_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        preprocess image and gt_mask

        image: (B, H, W, 3) H=W=1024
        gt_mask: (B, H, W) H=W=256

        returns:
            image: (B, 3, H, W) normalized image
            gt_mask_normalized: (B, H, W) -1 ~ 1 mask
        """
        image = (image - self.pixel_mean) / self.pixel_std
        image = image.permute(0, 3, 1, 2) # (B, 3, H, W)

        gt_mask_normalized = gt_mask * 2 - 1
        return image, gt_mask_normalized

    def forward_pass(self, image, gt_mask_normalized):
        """
        second stage non-interactive training

        image: (B, 3, H, W) H=W=1024
        gt_mask_normalized: (B, 1, H, W) -1 ~ 1 mask
        """
        full_size_image = image
        image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)

        B, _, H, W = gt_mask_normalized.shape
        L = 1024
        C = self.dim
        N_iter = len(self.gen_tokens) # num of iterations for generating

        clickers = [Clicker(num_random_clicks=self.num_init_clicks) for _ in range(B)]
        for i, clicker in enumerate(clickers):
            clicker.set_gt_mask(gt_mask_normalized[i, 0].cpu().numpy())
            clicks = clicker.init_clicks()

        image_embed = self.maskseg.image_encoder(full_size_image) # (B, H, W, C) h_embed*w_embed = L

        gt_idx = self.maskseg.maskgit.vqvae.img_to_idxBl(gt_mask_normalized.float())[-1] # (B, L)

        blank_tokens = torch.zeros_like(gt_idx) + self.maskseg.maskgit.vocab_size # (B, L)
        
        # shuffle positions
        positions = torch.randperm(L) # (L,)

        masks = torch.zeros_like(gt_idx) # (B, L)
        masks = masks.view(B, 1, L).repeat(1, N_iter, 1) # (B, N_iter, L)

        for i, num_tokens in enumerate(self.gen_tokens):
            masks[:, i, positions[:num_tokens]] = 1

        x = torch.where(masks == 1, gt_idx, blank_tokens) # (B, N_iter, L)
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
        image_embed = image_embed.view(B * N_iter, -1, C) # (B*N_iter, L, C)

        logits = self.maskseg.maskgit(x.view(-1, L), x_pos.view(-1, L), image_embed, prompt_embed, dense_pe.view(1, L, C)) # (B*N_iter, L, vocal_size)
        
        gt_idx_for_loss = gt_idx.unsqueeze(1).expand(-1, N_iter, -1).reshape(B*N_iter*L) # (B*N_iter*L,)
        logits_for_loss = logits.reshape(B*N_iter*L, -1) # (B*N_iter*L, vocal_size)

        loss = F.cross_entropy(logits_for_loss, gt_idx_for_loss)

        return loss, logits