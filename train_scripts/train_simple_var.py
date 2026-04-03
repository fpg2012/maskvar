from itertools import islice
from pathlib import Path
import json
import sys
import time
from datetime import datetime
import os

import torch
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import tqdm
from einops import rearrange, repeat

from maskvar.models.vqvae_single import VQVAE_Single
from maskvar.models.simple_ar import (
    SimpleVAR,
    simple_var_train_pass,
    simple_var_inference,
)
from maskvar.datasets import (
    MaskLevelDataset,
    MaskLevelDatasetDummy,
    MaskLevelDatasetRandom,
)
from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset
from maskvar.datasets.image_feature_cache import ImageFeatureCache
from maskvar.datasets.sharded_distributed_sampler import ShardedDistributedSampler
from maskvar.utils.clicker import init_clicks, to_sam_format
import matplotlib.pyplot as plt
import io
import torchvision


torch.set_float32_matmul_precision('high')

def save_train_configuration(args, outdir: Path):
    cur_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # create outdir
    outdir.mkdir(parents=True, exist_ok=True)
    
    # save train config
    with open(outdir / 'train_config.json', 'w') as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=2)
    
    # save train command
    with open(outdir / 'train_command.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# train start time: ' + cur_time + '\n')
        f.write(' '.join(sys.argv))
    
    # print train config
    print('train config: ')
    for k, v in args.__dict__.items():
        print(f'    {k}: {v}')
    
    # print train command
    print('train command: ')
    print(' '.join(sys.argv))
    
    # print start time
    print('train start time: ' + cur_time)

    time.sleep(2)


class SimpleARTrainer:

    def __init__(
        self,
        simple_var: SimpleVAR,
        vqvae: VQVAE_Single,
        lr: float,
        train_set: MaskLevelDataset,
        val_set: MaskLevelDataset,
        batch_size: int,
        accumulate_steps: int,
        device: str,
        log_dir: Path,
        checkpoint_dir: Path,
        skip_eval: bool = True,
        loss_weight_per_level=[1, 1, 1, 1, 1],
        dtype=torch.float32,
        opt_checkpoint: Path | None = None,
        dataloader_workers: int = 4,
        prefetch_factor: int = 2,
        shuffle_dataloader: bool = True,
        find_unused_parameters: bool = True,
        prompt_encoder=None,
        log_interval: int = 32,
    ):
        # models
        self.simple_var: SimpleVAR = simple_var
        self.vqvae: VQVAE_Single = vqvae
        self.prompt_encoder = prompt_encoder

        # optimizer
        self.optimizer = torch.optim.AdamW(simple_var.parameters(), lr=lr)

        # device
        self.device = device
        self.dtype = dtype

        # dataset
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.accumulate_steps = accumulate_steps
        self.sampler = None
        self.val_sampler = None
        
        # loss
        self.loss_function = nn.CrossEntropyLoss(reduction='none')
        if opt_checkpoint is not None:
            optimizer_state_dict = torch.load(opt_checkpoint)
            self.optimizer.load_state_dict(optimizer_state_dict)

        # loss weight
        with torch.no_grad():
            patch_num = simple_var.patch_num
            loss_weight_per_token = []
            for level, pn in enumerate(patch_num):
                loss_weight_per_token.extend(
                    [loss_weight_per_level[level] / pn**2] * pn**2
                )
            # print(f'loss weight per token: {loss_weight_per_token}')
            self.loss_weight_per_token = torch.tensor(loss_weight_per_token, device=self.device)
            self.loss_weight_per_token = F.normalize(self.loss_weight_per_token, p=1, dim=-1)

        # logger
        self.logger = SummaryWriter(log_dir=str(log_dir))
        self.output_dir = checkpoint_dir
        self.log_duration = log_interval

        self.skip_eval = skip_eval

        # torch distributed
        if tdist.is_initialized():
            self.rank = tdist.get_rank()
            self.world_size = tdist.get_world_size()
            self.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
        
        self.find_unused_parameters = find_unused_parameters

        self.compile_model()
        self.ddp_wrap()

        # dataloader
        self.dataloader_workers = dataloader_workers
        self.prefetch_factor = prefetch_factor
        self.train_dataloader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=(self.sampler is None) and shuffle_dataloader,
            sampler=self.sampler,
            drop_last=True,
            num_workers=self.dataloader_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=True
        )
        self.val_dataloader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=(self.val_sampler is None) and shuffle_dataloader,
            drop_last=True,
            sampler=self.val_sampler,
            num_workers=self.dataloader_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=True
        )
    
    def compile_model(self):
        self.simple_var.to(self.device)
        self.vqvae.to(self.device)
        self.simple_var = torch.compile(self.simple_var)
        self.vqvae = torch.compile(self.vqvae)
    
    def ddp_wrap(self):
        if self.world_size > 1:
            self.simple_var = DDP(self.simple_var, device_ids=[self.rank], find_unused_parameters=self.find_unused_parameters)
            # self.sampler = DistributedSampler(self.train_set)
            # self.val_sampler = DistributedSampler(self.val_set)
            self.sampler = ShardedDistributedSampler(self.train_set, rank=self.rank, world_size=self.world_size, shard_size=1024)
            self.val_sampler = DistributedSampler(self.val_set)

    @torch.no_grad()
    def visualize_masks_to_tensorboard(self, single_mask_normalized, gt_idx, logits, batch_clicks, writer, global_step, tag='val/mask_viz', sample_idx=0, inf_idx_list=None):
        """
        Visualize masks, predictions, and clicks from a single sample to tensorboard.

        Args:
            single_mask_normalized: (B, 1, H, W) normalized ground truth masks
            gt_idx: List of (B, l) ground truth token indices per scale
            logits: (B, L, vocab_size) model output logits (teacher forcing)
            batch_clicks: List of click lists for each sample
            writer: SummaryWriter instance
            global_step: global step for logging
            tag: tag prefix for tensorboard
            sample_idx: which sample in the batch to visualize (default: 0)
            inf_idx_list: List of (B, l) optional pure inference token indices (no teacher forcing)
        """
        B = single_mask_normalized.shape[0]
        if sample_idx >= B:
            sample_idx = 0

        # Get predicted token indices from logits (teacher forcing)
        pred_idx_flat = logits.argmax(dim=-1)  # (B, L)

        # Split flat indices into per-scale indices
        pred_idx = []
        start = 0
        for level_idx in gt_idx:
            l = level_idx.shape[1]
            pred_idx.append(pred_idx_flat[:, start:start+l])
            start += l

        # Decode predicted tokens to mask
        pred_masks = self.vqvae.idxBl_to_img(pred_idx, same_shape=False, last_one=True)  # (B, 1, H, W)
        pred_masks = (pred_masks + 1) / 2  # Denormalize: [-1, 1] -> [0, 1]

        # Decode ground truth tokens to mask (for comparison)
        gt_masks_decoded = self.vqvae.idxBl_to_img(gt_idx, same_shape=False, last_one=True)
        gt_masks_decoded = (gt_masks_decoded + 1) / 2

        # Handle inference results if provided
        if inf_idx_list is not None:
            inf_masks = self.vqvae.idxBl_to_img(inf_idx_list, same_shape=False, last_one=True)
            inf_masks = (inf_masks + 1) / 2
            ncols = 4
        else:
            ncols = 3

        # Create figure for single sample
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        if ncols == 3:
            axes = [axes[0], axes[1], axes[2]]

        # GT mask
        gt_mask_np = single_mask_normalized[sample_idx, 0].cpu().numpy()
        axes[0].imshow(gt_mask_np > 0, cmap='gray')
        axes[0].set_title('GT Mask')
        axes[0].axis('off')

        # GT reconstructed from tokens
        gt_decoded_np = gt_masks_decoded[sample_idx, 0].cpu().numpy()
        axes[1].imshow(gt_decoded_np > 0.5, cmap='gray')
        axes[1].set_title('GT from Tokens')
        axes[1].axis('off')

        # Teacher forced prediction
        pred_mask_np = pred_masks[sample_idx, 0].cpu().numpy()
        axes[2].imshow(pred_mask_np > 0.5, cmap='gray')
        axes[2].set_title('Teacher Forced')
        axes[2].axis('off')

        # Pure inference prediction (if provided)
        if inf_idx_list is not None:
            inf_mask_np = inf_masks[sample_idx, 0].cpu().numpy()
            axes[3].imshow(inf_mask_np > 0.5, cmap='gray')
            # Overlay clicks on inference prediction
            if sample_idx < len(batch_clicks) and len(batch_clicks[sample_idx]) > 0:
                for (y, x, label) in batch_clicks[sample_idx]:
                    color = 'green' if label == 1 else 'red'
                    axes[3].scatter(x, y, c=color, s=100, marker='x', linewidths=2)
            axes[3].set_title('Pure Inference')
            axes[3].axis('off')
        else:
            # Overlay clicks on teacher forced prediction
            if sample_idx < len(batch_clicks) and len(batch_clicks[sample_idx]) > 0:
                for (y, x, label) in batch_clicks[sample_idx]:
                    color = 'green' if label == 1 else 'red'
                    axes[2].scatter(x, y, c=color, s=100, marker='x', linewidths=2)

        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)

        # Convert to tensor
        from PIL import Image
        img = Image.open(buf)
        img_tensor = torchvision.transforms.ToTensor()(img)

        plt.close(fig)
        buf.close()

        # Log to tensorboard
        writer.add_image(tag, img_tensor, global_step)

    @torch.no_grad()
    def eval_iou(self, logits, gt_idx, gt_mask):
        """
        Calculate IoU between predicted mask and ground truth mask.

        Args:
            logits: (B, L, vocab_size) - model output logits
            gt_idx: List of (B, l) - ground truth token indices per scale
            gt_mask: (B, 1, H, W) - ground truth binary masks

        Returns:
            iou: (B,) - IoU score for each sample in batch
        """
        # Get predicted token indices from logits
        pred_idx_flat = logits.argmax(dim=-1)  # (B, L)

        # Split flat indices into per-scale indices
        pred_idx = []
        start = 0
        for level_idx in gt_idx:
            l = level_idx.shape[1]
            pred_idx.append(pred_idx_flat[:, start:start+l])
            start += l

        # Decode predicted tokens to mask
        pred_mask = self.vqvae.idxBl_to_img(pred_idx, same_shape=False, last_one=True)  # (B, 1, H, W)

        # Denormalize: [-1, 1] -> [0, 1]
        pred_mask = (pred_mask + 1) / 2

        # Binarize predictions
        pred_mask = (pred_mask > 0.5).float()
        gt_mask = (gt_mask > 0.5).float()

        # Calculate IoU
        intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))  # (B,)
        union = ((pred_mask + gt_mask) > 0).float().sum(dim=(1, 2, 3))  # (B,)

        # Avoid division by zero
        iou = intersection / (union + 1e-8)

        return iou

    def get_clicks_in_batch(self, single_mask, num_clicks=2):
        """
        Generate initial clicks for each sample in the batch.

        Args:
            single_mask: (B, 1, H, W) torch tensor, 0-1 binary masks
            num_clicks: number of initial clicks to generate per sample

        Returns:
            List[List[Tuple[int, int, int]]]: List of click lists, one per sample.
                Each click is (y, x, label) where label=1 for positive click.
        """
        # Convert to numpy and squeeze channel dimension: (B, 1, H, W) -> (B, H, W)
        masks_np = single_mask.squeeze(1).cpu().numpy()

        batch_clicks = []
        for mask in masks_np:
            # Generate num_clicks initial positive clicks
            click_list, _, _ = init_clicks(
                gt_mask=mask,
                num_random_clicks=num_clicks,
                random_sample=True
            )
            batch_clicks.append(click_list)

        return batch_clicks
    
    def clicks_to_prompt_embedding(self, batch_clicks):
        """
        Convert batch of clicks to SAM prompt embeddings.

        Args:
            batch_clicks: List[List[Tuple[int, int, int]]], clicks for each sample in batch
                         Each click is (y, x, label) where label=1 for positive, 0 for negative

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (sparse_embeddings, dense_embeddings)
                - sparse_embeddings: (B, N, embed_dim) - point and box embeddings
                - dense_embeddings: (B, embed_dim, H, W) - mask embeddings (no mask used here)
        """
        if self.prompt_encoder is None:
            raise ValueError("prompt_encoder is not initialized. Please pass prompt_encoder to SimpleARTrainer.")

        batch_size = len(batch_clicks)
        all_coords = []
        all_labels = []

        # Find max number of clicks in batch for padding, ensure at least 4
        max_clicks = max(max(len(clicks) for clicks in batch_clicks), 4)

        for clicks in batch_clicks:
            # Convert clicks to SAM format using to_sam_format from clicker.py
            coords, labels = to_sam_format(clicks, pad_size=max_clicks, device=self.device)
            all_coords.append(coords)
            all_labels.append(labels)

        # Stack to batch tensors: (B, N, 2) and (B, N)
        coords_batch = torch.stack(all_coords, dim=0)  # (B, N, 2)
        labels_batch = torch.stack(all_labels, dim=0)  # (B, N)

        # Pass through SAM prompt encoder
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(coords_batch, labels_batch),
                boxes=None,
                masks=None
            )

        return sparse_embeddings, dense_embeddings

    def train_step(self, inner_iter_count, image, image_embed_sam, single_mask_normalized, single_mask):
        image_embed_sam = image_embed_sam.to(self.device, non_blocking=True)
        single_mask_normalized = single_mask_normalized.to(self.device, non_blocking=True)

        # Generate initial clicks from ground truth masks (on CPU)
        batch_clicks = self.get_clicks_in_batch(single_mask, num_clicks=2)

        # Convert clicks to SAM prompt embeddings (if prompt_encoder is available)
        if self.prompt_encoder is not None:
            sparse_embeddings, dense_embeddings = self.clicks_to_prompt_embedding(batch_clicks)
            # TODO: Use prompt embeddings in the model
            # sparse_embeddings: (B, N, 256) - point embeddings
            # dense_embeddings: (B, 256, 32, 32) - dense positional encoding
        else:
            sparse_embeddings = None
            dense_embeddings = None

        gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized) # List of (B, l)
        gt_idx_flat = torch.cat(gt_idx, dim=1) # (B, L)

        with torch.autocast(self.device, dtype=self.dtype):

            logits = self.simple_var(
                idx=gt_idx,
                image_feat=image_embed_sam,
                vqvae=self.vqvae,
                sparse_embeddings=sparse_embeddings
            )

            acc = (logits.argmax(dim=-1) == gt_idx_flat).float()

            logits = rearrange(logits, 'B L C -> B C L')

            loss = self.loss_function(logits, gt_idx_flat) # (B, L)
            loss = loss * rearrange(self.loss_weight_per_token, 'L -> 1 L') # (B, L)

            loss = loss.mean()
            loss = loss / self.accumulate_steps
            loss.backward()

            if (inner_iter_count + 1) % self.accumulate_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return loss, acc, logits, batch_clicks

    def train(self, num_iters: int, outer_iter: int = 0, resume_iters: int = 0, val_iters=0):
        if num_iters <= 0:
            num_iters = len(self.train_dataloader)

        self.simple_var.train()
        
        if self.rank == 0:
            pbar = tqdm.tqdm(range(num_iters), desc="Training", total=num_iters)
        
        iters_count = 0

        while iters_count < num_iters:
            if tdist.is_initialized():
                self.sampler.set_epoch(outer_iter)
            for i, (image, image_embed_sam, single_mask_normalized, single_mask) in enumerate(self.train_dataloader):
                if iters_count >= num_iters:
                    break
                if self.rank == 0:
                    pbar.update(1)

                global_iters = iters_count + num_iters * outer_iter + resume_iters
                iters_count += 1

                loss, acc, logits, batch_clicks = self.train_step(
                    i,
                    image,
                    image_embed_sam,
                    single_mask_normalized,
                    single_mask,
                )

                if global_iters % self.log_duration == 0:
                    # Calculate IoU
                    # Reconstruct gt_idx for eval_iou
                    with torch.no_grad():
                        single_mask_normalized_iou = single_mask_normalized.to(self.device, non_blocking=True)
                        gt_idx_iou = self.vqvae.img_to_idxBl(single_mask_normalized_iou)
                        iou = self.eval_iou(logits, gt_idx_iou, single_mask_normalized_iou)
                        iou_mean = iou.mean()

                    acc_mean = acc.mean()
                    acc_sos = acc[:, 0].mean()
                    if tdist.is_initialized():
                        tdist.all_reduce(acc_mean, op=tdist.ReduceOp.AVG)
                        tdist.all_reduce(acc_sos, op=tdist.ReduceOp.AVG)
                        tdist.all_reduce(loss, op=tdist.ReduceOp.AVG)
                        tdist.all_reduce(iou_mean, op=tdist.ReduceOp.AVG)
                    loss = loss.item()
                    acc_mean = acc_mean.item()
                    acc_sos = acc_sos.item()
                    iou_mean = iou_mean.item()

                    if self.rank == 0:
                        # update loss and acc in progressive bar
                        pbar.set_postfix({'loss': f'{loss:.4f}', 'acc_mean': f'{acc_mean:.4f}', 'acc_sos': f'{acc_sos:.4f}', 'iou': f'{iou_mean:.4f}'})
                        # log to tensorboard
                        self.logger.add_scalar('train/loss', loss, global_step=global_iters)
                        self.logger.add_scalar('train/acc_mean', acc_mean, global_step=global_iters)
                        self.logger.add_scalar('train/acc_sos', acc_sos, global_step=global_iters)
                        self.logger.add_scalar('train/iou', iou_mean, global_step=global_iters)
                        # visualize masks (only first sample in batch)
                        self.visualize_masks_to_tensorboard(
                            single_mask_normalized_iou, gt_idx_iou, logits, batch_clicks,
                            self.logger, global_iters, tag='train/mask_viz', sample_idx=0
                        )
                
        if self.rank == 0:
            global_iters = (outer_iter + 1)*num_iters + resume_iters
            self.save_checkpoint(iters=global_iters)
        if tdist.is_initialized():
            tdist.barrier()
        self.eval(args.val_iters // world_size, global_step=global_iters)
    
    @torch.no_grad()
    def eval(self, num_iters: int, global_step: int = 0):
        if num_iters < 0:
            print(f'num_iters={num_iters}, skip val')
            return
        # Handle dataloader without __len__ (e.g., dummy dataset)
        try:
            dataloader_len = len(self.val_dataloader)
        except TypeError:
            # No __len__ method, use num_iters directly if > 0
            if num_iters <= 0:
                print('dataloader has no __len__ and num_iters <= 0, skip val')
                return
            dataloader_len = num_iters
        if num_iters == 0 or num_iters > dataloader_len:
            num_iters = dataloader_len
        self.simple_var.eval()

        total_loss = torch.tensor(0.0, device=self.device)
        total_acc_mean = torch.tensor(0.0, device=self.device)
        total_acc_sos = torch.tensor(0.0, device=self.device)
        total_iou = torch.tensor(0.0, device=self.device)
        if self.rank == 0:
            pbar = tqdm.tqdm(range(num_iters), desc="Val: ", total=num_iters)

        # Store first batch for visualization
        first_batch_for_viz = None

        for i, (image, image_embed_sam, single_mask_normalized, single_mask) in enumerate(self.val_dataloader):
            if self.rank == 0:
                pbar.update(1)

            if i >= num_iters:
                break
            image_embed_sam = image_embed_sam.to(self.device, non_blocking=True)
            single_mask_normalized = single_mask_normalized.to(self.device, non_blocking=True)

            # Save first batch for visualization (only on rank 0)
            if i == 0 and self.rank == 0:
                first_batch_for_viz = (image_embed_sam.clone(), single_mask_normalized.clone(), single_mask.clone())

            # Generate initial clicks from ground truth masks (consistent with training)
            batch_clicks = self.get_clicks_in_batch(single_mask, num_clicks=2)

            # Convert clicks to SAM prompt embeddings (if prompt_encoder is available)
            if self.prompt_encoder is not None:
                sparse_embeddings, _ = self.clicks_to_prompt_embedding(batch_clicks)
            else:
                sparse_embeddings = None

            gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized)
            gt_idx_flat = torch.cat(gt_idx, dim=1)

            with torch.autocast(self.device, dtype=self.dtype):
                logits = self.simple_var(
                    idx=gt_idx,
                    image_feat=image_embed_sam,
                    vqvae=self.vqvae,
                    sparse_embeddings=sparse_embeddings
                )
                
                acc = (logits.argmax(dim=-1) == gt_idx_flat).float()

                acc_mean = acc.mean()
                acc_sos = acc[:, 0].mean()

                # Calculate IoU
                iou = self.eval_iou(logits, gt_idx, single_mask_normalized)
                iou_mean = iou.mean()

                logits = rearrange(logits, 'b l c -> b c l')
                loss = self.loss_function(logits, gt_idx_flat)
                loss = loss * rearrange(self.loss_weight_per_token, 'L -> 1 L') # will be automatically broadcasted to [B, L]

                loss_mean = loss.mean()

            if self.rank == 0:
                pbar.set_postfix({'loss': f'{loss_mean:.4f}', 'acc_mean': f'{acc_mean:.4f}', 'acc_sos': f'{acc_sos:.4f}', 'iou': f'{iou_mean:.4f}'})

            total_loss += loss_mean
            total_acc_mean += acc_mean
            total_acc_sos += acc_sos
            total_iou += iou_mean

        if tdist.is_initialized():
            tdist.all_reduce(total_loss, op=tdist.ReduceOp.AVG)
            tdist.all_reduce(total_acc_mean, op=tdist.ReduceOp.AVG)
            tdist.all_reduce(total_acc_sos, op=tdist.ReduceOp.AVG)
            tdist.all_reduce(total_iou, op=tdist.ReduceOp.AVG)

        mean_loss = total_loss / num_iters
        mean_acc_mean = total_acc_mean / num_iters
        mean_acc_sos = total_acc_sos / num_iters
        mean_iou = total_iou / num_iters

        if self.rank == 0:
            self.logger.add_scalar('val/loss', mean_loss.item(), global_step=global_step)
            self.logger.add_scalar('val/acc_mean', mean_acc_mean.item(), global_step=global_step)
            self.logger.add_scalar('val/acc_sos', mean_acc_sos.item(), global_step=global_step)
            self.logger.add_scalar('val/iou', mean_iou.item(), global_step=global_step)

            # Visualize first batch
            if first_batch_for_viz is not None:
                image_embed_sam_viz, single_mask_normalized_viz, single_mask_viz = first_batch_for_viz
                batch_clicks_viz = self.get_clicks_in_batch(single_mask_viz, num_clicks=2)
                if self.prompt_encoder is not None:
                    sparse_embeddings_viz, _ = self.clicks_to_prompt_embedding(batch_clicks_viz)
                else:
                    sparse_embeddings_viz = None
                gt_idx_viz = self.vqvae.img_to_idxBl(single_mask_normalized_viz)
                with torch.autocast(self.device, dtype=self.dtype):
                    # Teacher forcing
                    logits_viz = self.simple_var(
                        idx=gt_idx_viz,
                        image_feat=image_embed_sam_viz,
                        vqvae=self.vqvae,
                        sparse_embeddings=sparse_embeddings_viz
                    )
                    # Pure inference (autoregressive generation)
                    from maskvar.models.simple_ar import simple_var_inference
                    inf_idx_list = simple_var_inference(
                        image_feat=image_embed_sam_viz,
                        simple_var=self.simple_var.module if hasattr(self.simple_var, 'module') else self.simple_var,
                        vqvae=self.vqvae,
                        sparse_embeddings=sparse_embeddings_viz
                    )
                    # Convert inf_idx_list to logits format (just for shape compatibility in visualization)
                    # Actually we need to reconstruct a tensor - but simple_var_inference returns idx
                    # So we'll handle this differently - pass inf_idx_list separately
                self.visualize_masks_to_tensorboard(
                    single_mask_normalized_viz, gt_idx_viz, logits_viz, batch_clicks_viz,
                    self.logger, global_step, tag='val/mask_viz', sample_idx=0,
                    inf_idx_list=inf_idx_list
                )

        return mean_loss, mean_acc_mean, mean_acc_sos, mean_iou
    
    def save_checkpoint(self, iters: int):
        if tdist.is_initialized():
            torch.save(self.optimizer.state_dict(), self.output_dir / f'.optimizer.{iters}.pt')
            torch.save(self.simple_var.module.state_dict(), self.output_dir / f'.simple_var.{iters}.pt')
        else:
            torch.save(self.optimizer.state_dict(), self.output_dir / f'.optimizer.{iters}.pt')
            torch.save(self.simple_var.state_dict(), self.output_dir / f'.simple_var.{iters}.pt')

def setup_dist():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        tdist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="env://")
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def cleanup():
    tdist.destroy_process_group()

if __name__ == "__main__":
    rank, local_rank, world_size = setup_dist()

    import argparse
    from maskvar.maskseg_build_everything import (
        builder_map
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str)
    # hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--outer_iters', type=int, default=2)
    parser.add_argument('--val_iters', type=int, default=0)
    parser.add_argument('--inner_iters', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulate_steps', type=int, default=1)
    # resume
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--resume_iters', type=int, default=0)
    # dataset
    parser.add_argument('--dataset', choices=['hqseg44k', 'cocolvis', 'coconut_hf'], type=str, default='hqseg44k')
    parser.add_argument('--use_dummy_dataset_for_debug', action='store_true')
    parser.add_argument('--dl_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--log_interval', type=int, default=32, help='Log to tensorboard every N iterations')
    # configs
    parser.add_argument('--simple_var', type=str, default='simple_var')
    parser.add_argument('--simple_var_init_checkpoint', type=str, default=None)
    parser.add_argument('--image_encoder', choices=['sam_vitb', 'mobile_sam'], type=str, default='mobile_sam')
    parser.add_argument('--image_encoder_checkpoint', type=str, default='ckpt/mobile_sam.pt')
    parser.add_argument('--vqvae', choices=builder_map['vqvae'].keys(), type=str, default='vqvae_single_5_stages_v1')
    parser.add_argument('--vqvae_checkpoint', type=str, default='out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth')
    parser.add_argument('--use_sam_pe', action='store_true')
    parser.add_argument('--prompt_encoder_checkpoint', type=str, default=None)
    parser.add_argument('--enable_clicks', action='store_true', help='Enable prompt encoder with click embeddings')
    parser.add_argument('--disable_find_unused_parameters', action='store_true')
    parser.add_argument('--exponential_loss_weight', action='store_true')
    # image embedding caching
    parser.add_argument('--image_feature_cache_dir', type=str, default="")
    # dtype
    parser.add_argument('--dtype', choices=['float16', 'float32', 'bfloat16'], type=str, default='float32')
    args = parser.parse_args()

    if tdist.is_initialized():
        device = f"cuda:{local_rank}"
    else:
        device = args.device
    dtype = getattr(torch, args.dtype)

    outdir = Path(args.outdir)

    if args.resume_from is not None:
        checkpoint_path = Path(args.resume_from) / "checkpoints" / f".simple_var.{args.resume_iters}.pt"
        opt_checkpoint = Path(args.resume_from) / "checkpoints" / f".optimizer.{args.resume_iters}.pt"
    else:
        checkpoint_path = Path(args.simple_var_init_checkpoint) if args.simple_var_init_checkpoint else None
        opt_checkpoint = None

    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        save_train_configuration(args, outdir)
        (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (outdir / "logs").mkdir(parents=True, exist_ok=True)

    # sam_image_encoder = build_mobile_sam_image_encoder('ckpt/mobile_sam.pt')
    if args.image_feature_cache_dir:
        print("Using image feature cache: ", args.image_feature_cache_dir)
        image_feature_cache_train = ImageFeatureCache(Path(args.image_feature_cache_dir), f"{args.dataset}_train", args.image_encoder)
        image_feature_cache_val = ImageFeatureCache(Path(args.image_feature_cache_dir), f"{args.dataset}_val", args.image_encoder)
        sam_image_encoder = None
    else:
        print("Not using image feature cache")
        raise NotImplementedError
        # image_feature_cache_train = None
        # image_feature_cache_val = None
        # sam_image_encoder = builder_map['image_encoder'][args.image_encoder](args.image_encoder_checkpoint)
        # sam_image_encoder = sam_image_encoder.to(device)
        # sam_image_encoder = torch.compile(sam_image_encoder)

    dataset_dir_map = {
        "hqseg44k": "data/sam-hq",
        "coco_lvis": "data/coco_lvis",
        "coconut_hf": "data/coconut_hf",
    }
    dataset_dir = dataset_dir_map[args.dataset]
    index_mapping_path = f'data/flat/{args.dataset}'
    # train_set, val_set = build_hqseg44k_dataset('data/sam-hq') # validate on train set
    train_set, val_set = builder_map['dataset'][args.dataset](dataset_dir)
    if args.use_dummy_dataset_for_debug:
        train_set_masklevel = MaskLevelDatasetDummy(
            dataset=train_set,
            with_image_embed=True,
            device=device,
            mask_filter_thresh=0.1,
            seed=42,
            count=5,
            image_feature_cache=image_feature_cache_train
        )
        val_set_masklevel = MaskLevelDatasetDummy(
            dataset=train_set,
            with_image_embed=True,
            device=device,
            mask_filter_thresh=0.1,
            seed=42,
            count=5,
            image_feature_cache=image_feature_cache_train
        )
    else:
        val_set_masklevel = MaskLevelFlatDataset(
            index_mapping_path=Path(index_mapping_path) / "val_index_mapping.npy",
            dataset=val_set,
            with_image_embed=True,
            image_feature_cache=image_feature_cache_val,
            mask_filter_thresh=0.1,
            dtype=torch.float32,
        )
        train_set_masklevel = MaskLevelFlatDataset(
            index_mapping_path=Path(index_mapping_path) / "train_index_mapping.npy",
            dataset=train_set,
            with_image_embed=True,
            image_feature_cache=image_feature_cache_train,
            mask_filter_thresh=0.1,
            dtype=torch.float32,
        )

    if args.use_sam_pe:
        prompt_encoder = builder_map['prompt_encoder'](args.prompt_encoder_checkpoint)
        sam_pe = prompt_encoder.get_dense_pe() # BCHW
        if args.enable_clicks:
            prompt_encoder = prompt_encoder.to(device).eval()
            for param in prompt_encoder.parameters():
                param.requires_grad = False
        else:
            prompt_encoder = None
    else:
        sam_pe = None
        prompt_encoder = None
        if args.enable_clicks:
            raise ValueError("--enable_clicks requires --use_sam_pe to be enabled")

    # simple_var = build_simple_var(simple_var_checkpoint_path=checkpoint_path, device=device)
    simple_var = builder_map['simple_var'][args.simple_var](simple_var_checkpoint_path=checkpoint_path, sam_pe=sam_pe, device=device, enable_prompt_tokens=args.enable_clicks)
    # vqvae = build_vqvae_single_5_stages_v1('out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth', require_grad=False)
    vqvae = builder_map['vqvae'][args.vqvae](vqvae_checkpoint_path=args.vqvae_checkpoint, require_grad=False).to(device)

    lr = args.lr
    batch_size = args.batch_size
    accumulate_steps = args.accumulate_steps

    local_batch_size = batch_size // world_size

    # loss weight
    n_stages = len(simple_var.patch_num)
    if args.exponential_loss_weight:
        loss_weight = [2**i for i in range(n_stages)][::-1]
    else:
        loss_weight = [1 for i in range(n_stages)]
    print(f'loss weight: {loss_weight}')

    trainer = SimpleARTrainer(
        simple_var=simple_var,
        vqvae=vqvae,
        lr=lr,
        train_set=train_set_masklevel,
        val_set=val_set_masklevel,
        batch_size=local_batch_size,
        accumulate_steps=accumulate_steps,
        device=device,
        log_dir=outdir / "logs",
        checkpoint_dir=outdir / "checkpoints",
        opt_checkpoint=opt_checkpoint,
        dtype=dtype,
        dataloader_workers=args.dl_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle_dataloader=(not args.use_dummy_dataset_for_debug),
        find_unused_parameters=(not args.disable_find_unused_parameters),
        prompt_encoder=prompt_encoder,
        loss_weight_per_level=loss_weight,
        log_interval=args.log_interval,
    )

    outer_iters = args.outer_iters
    inner_iters = args.inner_iters
    resume_iters = args.resume_iters
    for i in range(outer_iters):
        if rank == 0:
            print(f'=== outer iter {i} ===')
        trainer.train(num_iters=inner_iters // world_size, outer_iter=i, resume_iters=resume_iters, val_iters=args.val_iters // world_size)
    
    if rank == 0:
        print(f"Training finish at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Training complete. Checkpoints saved to {outdir / 'checkpoints'}")
        print(f"Logs saved to {outdir / 'logs'}")
    
    del trainer
    
    if tdist.is_initialized():
        cleanup()
