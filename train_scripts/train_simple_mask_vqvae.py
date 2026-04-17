"""
Training script for SimpleMaskVqvae.

Example usage:
    # Single GPU
    python train_scripts/train_simple_mask_vqvae.py --out_dir out_simple_mask_vqvae_v0 --num_iters 10000 \
        --sam_checkpoint_path checkpoints/mobile_sam.pt

    # Multi-GPU DDP
    torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=11134 \
        train_scripts/train_simple_mask_vqvae.py --out_dir out_simple_mask_vqvae_v0 --num_iters 10000 \
        --sam_checkpoint_path checkpoints/mobile_sam.pt
"""

import os
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from maskvar.datasets.sharded_distributed_sampler import ShardedDistributedSampler
from maskvar.maskseg_build_everything import builder_map
from maskvar.datasets.mask_level_dataset import MaskLevelDatasetDummy
from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset, MaskLevelFlatSubsetDataset
from maskvar.utils.metrics import calc_iou
from maskvar.utils import restore_normalized_image

torch.set_float32_matmul_precision('high')

def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        tdist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if tdist.is_initialized():
        tdist.destroy_process_group()


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims

# copied from RITM
class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred:   (B, 1, H, W), raw logits
        target: (B, 1, H, W), values in [0, 1]
        """
        pred = pred.float()

        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )

        # p_t = p if y==1 else (1-p)
        prob = torch.sigmoid(pred)
        p_t = prob * target + (1 - prob) * (1 - target)

        # alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # focal loss
        loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DICELoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W)
            target: (B, 1, H, W) [0, 1]
        """

        pred = torch.sigmoid(pred)          # 如果是 logits
        intersection = (pred * target).sum(dim=(2, 3))   # [B, 1]
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()              # 平均 batch


class DICEFocalLoss(nn.Module):

    def __init__(self, smooth=1.0, alpha=0.25, gamma=2.0, weight_dice=2.0):
        super().__init__()
        self.dice_loss = DICELoss(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.weight_dice = weight_dice
    
    def forward(self, pred, target):
        return self.weight_dice * self.dice_loss(pred, target) + self.focal_loss(pred, target)

class DICEBCELoss(nn.Module):

    def __init__(self, smooth=1.0, weight_dice=2.0):
        super().__init__()
        self.dice_loss = DICELoss(smooth=smooth)
        self.weight_dice = weight_dice
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='mean'
        )
        return self.weight_dice * self.dice_loss(pred, target) + bce_loss

class DiceNFLoss(nn.Module):

    def __init__(self, smooth=1.0, weight_dice=1.0):
        super().__init__()
        self.dice_loss = DICELoss(smooth=smooth)
        self.focal_loss = NormalizedFocalLossSigmoid()
        self.weight_dice = weight_dice
    
    def forward(self, pred, target):
        return self.weight_dice * self.dice_loss(pred, target) + self.focal_loss(pred, target)

class SimpleMaskVqvaeTrainer:
    """
    Trainer for SimpleMaskVqvae.

    Supports:
    - Distributed Data Parallel (DDP)
    - Mixed precision training
    - Gradient accumulation
    - TensorBoard logging
    - Checkpoint saving/loading
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        batch_size: int,
        learning_rate: float,
        device: str,
        out_dir: Path,
        accumulate_steps: int = 1,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        loss: str = 'nfl',
        vq_loss_weight: float = 1.0,
        dtype: torch.dtype = torch.float32,
):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.accumulate_steps = accumulate_steps
        self.out_dir = out_dir

        # Distributed training setup
        if tdist.is_initialized():
            self.rank = tdist.get_rank()
            self.world_size = tdist.get_world_size()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

        # Model to device (avoid changing parameter layout/strides if it's already on the target device)
        try:
            first_param = next(self.model.parameters())
            if str(first_param.device) != str(self.device):
                self.model.to(self.device)
        except StopIteration:
            # Model with no parameters
            pass

        # Wrap with DDP if using distributed training
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=True,
                gradient_as_bucket_view=False,
            )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Loss function
        if loss == 'nfl':
            self.criterion = NormalizedFocalLossSigmoid()
        elif loss == 'mse':
            self.criterion = nn.MSELoss()
        elif loss == 'dice':
            self.criterion = DICELoss()
        elif loss == 'fl':
            self.criterion = FocalLoss(alpha=0.75, gamma=2.0)
        elif loss == 'dicefl':
            self.criterion = DICEFocalLoss(smooth=1.0, alpha=0.75, gamma=2.0)
        elif loss == 'dicebce':
            self.criterion = DICEBCELoss()
        elif loss == 'dicenfl':
            self.criterion = DiceNFLoss()
        else:
            raise ValueError(f"Unknown loss: {loss}")

        self.vq_loss_weight = vq_loss_weight

        # DataLoader
        self.train_sampler = None
        self.val_sampler = None

        # Check if using dummy dataset (no sampler needed)
        is_dummy_dataset = getattr(train_dataset, 'is_dummy', False)

        if self.world_size > 1 and not is_dummy_dataset:
            # self.train_sampler = DistributedSampler(
            #     train_dataset,
            #     num_replicas=self.world_size,
            #     rank=self.rank,
            #     shuffle=True,
            # )
            self.train_sampler = ShardedDistributedSampler(
                train_dataset,
                rank=self.rank,
                world_size=self.world_size,
                epoch=0,
                shard_size=1024,
                seed=42,
            )
            self.val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )

        # For IterableDataset (dummy), disable shuffle and persistent_workers
        is_iterable = isinstance(train_dataset, torch.utils.data.IterableDataset)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None) and not is_iterable,
            sampler=self.train_sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(num_workers > 0) and not is_iterable,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(num_workers > 0) and not is_iterable,
        )

        # Logger
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=str(out_dir / 'logs'))
            (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        else:
            self.writer = None

        self.global_step = 0

    def train(self, num_iters: int, outer_iter: int = 0, resume_iters: int = 0, val_iters: int = 0, log_interval: int = 10):
        """
        Main training loop based on iterations (like train_simple_var).

        Args:
            num_iters: Number of iterations for this outer_iter (inner_iters)
            outer_iter: Current outer iteration (like epoch)
            resume_iters: Starting iteration (for resuming)
            val_iters: Number of validation iterations (0 to skip)
            log_interval: Log to tensorboard every N iterations
        """
        if num_iters <= 0:
            num_iters = len(self.train_loader)

        self.model.train()

        if self.rank == 0:
            pbar = tqdm(total=num_iters, desc=f"Training outer_iter {outer_iter}")

        iters_count = 0
        best_val_loss = float('inf')

        while iters_count < num_iters:
            # Set epoch for sampler
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(outer_iter)

            for batch in self.train_loader:
                if iters_count >= num_iters:
                    break

                # Unpack batch
                image, _, single_mask_normalized, single_mask = batch
                image = image.to(self.device, non_blocking=True)
                single_mask_normalized = single_mask_normalized.to(self.device, non_blocking=True)
                single_mask = single_mask.to(self.device, non_blocking=True)

                # Forward pass
                with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.dtype != torch.float32):
                    rec_mask, vq_loss, vq_usage = self.model(single_mask_normalized, image, return_usage=True)
                    recon_loss = self.criterion(rec_mask, (single_mask > 0.5).float()).mean()
                    loss = (recon_loss + self.vq_loss_weight * vq_loss) / self.accumulate_steps

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (self.global_step + 1) % self.accumulate_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Calculate global iteration count
                global_iters = iters_count + num_iters * outer_iter + resume_iters
                self.global_step = global_iters
                iters_count += 1

                # Logging
                if self.rank == 0:
                    pbar.update(1)
                    if global_iters % log_interval == 0:
                        # Calculate metrics
                        with torch.no_grad():
                            iou = calc_iou((rec_mask > 0).float(), single_mask)
                            iou_mean = iou.mean().item()

                        loss_val = loss.item() * self.accumulate_steps
                        recon_val = recon_loss.item()
                        vq_val = vq_loss.item()
                        # vq_usage is returned as tensor to avoid graph break with torch.compile
                        # Call .item() here outside the compiled region
                        vq_usage_val = vq_usage.item() if isinstance(vq_usage, torch.Tensor) else vq_usage

                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f'{loss_val:.4f}',
                            'recon': f'{recon_val:.4f}',
                            'vq': f'{vq_val:.4f}',
                            'iou': f'{iou_mean:.4f}',
                            'usage': f'{vq_usage_val:.1f}%',
                        })

                        # Log to tensorboard
                        self.writer.add_scalar('train/loss', loss_val, global_step=global_iters)
                        self.writer.add_scalar('train/recon_loss', recon_val, global_step=global_iters)
                        self.writer.add_scalar('train/vq_loss', vq_val, global_step=global_iters)
                        self.writer.add_scalar('train/iou', iou_mean, global_step=global_iters)
                        self.writer.add_scalar('train/vq_usage_percent', vq_usage_val, global_step=global_iters)

                        # Visualize training samples periodically
                        if global_iters % (log_interval * 5) == 0:
                            self._visualize_train_samples(image, single_mask_normalized, rec_mask, rec_mask, iou, num_samples=2)

        if self.rank == 0:
            pbar.close()
            # Save checkpoint at end of outer_iter
            global_iters = (outer_iter + 1) * num_iters + resume_iters
            self.save_checkpoint(global_iters)

        # Barrier for synchronization
        if self.world_size > 1:
            tdist.barrier()

        # Run validation
        self.validate(num_val_iters=val_iters, outer_iter=outer_iter)

    @torch.no_grad()
    def validate(self, num_val_iters: int = 0, outer_iter: int = 0, num_vis_samples: int = 8) -> dict:
        """Validate on validation set.

        Args:
            num_val_iters: Number of validation iterations (0 to use full val set)
            outer_iter: Current outer iteration for logging
            num_vis_samples: Number of samples to visualize
        """
        if num_val_iters < 0:
            return {'loss': 0.0, 'recon_loss': 0.0, 'vq_loss': 0.0, 'iou': 0.0}

        self.model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        total_iou = 0.0
        num_iou_samples = 0

        # For visualization (only on rank 0)
        vis_samples = []

        # Determine number of iterations
        if num_val_iters == 0:
            try:
                num_val_iters = len(self.val_loader)
            except TypeError:
                num_val_iters = 100  # Default for iterable dataset

        iters_count = 0

        if self.rank == 0:
            pbar = tqdm(total=num_val_iters, desc=f"Val outer_iter {outer_iter}")

        for batch in self.val_loader:
            if iters_count >= num_val_iters:
                break

            image, _, single_mask_normalized, single_mask = batch

            image = image.to(self.device, non_blocking=True)
            single_mask_normalized = single_mask_normalized.to(self.device, non_blocking=True)
            single_mask = single_mask.to(self.device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.dtype != torch.float32):
                rec_mask, vq_loss = self.model(
                    single_mask_normalized,
                    image,
                )

                recon_loss = self.criterion(rec_mask, single_mask_normalized)
                loss = recon_loss + vq_loss
            loss = loss.mean()
            recon_loss = recon_loss.mean()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

            # Calculate IoU
            iou = calc_iou(rec_mask, single_mask)
            total_iou += iou.sum().item()
            num_iou_samples += iou.shape[0]

            # Collect samples for visualization
            if self.rank == 0 and len(vis_samples) < num_vis_samples:
                vis_samples.append({
                    'gt': single_mask_normalized.detach(),
                    'pred': rec_mask.detach(),
                    'pred_logits': rec_mask.detach(),  # Raw logits before sigmoid
                    'image': image.detach(),
                    'iou': iou.detach(),
                })

            if self.rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'vq': f'{vq_loss.item():.4f}',
                    'iou': f'{iou.mean().item():.4f}',
                })

            iters_count += 1

        if self.rank == 0:
            pbar.close()

        # Average losses and IoU
        avg_loss = total_loss / iters_count if iters_count > 0 else 0.0
        avg_recon_loss = total_recon_loss / iters_count if iters_count > 0 else 0.0
        avg_vq_loss = total_vq_loss / iters_count if iters_count > 0 else 0.0
        avg_iou = total_iou / num_iou_samples if num_iou_samples > 0 else 0.0

        # All-reduce for distributed training
        if self.world_size > 1:
            metrics = torch.tensor([total_loss, total_recon_loss, total_vq_loss, total_iou, num_iou_samples], device=self.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            total_val_iters = iters_count * self.world_size
            avg_loss = metrics[0].item() / total_val_iters
            avg_recon_loss = metrics[1].item() / total_val_iters
            avg_vq_loss = metrics[2].item() / total_val_iters
            avg_iou = metrics[3].item() / metrics[4].item()

        # Visualize on rank 0
        if self.rank == 0 and vis_samples:
            self._visualize_validation(vis_samples, outer_iter=outer_iter)

        # Log validation metrics
        if self.rank == 0:
            global_step = (outer_iter + 1) * num_val_iters if hasattr(self, 'train_loader') and num_val_iters > 0 else 0
            print(f"\nVal outer_iter {outer_iter}: Loss={avg_loss:.4f}, Recon={avg_recon_loss:.4f}, VQ={avg_vq_loss:.4f}, IoU={avg_iou:.4f}")
            self.writer.add_scalar('val/loss', avg_loss, global_step=global_step)
            self.writer.add_scalar('val/recon_loss', avg_recon_loss, global_step=global_step)
            self.writer.add_scalar('val/vq_loss', avg_vq_loss, global_step=global_step)
            self.writer.add_scalar('val/iou', avg_iou, global_step=global_step)

        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'vq_loss': avg_vq_loss,
            'iou': avg_iou,
        }

    def _visualize_validation(self, vis_samples: list, outer_iter: int = 0):
        """Visualize validation samples with color-coded error map and logits heatmap.

        Color scheme (overlay on original image):
        - Blue: GT and Pred agree (TP)
        - Red: False Positive (GT background, Pred foreground)
        - Green: False Negative (GT foreground, Pred background)

        Args:
            vis_samples: List of validation samples to visualize
            outer_iter: Current outer iteration for logging
        """
        num_samples = len(vis_samples)

        # Create figure: 5 columns [Image, GT, Pred Mask, Error Map Overlay, Logits Heatmap]
        num_cols = 5
        fig, axes = plt.subplots(num_samples, num_cols, figsize=(20, 4 * num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        # Define soft colors for overlay (normalized RGB)
        COLOR_TP = np.array([0.2, 0.6, 1.0])   # Soft blue for correct foreground
        COLOR_FP = np.array([1.0, 0.3, 0.3])   # Soft red for false positive
        COLOR_FN = np.array([0.3, 0.9, 0.3])   # Soft green for false negative

        for row, sample in enumerate(vis_samples):
            # Detach tensors before converting to numpy
            # Handle both (B, 1, H, W) and (B, H, W) shapes
            # Convert to float32 first to avoid BFloat16 numpy conversion error
            def get_first_mask(tensor):
                if tensor.dim() == 4:
                    return tensor[0, 0].detach().float().cpu().numpy()
                else:
                    return tensor[0].detach().float().cpu().numpy()

            gt = get_first_mask(sample['gt']) > 0
            pred = get_first_mask(sample['pred']) > 0
            pred_logits = get_first_mask(sample['pred_logits'])  # Raw logits
            iou_val = sample['iou'][0]
            iou = iou_val.item() if not iou_val.requires_grad else iou_val.detach().item()

            # Get original image and restore normalization
            if 'image' in sample:
                image = restore_normalized_image(sample['image'][0])
                image = image.detach().float().cpu().numpy().transpose(1, 2, 0)
                image = image / 255.0 if image.max() > 1 else image
            else:
                # Use gray background
                image = np.ones((*gt.shape, 3)) * 0.5

            # Create color-coded error map with soft colors
            error_map = np.zeros((*gt.shape, 3))
            tp_mask = gt & pred
            fp_mask = (~gt) & pred
            fn_mask = gt & (~pred)

            error_map[tp_mask] = COLOR_TP
            error_map[fp_mask] = COLOR_FP
            error_map[fn_mask] = COLOR_FN

            # Overlay on original image with alpha blending
            alpha = 0.35  # Transparency for error colors
            overlay = image.copy().astype(np.float32) / 255.0 if image.max() > 1 else image.copy()

            # Apply each color mask with alpha blending
            for mask, color in [(tp_mask, COLOR_TP), (fp_mask, COLOR_FP), (fn_mask, COLOR_FN)]:
                if mask.any():
                    overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

            overlay = np.clip(overlay, 0, 1)

            # Column 0: Original Image
            axes[row, 0].imshow(image)
            axes[row, 0].set_title('Image' if row == 0 else '')
            axes[row, 0].axis('off')

            # Column 1: Ground Truth
            axes[row, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
            axes[row, 1].set_title('GT' if row == 0 else '')
            axes[row, 1].axis('off')

            # Column 2: Predicted Mask (binary)
            axes[row, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[row, 2].set_title(f'Pred (IoU={iou:.3f})' if row == 0 else f'IoU={iou:.3f}')
            axes[row, 2].axis('off')

            # Column 3: Error Map Overlay on Original Image
            axes[row, 3].imshow(overlay)
            axes[row, 3].set_title('Overlay (Blue=TP, Red=FP, Green=FN)' if row == 0 else '')
            axes[row, 3].axis('off')

            # Column 4: Logits Heatmap
            # Use diverging colormap centered at 0 (since logits range roughly [-1, 1] before sigmoid)
            vmin, vmax = pred_logits.min(), pred_logits.max()
            vmax_abs = max(abs(vmin), abs(vmax))
            im = axes[row, 4].imshow(pred_logits, cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs)
            axes[row, 4].set_title('Logits Heatmap' if row == 0 else '')
            axes[row, 4].axis('off')
            # Add colorbar for the last row
            if row == num_samples - 1:
                plt.colorbar(im, ax=axes[row, 4], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save to TensorBoard
        self.writer.add_figure('val/visualization', fig, self.global_step)

        # Save to file
        vis_dir = self.out_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        fig_path = vis_dir / f'iter_{self.global_step}_val_vis.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved validation visualization to {fig_path}")

    def _visualize_train_samples(self, image: torch.Tensor, gt_mask: torch.Tensor, pred_mask: torch.Tensor, pred_logits: torch.Tensor, iou: torch.Tensor, num_samples: int = 2):
        """Visualize training samples with 5-column layout matching test script.

        Columns: [Image, GT Mask, Pred Mask, Error Overlay, Logits Heatmap]
        Error Overlay: Blue=TP, Red=FP, Green=FN (alpha blended on original image)
        """
        if self.writer is None:
            return

        # Take first num_samples from batch
        image = image[:num_samples]
        gt_mask = gt_mask[:num_samples]
        pred_mask = pred_mask[:num_samples]
        pred_logits = pred_logits[:num_samples]
        iou = iou[:num_samples]

        # Create figure: 5 columns [Image, GT, Pred Mask, Error Overlay, Logits Heatmap]
        fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        # Define soft colors for overlay (normalized RGB)
        COLOR_TP = np.array([0.2, 0.6, 1.0])   # Soft blue for correct foreground
        COLOR_FP = np.array([1.0, 0.3, 0.3])   # Soft red for false positive
        COLOR_FN = np.array([0.3, 0.9, 0.3])   # Soft green for false negative

        for i in range(num_samples):
            # Restore and display image
            img = restore_normalized_image(image[i])
            # Convert to float32 first to avoid BFloat16 numpy conversion error
            img = img.float().cpu().numpy().transpose(1, 2, 0)
            img_display = img / 255.0 if img.max() > 1 else img.copy()

            # GT mask and pred mask - detach before cpu() to avoid "requires grad" error
            # Handle both (B, 1, H, W) and (B, H, W) shapes
            # Convert to float32 first to avoid BFloat16 numpy conversion error
            gt_tensor = gt_mask[i, 0] if gt_mask.dim() == 4 else gt_mask[i]
            gt = gt_tensor.detach().float().cpu().numpy() > 0
            pred_tensor = pred_mask[i, 0] if pred_mask.dim() == 4 else pred_mask[i]
            pred = pred_tensor.detach().float().cpu().numpy() > 0
            logits_tensor = pred_logits[i, 0] if pred_logits.dim() == 4 else pred_logits[i]
            logits = logits_tensor.detach().float().cpu().numpy()

            # Create color-coded error map with soft colors
            error_map = np.zeros((*gt.shape, 3))
            tp_mask = gt & pred
            fp_mask = (~gt) & pred
            fn_mask = gt & (~pred)

            error_map[tp_mask] = COLOR_TP
            error_map[fp_mask] = COLOR_FP
            error_map[fn_mask] = COLOR_FN

            # Overlay on original image with alpha blending
            alpha = 0.35  # Transparency for error colors
            overlay = img_display.copy()

            # Apply each color mask with alpha blending
            for mask, color in [(tp_mask, COLOR_TP), (fp_mask, COLOR_FP), (fn_mask, COLOR_FN)]:
                if mask.any():
                    overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

            overlay = np.clip(overlay, 0, 1)

            # Column 0: Original Image
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Image' if i == 0 else '')
            axes[i, 0].axis('off')

            # Column 1: Ground Truth
            axes[i, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('GT Mask' if i == 0 else '')
            axes[i, 1].axis('off')

            # Column 2: Predicted Mask (binary)
            axes[i, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Pred (IoU={iou[i].item():.3f})' if i == 0 else f'IoU={iou[i].item():.3f}')
            axes[i, 2].axis('off')

            # Column 3: Error Map Overlay on Original Image
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay (Blue=TP, Red=FP, Green=FN)' if i == 0 else '')
            axes[i, 3].axis('off')

            # Column 4: Logits Heatmap
            vmin, vmax = logits.min(), logits.max()
            vmax_abs = max(abs(vmin), abs(vmax))
            im = axes[i, 4].imshow(logits, cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs)
            axes[i, 4].set_title('Logits Heatmap' if i == 0 else '')
            axes[i, 4].axis('off')
            # Add colorbar for the last row
            if i == num_samples - 1:
                plt.colorbar(im, ax=axes[i, 4], fraction=0.046, pad=0.04)

        plt.tight_layout()
        self.writer.add_figure('train/samples', fig, self.global_step)
        plt.close()

    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return

        checkpoint_dir = self.out_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get model state dict (handle DDP wrapper)
        model = self.model.module if isinstance(self.model, DDP) else self.model

        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }

        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')

        # Save step checkpoint
        torch.save(checkpoint, checkpoint_dir / f'iter_{step}.pth')

        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')

        print(f"Checkpoint saved to {checkpoint_dir}/iter_{step}.pth")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint. Returns starting epoch."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        # Get model (handle DDP wrapper)
        model = self.model.module if isinstance(self.model, DDP) else self.model

        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)

        start_epoch = checkpoint.get('epoch', 0)

        if self.rank == 0:
            print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")

        return start_epoch


def main():
    parser = argparse.ArgumentParser(description='Train SimpleMaskVqvae')

    # Output
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')

    # Training hyperparameters
    parser.add_argument('--outer_iters', type=int, default=10, help='Number of outer iterations (like epochs)')
    parser.add_argument('--inner_iters', type=int, default=1000, help='Number of inner iterations per outer_iter')
    parser.add_argument('--val_iters', type=int, default=100, help='Number of validation iterations (0 for full val set)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--accumulate_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--log_interval', type=int, default=10, help='Log to tensorboard every N iterations')

    # Data
    parser.add_argument('--dataset', type=str, default='hqseg44k',
                        choices=['hqseg44k', 'cocolvis', 'coconut_hf'],
                        help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path (auto-detected if None)')
    parser.add_argument('--train_subset_index', type=str, default=None, help='Path to train subset indices (.npy file)')
    parser.add_argument('--val_subset_index', type=str, default=None, help='Path to val subset indices (.npy file)')

    # Training settings
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader prefetch factor')
    parser.add_argument('--loss', default='nfl', type=str,
                        help='loss')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float16', 'float32', 'bfloat16'],
                        help='Training dtype')
    parser.add_argument('--enable_vq', action='store_true', help='Enable VQ training')
    parser.add_argument('--vq_loss_weight', type=float, default=1.0, help='Weight for VQ loss')

    # Checkpoint
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='model init checkpoint path, will override --resume_from')
    parser.add_argument('--image_encoder_checkpoint', type=str, default=None,
                        help='Path to SAM/MobileSAM checkpoint for initializing encoders')
    parser.add_argument('--config', type=str, default='simple_mask_vqvae')
    parser.add_argument('--image_encoder_config', type=str, default='mobile_sam')

    # Optimization
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile for model acceleration')

    # Debug
    parser.add_argument('--debug', action='store_true', help='Debug mode (use dummy dataset)')
    parser.add_argument('--debug_iters', type=int, default=100, help='Number of iterations for debug mode')

    parser.add_argument('--freeze_image_encoder', action='store_true')
    parser.add_argument('--freeze_mask_encoder', action='store_true')

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()

    # Device and dtype
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.dtype)

    # Output directory
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (out_dir / 'logs').mkdir(parents=True, exist_ok=True)

        # Save config
        with open(out_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

        print(f"Training configuration:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        print(f"World size: {world_size}, Rank: {rank}")

    # Build dataset
    dataset_path_map = {
        'hqseg44k': 'data/sam-hq',
        'cocolvis': 'data/coco_lvis',
        'coconut_hf': 'data/coconut_hf',
    }
    dataset_path = args.dataset_path or dataset_path_map[args.dataset]

    # Build base dataset
    train_set_base, val_set_base = builder_map['dataset'][args.dataset](dataset_path)

    # Build MaskLevelDataset
    index_mapping_path = f'data/flat/{args.dataset}'

    if args.train_subset_index:
        train_set = MaskLevelFlatSubsetDataset(
            subset_list=Path(args.train_subset_index),
            index_mapping_path=Path(index_mapping_path) / "train_index_mapping.npy",
            dataset=train_set_base,
            with_image_embed=False,  # SimpleMaskVqvae encodes images on-the-fly
            image_feature_cache=None,
            mask_filter_thresh=0.1,
            dtype=torch.float32,
            image_size_encoder=1024,
            image_size_mask=1024,
        )
    else:
        train_set = MaskLevelFlatDataset(
            index_mapping_path=Path(index_mapping_path) / "train_index_mapping.npy",
            dataset=train_set_base,
            with_image_embed=False,  # SimpleMaskVqvae encodes images on-the-fly
            image_feature_cache=None,
            mask_filter_thresh=0.1,
            dtype=torch.float32,
            image_size_encoder=1024,
            image_size_mask=1024,
        )

    if args.val_subset_index:
        val_set = MaskLevelFlatSubsetDataset(
            subset_list=Path(args.val_subset_index),
            index_mapping_path=Path(index_mapping_path) / "val_index_mapping.npy",
            dataset=val_set_base,
            with_image_embed=False,  # SimpleMaskVqvae encodes images on-the-fly
            image_feature_cache=None,
            mask_filter_thresh=0.1,
            dtype=torch.float32,
            image_size_encoder=1024,
            image_size_mask=1024,
        )
    else:
        val_set = MaskLevelFlatDataset(
            index_mapping_path=Path(index_mapping_path) / "val_index_mapping.npy",
            dataset=val_set_base,
            with_image_embed=False,  # SimpleMaskVqvae encodes images on-the-fly
            image_feature_cache=None,
            mask_filter_thresh=0.1,
            dtype=torch.float32,
            image_size_encoder=1024,
            image_size_mask=1024,
        )

    # Debug mode: use dummy dataset with fixed samples
    if args.debug:
        device_for_dummy = torch.device(device)
        # Use different seed for each rank to ensure different data
        train_set = MaskLevelDatasetDummy(
            dataset=train_set_base,
            device=device_for_dummy,
            with_image_embed=False,
            mask_filter_thresh=0.1,
            seed=42 + rank,  # Different seed for each rank
            count=20,
            image_size_encoder=1024,
            image_size_mask=1024,
        )
        val_set = MaskLevelDatasetDummy(
            dataset=val_set_base,
            device=device_for_dummy,
            with_image_embed=False,
            mask_filter_thresh=0.1,
            seed=100 + rank,  # Different seed for each rank
            count=5,
            image_size_encoder=1024,
            image_size_mask=1024,
        )
        # Mark as dummy dataset to disable sampler in trainer
        train_set.is_dummy = True
        val_set.is_dummy = True
        if rank == 0:
            print(f"Debug mode: using dummy dataset with different seeds per rank")

    # Build model using builder
    print(f'Using config: {args.config}')
    print(f'Using image_encoder: {args.image_encoder_config}')

    checkpoint_to_use = args.checkpoint
    if checkpoint_to_use is None:
        checkpoint_to_use = args.resume_from

    print(f'Loading checkpoint: {checkpoint_to_use}')

    model = builder_map['simple_mask_vqvae'][args.config](
        simple_mask_vqvae_checkpoint_path=checkpoint_to_use,
        image_encoder_checkpoint=args.image_encoder_checkpoint,
        image_encoder_config_name=args.image_encoder_config,
        enable_vq=args.enable_vq,
        device=device,
    )

    # Freeze image_encoder parameters (mask_encoder remains trainable)
    if args.freeze_image_encoder:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("Frozen image_encoder parameters (mask_encoder is trainable)")
    
    if args.freeze_mask_encoder:
        for param in model.mask_encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("Frozen mask_encoder parameters (image_encoder is trainable)")

    # Apply torch.compile by default (can be disabled with --no_compile)
    if not args.no_compile:
        model = torch.compile(model)
        if rank == 0:
            print("Applied torch.compile to model")

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable, {frozen_params:,} frozen")

    # Build trainer
    trainer = SimpleMaskVqvaeTrainer(
        model=model,
        train_dataset=train_set,
        val_dataset=val_set,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        out_dir=out_dir,
        accumulate_steps=args.accumulate_steps,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        loss=args.loss,
        dtype=dtype,
    )

    # Resume from checkpoint (already handled by builder, but trainer needs to load optimizer state)
    resume_iters = 0
    if args.resume_from and os.path.exists(args.resume_from):
        resume_iters = trainer.load_checkpoint(args.resume_from)

    # Determine inner iters (debug mode uses debug_iters)
    inner_iters = args.debug_iters if args.debug else args.inner_iters

    # Train with outer_iters loop (like train_simple_var.py)
    try:
        for i in range(args.outer_iters):
            if rank == 0:
                print(f"\n{'='*50}")
                print(f"Outer iteration {i+1}/{args.outer_iters}")
                print(f"{'='*50}")

            trainer.train(
                num_iters=inner_iters // world_size,
                outer_iter=i,
                resume_iters=resume_iters,
                val_iters=args.val_iters // world_size if args.val_iters > 0 else 0,
                log_interval=args.log_interval,
            )
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user")
    finally:
        cleanup_distributed()

        if rank == 0:
            print(f"Training complete. Outputs saved to {out_dir}")


if __name__ == '__main__':
    main()
