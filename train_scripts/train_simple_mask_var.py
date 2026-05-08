"""
Training script for SimpleMaskVAR.

Example usage:
    # Single GPU
    python train_scripts/train_simple_mask_var.py --out_dir out_simple_mask_var_v0 \
        --vqvae_checkpoint checkpoints/simple_mask_vqvae_multiscale_v2.pth

    # Multi-GPU DDP
    torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=11134 \
        train_scripts/train_simple_mask_var.py --out_dir out_simple_mask_var_v0 \
        --vqvae_checkpoint checkpoints/simple_mask_vqvae_multiscale_v2.pth
"""

import os
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as tdist
import torch.profiler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
from einops import rearrange

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


def load_vqvae_config_overrides(vqvae_checkpoint_path: str) -> dict:
    """Load VQVAE training config from the checkpoint output directory if available."""
    checkpoint_path = Path(vqvae_checkpoint_path)
    config_path = checkpoint_path.parent.parent / 'config.json'
    if not config_path.exists():
        return {}

    with open(config_path, 'r') as f:
        return json.load(f)


class SimpleMaskVARTrainer:
    """
    Trainer for SimpleMaskVAR.

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
        vqvae_model,
        train_dataset,
        val_dataset,
        batch_size: int,
        learning_rate: float,
        device: str,
        out_dir: Path,
        accumulate_steps: int = 1,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        dtype: torch.dtype = torch.float32,
        find_unused_parameters=True,
        train_vis_interval_mult: int = 20,
        enable_timing: bool = False,
        profiler=None,
        infer_val_batches: int = 4,
        enable_infer_iou: bool = False,
    ):
        self.model = model
        self.vqvae_model = vqvae_model
        self.device = device
        self.dtype = dtype
        self.accumulate_steps = accumulate_steps
        self.out_dir = out_dir
        self.train_vis_interval_mult = train_vis_interval_mult
        self.enable_timing = enable_timing
        self.profiler = profiler
        self.infer_val_batches = infer_val_batches
        self.enable_infer_iou = enable_infer_iou

        # Distributed training setup
        if tdist.is_initialized():
            self.rank = tdist.get_rank()
            self.world_size = tdist.get_world_size()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

        # Model to device
        try:
            first_param = next(self.model.parameters())
            if str(first_param.device) != str(self.device):
                self.model.to(self.device)
        except StopIteration:
            pass

        # VQVAE to device (frozen, for encoding)
        if self.vqvae_model is not None:
            self.vqvae_model.to(self.device)
            self.vqvae_model.eval()
            for param in self.vqvae_model.parameters():
                param.requires_grad = False
        self._get_var_model().set_vqvae_model(self.vqvae_model)

        # Wrap with DDP if using distributed training
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=False,
            )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # DataLoader
        self.train_sampler = None
        self.val_sampler = None

        is_dummy_dataset = getattr(train_dataset, 'is_dummy', False)

        if self.world_size > 1 and not is_dummy_dataset:
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

    def _new_timer(self):
        if not self.enable_timing:
            return None
        if self.device.startswith('cuda'):
            return {
                'kind': 'cuda',
                'start': torch.cuda.Event(enable_timing=True),
                'end': torch.cuda.Event(enable_timing=True),
            }
        return {'kind': 'cpu', 'start': 0.0, 'end': 0.0}

    def _timer_start(self, timer):
        if timer is None:
            return
        if timer['kind'] == 'cuda':
            timer['start'].record()
        else:
            timer['start'] = time.perf_counter()

    def _timer_end(self, timer):
        if timer is None:
            return
        if timer['kind'] == 'cuda':
            timer['end'].record()
        else:
            timer['end'] = time.perf_counter()

    def _timer_ms(self, timer):
        if timer is None:
            return None
        if timer['kind'] == 'cuda':
            torch.cuda.synchronize(device=self.device)
            return timer['start'].elapsed_time(timer['end'])
        return (timer['end'] - timer['start']) * 1000.0

    def encode_mask_to_tokens(self, mask_normalized, image):
        """
        Encode mask to token ids using VQVAE.

        Args:
            mask_normalized: (B, 1, H, W)
            image: (B, 3, H, W)

        Returns:
            token_ids: (B, L) - token indices in sequence format
        """
        with torch.no_grad():
            with torch.autocast(
                device_type='cuda',
                dtype=self.dtype,
                enabled=(self.device.startswith('cuda') and self.dtype != torch.float32),
            ):
                token_ids = self.vqvae_model.encode_to_multiscale_token_ids(mask_normalized)
                var_input = self.vqvae_model.to_var_input(token_ids)
                image_tokens = self.vqvae_model.image_encoder(image)
                image_tokens = rearrange(image_tokens, 'b c h w -> b (h w) c')

        return token_ids, var_input, image_tokens

    def _get_ar_model(self):
        return self._get_var_model()

    def _get_var_model(self):
        model = self.model.module if isinstance(self.model, DDP) else self.model
        return getattr(model, '_orig_mod', model)

    def _token_loss_and_accuracy(self, logits_by_scale, token_ids_by_scale):
        total_loss = None
        total_correct = token_ids_by_scale[0].new_tensor(0.0, dtype=torch.float32)
        total_tokens = 0
        scale_losses = {}
        scale_accs = {}
        scale_weights = {}

        total_target_tokens = sum(token_ids.numel() for token_ids in token_ids_by_scale)

        for scale_idx, (logits, token_ids) in enumerate(zip(logits_by_scale, token_ids_by_scale)):
            scale = self._get_var_model().scales[scale_idx]
            loss = F.cross_entropy(
                rearrange(logits, 'b l vocab -> (b l) vocab'),
                rearrange(token_ids, 'b l -> (b l)'),
                reduction='mean',
            )
            pred_ids = logits.argmax(dim=-1)
            correct_count = (pred_ids == token_ids).float().sum()
            acc = correct_count / token_ids.numel()

            weight = token_ids.numel() / total_target_tokens
            weighted_loss = loss * weight
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
            total_correct = total_correct + correct_count
            total_tokens += token_ids.numel()
            scale_losses[f'loss_s{scale}'] = loss.detach()
            scale_accs[f'acc_s{scale}'] = acc.detach()
            scale_weights[f'weight_s{scale}'] = token_ids.new_tensor(weight, dtype=torch.float32)

        return total_loss, total_correct / total_tokens, scale_losses, scale_accs, scale_weights

    @torch.no_grad()
    def decode_token_ids_to_mask_logits(self, token_ids, image_tokens, output_size):
        """Decode VQ token ids back to mask logits using the frozen VQVAE decoder."""
        return self.vqvae_model.decode_from_multiscale_token_ids(
            token_ids,
            image_tokens=image_tokens,
            output_size=output_size,
        )

    @torch.no_grad()
    def compute_mask_predictions(
        self,
        token_ids,
        logits,
        image_tokens,
        gt_mask,
        compute_infer: bool = False,
        compute_multiscale: bool = False,
        click_coords=None,
        click_labels=None,
    ):
        """Compute GT-token reconstruction, teacher predictions and optional pure-autoregressive predictions plus IoU."""
        gt_token_mask_logits = self.decode_token_ids_to_mask_logits(
            token_ids,
            image_tokens,
            output_size=gt_mask.shape[-2:],
        )
        gt_token_iou = calc_iou(gt_token_mask_logits, gt_mask)

        teacher_token_ids = [scale_logits.argmax(dim=-1) for scale_logits in logits]
        teacher_mask_logits = self.decode_token_ids_to_mask_logits(
            teacher_token_ids,
            image_tokens,
            output_size=gt_mask.shape[-2:],
        )
        teacher_iou = calc_iou(teacher_mask_logits, gt_mask)
        teacher_cumulative_logits = None
        teacher_cumulative_iou = None

        infer_token_ids = None
        infer_mask_logits = None
        infer_iou = None
        infer_cumulative_logits = None
        infer_cumulative_iou = None

        if compute_infer:
            infer_token_ids = self._get_var_model().autoregressive_infer(
                image_tokens,
            )
            infer_mask_logits = self.decode_token_ids_to_mask_logits(
                infer_token_ids,
                image_tokens,
                output_size=gt_mask.shape[-2:],
            )
            infer_iou = calc_iou(infer_mask_logits, gt_mask)

        if compute_multiscale:
            teacher_cumulative_logits = []
            teacher_cumulative_iou = []
            for scale_idx in range(len(teacher_token_ids)):
                cumulative_logits = self.decode_token_ids_to_mask_logits(
                    teacher_token_ids[: scale_idx + 1],
                    image_tokens,
                    output_size=gt_mask.shape[-2:],
                )
                teacher_cumulative_logits.append(cumulative_logits)
                teacher_cumulative_iou.append(calc_iou(cumulative_logits, gt_mask))

            if infer_token_ids is not None:
                infer_cumulative_logits = []
                infer_cumulative_iou = []
                for scale_idx in range(len(infer_token_ids)):
                    cumulative_logits = self.decode_token_ids_to_mask_logits(
                        infer_token_ids[: scale_idx + 1],
                        image_tokens,
                        output_size=gt_mask.shape[-2:],
                    )
                    infer_cumulative_logits.append(cumulative_logits)
                    infer_cumulative_iou.append(calc_iou(cumulative_logits, gt_mask))

        return {
            'gt_token_mask_logits': gt_token_mask_logits,
            'gt_token_iou': gt_token_iou,
            'teacher_token_ids': teacher_token_ids,
            'teacher_mask_logits': teacher_mask_logits,
            'infer_token_ids': infer_token_ids,
            'infer_mask_logits': infer_mask_logits,
            'teacher_iou': teacher_iou,
            'infer_iou': infer_iou,
            'teacher_cumulative_logits': teacher_cumulative_logits,
            'teacher_cumulative_iou': teacher_cumulative_iou,
            'infer_cumulative_logits': infer_cumulative_logits,
            'infer_cumulative_iou': infer_cumulative_iou,
        }

    def _robust_logit_limits(self, array):
        finite = np.asarray(array)[np.isfinite(array)]
        if finite.size == 0:
            return -1.0, 1.0
        lo, hi = np.percentile(finite, [1, 99])
        vmax = max(abs(float(lo)), abs(float(hi)), 1e-6)
        return -vmax, vmax

    def _visualize_prediction_samples(
        self,
        tag: str,
        vis_samples: list,
        save_name: str | None = None,
        include_infer: bool = False,
    ):
        """Visualize GT mask, GT-token reconstruction, teacher prediction and optional pure autoregressive prediction."""
        if self.writer is None or not vis_samples:
            return

        num_samples = len(vis_samples)
        num_cols = 8 if include_infer else 6
        fig, axes = plt.subplots(num_samples, num_cols, figsize=(4 * num_cols, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        color_tp = np.array([0.2, 0.6, 1.0])
        color_fp = np.array([1.0, 0.3, 0.3])
        color_fn = np.array([0.3, 0.9, 0.3])

        def build_overlay(image_np, gt_mask_np, pred_mask_np):
            overlay = image_np.copy()
            alpha = 0.35
            tp_mask = gt_mask_np & pred_mask_np
            fp_mask = (~gt_mask_np) & pred_mask_np
            fn_mask = gt_mask_np & (~pred_mask_np)

            for mask, color in ((tp_mask, color_tp), (fp_mask, color_fp), (fn_mask, color_fn)):
                if mask.any():
                    overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

            return np.clip(overlay, 0, 1)

        for row, sample in enumerate(vis_samples):
            image = restore_normalized_image(sample['image'][0]).float().cpu().numpy().transpose(1, 2, 0) / 255.0
            gt = sample['gt'][0, 0].float().cpu().numpy() > 0
            gt_token_pred = sample['gt_token_pred'][0, 0].float().cpu().numpy() > 0
            gt_token_iou = sample['gt_token_iou'][0].item()
            teacher_pred = sample['teacher_pred'][0, 0].float().cpu().numpy() > 0
            teacher_iou = sample['teacher_iou'][0].item()

            gt_token_overlay = build_overlay(image, gt, gt_token_pred)
            teacher_overlay = build_overlay(image, gt, teacher_pred)

            axes[row, 0].imshow(image)
            axes[row, 0].set_title('Image' if row == 0 else '')
            axes[row, 0].axis('off')

            axes[row, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
            axes[row, 1].set_title('GT Mask' if row == 0 else '')
            axes[row, 1].axis('off')

            axes[row, 2].imshow(gt_token_pred, cmap='gray', vmin=0, vmax=1)
            axes[row, 2].set_title(
                f'GT Token Recon (IoU={gt_token_iou:.3f})' if row == 0 else f'IoU={gt_token_iou:.3f}'
            )
            axes[row, 2].axis('off')

            axes[row, 3].imshow(gt_token_overlay)
            axes[row, 3].set_title('GT Token Overlay' if row == 0 else '')
            axes[row, 3].axis('off')

            axes[row, 4].imshow(teacher_pred, cmap='gray', vmin=0, vmax=1)
            axes[row, 4].set_title(
                f'Teacher Pred (IoU={teacher_iou:.3f})' if row == 0 else f'IoU={teacher_iou:.3f}'
            )
            axes[row, 4].axis('off')

            axes[row, 5].imshow(teacher_overlay)
            axes[row, 5].set_title('Teacher Overlay' if row == 0 else '')
            axes[row, 5].axis('off')

            if include_infer:
                infer_pred = sample['infer_pred'][0, 0].float().cpu().numpy() > 0
                infer_iou = sample['infer_iou'][0].item()
                infer_overlay = build_overlay(image, gt, infer_pred)

                axes[row, 6].imshow(infer_pred, cmap='gray', vmin=0, vmax=1)
                axes[row, 6].set_title(
                    f'Infer Pred (IoU={infer_iou:.3f})' if row == 0 else f'IoU={infer_iou:.3f}'
                )
                axes[row, 6].axis('off')

                axes[row, 7].imshow(infer_overlay)
                axes[row, 7].set_title('Infer Overlay' if row == 0 else '')
                axes[row, 7].axis('off')

        plt.tight_layout()
        self.writer.add_figure(tag, fig, self.global_step)

        if save_name is not None:
            vis_dir = self.out_dir / 'visualizations'
            vis_dir.mkdir(parents=True, exist_ok=True)
            fig_path = vis_dir / save_name
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')

        plt.close(fig)

    def _visualize_train_samples(
        self,
        image: torch.Tensor,
        gt_mask: torch.Tensor,
        gt_token_mask_logits: torch.Tensor,
        gt_token_iou: torch.Tensor,
        teacher_mask_logits: torch.Tensor,
        teacher_iou: torch.Tensor,
        num_samples: int = 2,
    ):
        vis_samples = []
        num_samples = min(num_samples, image.shape[0])

        for i in range(num_samples):
            vis_samples.append({
                'image': image[i:i + 1].detach().cpu(),
                'gt': gt_mask[i:i + 1].detach().cpu(),
                'gt_token_pred': gt_token_mask_logits[i:i + 1].detach().cpu(),
                'gt_token_iou': gt_token_iou[i:i + 1].detach().cpu(),
                'teacher_pred': teacher_mask_logits[i:i + 1].detach().cpu(),
                'teacher_iou': teacher_iou[i:i + 1].detach().cpu(),
            })

        self._visualize_prediction_samples('train/visualization', vis_samples, include_infer=False)

    def _visualize_validation(self, vis_samples: list):
        self._visualize_prediction_samples(
            'val/visualization',
            vis_samples,
            save_name=f'iter_{self.global_step}_val_vis.png',
            include_infer=self.enable_infer_iou,
        )

    def train(self, num_iters: int, outer_iter: int = 0, resume_iters: int = 0, val_iters: int = 0, log_interval: int = 10):
        """
        Main training loop based on iterations.

        Args:
            num_iters: Number of iterations for this outer_iter
            outer_iter: Current outer iteration
            resume_iters: Starting iteration (for resuming)
            val_iters: Number of validation iterations
            log_interval: Log to tensorboard every N iterations
        """
        if num_iters <= 0:
            num_iters = len(self.train_loader)

        self.model.train()

        if self.rank == 0:
            pbar = tqdm(total=num_iters, desc=f"Training outer_iter {outer_iter}")

        iters_count = 0
        self.optimizer.zero_grad(set_to_none=True)

        while iters_count < num_iters:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(outer_iter)

            data_wait_start = time.perf_counter()
            for batch in self.train_loader:
                if iters_count >= num_iters:
                    break

                data_time_ms = None
                if self.enable_timing:
                    data_time_ms = (time.perf_counter() - data_wait_start) * 1000.0

                image, _, single_mask_normalized, single_mask = batch
                image = image.to(self.device, non_blocking=True)
                single_mask_normalized = single_mask_normalized.to(self.device, non_blocking=True)
                single_mask = single_mask.to(self.device, non_blocking=True)

                # Encode to tokens using VQVAE
                encode_timer = self._new_timer()
                self._timer_start(encode_timer)
                with torch.no_grad():
                    token_ids, var_input, image_tokens = self.encode_mask_to_tokens(single_mask_normalized, image)
                    # token_ids: list[(B, scale*scale)], var_input: BLC for scales 1..K-1, image_tokens: (B, L_img, C)
                self._timer_end(encode_timer)

                # Forward pass
                step_timer = self._new_timer()
                self._timer_start(step_timer)
                with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.dtype != torch.float32):
                    logits = self.model(
                        var_input,
                        image_tokens,
                    )
                    loss, correct, scale_losses, scale_accs, scale_weights = self._token_loss_and_accuracy(logits, token_ids)
                    loss = loss / self.accumulate_steps

                # Backward pass
                loss.backward()

                iters_count += 1
                global_iters = num_iters * outer_iter + resume_iters + iters_count

                should_step = (
                    (iters_count % self.accumulate_steps == 0) or
                    (iters_count == num_iters)
                )

                # Gradient accumulation
                if should_step:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                self._timer_end(step_timer)
                if self.profiler is not None:
                    self.profiler.step()

                # Calculate global iteration count
                self.global_step = global_iters

                # Logging
                if self.rank == 0:
                    pbar.update(1)
                    if global_iters % log_interval == 0:
                        loss_val = loss.item() * self.accumulate_steps

                        # Calculate accuracy
                        with torch.no_grad():
                            vis_outputs = self.compute_mask_predictions(
                                token_ids,
                                logits,
                                image_tokens,
                                single_mask,
                                compute_infer=False,
                            )
                            teacher_iou_mean = vis_outputs['teacher_iou'].mean().item()

                        pbar.set_postfix({
                            'loss': f'{loss_val:.4f}',
                            'acc': f'{correct.item():.4f}',
                            'teacher_iou': f'{teacher_iou_mean:.4f}',
                        })

                        self.writer.add_scalar('train/loss', loss_val, global_step=global_iters)
                        self.writer.add_scalar('train/accuracy', correct.item(), global_step=global_iters)
                        self.writer.add_scalar('train/teacher_iou', teacher_iou_mean, global_step=global_iters)
                        for name, value in scale_losses.items():
                            self.writer.add_scalar(f'train/{name}', value.item(), global_step=global_iters)
                        for name, value in scale_accs.items():
                            self.writer.add_scalar(f'train/{name}', value.item(), global_step=global_iters)
                        for name, value in scale_weights.items():
                            self.writer.add_scalar(f'train/{name}', value.item(), global_step=global_iters)

                        vis_time_ms = None

                        if global_iters % (log_interval * self.train_vis_interval_mult) == 0:
                            vis_timer = self._new_timer()
                            self._timer_start(vis_timer)
                            self._timer_end(vis_timer)
                            vis_time_ms = self._timer_ms(vis_timer)

                            pbar.set_postfix({
                                'loss': f'{loss_val:.4f}',
                                'acc': f'{correct.item():.4f}',
                                'teacher_iou': f'{teacher_iou_mean:.4f}',
                            })

                            self._visualize_train_samples(
                                image,
                                single_mask,
                                vis_outputs['gt_token_mask_logits'],
                                vis_outputs['gt_token_iou'],
                                vis_outputs['teacher_mask_logits'],
                                vis_outputs['teacher_iou'],
                                num_samples=2,
                            )

                        if self.enable_timing:
                            encode_ms = self._timer_ms(encode_timer)
                            step_ms = self._timer_ms(step_timer)
                            timing_msg = (
                                f"timing step={global_iters}: "
                                f"data={data_time_ms:.1f}ms, "
                                f"encode={encode_ms:.1f}ms, "
                                f"train={step_ms:.1f}ms"
                            )
                            if vis_time_ms is not None:
                                timing_msg += f", vis={vis_time_ms:.1f}ms"
                            print(timing_msg)

                if self.enable_timing:
                    data_wait_start = time.perf_counter()

        if self.rank == 0:
            pbar.close()
            global_iters = (outer_iter + 1) * num_iters + resume_iters
            self.save_checkpoint(global_iters)

        if self.world_size > 1:
            tdist.barrier()

        self.validate(
            num_val_iters=val_iters,
            outer_iter=outer_iter,
            infer_val_batches=getattr(self, 'infer_val_batches', 4),
        )

    @torch.no_grad()
    def validate(self, num_val_iters: int = 0, outer_iter: int = 0, infer_val_batches: int = 4) -> dict:
        """Validate on validation set."""
        if num_val_iters < 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'teacher_iou': 0.0, 'infer_iou': 0.0}

        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_teacher_iou = 0.0
        total_infer_iou = 0.0
        num_batches = 0
        num_iou_samples = 0
        num_infer_iou_samples = 0
        vis_samples = []

        if num_val_iters == 0:
            try:
                num_val_iters = len(self.val_loader)
            except TypeError:
                num_val_iters = 100

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

            # Encode to tokens
            token_ids, var_input, image_tokens = self.encode_mask_to_tokens(single_mask_normalized, image)

            with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.dtype != torch.float32):
                logits = self.model(
                    var_input,
                    image_tokens,
                )
                loss, correct, _, _, _ = self._token_loss_and_accuracy(logits, token_ids)

            # Calculate accuracy
            compute_infer = self.enable_infer_iou and (iters_count < infer_val_batches)
            vis_outputs = self.compute_mask_predictions(
                token_ids,
                logits,
                image_tokens,
                single_mask,
                compute_infer=compute_infer,
            )
            teacher_iou = vis_outputs['teacher_iou']
            infer_iou = vis_outputs['infer_iou']

            total_loss += loss.item()
            total_acc += correct.item()
            total_teacher_iou += teacher_iou.sum().item()
            num_batches += 1
            num_iou_samples += teacher_iou.shape[0]
            if infer_iou is not None:
                total_infer_iou += infer_iou.sum().item()
                num_infer_iou_samples += infer_iou.shape[0]

            if self.rank == 0 and infer_iou is not None and len(vis_samples) < 8:
                remain = 8 - len(vis_samples)
                for idx in range(min(remain, image.shape[0])):
                    vis_samples.append({
                        'image': image[idx:idx + 1].detach().cpu(),
                        'gt': single_mask[idx:idx + 1].detach().cpu(),
                        'gt_token_pred': vis_outputs['gt_token_mask_logits'][idx:idx + 1].detach().cpu(),
                        'gt_token_iou': vis_outputs['gt_token_iou'][idx:idx + 1].detach().cpu(),
                        'teacher_pred': vis_outputs['teacher_mask_logits'][idx:idx + 1].detach().cpu(),
                        'infer_pred': vis_outputs['infer_mask_logits'][idx:idx + 1].detach().cpu(),
                        'teacher_iou': teacher_iou[idx:idx + 1].detach().cpu(),
                        'infer_iou': infer_iou[idx:idx + 1].detach().cpu(),
                    })

            if self.rank == 0:
                pbar.update(1)
                postfix = {
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct.item():.4f}',
                    'teacher_iou': f'{teacher_iou.mean().item():.4f}',
                }
                if infer_iou is not None:
                    postfix['infer_iou'] = f'{infer_iou.mean().item():.4f}'
                pbar.set_postfix(postfix)

            iters_count += 1

        if self.rank == 0:
            pbar.close()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
        avg_teacher_iou = total_teacher_iou / num_iou_samples if num_iou_samples > 0 else 0.0
        avg_infer_iou = total_infer_iou / num_infer_iou_samples if num_infer_iou_samples > 0 else 0.0

        if self.world_size > 1:
            metrics = torch.tensor(
                [total_loss, total_acc, total_teacher_iou, total_infer_iou, num_batches, num_iou_samples, num_infer_iou_samples],
                device=self.device,
            )
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            total_val_iters = metrics[4].item()
            avg_loss = metrics[0].item() / total_val_iters
            avg_acc = metrics[1].item() / total_val_iters
            avg_teacher_iou = metrics[2].item() / metrics[5].item()
            avg_infer_iou = metrics[3].item() / metrics[6].item() if metrics[6].item() > 0 else 0.0

        if self.rank == 0:
            if vis_samples:
                self._visualize_validation(vis_samples)
            print(
                f"\nVal outer_iter {outer_iter}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, "
                f"TeacherIoU={avg_teacher_iou:.4f}, InferIoU={avg_infer_iou:.4f}"
            )
            self.writer.add_scalar('val/loss', avg_loss, global_step=self.global_step)
            self.writer.add_scalar('val/accuracy', avg_acc, global_step=self.global_step)
            self.writer.add_scalar('val/teacher_iou', avg_teacher_iou, global_step=self.global_step)
            if self.enable_infer_iou:
                self.writer.add_scalar('val/infer_iou', avg_infer_iou, global_step=self.global_step)

        self.model.train()

        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'teacher_iou': avg_teacher_iou,
            'infer_iou': avg_infer_iou,
        }

    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return

        checkpoint_dir = self.out_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model = self.model.module if isinstance(self.model, DDP) else self.model

        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }

        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        torch.save(checkpoint, checkpoint_dir / f'iter_{step}.pth')

        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')

        print(f"Checkpoint saved to {checkpoint_dir}/iter_{step}.pth")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)

        start_iter = checkpoint.get('step', 0)

        if self.rank == 0:
            print(f"Loaded checkpoint from {checkpoint_path}, resuming from iter {start_iter}")

        return start_iter


def main():
    parser = argparse.ArgumentParser(description='Train SimpleMaskVAR')

    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')

    parser.add_argument('--outer_iters', type=int, default=10, help='Number of outer iterations')
    parser.add_argument('--inner_iters', type=int, default=1000, help='Number of inner iterations per outer_iter')
    parser.add_argument('--val_iters', type=int, default=100, help='Number of validation iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--accumulate_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--log_interval', type=int, default=10, help='Log to tensorboard every N iterations')

    parser.add_argument('--dataset', type=str, default='hqseg44k',
                        choices=['hqseg44k', 'cocolvis', 'coconut_hf'],
                        help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('--train_subset_index', type=str, default=None, help='Path to train subset indices')
    parser.add_argument('--val_subset_index', type=str, default=None, help='Path to val subset indices')

    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader prefetch factor')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float16', 'float32', 'bfloat16'],
                        help='Training dtype')

    # Model config
    parser.add_argument('--config', type=str, default='simple_mask_var',
                        help='Model config name in builder_map')
    parser.add_argument(
        '--vqvae_config',
        type=str,
        default='simple_mask_vqvae_multiscale_v2_dim384',
        choices=sorted(builder_map['simple_mask_vqvae'].keys()),
        help='SimpleMaskVqvae config name in builder_map',
    )
    parser.add_argument('--vqvae_image_encoder_checkpoint', type=str, default=None,
                        help='Image encoder checkpoint used by the VQVAE builder')
    parser.add_argument('--vqvae_image_encoder_config', type=str, default=None,
                        help='Image encoder config name used by the VQVAE builder')

    # Checkpoints
    parser.add_argument('--vqvae_checkpoint', type=str, required=True, help='Path to VQVAE checkpoint')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model init checkpoint')

    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile')
    parser.add_argument('--disable_find_unused_parameters', action='store_true')

    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--debug_iters', type=int, default=100, help='Debug iterations')
    parser.add_argument(
        '--debug_smoketest',
        action='store_true',
        help='Run a very short end-to-end smoketest that forces frequent logging and visualization',
    )
    parser.add_argument('--timing', action='store_true', help='Print lightweight timing breakdown on log steps')
    parser.add_argument('--val_infer_batches', type=int, default=4,
                        help='Number of validation batches to run pure inference IoU and visualization on')
    parser.add_argument('--enable_infer_iou', action='store_true',
                        help='Enable pure autoregressive inference IoU. Default is disabled.')
    parser.add_argument('--profile', action='store_true', help='Enable short torch profiler trace at startup')
    parser.add_argument('--profile_wait', type=int, default=1, help='Profiler wait steps')
    parser.add_argument('--profile_warmup', type=int, default=1, help='Profiler warmup steps')
    parser.add_argument('--profile_active', type=int, default=2, help='Profiler active steps')

    args = parser.parse_args()

    if args.debug_smoketest:
        args.debug = True
        args.outer_iters = 1
        args.debug_iters = min(args.debug_iters, 4)
        args.val_iters = 4
        args.log_interval = 1
        args.batch_size = min(args.batch_size, 2)
        args.num_workers = 0
        args.prefetch_factor = 2
        args.no_compile = True
        args.profile = True

    rank, world_size, local_rank = setup_distributed()

    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.dtype)

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (out_dir / 'logs').mkdir(parents=True, exist_ok=True)

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

    train_set_base, val_set_base = builder_map['dataset'][args.dataset](dataset_path)

    index_mapping_path = f'data/flat/{args.dataset}'

    if args.train_subset_index:
        train_set = MaskLevelFlatSubsetDataset(
            subset_list=Path(args.train_subset_index),
            index_mapping_path=Path(index_mapping_path) / "train_index_mapping.npy",
            dataset=train_set_base,
            with_image_embed=False,
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
            with_image_embed=False,
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
            with_image_embed=False,
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
            with_image_embed=False,
            image_feature_cache=None,
            mask_filter_thresh=0.1,
            dtype=torch.float32,
            image_size_encoder=1024,
            image_size_mask=1024,
        )

    if args.debug:
        device_for_dummy = torch.device(device)
        train_set = MaskLevelDatasetDummy(
            dataset=train_set_base,
            device=device_for_dummy,
            with_image_embed=False,
            mask_filter_thresh=0.1,
            seed=42 + rank,
            count=8,
            image_size_encoder=1024,
            image_size_mask=1024,
        )
        val_set = MaskLevelDatasetDummy(
            dataset=val_set_base,
            device=device_for_dummy,
            with_image_embed=False,
            mask_filter_thresh=0.1,
            seed=100 + rank,
            count=5,
            image_size_encoder=1024,
            image_size_mask=1024,
        )
        train_set.is_dummy = True
        val_set.is_dummy = True
        if rank == 0:
            print(f"Debug mode: using dummy dataset")

    # Build models
    # Load VQVAE for encoding
    vqvae_config_overrides = load_vqvae_config_overrides(args.vqvae_checkpoint)
    vqvae_image_encoder_checkpoint = (
        args.vqvae_image_encoder_checkpoint or
        vqvae_config_overrides.get('image_encoder_checkpoint')
    )
    vqvae_image_encoder_config = (
        args.vqvae_image_encoder_config or
        vqvae_config_overrides.get('image_encoder_config')
    )

    vqvae_builder_kwargs = {
        'simple_mask_vqvae_checkpoint_path': args.vqvae_checkpoint,
        'device': device,
    }
    if vqvae_image_encoder_checkpoint is not None:
        vqvae_builder_kwargs['image_encoder_checkpoint'] = vqvae_image_encoder_checkpoint
    if vqvae_image_encoder_config is not None:
        vqvae_builder_kwargs['image_encoder_config_name'] = vqvae_image_encoder_config

    if 'shape_only' in args.vqvae_config:
        raise ValueError(f"VQVAE config '{args.vqvae_config}' is shape-only and cannot provide image tokens.")
    if 'multiscale_v2' not in args.vqvae_config:
        raise ValueError("SimpleMaskVAR requires a multiscale V2 SimpleMaskVqvae config.")

    vqvae_model = builder_map['simple_mask_vqvae'][args.vqvae_config](
        **vqvae_builder_kwargs,
    )
    vqvae_model.eval()

    # Build VAR model using builder (hyperparameters are fixed in builder)
    checkpoint_to_use = args.checkpoint or args.resume_from
    model = builder_map['simple_mask_ar'][args.config](
        checkpoint_path=checkpoint_to_use if checkpoint_to_use and os.path.exists(checkpoint_to_use) else None,
        device=device,
        enable_click=False,
    )
    model.set_vqvae_model(vqvae_model)
    if rank == 0:
        print(f"Using config: {args.config}")
        print(f"Using VQVAE config: {args.vqvae_config}")
        if vqvae_image_encoder_config is not None:
            print(f"Using VQVAE image encoder config: {vqvae_image_encoder_config}")
        if vqvae_image_encoder_checkpoint is not None:
            print(f"Using VQVAE image encoder checkpoint: {vqvae_image_encoder_checkpoint}")
    if checkpoint_to_use and rank == 0:
        print(f"Loaded checkpoint from {checkpoint_to_use}")

    if not args.no_compile:
        model = torch.compile(model)
        if rank == 0:
            print("Applied torch.compile to model")

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Build trainer
    profiler = None
    if args.profile and rank == 0:
        profile_dir = out_dir / 'profiler'
        profile_dir.mkdir(parents=True, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=args.profile_wait,
                warmup=args.profile_warmup,
                active=args.profile_active,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()
        if rank == 0:
            print(f"Profiler enabled. Traces will be written to {profile_dir}")

    trainer = SimpleMaskVARTrainer(
        model=model,
        vqvae_model=vqvae_model,
        train_dataset=train_set,
        val_dataset=val_set,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        out_dir=out_dir,
        accumulate_steps=args.accumulate_steps,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        dtype=dtype,
        find_unused_parameters=not args.disable_find_unused_parameters,
        train_vis_interval_mult=1 if args.debug_smoketest else 20,
        enable_timing=args.timing or args.debug_smoketest,
        profiler=profiler,
        infer_val_batches=args.val_infer_batches,
        enable_infer_iou=args.enable_infer_iou,
    )

    # Resume from checkpoint
    resume_iters = 0
    if args.resume_from and os.path.exists(args.resume_from):
        resume_iters = trainer.load_checkpoint(args.resume_from)

    inner_iters = args.debug_iters if args.debug else args.inner_iters

    training_succeeded = False

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
        training_succeeded = True
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user")
    finally:
        if profiler is not None:
            profiler.stop()
        cleanup_distributed()
        if rank == 0:
            if training_succeeded:
                print(f"Training complete. Outputs saved to {out_dir}")
            else:
                print(f"Training stopped before completion. Partial outputs may be in {out_dir}")


if __name__ == '__main__':
    main()
