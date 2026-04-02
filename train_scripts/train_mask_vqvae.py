"""
Training script for MaskVQVAE.

Example usage:
    # Single GPU
    python train_scripts/train_mask_vqvae.py --out_dir out_mask_vqvae_v0 --num_epochs 50

    # Multi-GPU DDP
    torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=11134 \
        train_scripts/train_mask_vqvae.py --out_dir out_mask_vqvae_v0 --num_epochs 50
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from maskvar.models.mask_vqvae import MaskVQVAE
from maskvar.maskseg_build_everything import builder_map


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


class NormalizedFocalLoss(nn.Module):
    """
    Normalized Focal Loss for handling class imbalance in mask reconstruction.

    Args:
        alpha: Balance weight for positive/negative samples
        gamma: Focusing parameter
        eps: Small value for numerical stability
    """
    def __init__(self, alpha=0.5, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted values, range in [-1, 1]
            target: Target values, range in [-1, 1]
        """
        # Apply sigmoid to map [-1, 1] to [0, 1]
        pred = torch.sigmoid(pred)

        # Binarize target (threshold at 0)
        target = (target > 0).float()

        # Compute alpha weights
        alpha = torch.where(target > 0, self.alpha, 1 - self.alpha)

        # Compute pt (probability of correct prediction)
        pt = 1.0 - (pred - target).abs()

        # Compute beta (hard/easy sample weighting)
        beta = (1.0 - pt) ** self.gamma

        # Normalization factor
        scale = target.numel() / (beta.sum() + self.eps)
        scale = scale.detach()

        beta = scale * beta
        loss = -alpha * beta * (pt + self.eps).log()

        return loss.mean()


class MaskVQVAETrainer:
    """
    Trainer for MaskVQVAE.

    Supports:
    - Distributed Data Parallel (DDP)
    - Mixed precision training
    - Gradient accumulation
    - TensorBoard logging
    - Checkpoint saving/loading
    """

    def __init__(
        self,
        model: MaskVQVAE,
        train_dataset,
        val_dataset,
        batch_size: int,
        learning_rate: float,
        device: str,
        out_dir: Path,
        accumulate_steps: int = 1,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        use_focal_loss: bool = True,
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

        # Model to device
        self.model.to(self.device)

        # Wrap with DDP if using distributed training
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Loss function
        if use_focal_loss:
            self.criterion = NormalizedFocalLoss(alpha=0.5, gamma=2.0)
        else:
            self.criterion = nn.MSELoss()

        # DataLoader
        self.train_sampler = None
        self.val_sampler = None

        if self.world_size > 1:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            self.val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

        # Logger
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=str(out_dir / 'logs'))
            (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        else:
            self.writer = None

        self.global_step = 0

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        total_loss = 0.0
        total_recon_loss = 0.0
        total_vq_loss = 0.0

        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        else:
            pbar = self.train_loader

        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            # Expected: (image, image_embed_sam, single_mask_normalized, single_mask)
            # We only need image_embed_sam and single_mask_normalized
            _, image_embed_sam, single_mask_normalized, _ = batch

            image_embed_sam = image_embed_sam.to(self.device, non_blocking=True)
            single_mask_normalized = single_mask_normalized.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.dtype != torch.float32):
                rec_mask, vq_loss = self.model(
                    single_mask_normalized,
                    image_embed_sam,
                    use_image_features=True,
                )

                # Reconstruction loss
                recon_loss = self.criterion(rec_mask, single_mask_normalized)

                # Total loss
                loss = recon_loss + vq_loss
                loss = loss / self.accumulate_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Logging
            total_loss += loss.item() * self.accumulate_steps
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

            if self.rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.accumulate_steps:.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'vq': f'{vq_loss.item():.4f}',
                })

                # TensorBoard logging
                if self.global_step % 10 == 0:
                    self.writer.add_scalar('train/loss', loss.item() * self.accumulate_steps, self.global_step)
                    self.writer.add_scalar('train/recon_loss', recon_loss.item(), self.global_step)
                    self.writer.add_scalar('train/vq_loss', vq_loss.item(), self.global_step)

                self.global_step += 1

        # Average losses
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_vq_loss = total_vq_loss / num_batches

        # All-reduce for distributed training
        if self.world_size > 1:
            metrics = torch.tensor([avg_loss, avg_recon_loss, avg_vq_loss], device=self.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.AVG)
            avg_loss, avg_recon_loss, avg_vq_loss = metrics.tolist()

        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'vq_loss': avg_vq_loss,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_vq_loss = 0.0

        if self.rank == 0:
            pbar = tqdm(self.val_loader, desc=f"Val {epoch}")
        else:
            pbar = self.val_loader

        for batch in pbar:
            _, image_embed_sam, single_mask_normalized, _ = batch

            image_embed_sam = image_embed_sam.to(self.device, non_blocking=True)
            single_mask_normalized = single_mask_normalized.to(self.device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.dtype != torch.float32):
                rec_mask, vq_loss = self.model(
                    single_mask_normalized,
                    image_embed_sam,
                    use_image_features=True,
                )

                recon_loss = self.criterion(rec_mask, single_mask_normalized)
                loss = recon_loss + vq_loss

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

            if self.rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'vq': f'{vq_loss.item():.4f}',
                })

        # Average losses
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_vq_loss = total_vq_loss / num_batches

        # All-reduce for distributed training
        if self.world_size > 1:
            metrics = torch.tensor([avg_loss, avg_recon_loss, avg_vq_loss], device=self.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.AVG)
            avg_loss, avg_recon_loss, avg_vq_loss = metrics.tolist()

        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'vq_loss': avg_vq_loss,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return

        checkpoint_dir = self.out_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get model state dict (handle DDP wrapper)
        model = self.model.module if isinstance(self.model, DDP) else self.model

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }

        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')

        # Save epoch checkpoint
        torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch}.pth')

        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')

        print(f"Checkpoint saved to {checkpoint_dir}/epoch_{epoch}.pth")

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

    def train(self, num_epochs: int, start_epoch: int = 0, val_interval: int = 5):
        """
        Main training loop.

        Args:
            num_epochs: Total number of epochs to train
            start_epoch: Starting epoch (for resuming)
            val_interval: Run validation every N epochs
        """
        best_val_loss = float('inf')

        for epoch in range(start_epoch, num_epochs):
            if self.rank == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(epoch)

            if self.rank == 0:
                print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                      f"Recon: {train_metrics['recon_loss']:.4f}, "
                      f"VQ: {train_metrics['vq_loss']:.4f}")

                # Log to TensorBoard
                self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('epoch/train_recon', train_metrics['recon_loss'], epoch)
                self.writer.add_scalar('epoch/train_vq', train_metrics['vq_loss'], epoch)

            # Validation
            if (epoch + 1) % val_interval == 0:
                val_metrics = self.validate(epoch)

                if self.rank == 0:
                    print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                          f"Recon: {val_metrics['recon_loss']:.4f}, "
                          f"VQ: {val_metrics['vq_loss']:.4f}")

                    # Log to TensorBoard
                    self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
                    self.writer.add_scalar('epoch/val_recon', val_metrics['recon_loss'], epoch)
                    self.writer.add_scalar('epoch/val_vq', val_metrics['vq_loss'], epoch)

                    # Save best checkpoint
                    is_best = val_metrics['loss'] < best_val_loss
                    if is_best:
                        best_val_loss = val_metrics['loss']
                        print(f"New best validation loss: {best_val_loss:.4f}")
                else:
                    is_best = False
            else:
                is_best = False

            # Save checkpoint
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best=is_best)

            # Barrier for synchronization
            if self.world_size > 1:
                tdist.barrier()

        if self.rank == 0:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train MaskVQVAE')

    # Output
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--accumulate_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval (epochs)')

    # Model configuration
    parser.add_argument('--vocab_size', type=int, default=4096, help='Codebook size')
    parser.add_argument('--z_channels', type=int, default=32, help='Latent space channels')
    parser.add_argument('--ch', type=int, default=128, help='Base channels')
    parser.add_argument('--beta', type=float, default=0.25, help='VQ commitment loss weight')
    parser.add_argument('--v_patch_nums', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                        help='Patch numbers for each scale')
    parser.add_argument('--img_feat_dim', type=int, default=256, help='Image feature dimension')
    parser.add_argument('--transformer_dim', type=int, default=256, help='Transformer dimension')
    parser.add_argument('--transformer_depth', type=int, default=2, help='Transformer depth')
    parser.add_argument('--transformer_num_heads', type=int, default=8, help='Transformer num heads')
    parser.add_argument('--fusion_type', type=str, default='sum', choices=['sum', 'weighted'],
                        help='Multi-scale fusion type')

    # Data
    parser.add_argument('--dataset', type=str, default='hqseg44k',
                        choices=['hqseg44k', 'cocolvis', 'coconut_hf'],
                        help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path (auto-detected if None)')
    parser.add_argument('--image_feature_cache_dir', type=str, required=True,
                        help='Image feature cache directory')
    parser.add_argument('--image_encoder', type=str, default='mobile_sam',
                        choices=['sam_vitb', 'mobile_sam'],
                        help='Image encoder type (for cache naming)')

    # Training settings
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader prefetch factor')
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                        help='Use focal loss for reconstruction')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float16', 'float32', 'bfloat16'],
                        help='Training dtype')

    # Checkpoint
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--vqvae_init_checkpoint', type=str, default=None,
                        help='Initialize from pretrained VQVAE checkpoint (encoder/quantizer only)')

    # Debug
    parser.add_argument('--debug', action='store_true', help='Debug mode (small dataset)')

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

    # Import MaskLevelDataset
    from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset
    from maskvar.datasets.image_feature_cache import ImageFeatureCache

    # Image feature cache
    image_feature_cache_train = ImageFeatureCache(
        Path(args.image_feature_cache_dir),
        f"{args.dataset}_train",
        args.image_encoder,
    )
    image_feature_cache_val = ImageFeatureCache(
        Path(args.image_feature_cache_dir),
        f"{args.dataset}_val",
        args.image_encoder,
    )

    # Build base dataset
    train_set_base, val_set_base = builder_map['dataset'][args.dataset](dataset_path)

    # Build MaskLevelDataset
    index_mapping_path = f'data/flat/{args.dataset}'

    train_set = MaskLevelFlatDataset(
        index_mapping_path=Path(index_mapping_path) / "train_index_mapping.npy",
        dataset=train_set_base,
        with_image_embed=True,
        image_feature_cache=image_feature_cache_train,
        mask_filter_thresh=0.1,
        dtype=torch.float32,
    )

    val_set = MaskLevelFlatDataset(
        index_mapping_path=Path(index_mapping_path) / "val_index_mapping.npy",
        dataset=val_set_base,
        with_image_embed=True,
        image_feature_cache=image_feature_cache_val,
        mask_filter_thresh=0.1,
        dtype=torch.float32,
    )

    # Debug mode: use small subset
    if args.debug:
        from torch.utils.data import Subset
        train_set = Subset(train_set, range(min(100, len(train_set))))
        val_set = Subset(val_set, range(min(20, len(val_set))))
        if rank == 0:
            print(f"Debug mode: using {len(train_set)} train samples, {len(val_set)} val samples")

    # Build model
    model = MaskVQVAE(
        vocab_size=args.vocab_size,
        z_channels=args.z_channels,
        ch=args.ch,
        beta=args.beta,
        v_patch_nums=tuple(args.v_patch_nums),
        img_feat_dim=args.img_feat_dim,
        transformer_dim=args.transformer_dim,
        transformer_depth=args.transformer_depth,
        transformer_num_heads=args.transformer_num_heads,
        fusion_type=args.fusion_type,
        use_sam_mask_decoder=True,
        test_mode=False,
    )

    # Load pretrained VQVAE weights if specified
    if args.vqvae_init_checkpoint:
        if rank == 0:
            print(f"Initializing from pretrained VQVAE: {args.vqvae_init_checkpoint}")

        checkpoint = torch.load(args.vqvae_init_checkpoint, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']

        # Load only encoder/quantizer weights
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in checkpoint.items() if k in model_state}
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)

        if rank == 0:
            print(f"Loaded {len(pretrained_state)} parameters from pretrained VQVAE")

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Build trainer
    trainer = MaskVQVAETrainer(
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
        use_focal_loss=args.use_focal_loss,
        dtype=dtype,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume_from:
        start_epoch = trainer.load_checkpoint(args.resume_from)

    # Train
    try:
        trainer.train(
            num_epochs=args.num_epochs,
            start_epoch=start_epoch,
            val_interval=args.val_interval,
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
