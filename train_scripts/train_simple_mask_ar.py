"""
Training script for SimpleMaskAR.

Example usage:
    # Single GPU
    python train_scripts/train_simple_mask_ar.py --out_dir out_simple_mask_ar_v0 \
        --vqvae_checkpoint checkpoints/simple_mask_vqvae.pth

    # Multi-GPU DDP
    torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=11134 \
        train_scripts/train_simple_mask_ar.py --out_dir out_simple_mask_ar_v0 \
        --vqvae_checkpoint checkpoints/simple_mask_vqvae.pth
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
from einops import rearrange, repeat

from maskvar.datasets.sharded_distributed_sampler import ShardedDistributedSampler
from maskvar.maskseg_build_everything import builder_map
from maskvar.datasets.mask_level_dataset import MaskLevelDatasetDummy
from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset, MaskLevelFlatSubsetDataset

# Model imports (for direct use if needed)
from maskvar.models.simple_mask_ar.simple_mask_ar import SimpleMaskAR
from maskvar.models.simple_mask_vqvae.simple_mask_vqvae import SimpleMaskVqvae

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


class SimpleMaskARTrainer:
    """
    Trainer for SimpleMaskAR.

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
    ):
        self.model = model
        self.vqvae_model = vqvae_model
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

    def encode_mask_to_tokens(self, mask_normalized, image):
        """
        Encode mask to token ids using VQVAE.

        Args:
            mask_normalized: (B, 1, H, W)
            image: (B, 3, H, W)

        Returns:
            token_ids: (B, h, w) - token indices in spatial format
        """
        with torch.no_grad():
            # Get mask tokens from encoder
            mask_tokens = self.vqvae_model.mask_encoder(mask_normalized)  # (B, C, h, w)
            image_tokens = self.vqvae_model.image_encoder(image)  # (B, C, h, w)

            # Convert to BLC format for quantization
            B, C, h, w = mask_tokens.shape
            mask_tokens_blc = rearrange(mask_tokens, 'b c h w -> b (h w) c')

            # Quantize to get token ids
            token_ids = self.vqvae_model.quant.x_to_idx(mask_tokens_blc)  # (B, h*w)
            token_ids = token_ids.view(B, h, w)  # (B, h, w)

            # Also return image tokens in spatial format (B, h, w, C)
            image_tokens_spatial = rearrange(image_tokens, 'b c h w -> b h w c')

        return token_ids, image_tokens_spatial

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

            for batch in self.train_loader:
                if iters_count >= num_iters:
                    break

                # Unpack batch
                image, _, single_mask_normalized, single_mask = batch
                image = image.to(self.device, non_blocking=True)
                single_mask_normalized = single_mask_normalized.to(self.device, non_blocking=True)

                # Encode to tokens using VQVAE
                with torch.no_grad():
                    token_ids, image_tokens = self.encode_mask_to_tokens(single_mask_normalized, image)
                    # token_ids: (B, h, w), image_tokens: (B, h, w, C)

                # Forward pass
                with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.dtype != torch.float32):
                    # Model predicts logits for all positions
                    # Input: token_ids (B, h, w), last token will be dropped internally
                    logits = self.model(token_ids, image_tokens)  # (B, h, w, vocab_size)

                    # Compute loss against the original token ids.
                    # The model internally prepends sos and drops the last input token.
                    B, h, w, vocab_size = logits.shape

                    # Flatten logits and targets
                    logits_flat = rearrange(logits, 'b h w vocab -> (b h w) vocab')
                    token_ids_flat = rearrange(token_ids, 'b h w -> (b h w)')

                    # No explicit target shift is needed here because preprocess aligns
                    # [sos, t0, ..., t(L-2)] with targets [t0, ..., t(L-1)].
                    targets = token_ids_flat

                    # Compute cross-entropy loss
                    loss = F.cross_entropy(logits_flat, targets, reduction='mean')
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

                # Calculate global iteration count
                self.global_step = global_iters

                # Logging
                if self.rank == 0:
                    pbar.update(1)
                    if global_iters % log_interval == 0:
                        loss_val = loss.item() * self.accumulate_steps

                        # Calculate accuracy
                        with torch.no_grad():
                            pred_ids = logits.argmax(dim=-1)  # (B, h, w)
                            correct = (pred_ids == token_ids).float().mean()

                        pbar.set_postfix({
                            'loss': f'{loss_val:.4f}',
                            'acc': f'{correct.item():.4f}',
                        })

                        self.writer.add_scalar('train/loss', loss_val, global_step=global_iters)
                        self.writer.add_scalar('train/accuracy', correct.item(), global_step=global_iters)

        if self.rank == 0:
            pbar.close()
            global_iters = (outer_iter + 1) * num_iters + resume_iters
            self.save_checkpoint(global_iters)

        if self.world_size > 1:
            tdist.barrier()

        self.validate(num_val_iters=val_iters, outer_iter=outer_iter)

    @torch.no_grad()
    def validate(self, num_val_iters: int = 0, outer_iter: int = 0) -> dict:
        """Validate on validation set."""
        if num_val_iters < 0:
            return {'loss': 0.0, 'accuracy': 0.0}

        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

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

            # Encode to tokens
            token_ids, image_tokens = self.encode_mask_to_tokens(single_mask_normalized, image)

            with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.dtype != torch.float32):
                logits = self.model(token_ids, image_tokens)
                B, h, w, vocab_size = logits.shape

                logits_flat = rearrange(logits, 'b h w vocab -> (b h w) vocab')
                token_ids_flat = rearrange(token_ids, 'b h w -> (b h w)')
                targets = token_ids_flat

                loss = F.cross_entropy(logits_flat, targets, reduction='mean')

            # Calculate accuracy
            pred_ids = logits.argmax(dim=-1)
            correct = (pred_ids == token_ids).float().mean()

            total_loss += loss.item()
            total_acc += correct.item()
            num_batches += 1

            if self.rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct.item():.4f}',
                })

            iters_count += 1

        if self.rank == 0:
            pbar.close()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0

        if self.world_size > 1:
            metrics = torch.tensor([total_loss, total_acc, num_batches], device=self.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            total_val_iters = metrics[2].item()
            avg_loss = metrics[0].item() / total_val_iters
            avg_acc = metrics[1].item() / total_val_iters

        if self.rank == 0:
            print(f"\nVal outer_iter {outer_iter}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
            self.writer.add_scalar('val/loss', avg_loss, global_step=self.global_step)
            self.writer.add_scalar('val/accuracy', avg_acc, global_step=self.global_step)

        self.model.train()

        return {'loss': avg_loss, 'accuracy': avg_acc}

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
    parser = argparse.ArgumentParser(description='Train SimpleMaskAR')

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
    parser.add_argument('--config', type=str, default='simple_mask_ar',
                        help='Model config name in builder_map')

    # Checkpoints
    parser.add_argument('--vqvae_checkpoint', type=str, required=True, help='Path to VQVAE checkpoint')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model init checkpoint')

    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile')
    parser.add_argument('--disable_find_unused_parameters', action='store_true')

    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--debug_iters', type=int, default=100, help='Debug iterations')

    args = parser.parse_args()

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
            count=20,
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
    vqvae_model = builder_map['simple_mask_vqvae']['simple_mask_vqvae'](
        simple_mask_vqvae_checkpoint_path=args.vqvae_checkpoint,
        device=device,
    )
    vqvae_model.eval()

    # Build AR model using builder (hyperparameters are fixed in builder)
    checkpoint_to_use = args.checkpoint or args.resume_from
    model = builder_map['simple_mask_ar'][args.config](
        checkpoint_path=checkpoint_to_use if checkpoint_to_use and os.path.exists(checkpoint_to_use) else None,
        device=device,
    )
    if rank == 0:
        print(f"Using config: {args.config}")
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
    trainer = SimpleMaskARTrainer(
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
    )

    # Resume from checkpoint
    resume_iters = 0
    if args.resume_from and os.path.exists(args.resume_from):
        resume_iters = trainer.load_checkpoint(args.resume_from)

    inner_iters = args.debug_iters if args.debug else args.inner_iters

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
