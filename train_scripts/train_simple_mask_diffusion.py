"""
Train SimpleMaskLatentDiT on frozen SimpleMaskVAEV2 latents.

The command-line and dataset defaults mirror train_simple_mask_ar.py.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from maskvar.datasets.mask_level_dataset import MaskLevelDatasetDummy, MaskLevelFlatDataset, MaskLevelFlatSubsetDataset
from maskvar.datasets.sharded_distributed_sampler import ShardedDistributedSampler
from maskvar.maskseg_build_everything import builder_map
from maskvar.utils import restore_normalized_image
from maskvar.utils.metrics import calc_iou

torch.set_float32_matmul_precision("high")


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        tdist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if tdist.is_initialized():
        tdist.destroy_process_group()


def load_vae_config_overrides(vae_checkpoint: str) -> dict:
    config_path = Path(vae_checkpoint).parent.parent / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def build_flat_datasets(args, rank):
    dataset_path_map = {
        "hqseg44k": "data/sam-hq",
        "cocolvis": "data/coco_lvis",
        "coconut_hf": "data/coconut_hf",
    }
    dataset_path = args.dataset_path or dataset_path_map[args.dataset]
    train_set_base, val_set_base = builder_map["dataset"][args.dataset](dataset_path)
    index_mapping_path = Path(f"data/flat/{args.dataset}")
    dataset_kwargs = dict(
        with_image_embed=False,
        image_feature_cache=None,
        mask_filter_thresh=0.1,
        dtype=torch.float32,
        image_size_encoder=1024,
        image_size_mask=1024,
    )
    train_cls = MaskLevelFlatSubsetDataset if args.train_subset_index else MaskLevelFlatDataset
    val_cls = MaskLevelFlatSubsetDataset if args.val_subset_index else MaskLevelFlatDataset
    train_kwargs = dict(index_mapping_path=index_mapping_path / "train_index_mapping.npy", dataset=train_set_base, **dataset_kwargs)
    val_kwargs = dict(index_mapping_path=index_mapping_path / "val_index_mapping.npy", dataset=val_set_base, **dataset_kwargs)
    if args.train_subset_index:
        train_kwargs["subset_list"] = Path(args.train_subset_index)
    if args.val_subset_index:
        val_kwargs["subset_list"] = Path(args.val_subset_index)
    train_set = train_cls(**train_kwargs)
    val_set = val_cls(**val_kwargs)

    if args.debug:
        device_for_dummy = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    return train_set, val_set


class SimpleMaskDiffusionTrainer:
    def __init__(
        self,
        model,
        vae_model,
        train_dataset,
        val_dataset,
        batch_size,
        learning_rate,
        device,
        out_dir,
        accumulate_steps=1,
        num_workers=4,
        prefetch_factor=2,
        dtype=torch.float32,
        find_unused_parameters=True,
        sample_val_batches=2,
        sample_steps=50,
    ):
        self.model = model
        self.vae_model = vae_model
        self.device = device
        self.out_dir = out_dir
        self.accumulate_steps = accumulate_steps
        self.dtype = dtype
        self.sample_val_batches = sample_val_batches
        self.sample_steps = sample_steps
        self.global_step = 0

        if tdist.is_initialized():
            self.rank = tdist.get_rank()
            self.world_size = tdist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

        self.vae_model.to(device).eval()
        for p in self.vae_model.parameters():
            p.requires_grad = False
        self.model.to(device)
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=False,
            )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)

        self.train_sampler = None
        self.val_sampler = None
        is_dummy = getattr(train_dataset, "is_dummy", False)
        if self.world_size > 1 and not is_dummy:
            self.train_sampler = ShardedDistributedSampler(train_dataset, rank=self.rank, world_size=self.world_size, epoch=0, shard_size=1024, seed=42)
            self.val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
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
            drop_last=False,
            persistent_workers=(num_workers > 0) and not is_iterable,
        )

        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=str(out_dir / "logs"))
            (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (out_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        else:
            self.writer = None

    def _get_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    @torch.no_grad()
    def encode_batch(self, mask_normalized, image):
        with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.device.startswith("cuda") and self.dtype != torch.float32):
            image_tokens = self.vae_model.image_encoder(image)
            mu, logvar = self.vae_model.encode(mask_normalized)
        return mu.float(), image_tokens

    @torch.no_grad()
    def decode_latents(self, z, image_tokens, output_size):
        with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.device.startswith("cuda") and self.dtype != torch.float32):
            return self.vae_model.decode(z.to(image_tokens.dtype), image_tokens=image_tokens, output_size=output_size)

    def train(self, num_iters, outer_iter=0, resume_iters=0, val_iters=0, log_interval=10):
        if num_iters <= 0:
            num_iters = len(self.train_loader)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        iters_count = 0
        if self.rank == 0:
            pbar = tqdm(total=num_iters, desc=f"Training outer_iter {outer_iter}")

        while iters_count < num_iters:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(outer_iter)
            for batch in self.train_loader:
                if iters_count >= num_iters:
                    break
                image, _, mask_normalized, mask = batch
                image = image.to(self.device, non_blocking=True)
                mask_normalized = mask_normalized.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                z0, image_tokens = self.encode_batch(mask_normalized, image)

                with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.device.startswith("cuda") and self.dtype != torch.float32):
                    loss, pred_noise, noise, timesteps = self.model(z0, image_tokens)
                    loss = loss / self.accumulate_steps
                loss.backward()

                iters_count += 1
                global_iters = num_iters * outer_iter + resume_iters + iters_count
                self.global_step = global_iters
                if (iters_count % self.accumulate_steps == 0) or (iters_count == num_iters):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.rank == 0:
                    pbar.update(1)
                    if global_iters % log_interval == 0:
                        loss_val = loss.item() * self.accumulate_steps
                        pbar.set_postfix({"loss": f"{loss_val:.4f}", "t": f"{timesteps.float().mean().item():.1f}"})
                        self.writer.add_scalar("train/loss", loss_val, global_iters)
                        self.writer.add_scalar("train/timestep_mean", timesteps.float().mean().item(), global_iters)
                        if global_iters % (log_interval * 5) == 0:
                            self._visualize_train_samples(image, mask, z0, image_tokens)

        if self.rank == 0:
            pbar.close()
            self.save_checkpoint((outer_iter + 1) * num_iters + resume_iters)
        if self.world_size > 1:
            tdist.barrier()
        self.validate(val_iters, outer_iter)

    @torch.no_grad()
    def validate(self, num_val_iters=0, outer_iter=0):
        if num_val_iters < 0:
            return {}
        self.model.eval()
        if num_val_iters == 0:
            num_val_iters = len(self.val_loader)
        total_loss = 0.0
        count = 0
        sample_iou_sum = 0.0
        sample_iou_count = 0
        vis_sample = None
        if self.rank == 0:
            pbar = tqdm(total=num_val_iters, desc=f"Val outer_iter {outer_iter}")

        for batch in self.val_loader:
            if count >= num_val_iters:
                break
            image, _, mask_normalized, mask = batch
            image = image.to(self.device, non_blocking=True)
            mask_normalized = mask_normalized.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            z0, image_tokens = self.encode_batch(mask_normalized, image)
            loss, _, _, _ = self.model(z0, image_tokens)
            total_loss += loss.item()

            sample_logits = None
            sample_iou = None
            teacher_logits = None
            teacher_iou = None
            if count < self.sample_val_batches:
                teacher_logits = self.decode_latents(z0, image_tokens, mask.shape[-2:])
                teacher_iou = calc_iou(teacher_logits, mask)
                z_sample = self._get_model().sample(image_tokens, shape=z0.shape, num_steps=self.sample_steps)
                sample_logits = self.decode_latents(z_sample, image_tokens, mask.shape[-2:])
                sample_iou = calc_iou(sample_logits, mask)
                sample_iou_sum += sample_iou.sum().item()
                sample_iou_count += sample_iou.shape[0]
                if vis_sample is None and self.rank == 0:
                    vis_sample = (
                        image.detach(),
                        mask.detach(),
                        teacher_logits.detach(),
                        teacher_iou.detach(),
                        sample_logits.detach(),
                        sample_iou.detach(),
                    )

            count += 1
            if self.rank == 0:
                pbar.update(1)
                postfix = {"loss": f"{loss.item():.4f}"}
                if sample_iou is not None:
                    postfix["sample_iou"] = f"{sample_iou.mean().item():.4f}"
                pbar.set_postfix(postfix)

        if self.rank == 0:
            pbar.close()
        metrics = torch.tensor([total_loss, count, sample_iou_sum, sample_iou_count], device=self.device)
        if self.world_size > 1:
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
        avg_loss = metrics[0].item() / max(metrics[1].item(), 1)
        avg_sample_iou = metrics[2].item() / max(metrics[3].item(), 1)
        if self.rank == 0:
            if vis_sample is not None:
                self._visualize(*vis_sample, save_name=f"iter_{self.global_step}_val_vis.png")
            print(f"\nVal outer_iter {outer_iter}: Loss={avg_loss:.4f}, SampleIoU={avg_sample_iou:.4f}")
            self.writer.add_scalar("val/loss", avg_loss, self.global_step)
            self.writer.add_scalar("val/sample_iou", avg_sample_iou, self.global_step)
        self.model.train()
        return {"loss": avg_loss, "sample_iou": avg_sample_iou}

    def _visualize(self, image, gt_mask, teacher_logits, teacher_iou, sample_logits, sample_iou, tag="val/visualization", save_name=None, num_samples=4):
        if self.writer is None:
            return
        n = min(num_samples, image.shape[0])
        fig, axes = plt.subplots(n, 6, figsize=(24, 4 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
        for r in range(n):
            img = restore_normalized_image(image[r]).float().cpu().numpy().transpose(1, 2, 0)
            img = img / 255.0 if img.max() > 1 else img
            gt = gt_mask[r, 0].float().cpu().numpy() > 0
            teacher = teacher_logits[r, 0].float().cpu().numpy() > 0
            sample = sample_logits[r, 0].float().cpu().numpy() > 0
            axes[r, 0].imshow(img); axes[r, 0].set_title("Image"); axes[r, 0].axis("off")
            axes[r, 1].imshow(gt, cmap="gray", vmin=0, vmax=1); axes[r, 1].set_title("GT"); axes[r, 1].axis("off")
            axes[r, 2].imshow(teacher, cmap="gray", vmin=0, vmax=1); axes[r, 2].set_title(f"VAE Recon IoU={teacher_iou[r].item():.3f}"); axes[r, 2].axis("off")
            axes[r, 3].imshow(sample, cmap="gray", vmin=0, vmax=1); axes[r, 3].set_title(f"DiT Sample IoU={sample_iou[r].item():.3f}"); axes[r, 3].axis("off")
            axes[r, 4].imshow(teacher_logits[r, 0].float().cpu().numpy(), cmap="RdBu_r"); axes[r, 4].set_title("VAE Logits"); axes[r, 4].axis("off")
            axes[r, 5].imshow(sample_logits[r, 0].float().cpu().numpy(), cmap="RdBu_r"); axes[r, 5].set_title("Sample Logits"); axes[r, 5].axis("off")
        plt.tight_layout()
        self.writer.add_figure(tag, fig, self.global_step)
        if save_name is not None:
            fig.savefig(self.out_dir / "visualizations" / save_name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @torch.no_grad()
    def _visualize_train_samples(self, image, gt_mask, z0, image_tokens, num_samples=2):
        if self.writer is None:
            return

        n = min(num_samples, image.shape[0])
        image_vis = image[:n].detach()
        gt_vis = gt_mask[:n].detach()
        z0_vis = z0[:n].detach()
        image_tokens_vis = image_tokens[:n].detach()

        teacher_logits = self.decode_latents(z0_vis, image_tokens_vis, gt_vis.shape[-2:])
        teacher_iou = calc_iou(teacher_logits, gt_vis)
        z_sample = self._get_model().sample(image_tokens_vis, shape=z0_vis.shape, num_steps=self.sample_steps)
        sample_logits = self.decode_latents(z_sample, image_tokens_vis, gt_vis.shape[-2:])
        sample_iou = calc_iou(sample_logits, gt_vis)

        self.writer.add_scalar("train/teacher_iou", teacher_iou.mean().item(), self.global_step)
        self.writer.add_scalar("train/sample_iou", sample_iou.mean().item(), self.global_step)
        self._visualize(
            image_vis,
            gt_vis,
            teacher_logits.detach(),
            teacher_iou.detach(),
            sample_logits.detach(),
            sample_iou.detach(),
            tag="train/visualization",
            save_name=f"iter_{self.global_step}_train_vis.png",
            num_samples=n,
        )

    def save_checkpoint(self, step):
        if self.rank != 0:
            return
        checkpoint = {
            "step": step,
            "model_state_dict": self._get_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        ckpt_dir = self.out_dir / "checkpoints"
        torch.save(checkpoint, ckpt_dir / "latest.pth")
        torch.save(checkpoint, ckpt_dir / f"iter_{step}.pth")
        print(f"Checkpoint saved to {ckpt_dir}/iter_{step}.pth")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self._get_model().load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        return checkpoint.get("step", 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimpleMaskLatentDiT")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--outer_iters", type=int, default=10)
    parser.add_argument("--inner_iters", type=int, default=1000)
    parser.add_argument("--val_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="coconut_hf", choices=["hqseg44k", "cocolvis", "coconut_hf"])
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_subset_index", type=str, default=None)
    parser.add_argument("--val_subset_index", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--config", type=str, default="simple_mask_latent_dit", choices=sorted(builder_map["simple_mask_diffusion"].keys()))
    parser.add_argument("--vae_config", type=str, default="simple_mask_vae_v2_dim384", choices=sorted(builder_map["simple_mask_vae"].keys()))
    parser.add_argument("--vae_checkpoint", type=str, required=True)
    parser.add_argument("--vae_image_encoder_checkpoint", type=str, default=None)
    parser.add_argument("--vae_image_encoder_config", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sample_val_batches", type=int, default=2)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--disable_find_unused_parameters", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_iters", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        print("Training configuration:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        print(f"World size: {world_size}, Rank: {rank}")

    train_set, val_set = build_flat_datasets(args, rank)

    vae_overrides = load_vae_config_overrides(args.vae_checkpoint)
    vae_image_encoder_checkpoint = args.vae_image_encoder_checkpoint or vae_overrides.get("image_encoder_checkpoint")
    vae_image_encoder_config = args.vae_image_encoder_config or vae_overrides.get("image_encoder_config")
    vae_model = builder_map["simple_mask_vae"][args.vae_config](
        checkpoint_path=args.vae_checkpoint,
        image_encoder_checkpoint=vae_image_encoder_checkpoint,
        image_encoder_config_name=vae_image_encoder_config or "dino_v3_vits",
        device=device,
    )
    checkpoint_to_use = args.checkpoint or args.resume_from
    model = builder_map["simple_mask_diffusion"][args.config](
        checkpoint_path=checkpoint_to_use if checkpoint_to_use and os.path.exists(checkpoint_to_use) else None,
        device=device,
    )
    if not args.no_compile:
        model = torch.compile(model)
    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total:,} total, {trainable:,} trainable")

    trainer = SimpleMaskDiffusionTrainer(
        model=model,
        vae_model=vae_model,
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
        sample_val_batches=args.sample_val_batches,
        sample_steps=args.sample_steps,
    )
    resume_iters = 0
    if args.resume_from and os.path.exists(args.resume_from):
        resume_iters = trainer.load_checkpoint(args.resume_from)
    inner_iters = args.debug_iters if args.debug else args.inner_iters
    try:
        for i in range(args.outer_iters):
            if rank == 0:
                print(f"\n{'='*50}\nOuter iteration {i+1}/{args.outer_iters}\n{'='*50}")
            trainer.train(inner_iters // world_size, i, resume_iters, args.val_iters // world_size if args.val_iters > 0 else 0, args.log_interval)
    finally:
        cleanup_distributed()
        if rank == 0:
            print(f"Training complete. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
