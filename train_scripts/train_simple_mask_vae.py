"""
Train SimpleMaskVAEV2.

The command-line and dataset defaults mirror train_simple_mask_vqvae.py.
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
import torch.nn as nn
import torch.nn.functional as F
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


class DiceNFLoss(nn.Module):
    def __init__(self, smooth=1.0, weight_dice=1.0):
        super().__init__()
        self.smooth = smooth
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        target = (target > 0.5).float()
        pred_prob = torch.sigmoid(pred.float())
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - ((2 * intersection + self.smooth) / (union + self.smooth)).mean()
        bce = F.binary_cross_entropy_with_logits(pred.float(), target, reduction="mean")
        return self.weight_dice * dice + bce


def make_criterion(loss: str):
    if loss == "dicenfl":
        return DiceNFLoss()
    if loss == "bce":
        return nn.BCEWithLogitsLoss()
    if loss == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {loss}")


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
    if args.train_subset_index:
        train_set = MaskLevelFlatSubsetDataset(
            subset_list=Path(args.train_subset_index),
            index_mapping_path=index_mapping_path / "train_index_mapping.npy",
            dataset=train_set_base,
            **dataset_kwargs,
        )
    else:
        train_set = MaskLevelFlatDataset(
            index_mapping_path=index_mapping_path / "train_index_mapping.npy",
            dataset=train_set_base,
            **dataset_kwargs,
        )
    if args.val_subset_index:
        val_set = MaskLevelFlatSubsetDataset(
            subset_list=Path(args.val_subset_index),
            index_mapping_path=index_mapping_path / "val_index_mapping.npy",
            dataset=val_set_base,
            **dataset_kwargs,
        )
    else:
        val_set = MaskLevelFlatDataset(
            index_mapping_path=index_mapping_path / "val_index_mapping.npy",
            dataset=val_set_base,
            **dataset_kwargs,
        )

    if args.debug:
        train_set = MaskLevelDatasetDummy(
            dataset=train_set_base,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            with_image_embed=False,
            mask_filter_thresh=0.1,
            seed=42 + rank,
            count=20,
            image_size_encoder=1024,
            image_size_mask=1024,
        )
        val_set = MaskLevelDatasetDummy(
            dataset=val_set_base,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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


class SimpleMaskVAETrainer:
    def __init__(
        self,
        model,
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
        loss="dicenfl",
        kl_weight=1e-4,
        kl_warmup_iters=0,
        find_unused_parameters=True,
    ):
        self.model = model
        self.device = device
        self.out_dir = out_dir
        self.accumulate_steps = accumulate_steps
        self.dtype = dtype
        self.criterion = make_criterion(loss)
        self.kl_weight = kl_weight
        self.kl_warmup_iters = kl_warmup_iters
        self.global_step = 0

        if tdist.is_initialized():
            self.rank = tdist.get_rank()
            self.world_size = tdist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

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
            drop_last=True,
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

    def _current_kl_weight(self, step):
        if self.kl_warmup_iters <= 0:
            return self.kl_weight
        return self.kl_weight * min(1.0, max(0.0, step / self.kl_warmup_iters))

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

                global_iters = num_iters * outer_iter + resume_iters + iters_count + 1
                kl_weight = self._current_kl_weight(global_iters)
                with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.device.startswith("cuda") and self.dtype != torch.float32):
                    out = self.model(mask_normalized, image, sample=True)
                    recon_loss = self.criterion(out["mask_logits"], mask)
                    kl_loss = out["kl_loss"]
                    loss = (recon_loss + kl_weight * kl_loss) / self.accumulate_steps
                loss.backward()

                iters_count += 1
                self.global_step = global_iters
                if (iters_count % self.accumulate_steps == 0) or (iters_count == num_iters):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.rank == 0:
                    pbar.update(1)
                    if global_iters % log_interval == 0:
                        with torch.no_grad():
                            iou = calc_iou(out["mask_logits"], mask).mean().item()
                        loss_val = loss.item() * self.accumulate_steps
                        pbar.set_postfix({"loss": f"{loss_val:.4f}", "recon": f"{recon_loss.item():.4f}", "kl": f"{kl_loss.item():.4f}", "iou": f"{iou:.4f}"})
                        self.writer.add_scalar("train/loss", loss_val, global_iters)
                        self.writer.add_scalar("train/recon_loss", recon_loss.item(), global_iters)
                        self.writer.add_scalar("train/kl_loss", kl_loss.item(), global_iters)
                        self.writer.add_scalar("train/kl_weight", kl_weight, global_iters)
                        self.writer.add_scalar("train/iou", iou, global_iters)
                        if global_iters % (log_interval * 5) == 0:
                            self._visualize("train/visualization", image, mask, out["mask_logits"], calc_iou(out["mask_logits"], mask), f"iter_{global_iters}_train_vis.png")

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
        totals = torch.zeros(4, device=self.device)
        count = 0
        vis = None
        if self.rank == 0:
            pbar = tqdm(total=num_val_iters, desc=f"Val outer_iter {outer_iter}")
        for batch in self.val_loader:
            if count >= num_val_iters:
                break
            image, _, mask_normalized, mask = batch
            image = image.to(self.device, non_blocking=True)
            mask_normalized = mask_normalized.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.device.startswith("cuda") and self.dtype != torch.float32):
                out = self.model(mask_normalized, image, sample=False)
                recon_loss = self.criterion(out["mask_logits"], mask)
                kl_loss = out["kl_loss"]
                loss = recon_loss + self.kl_weight * kl_loss
            iou = calc_iou(out["mask_logits"], mask)
            totals += torch.tensor([loss.item(), recon_loss.item(), kl_loss.item(), iou.sum().item()], device=self.device)
            if vis is None and self.rank == 0:
                vis = (image.detach(), mask.detach(), out["mask_logits"].detach(), iou.detach())
            count += 1
            if self.rank == 0:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou.mean().item():.4f}"})
        if self.rank == 0:
            pbar.close()

        sample_count = count * self.val_loader.batch_size
        metrics = torch.tensor([totals[0], totals[1], totals[2], totals[3], count, sample_count], device=self.device)
        if self.world_size > 1:
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
        avg_loss = metrics[0].item() / max(metrics[4].item(), 1)
        avg_recon = metrics[1].item() / max(metrics[4].item(), 1)
        avg_kl = metrics[2].item() / max(metrics[4].item(), 1)
        avg_iou = metrics[3].item() / max(metrics[5].item(), 1)

        if self.rank == 0:
            if vis is not None:
                self._visualize("val/visualization", *vis, save_name=f"iter_{self.global_step}_val_vis.png")
            print(f"\nVal outer_iter {outer_iter}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}, IoU={avg_iou:.4f}")
            self.writer.add_scalar("val/loss", avg_loss, self.global_step)
            self.writer.add_scalar("val/recon_loss", avg_recon, self.global_step)
            self.writer.add_scalar("val/kl_loss", avg_kl, self.global_step)
            self.writer.add_scalar("val/iou", avg_iou, self.global_step)
        self.model.train()
        return {"loss": avg_loss, "recon_loss": avg_recon, "kl_loss": avg_kl, "iou": avg_iou}

    def _visualize(self, tag, image, gt_mask, pred_logits, iou, save_name=None, num_samples=4):
        if self.writer is None:
            return
        n = min(num_samples, image.shape[0])
        fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
        colors = {"tp": np.array([0.2, 0.6, 1.0]), "fp": np.array([1.0, 0.3, 0.3]), "fn": np.array([0.3, 0.9, 0.3])}
        for r in range(n):
            img = restore_normalized_image(image[r]).detach().float().cpu().numpy().transpose(1, 2, 0)
            img = img / 255.0 if img.max() > 1 else img
            gt = gt_mask[r, 0].detach().float().cpu().numpy() > 0
            pred = pred_logits[r, 0].detach().float().cpu().numpy() > 0
            logits = pred_logits[r, 0].detach().float().cpu().numpy()
            overlay = img.copy()
            for m, c in ((gt & pred, colors["tp"]), ((~gt) & pred, colors["fp"]), (gt & (~pred), colors["fn"])):
                overlay[m] = overlay[m] * 0.65 + c * 0.35
            axes[r, 0].imshow(img); axes[r, 0].set_title("Image"); axes[r, 0].axis("off")
            axes[r, 1].imshow(gt, cmap="gray", vmin=0, vmax=1); axes[r, 1].set_title("GT"); axes[r, 1].axis("off")
            axes[r, 2].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[r, 2].set_title(f"Pred IoU={iou[r].item():.3f}"); axes[r, 2].axis("off")
            axes[r, 3].imshow(np.clip(overlay, 0, 1)); axes[r, 3].set_title("Overlay"); axes[r, 3].axis("off")
            axes[r, 4].imshow(logits, cmap="RdBu_r"); axes[r, 4].set_title("Logits"); axes[r, 4].axis("off")
        plt.tight_layout()
        self.writer.add_figure(tag, fig, self.global_step)
        if save_name is not None:
            fig.savefig(self.out_dir / "visualizations" / save_name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_checkpoint(self, step, is_best=False):
        if self.rank != 0:
            return
        model = self._get_model()
        checkpoint = {"step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(), "global_step": self.global_step}
        ckpt_dir = self.out_dir / "checkpoints"
        torch.save(checkpoint, ckpt_dir / "latest.pth")
        torch.save(checkpoint, ckpt_dir / f"iter_{step}.pth")
        if is_best:
            torch.save(checkpoint, ckpt_dir / "best.pth")
        print(f"Checkpoint saved to {ckpt_dir}/iter_{step}.pth")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self._get_model().load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        return checkpoint.get("step", 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimpleMaskVAEV2")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--outer_iters", type=int, default=10)
    parser.add_argument("--inner_iters", type=int, default=1000)
    parser.add_argument("--val_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="coconut_hf", choices=["hqseg44k", "cocolvis", "coconut_hf"])
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_subset_index", type=str, default=None)
    parser.add_argument("--val_subset_index", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--loss", type=str, default="dicenfl", choices=["dicenfl", "bce", "mse"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--config", type=str, default="simple_mask_vae_v2_dim384", choices=sorted(builder_map["simple_mask_vae"].keys()))
    parser.add_argument("--image_encoder_checkpoint", type=str, default="ckpt/dino_v3_vits.safetensors")
    parser.add_argument("--image_encoder_config", type=str, default="dino_v3_vits", choices=sorted(builder_map["image_encoder"].keys()))
    parser.add_argument("--kl_weight", type=float, default=1e-4)
    parser.add_argument("--kl_warmup_iters", type=int, default=10000)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--freeze_image_encoder", action="store_true")
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
    checkpoint_to_use = args.checkpoint or args.resume_from
    model = builder_map["simple_mask_vae"][args.config](
        checkpoint_path=checkpoint_to_use if checkpoint_to_use and os.path.exists(checkpoint_to_use) else None,
        image_encoder_checkpoint=args.image_encoder_checkpoint,
        image_encoder_config_name=args.image_encoder_config,
        device=device,
    )
    model.beta_kl = args.kl_weight
    if args.freeze_image_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
    if not args.no_compile:
        model = torch.compile(model)

    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total:,} total, {trainable:,} trainable")

    trainer = SimpleMaskVAETrainer(
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
        dtype=dtype,
        loss=args.loss,
        kl_weight=args.kl_weight,
        kl_warmup_iters=args.kl_warmup_iters,
        find_unused_parameters=not args.disable_find_unused_parameters,
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
