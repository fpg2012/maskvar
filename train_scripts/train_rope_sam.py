"""
Training script for RopeSAM.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from maskvar.datasets.mask_level_dataset import MaskLevelDatasetDummy
from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset, MaskLevelFlatSubsetDataset
from maskvar.datasets.sharded_distributed_sampler import ShardedDistributedSampler
from maskvar.maskseg_build_everything import builder_map
from maskvar.utils import restore_normalized_image
from maskvar.utils.clicker_v2 import init_clicks, predict_next_click, to_sam_format
from maskvar.utils.losses import (
    DICEBCELoss,
    DICEFocalLoss,
    DICELoss,
    DiceNFLoss,
    FocalLoss,
    NormalizedFocalLossSigmoid,
)
from maskvar.utils.metrics import calc_iou

torch.set_float32_matmul_precision("high")


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        tdist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if tdist.is_initialized():
        tdist.destroy_process_group()


def sample_click_condition(single_mask: torch.Tensor, ar_h: int, ar_w: int, max_clicks: int = 10):
    mask_np = single_mask[0].detach().cpu().numpy() > 0
    click_list, _, _ = init_clicks(mask_np, num_random_clicks=1, random_sample=True)
    coords_xy, labels = to_sam_format(click_list, pad_size=max_clicks)

    h, w = single_mask.shape[-2:]
    click_coords = torch.empty_like(coords_xy, dtype=torch.float32)
    click_coords[..., 0] = coords_xy[..., 1] * (ar_h / h)
    click_coords[..., 1] = coords_xy[..., 0] * (ar_w / w)
    click_coords = click_coords.clamp_min(0)
    click_coords[..., 0].clamp_(max=ar_h - 1)
    click_coords[..., 1].clamp_(max=ar_w - 1)
    return click_coords, labels.long()


class ClickConditionDataset(Dataset):
    def __init__(self, dataset, ar_h: int, ar_w: int, max_clicks: int = 2):
        self.dataset = dataset
        self.ar_h = ar_h
        self.ar_w = ar_w
        self.max_clicks = max_clicks
        self.is_dummy = getattr(dataset, "is_dummy", False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, image_embed_sam, single_mask_normalized, single_mask = self.dataset[index]
        click_coords, click_labels = sample_click_condition(single_mask, self.ar_h, self.ar_w, self.max_clicks)
        return image, image_embed_sam, single_mask_normalized, single_mask, click_coords, click_labels


class ClickConditionIterableDataset(IterableDataset):
    def __init__(self, dataset, ar_h: int, ar_w: int, max_clicks: int = 2):
        self.dataset = dataset
        self.ar_h = ar_h
        self.ar_w = ar_w
        self.max_clicks = max_clicks
        self.is_dummy = getattr(dataset, "is_dummy", False)

    def __iter__(self):
        for image, image_embed_sam, single_mask_normalized, single_mask in self.dataset:
            click_coords, click_labels = sample_click_condition(single_mask, self.ar_h, self.ar_w, self.max_clicks)
            yield image, image_embed_sam, single_mask_normalized, single_mask, click_coords, click_labels


def wrap_click_condition_dataset(dataset, ar_h: int, ar_w: int, max_clicks: int = 2):
    if isinstance(dataset, IterableDataset):
        return ClickConditionIterableDataset(dataset, ar_h=ar_h, ar_w=ar_w, max_clicks=max_clicks)
    return ClickConditionDataset(dataset, ar_h=ar_h, ar_w=ar_w, max_clicks=max_clicks)


def build_loss(loss: str):
    if loss == "nfl":
        return NormalizedFocalLossSigmoid()
    if loss == "mse":
        return torch.nn.MSELoss()
    if loss == "dice":
        return DICELoss()
    if loss == "fl":
        return FocalLoss(alpha=0.75, gamma=2.0)
    if loss == "dicefl":
        return DICEFocalLoss(smooth=1.0, alpha=0.75, gamma=2.0)
    if loss == "dicebce":
        return DICEBCELoss()
    if loss == "dicenfl":
        return DiceNFLoss()
    raise ValueError(f"Unknown loss: {loss}")


class RopeSAMTrainer:
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
        loss: str = "nfl",
        dtype: torch.dtype = torch.float32,
        find_unused_parameters: bool = True,
        interactive_click_warmup_iters: int = 10000,
    ):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.accumulate_steps = accumulate_steps
        self.out_dir = out_dir
        self.criterion = build_loss(loss)
        self.global_step = 0
        self.interactive_click_warmup_iters = interactive_click_warmup_iters

        if tdist.is_initialized():
            self.rank = tdist.get_rank()
            self.world_size = tdist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

        try:
            first_param = next(self.model.parameters())
            if str(first_param.device) != str(self.device):
                self.model.to(self.device)
        except StopIteration:
            pass

        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=False,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        self.train_sampler = None
        self.val_sampler = None
        is_dummy_dataset = getattr(train_dataset, "is_dummy", False)
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

        is_iterable = isinstance(train_dataset, IterableDataset)
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
        else:
            self.writer = None

    def _autocast(self):
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        return torch.autocast(device_type=device_type, dtype=self.dtype, enabled=self.dtype != torch.float32)

    def _unpack_batch(self, batch):
        image, _, single_mask_normalized, single_mask, click_coords, click_labels = batch
        return (
            image.to(self.device, non_blocking=True),
            single_mask_normalized.to(self.device, non_blocking=True),
            single_mask.to(self.device, non_blocking=True),
            click_coords.to(self.device, non_blocking=True),
            click_labels.to(self.device, non_blocking=True),
        )

    def _click_tensors_to_lists(self, click_coords, click_labels, mask_shape):
        mask_h, mask_w = mask_shape
        coords_cpu = click_coords.detach().float().cpu().numpy()
        labels_cpu = click_labels.detach().cpu().numpy()
        click_lists = []
        not_clicked_maps = []
        for sample_coords, sample_labels in zip(coords_cpu, labels_cpu):
            click_list = []
            not_clicked = np.ones((mask_h, mask_w), dtype=bool)
            for coord, label in zip(sample_coords, sample_labels):
                label = int(label)
                if label < 0:
                    continue
                y = int(np.clip(round(float(coord[0]) * mask_h / 64.0), 0, mask_h - 1))
                x = int(np.clip(round(float(coord[1]) * mask_w / 64.0), 0, mask_w - 1))
                click_list.append((y, x, label))
                not_clicked[y, x] = False
            click_lists.append(click_list)
            not_clicked_maps.append(not_clicked)
        return click_lists, not_clicked_maps

    def _click_lists_to_tensors(self, click_lists, max_clicks: int, mask_shape):
        mask_h, mask_w = mask_shape
        point_coords = []
        point_labels = []
        for click_list in click_lists:
            coords_xy, labels = to_sam_format(click_list, pad_size=max_clicks)
            click_coords = torch.empty_like(coords_xy, dtype=torch.float32)
            click_coords[..., 0] = coords_xy[..., 1] * (64.0 / mask_h)
            click_coords[..., 1] = coords_xy[..., 0] * (64.0 / mask_w)
            click_coords = click_coords.clamp_min(0)
            click_coords[..., 0].clamp_(max=63)
            click_coords[..., 1].clamp_(max=63)
            point_coords.append(click_coords)
            point_labels.append(labels.long())
        return (
            torch.stack(point_coords).to(self.device, non_blocking=True),
            torch.stack(point_labels).to(self.device, non_blocking=True),
        )

    @torch.no_grad()
    def _append_next_clicks(self, click_lists, not_clicked_maps, gt_mask, pred_logits):
        gt_cpu = (gt_mask.detach().cpu().numpy()[:, 0] > 0.5)
        pred_cpu = (pred_logits.detach().float().cpu().numpy()[:, 0] > 0.0)
        for i, click_list in enumerate(click_lists):
            fn_mask = np.logical_and(gt_cpu[i], np.logical_not(pred_cpu[i]))
            fp_mask = np.logical_and(np.logical_not(gt_cpu[i]), pred_cpu[i])
            error_mask = np.logical_and(np.logical_or(fn_mask, fp_mask), not_clicked_maps[i])
            if error_mask.any():
                predict_next_click(
                    gt_mask=gt_cpu[i],
                    pred_mask=pred_cpu[i],
                    click_list=click_list,
                    not_clicked_map=not_clicked_maps[i],
                )
            else:
                fallback_clicks, _, _ = init_clicks(
                    gt_cpu[i],
                    num_random_clicks=1,
                    not_clicked_map=not_clicked_maps[i],
                    random_sample=True,
                )
                click_list.extend(fallback_clicks)

    def _current_max_clicks(self):
        max_clicks = 10
        if hasattr(self.model, "module"):
            max_clicks = self.model.module.max_clicks
        elif hasattr(self.model, "max_clicks"):
            max_clicks = self.model.max_clicks

        if self.interactive_click_warmup_iters <= 0:
            return max_clicks
        progress = min(1.0, float(self.global_step) / float(self.interactive_click_warmup_iters))
        return max(1, min(max_clicks, 1 + int(progress * (max_clicks - 1))))

    def interactive_forward(self, image, single_mask, click_coords, click_labels):
        max_clicks = click_labels.shape[1]
        current_max_clicks = min(max_clicks, self._current_max_clicks())
        total_clicks = int(np.random.randint(1, current_max_clicks + 1))
        mask_shape = single_mask.shape[-2:]
        click_lists, not_clicked_maps = self._click_tensors_to_lists(click_coords, click_labels, mask_shape)

        prev_logits = None
        cur_coords, cur_labels = click_coords, click_labels
        for _ in range(1, total_clicks):
            with torch.no_grad():
                prev_logits = self.model(
                    image=image,
                    click_coords=cur_coords,
                    click_labels=cur_labels,
                    prev_mask_logits=prev_logits,
                    output_size=mask_shape,
                ).detach()
                self._append_next_clicks(click_lists, not_clicked_maps, single_mask, prev_logits)
                cur_coords, cur_labels = self._click_lists_to_tensors(click_lists, max_clicks, mask_shape)

        logits = self.model(
            image=image,
            click_coords=cur_coords,
            click_labels=cur_labels,
            prev_mask_logits=prev_logits,
            output_size=mask_shape,
        )
        return logits, cur_coords, cur_labels, total_clicks

    def train(self, num_iters: int, outer_iter: int = 0, resume_iters: int = 0, val_iters: int = 0, log_interval: int = 10):
        if num_iters <= 0:
            num_iters = len(self.train_loader)

        self.model.train()
        if self.rank == 0:
            pbar = tqdm(total=num_iters, desc=f"Training outer_iter {outer_iter}")

        iters_count = 0
        while iters_count < num_iters:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(outer_iter)

            for batch in self.train_loader:
                if iters_count >= num_iters:
                    break

                image, single_mask_normalized, single_mask, click_coords, click_labels = self._unpack_batch(batch)
                target_mask = (single_mask > 0.5).float()

                with self._autocast():
                    logits, click_coords_final, click_labels_final, total_clicks = self.interactive_forward(
                        image=image,
                        single_mask=single_mask,
                        click_coords=click_coords,
                        click_labels=click_labels,
                    )
                    recon_loss = self.criterion(logits, target_mask).mean()
                    loss = recon_loss / self.accumulate_steps

                loss.backward()

                if (self.global_step + 1) % self.accumulate_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                global_iters = iters_count + num_iters * outer_iter + resume_iters
                self.global_step = global_iters
                iters_count += 1

                if self.rank == 0:
                    pbar.update(1)
                    if global_iters % log_interval == 0:
                        with torch.no_grad():
                            iou = calc_iou((logits > 0).float(), single_mask)
                        loss_val = loss.item() * self.accumulate_steps
                        iou_mean = iou.mean().item()
                        pbar.set_postfix({"loss": f"{loss_val:.4f}", "iou": f"{iou_mean:.4f}"})
                        self.writer.add_scalar("train/loss", loss_val, global_step=global_iters)
                        self.writer.add_scalar("train/iou", iou_mean, global_step=global_iters)
                        self.writer.add_scalar("train/num_clicks", total_clicks, global_step=global_iters)
                        if global_iters % (log_interval * 5) == 0:
                            self._visualize_samples(
                                image,
                                single_mask_normalized,
                                logits,
                                iou,
                                click_coords_final,
                                click_labels_final,
                                tag="train/samples",
                            )

        if self.rank == 0:
            pbar.close()
            self.save_checkpoint((outer_iter + 1) * num_iters + resume_iters)

        if self.world_size > 1:
            tdist.barrier()

        self.validate(num_val_iters=val_iters, outer_iter=outer_iter)

    @torch.no_grad()
    def validate(self, num_val_iters: int = 0, outer_iter: int = 0):
        if num_val_iters < 0:
            return {"loss": 0.0, "iou": 0.0}

        self.model.eval()
        if num_val_iters == 0:
            try:
                num_val_iters = len(self.val_loader)
            except TypeError:
                num_val_iters = 100

        total_loss = 0.0
        total_iou = 0.0
        num_iou_samples = 0
        vis_batch = None

        if self.rank == 0:
            pbar = tqdm(total=num_val_iters, desc=f"Val outer_iter {outer_iter}")

        iters_count = 0
        for batch in self.val_loader:
            if iters_count >= num_val_iters:
                break

            image, single_mask_normalized, single_mask, click_coords, click_labels = self._unpack_batch(batch)
            target_mask = (single_mask > 0.5).float()
            with self._autocast():
                logits, click_coords_final, click_labels_final, _ = self.interactive_forward(
                    image=image,
                    single_mask=single_mask,
                    click_coords=click_coords,
                    click_labels=click_labels,
                )
                loss = self.criterion(logits, target_mask).mean()

            iou = calc_iou((logits > 0).float(), single_mask)
            total_loss += loss.item()
            total_iou += iou.sum().item()
            num_iou_samples += iou.shape[0]
            if self.rank == 0 and vis_batch is None:
                vis_batch = (
                    image.detach(),
                    single_mask_normalized.detach(),
                    logits.detach(),
                    iou.detach(),
                    click_coords_final.detach(),
                    click_labels_final.detach(),
                )

            if self.rank == 0:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou.mean().item():.4f}"})
            iters_count += 1

        if self.rank == 0:
            pbar.close()

        avg_loss = total_loss / iters_count if iters_count > 0 else 0.0
        avg_iou = total_iou / num_iou_samples if num_iou_samples > 0 else 0.0

        if self.world_size > 1:
            metrics = torch.tensor([total_loss, total_iou, num_iou_samples, iters_count], device=self.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            avg_loss = metrics[0].item() / max(metrics[3].item(), 1.0)
            avg_iou = metrics[1].item() / max(metrics[2].item(), 1.0)

        if self.rank == 0:
            print(f"\nVal outer_iter {outer_iter}: Loss={avg_loss:.4f}, IoU={avg_iou:.4f}")
            self.writer.add_scalar("val/loss", avg_loss, global_step=self.global_step)
            self.writer.add_scalar("val/iou", avg_iou, global_step=self.global_step)
            if vis_batch is not None:
                self._visualize_samples(*vis_batch, tag="val/samples")

        self.model.train()
        return {"loss": avg_loss, "iou": avg_iou}

    def _visualize_samples(self, image, gt_mask, logits, iou, click_coords, click_labels, tag: str, num_samples: int = 4):
        if self.writer is None:
            return
        num_samples = min(num_samples, image.shape[0])
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            img = restore_normalized_image(image[i]).detach().float().cpu().numpy().transpose(1, 2, 0)
            img = img / 255.0 if img.max() > 1 else img
            gt = gt_mask[i, 0].detach().float().cpu().numpy() > 0
            pred_logits = logits[i, 0].detach().float().cpu().numpy()
            pred = pred_logits > 0

            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Image" if i == 0 else "")
            self._draw_clicks(axes[i, 0], click_coords[i], click_labels[i], img.shape[:2])
            axes[i, 0].axis("off")

            axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title("GT" if i == 0 else "")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].set_title(f"Pred IoU={iou[i].item():.3f}")
            axes[i, 2].axis("off")

            vmax_abs = max(abs(float(pred_logits.min())), abs(float(pred_logits.max())))
            axes[i, 3].imshow(pred_logits, cmap="RdBu_r", vmin=-vmax_abs, vmax=vmax_abs)
            axes[i, 3].set_title("Logits" if i == 0 else "")
            axes[i, 3].axis("off")

        plt.tight_layout()
        self.writer.add_figure(tag, fig, self.global_step)
        vis_dir = self.out_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(vis_dir / f"iter_{self.global_step}_{tag.replace('/', '_')}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _draw_clicks(ax, click_coords, click_labels, image_shape):
        img_h, img_w = image_shape
        coords = click_coords.detach().float().cpu().numpy()
        labels = click_labels.detach().cpu().numpy()
        for coord, label in zip(coords, labels):
            if label < 0:
                continue
            y = coord[0] * img_h / 64.0
            x = coord[1] * img_w / 64.0
            color = "lime" if label == 1 else "red"
            marker = "+" if label == 1 else "x"
            ax.scatter([x], [y], c=color, marker=marker, s=80, linewidths=2)

    def save_checkpoint(self, step: int):
        if self.rank != 0:
            return
        checkpoint_dir = self.out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pth")
        torch.save(checkpoint, checkpoint_dir / f"iter_{step}.pth")
        print(f"Checkpoint saved to {checkpoint_dir / f'iter_{step}.pth'}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        if self.rank == 0:
            print(f"Loaded checkpoint from {checkpoint_path}")
        return self.global_step


def build_datasets(args, device: str, rank: int):
    dataset_path_map = {
        "hqseg44k": "data/sam-hq",
        "cocolvis": "data/coco_lvis",
        "coconut_hf": "data/coconut_hf",
    }
    dataset_path = args.dataset_path or dataset_path_map[args.dataset]
    train_set_base, val_set_base = builder_map["dataset"][args.dataset](dataset_path)
    index_mapping_path = Path("data/flat") / args.dataset

    dataset_kwargs = {
        "with_image_embed": False,
        "image_feature_cache": None,
        "mask_filter_thresh": 0.1,
        "dtype": torch.float32,
        "image_size_encoder": 1024,
        "image_size_mask": 1024,
    }

    train_cls = MaskLevelFlatSubsetDataset if args.train_subset_index else MaskLevelFlatDataset
    val_cls = MaskLevelFlatSubsetDataset if args.val_subset_index else MaskLevelFlatDataset
    train_kwargs = {
        "index_mapping_path": index_mapping_path / "train_index_mapping.npy",
        "dataset": train_set_base,
        **dataset_kwargs,
    }
    val_kwargs = {
        "index_mapping_path": index_mapping_path / "val_index_mapping.npy",
        "dataset": val_set_base,
        **dataset_kwargs,
    }
    if args.train_subset_index:
        train_kwargs["subset_list"] = Path(args.train_subset_index)
    if args.val_subset_index:
        val_kwargs["subset_list"] = Path(args.val_subset_index)

    train_set = train_cls(**train_kwargs)
    val_set = val_cls(**val_kwargs)

    if args.debug:
        train_set = MaskLevelDatasetDummy(
            dataset=train_set_base,
            device=torch.device(device),
            with_image_embed=False,
            mask_filter_thresh=0.1,
            seed=42 + rank,
            count=20,
            image_size_encoder=1024,
            image_size_mask=1024,
        )
        val_set = MaskLevelDatasetDummy(
            dataset=val_set_base,
            device=torch.device(device),
            with_image_embed=False,
            mask_filter_thresh=0.1,
            seed=100 + rank,
            count=5,
            image_size_encoder=1024,
            image_size_mask=1024,
        )
        train_set.is_dummy = True
        val_set.is_dummy = True

    train_set = wrap_click_condition_dataset(train_set, ar_h=64, ar_w=64, max_clicks=args.max_clicks)
    val_set = wrap_click_condition_dataset(val_set, ar_h=64, ar_w=64, max_clicks=args.max_clicks)
    return train_set, val_set


def main():
    parser = argparse.ArgumentParser(description="Train RopeSAM")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--outer_iters", type=int, default=10)
    parser.add_argument("--inner_iters", type=int, default=1000)
    parser.add_argument("--val_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--dataset", type=str, default="coconut_hf", choices=["hqseg44k", "cocolvis", "coconut_hf"])
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_subset_index", type=str, default=None)
    parser.add_argument("--val_subset_index", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--loss", default="nfl", type=str)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "bfloat16"])

    parser.add_argument("--config", type=str, default="rope_sam_dim384", choices=sorted(builder_map["rope_sam"].keys()))
    parser.add_argument("--image_encoder_config", type=str, default="dino_v3_vits", choices=sorted(builder_map["image_encoder"].keys()))
    parser.add_argument("--image_encoder_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--max_clicks", type=int, default=10)
    parser.add_argument("--interactive_click_warmup_iters", type=int, default=10000)

    parser.add_argument("--freeze_image_encoder", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--disable_find_unused_parameters", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_iters", type=int, default=100)

    args = parser.parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

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

    train_set, val_set = build_datasets(args, device=device, rank=rank)

    checkpoint_to_use = args.checkpoint or args.resume_from
    model = builder_map["rope_sam"][args.config](
        checkpoint_path=checkpoint_to_use if checkpoint_to_use and os.path.exists(checkpoint_to_use) else None,
        image_encoder_checkpoint=args.image_encoder_checkpoint,
        image_encoder_config_name=args.image_encoder_config,
        max_clicks=args.max_clicks,
        device=device,
    )

    if args.freeze_image_encoder:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("Frozen image_encoder parameters")

    if not args.no_compile:
        model = torch.compile(model)
        if rank == 0:
            print("Applied torch.compile to model")

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    trainer = RopeSAMTrainer(
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
        find_unused_parameters=not args.disable_find_unused_parameters,
        interactive_click_warmup_iters=args.interactive_click_warmup_iters,
    )

    resume_iters = 0
    if args.resume_from and os.path.exists(args.resume_from):
        resume_iters = trainer.load_checkpoint(args.resume_from)

    inner_iters = args.debug_iters if args.debug else args.inner_iters
    try:
        for outer_iter in range(args.outer_iters):
            trainer.train(
                num_iters=inner_iters,
                outer_iter=outer_iter,
                resume_iters=resume_iters,
                val_iters=args.val_iters,
                log_interval=args.log_interval,
            )
        if rank == 0:
            print(f"Training complete. Checkpoints saved to {out_dir / 'checkpoints'}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
