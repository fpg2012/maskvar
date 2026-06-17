"""
Fine-tune SAM's mask decoder on mask-level datasets.

The image encoder and prompt encoder stay frozen. Checkpoints are saved as full
SAM state_dict files so they can be passed directly to eval_interactive_seg.py
via --sam_checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from maskvar.build_sam import sam_model_registry
from maskvar.datasets.image_feature_cache import ImageFeatureCache
from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset, MaskLevelFlatSubsetDataset
from maskvar.maskseg_build_everything import builder_map
from maskvar.utils.clicker_v2 import init_clicks, predict_next_click, to_sam_format
from maskvar.utils.losses import DICEFocalLoss, NormalizedFocalLossSigmoid
from maskvar.utils.metrics import calc_iou

torch.set_float32_matmul_precision("high")


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_dataset(args, split: str, image_feature_cache: Optional[ImageFeatureCache]):
    dataset_path_map = {
        "hqseg44k": "data/sam-hq",
        "cocolvis": "data/coco_lvis",
        "coconut_hf": "data/coconut_hf",
    }
    dataset_path = args.dataset_path or dataset_path_map[args.dataset]
    train_set, val_set = builder_map["dataset"][args.dataset](dataset_path)
    base_set = train_set if split == "train" else val_set
    subset_index = args.train_subset_index if split == "train" else args.val_subset_index
    index_mapping_path = Path("data/flat") / args.dataset / f"{split}_index_mapping.npy"
    dataset_cls = MaskLevelFlatSubsetDataset if subset_index else MaskLevelFlatDataset
    kwargs = {
        "index_mapping_path": index_mapping_path,
        "dataset": base_set,
        "with_image_embed": image_feature_cache is not None,
        "image_feature_cache": image_feature_cache,
        "mask_filter_thresh": args.mask_filter_thresh,
        "dtype": torch.float32,
        "image_size_encoder": args.image_size_encoder,
        "image_size_mask": args.image_size_mask,
    }
    if subset_index:
        kwargs["subset_list"] = Path(subset_index)
    return dataset_cls(**kwargs)


def build_cache(args, split: str) -> Optional[ImageFeatureCache]:
    if not args.image_feature_cache_dir:
        return None
    return ImageFeatureCache(
        cache_dir=Path(args.image_feature_cache_dir),
        dataset=f"{args.dataset}_{split}",
        model_name=args.sam_cache_model_name,
        device=args.device,
        max_cache_shard=args.image_feature_cache_max_shard,
    )


def build_loss(name: str):
    if name == "nfl":
        return NormalizedFocalLossSigmoid()
    if name == "dicefl":
        return DICEFocalLoss(smooth=1.0, alpha=0.75, gamma=2.0)
    if name == "bce":
        return torch.nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss: {name}")


def init_click_state(gt_mask: torch.Tensor, deterministic_clicks: bool):
    gt_np = gt_mask.detach().cpu().numpy()[:, 0] > 0.5
    click_lists = []
    not_clicked_maps = []
    for gt in gt_np:
        not_clicked = np.ones_like(gt, dtype=bool)
        clicks, _, _ = init_clicks(
            gt,
            num_random_clicks=1,
            not_clicked_map=not_clicked,
            random_sample=not deterministic_clicks,
        )
        click_lists.append(clicks)
        not_clicked_maps.append(not_clicked)
    return gt_np, click_lists, not_clicked_maps


def append_next_clicks(gt_np, pred_np, click_lists, not_clicked_maps):
    for i, clicks in enumerate(click_lists):
        error_mask = np.logical_and(gt_np[i] != pred_np[i], not_clicked_maps[i])
        if not error_mask.any():
            continue
        predict_next_click(
            gt_mask=gt_np[i],
            pred_mask=pred_np[i],
            click_list=clicks,
            not_clicked_map=not_clicked_maps[i],
        )


class SAMDecoderTrainer:
    def __init__(self, args, sam, train_set, val_set, device: str, rank: int, world_size: int):
        self.args = args
        self.sam = sam.to(device)
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.global_step = 0
        self.criterion = build_loss(args.loss)
        self.dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
        self.train_max_clicks = args.train_max_clicks if args.train_max_clicks > 0 else args.max_clicks

        for param in self.sam.parameters():
            param.requires_grad = False
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True
        self.sam.image_encoder.eval()
        self.sam.prompt_encoder.eval()
        self.sam.mask_decoder.train()

        if world_size > 1:
            self.train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
            self.val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None

        self.train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )

        trainable = [p for p in self.sam.mask_decoder.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        if world_size > 1:
            self.ddp_mask_decoder = DDP(
                self.sam.mask_decoder,
                device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
                find_unused_parameters=True,
            )
        else:
            self.ddp_mask_decoder = self.sam.mask_decoder

        self.out_dir = Path(args.out_dir)
        self.image_pe = self.sam.prompt_encoder.get_dense_pe()
        if rank == 0:
            (self.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.out_dir / "logs"))
        else:
            self.writer = None

    def _autocast(self):
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        return torch.autocast(device_type=device_type, dtype=self.dtype, enabled=self.dtype != torch.float32)

    @torch.no_grad()
    def encode_images(self, image, image_embedding):
        if image_embedding is not None and torch.is_tensor(image_embedding) and image_embedding.ndim == 4:
            return image_embedding
        return self.sam.image_encoder(image)

    def predict_batch(self, image_embeddings, click_lists, prev_logits, output_size, multimask_output: bool):
        masks = []
        scale_x = self.sam.prompt_encoder.input_image_size[1] / float(output_size[1])
        scale_y = self.sam.prompt_encoder.input_image_size[0] / float(output_size[0])
        for i, clicks in enumerate(click_lists):
            coords_xy, labels = to_sam_format(clicks, device=self.device)
            coords_xy = coords_xy.float()
            coords_xy[:, 0] *= scale_x
            coords_xy[:, 1] *= scale_y
            mask_prompt = None
            if prev_logits is not None:
                mask_prompt = F.interpolate(
                    prev_logits[i : i + 1].float(),
                    size=self.sam.prompt_encoder.mask_input_size,
                    mode="bilinear",
                    align_corners=False,
                )
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=(coords_xy.unsqueeze(0), labels.long().unsqueeze(0)),
                    boxes=None,
                    masks=mask_prompt,
                )
            low_res_masks, iou_predictions = self.ddp_mask_decoder(
                image_embeddings=image_embeddings[i : i + 1],
                image_pe=self.image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            if multimask_output:
                best_idx = int(iou_predictions[0].detach().argmax().item())
                low_res_masks = low_res_masks[:, best_idx : best_idx + 1]
            masks.append(F.interpolate(low_res_masks, size=output_size, mode="bilinear", align_corners=False))
        return torch.cat(masks, dim=0)

    def interactive_forward(self, image, image_embedding, gt_mask, train: bool):
        output_size = tuple(gt_mask.shape[-2:])
        gt_np, click_lists, not_clicked_maps = init_click_state(gt_mask, self.args.deterministic_clicks)
        total_clicks = self.args.max_clicks if not train else int(np.random.randint(1, self.train_max_clicks + 1))
        prev_logits = None
        image_embeddings = self.encode_images(image, image_embedding)
        for click_idx in range(1, total_clicks + 1):
            logits = self.predict_batch(
                image_embeddings=image_embeddings,
                click_lists=click_lists,
                prev_logits=prev_logits if self.args.use_prev_mask else None,
                output_size=output_size,
                multimask_output=self.args.multimask_first_click and click_idx == 1,
            )
            if click_idx < total_clicks:
                pred_np = logits.detach().float().cpu().numpy()[:, 0] > 0.0
                append_next_clicks(gt_np, pred_np, click_lists, not_clicked_maps)
                prev_logits = logits.detach()
        return logits, total_clicks

    def train_one_outer(self, outer_iter: int):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(outer_iter)
        self.sam.mask_decoder.train()
        num_iters = self.args.inner_iters if self.args.inner_iters > 0 else len(self.train_loader)
        pbar = tqdm(total=num_iters, desc=f"Train SAM decoder {outer_iter}") if self.rank == 0 else None
        data_iter = iter(self.train_loader)
        for it in range(num_iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            image, image_embedding, _, gt_mask = batch
            image = image.to(self.device, non_blocking=True)
            gt_mask = gt_mask.to(self.device, non_blocking=True)
            image_embedding = image_embedding.to(self.device, non_blocking=True) if torch.is_tensor(image_embedding) and image_embedding.ndim > 1 else None
            target = (gt_mask > 0.5).float()

            with self._autocast():
                logits, num_clicks = self.interactive_forward(image, image_embedding, gt_mask, train=True)
                loss = self.criterion(logits, target)
                if loss.ndim > 0:
                    loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sam.mask_decoder.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1
            if self.rank == 0:
                pbar.update(1)
                if self.global_step % self.args.log_interval == 0:
                    with torch.no_grad():
                        iou = calc_iou((logits > 0).float(), gt_mask).mean().item()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}", "clicks": num_clicks})
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/iou", iou, self.global_step)
                    self.writer.add_scalar("train/clicks", num_clicks, self.global_step)
        if self.rank == 0:
            pbar.close()
            self.save_checkpoint()
        if self.world_size > 1:
            dist.barrier()
        if self.args.val_iters >= 0:
            self.validate(outer_iter)

    @torch.no_grad()
    def validate(self, outer_iter: int):
        self.sam.mask_decoder.eval()
        total_iou = 0.0
        total_loss = 0.0
        total_samples = 0
        total_batches = 0
        max_batches = len(self.val_loader) if self.args.val_iters == 0 else min(self.args.val_iters, len(self.val_loader))
        pbar = tqdm(total=max_batches, desc=f"Val SAM decoder {outer_iter}") if self.rank == 0 else None
        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= max_batches:
                break
            image, image_embedding, _, gt_mask = batch
            image = image.to(self.device, non_blocking=True)
            gt_mask = gt_mask.to(self.device, non_blocking=True)
            image_embedding = image_embedding.to(self.device, non_blocking=True) if torch.is_tensor(image_embedding) and image_embedding.ndim > 1 else None
            target = (gt_mask > 0.5).float()
            with self._autocast():
                logits, _ = self.interactive_forward(image, image_embedding, gt_mask, train=False)
                loss = self.criterion(logits, target)
                if loss.ndim > 0:
                    loss = loss.mean()
            iou = calc_iou((logits > 0).float(), gt_mask)
            total_loss += loss.item()
            total_iou += iou.sum().item()
            total_samples += iou.shape[0]
            total_batches += 1
            if self.rank == 0:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou.mean().item():.4f}"})

        metrics = torch.tensor([total_loss, total_iou, total_samples, total_batches], device=self.device)
        if self.world_size > 1:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        avg_loss = metrics[0].item() / max(metrics[3].item(), 1.0)
        avg_iou = metrics[1].item() / max(metrics[2].item(), 1.0)
        if self.rank == 0:
            pbar.close()
            print(f"Val outer_iter {outer_iter}: loss={avg_loss:.4f}, iou={avg_iou:.4f}")
            self.writer.add_scalar("val/loss", avg_loss, self.global_step)
            self.writer.add_scalar("val/iou", avg_iou, self.global_step)
        self.sam.mask_decoder.train()

    def save_checkpoint(self):
        sam = self.sam.module if hasattr(self.sam, "module") else self.sam
        state_dict = {k: v.detach().cpu() for k, v in sam.state_dict().items()}
        torch.save(state_dict, self.out_dir / "checkpoints" / "latest.pth")
        torch.save(
            {
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "args": vars(self.args),
            },
            self.out_dir / "checkpoints" / "trainer_latest.pth",
        )

    def load_resume(self, path: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "model_state_dict" in checkpoint:
            self.sam.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.global_step = int(checkpoint.get("global_step", 0))
        else:
            self.sam.load_state_dict(checkpoint, strict=True)
        if self.rank == 0:
            print(f"Resumed SAM decoder training from {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SAM mask decoder")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--sam_model_type", choices=sorted(sam_model_registry.keys()), default="vit_b")
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--dataset", choices=["hqseg44k", "cocolvis", "coconut_hf"], default="coconut_hf")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_subset_index", type=str, default=None)
    parser.add_argument("--val_subset_index", type=str, default=None)
    parser.add_argument("--mask_filter_thresh", type=float, default=0.1)
    parser.add_argument("--image_size_encoder", type=int, default=1024)
    parser.add_argument("--image_size_mask", type=int, default=1024)
    parser.add_argument("--image_feature_cache_dir", type=str, default="")
    parser.add_argument("--sam_cache_model_name", type=str, default="sam_vitb")
    parser.add_argument("--image_feature_cache_max_shard", type=int, default=2)

    parser.add_argument("--outer_iters", type=int, default=5)
    parser.add_argument("--inner_iters", type=int, default=0)
    parser.add_argument("--val_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_clicks", type=int, default=10)
    parser.add_argument(
        "--train_max_clicks",
        type=int,
        default=0,
        help="Maximum clicks sampled during training. 0 means use --max_clicks.",
    )
    parser.add_argument("--use_prev_mask", action="store_true")
    parser.add_argument("--multimask_first_click", action="store_true")
    parser.add_argument("--deterministic_clicks", action="store_true")
    parser.add_argument("--loss", choices=["nfl", "dicefl", "bce"], default="nfl")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    if torch.cuda.is_available() and args.device == "cuda":
        args.device = f"cuda:{local_rank}"
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    train_cache = build_cache(args, "train")
    val_cache = build_cache(args, "val")
    train_set = build_dataset(args, "train", train_cache)
    val_set = build_dataset(args, "val", val_cache)
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    trainer = SAMDecoderTrainer(args, sam, train_set, val_set, args.device, rank, world_size)
    if args.resume_from:
        trainer.load_resume(args.resume_from)

    try:
        for outer_iter in range(args.outer_iters):
            trainer.train_one_outer(outer_iter)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
