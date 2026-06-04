"""
Evaluate interactive segmentation on mask-level datasets.

Metrics:
  - NoC@T: mean number of clicks needed to reach IoU threshold T. Samples that
    do not reach T within max_clicks use max_clicks + 1.
  - IoU@k: mean IoU after k clicks.

Example:
    python misc_scripts/eval_interactive_seg.py \
        --model both \
        --dataset coconut_hf \
        --dataset_split val \
        --sam_checkpoint ckpt/sam_vit_b_01ec64.pth \
        --rope_sam_checkpoint out/ddp_rope_sam_coconut_hf_dino_click/checkpoints/latest.pth \
        --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
        --outdir out/interactive_eval_coconut_val \
        --val_iters 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from maskvar.build_sam import sam_model_registry
from maskvar.datasets.image_feature_cache import ImageFeatureCache
from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset, MaskLevelFlatSubsetDataset
from maskvar.maskseg_build_everything import builder_map
from maskvar.utils.clicker_v2 import init_clicks, predict_next_click, to_sam_format

torch.set_float32_matmul_precision("high")


THRESHOLDS = (60, 70, 75, 80, 85, 90, 95, 98)


def mask_iou_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def build_dataset(args, image_feature_cache: Optional[ImageFeatureCache]):
    dataset_path_map = {
        "hqseg44k": "data/sam-hq",
        "cocolvis": "data/coco_lvis",
        "coconut_hf": "data/coconut_hf",
    }
    dataset_path = args.dataset_path or dataset_path_map[args.dataset]
    train_set, val_set = builder_map["dataset"][args.dataset](dataset_path)
    base_set = train_set if args.dataset_split == "train" else val_set
    index_mapping_path = Path("data/flat") / args.dataset / f"{args.dataset_split}_index_mapping.npy"
    if not index_mapping_path.exists():
        raise FileNotFoundError(f"Index mapping not found: {index_mapping_path}")

    dataset_cls = MaskLevelFlatSubsetDataset if args.subset_index else MaskLevelFlatDataset
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
    if args.subset_index:
        kwargs["subset_list"] = Path(args.subset_index)
    return dataset_cls(**kwargs)


def build_cache(args, model_name: str) -> Optional[ImageFeatureCache]:
    if not args.image_feature_cache_dir:
        return None
    return ImageFeatureCache(
        cache_dir=Path(args.image_feature_cache_dir),
        dataset=f"{args.dataset}_{args.dataset_split}",
        model_name=model_name,
        device=args.device,
        max_cache_shard=args.image_feature_cache_max_shard,
    )


class InteractiveModel:
    name: str

    def to(self, device: str):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        image_embedding: Optional[torch.Tensor],
        click_lists: List[List[Tuple[int, int, int]]],
        prev_logits: Optional[torch.Tensor],
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        raise NotImplementedError


class SAMInteractiveModel(InteractiveModel):
    def __init__(self, checkpoint: str, model_type: str = "vit_b"):
        self.name = "sam"
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.image_embedding_cache: Optional[torch.Tensor] = None
        self.image_cache_key: Optional[int] = None

    def to(self, device: str):
        self.sam.to(device)
        return self

    def eval(self):
        self.sam.eval()
        return self

    @torch.no_grad()
    def _encode_image(self, image: torch.Tensor, image_embedding: Optional[torch.Tensor]) -> torch.Tensor:
        if image_embedding is not None:
            return image_embedding
        key = int(image.data_ptr())
        if self.image_cache_key != key:
            self.image_embedding_cache = self.sam.image_encoder(image)
            self.image_cache_key = key
        return self.image_embedding_cache

    @torch.no_grad()
    def predict(self, image, image_embedding, click_lists, prev_logits, output_size):
        image_embeddings = self._encode_image(image, image_embedding)
        masks = []
        scale_x = self.sam.prompt_encoder.input_image_size[1] / float(output_size[1])
        scale_y = self.sam.prompt_encoder.input_image_size[0] / float(output_size[0])
        for i, clicks in enumerate(click_lists):
            coords_xy, labels = to_sam_format(clicks, device=image.device)
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
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=(coords_xy.unsqueeze(0), labels.long().unsqueeze(0)),
                boxes=None,
                masks=mask_prompt,
            )
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i : i + 1],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks.append(
                F.interpolate(
                    low_res_masks,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.cat(masks, dim=0)


class RopeSAMInteractiveModel(InteractiveModel):
    def __init__(
        self,
        checkpoint: str,
        config: str,
        image_encoder_checkpoint: Optional[str],
        image_encoder_config: str,
        max_clicks: int,
        device: str,
    ):
        self.name = "rope_sam"
        self.model = builder_map["rope_sam"][config](
            checkpoint_path=checkpoint,
            image_encoder_checkpoint=image_encoder_checkpoint,
            image_encoder_config_name=image_encoder_config,
            max_clicks=max_clicks,
            device=device,
        )

    def to(self, device: str):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def _click_lists_to_tensors(self, click_lists, output_size, device):
        mask_h, mask_w = output_size
        point_coords = []
        point_labels = []
        for clicks in click_lists:
            coords_xy, labels = to_sam_format(clicks, pad_size=self.model.max_clicks, device=device)
            click_coords = torch.empty_like(coords_xy, dtype=torch.float32)
            click_coords[..., 0] = coords_xy[..., 1] * (self.model.h / float(mask_h))
            click_coords[..., 1] = coords_xy[..., 0] * (self.model.w / float(mask_w))
            click_coords = click_coords.clamp_min(0)
            click_coords[..., 0].clamp_(max=self.model.h - 1)
            click_coords[..., 1].clamp_(max=self.model.w - 1)
            point_coords.append(click_coords)
            point_labels.append(labels.long())
        return torch.stack(point_coords), torch.stack(point_labels)

    @torch.no_grad()
    def predict(self, image, image_embedding, click_lists, prev_logits, output_size):
        click_coords, click_labels = self._click_lists_to_tensors(click_lists, output_size, image.device)
        if image_embedding is not None and torch.is_tensor(image_embedding) and image_embedding.ndim != 4:
            image_embedding = None
        return self.model(
            image=image,
            image_embedding=image_embedding,
            click_coords=click_coords,
            click_labels=click_labels,
            prev_mask_logits=prev_logits,
            output_size=output_size,
        )


class InteractiveEvaluator:
    def __init__(
        self,
        model: InteractiveModel,
        dataset,
        batch_size: int,
        device: str,
        max_clicks: int,
        thresholds: Iterable[int],
        num_workers: int,
        deterministic_clicks: bool,
        use_prev_mask: bool,
        dtype: torch.dtype,
    ):
        self.model = model.to(device).eval()
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.max_clicks = max_clicks
        self.thresholds = tuple(thresholds)
        self.num_workers = num_workers
        self.deterministic_clicks = deterministic_clicks
        self.use_prev_mask = use_prev_mask
        self.dtype = dtype

    def _autocast(self):
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        return torch.autocast(device_type=device_type, dtype=self.dtype, enabled=self.dtype != torch.float32)

    @staticmethod
    def _init_click_state(gt_masks: torch.Tensor, deterministic_clicks: bool):
        gt_np = gt_masks.detach().cpu().numpy()[:, 0] > 0.5
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

    @staticmethod
    def _append_next_clicks(gt_np, pred_np, click_lists, not_clicked_maps):
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

    @torch.no_grad()
    def eval(self, val_iters: int = 0) -> Dict:
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )
        total_batches = len(loader) if val_iters <= 0 else min(val_iters, len(loader))
        iou_by_click = {k: [] for k in range(1, self.max_clicks + 1)}
        noc_hits = {thr: [] for thr in self.thresholds}
        sample_records = []

        pbar = tqdm(enumerate(loader), total=total_batches, desc=f"Eval {self.model.name}")
        for batch_idx, batch in pbar:
            if batch_idx >= total_batches:
                break
            image, image_embedding, _, gt_mask = batch
            image = image.to(self.device, non_blocking=True)
            gt_mask = gt_mask.to(self.device, non_blocking=True)
            image_embedding = image_embedding.to(self.device, non_blocking=True) if torch.is_tensor(image_embedding) and image_embedding.ndim > 1 else None
            output_size = tuple(gt_mask.shape[-2:])
            gt_np, click_lists, not_clicked_maps = self._init_click_state(gt_mask, self.deterministic_clicks)

            prev_logits = None
            sample_ious = [[] for _ in range(gt_mask.shape[0])]
            reached = [{thr: None for thr in self.thresholds} for _ in range(gt_mask.shape[0])]
            for click_idx in range(1, self.max_clicks + 1):
                with self._autocast():
                    logits = self.model.predict(
                        image=image,
                        image_embedding=image_embedding,
                        click_lists=click_lists,
                        prev_logits=prev_logits if self.use_prev_mask else None,
                        output_size=output_size,
                    )
                pred_np = logits.detach().float().cpu().numpy()[:, 0] > 0.0
                batch_ious = [mask_iou_np(pred_np[i], gt_np[i]) for i in range(len(gt_np))]
                for sample_idx, iou in enumerate(batch_ious):
                    iou_by_click[click_idx].append(iou)
                    sample_ious[sample_idx].append(iou)
                    for thr in self.thresholds:
                        if reached[sample_idx][thr] is None and iou >= thr / 100.0:
                            reached[sample_idx][thr] = click_idx

                if click_idx < self.max_clicks:
                    self._append_next_clicks(gt_np, pred_np, click_lists, not_clicked_maps)
                    prev_logits = logits.detach() if self.use_prev_mask else None

                pbar.set_postfix({"IoU": f"{safe_mean(batch_ious):.4f}", "click": click_idx})

            for sample_idx in range(gt_mask.shape[0]):
                for thr in self.thresholds:
                    noc_hits[thr].append(reached[sample_idx][thr] or (self.max_clicks + 1))
                sample_records.append(
                    {
                        "sample_index": batch_idx * self.batch_size + sample_idx,
                        "ious": sample_ious[sample_idx],
                        "noc": {str(thr): reached[sample_idx][thr] or (self.max_clicks + 1) for thr in self.thresholds},
                    }
                )

        return {
            "model": self.model.name,
            "num_samples": len(sample_records),
            "max_clicks": self.max_clicks,
            "thresholds": list(self.thresholds),
            "NoC": {f"NoC@{thr}": safe_mean(noc_hits[thr]) for thr in self.thresholds},
            "IoU": {f"IoU@{k}": safe_mean(iou_by_click[k]) for k in range(1, self.max_clicks + 1)},
            "samples": sample_records,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAM/RopeSAM interactive segmentation")
    parser.add_argument("--model", choices=["sam", "rope_sam", "both"], default="both")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--val_iters", type=int, default=0, help="Number of batches to evaluate; 0 means all")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_clicks", type=int, default=10)
    parser.add_argument("--thresholds", type=str, default="60,70,75,80,85,90,95,98")
    parser.add_argument("--deterministic_clicks", action="store_true", help="Use center-most initial click instead of random sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")

    parser.add_argument("--dataset", choices=["hqseg44k", "cocolvis", "coconut_hf"], default="coconut_hf")
    parser.add_argument("--dataset_split", choices=["train", "val"], default="val")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--subset_index", type=str, default=None)
    parser.add_argument("--mask_filter_thresh", type=float, default=0.1)
    parser.add_argument("--image_size_encoder", type=int, default=1024)
    parser.add_argument("--image_size_mask", type=int, default=1024)

    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--sam_model_type", choices=sorted(sam_model_registry.keys()), default="vit_b")
    parser.add_argument("--sam_no_prev_mask", action="store_true", help="Disable previous-mask feedback for SAM")

    parser.add_argument("--rope_sam_checkpoint", type=str, default=None)
    parser.add_argument("--rope_sam_config", choices=sorted(builder_map["rope_sam"].keys()), default="rope_sam_dim384")
    parser.add_argument("--image_encoder_checkpoint", type=str, default=None)
    parser.add_argument("--image_encoder_config", choices=sorted(builder_map["image_encoder"].keys()), default="dino_v3_vits")
    parser.add_argument("--rope_no_prev_mask", action="store_true", help="Disable previous-mask feedback for RopeSAM")

    parser.add_argument("--image_feature_cache_dir", type=str, default="")
    parser.add_argument("--sam_cache_model_name", type=str, default="sam_vitb")
    parser.add_argument("--rope_cache_model_name", type=str, default=None)
    parser.add_argument("--image_feature_cache_max_shard", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model in ("sam", "both") and not args.sam_checkpoint:
        raise ValueError("--sam_checkpoint is required when evaluating SAM")
    if args.model in ("rope_sam", "both") and not args.rope_sam_checkpoint:
        raise ValueError("--rope_sam_checkpoint is required when evaluating RopeSAM")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    thresholds = tuple(int(x) for x in args.thresholds.split(",") if x)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.model in ("sam", "both"):
        jobs.append(("sam", SAMInteractiveModel(args.sam_checkpoint, args.sam_model_type), args.sam_cache_model_name, not args.sam_no_prev_mask))
    if args.model in ("rope_sam", "both"):
        rope_cache_name = args.rope_cache_model_name or args.image_encoder_config
        jobs.append(
            (
                "rope_sam",
                RopeSAMInteractiveModel(
                    checkpoint=args.rope_sam_checkpoint,
                    config=args.rope_sam_config,
                    image_encoder_checkpoint=args.image_encoder_checkpoint,
                    image_encoder_config=args.image_encoder_config,
                    max_clicks=args.max_clicks,
                    device=args.device,
                ),
                rope_cache_name,
                not args.rope_no_prev_mask,
            )
        )

    all_results = {}
    for model_name, model, cache_model_name, use_prev_mask in jobs:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        image_feature_cache = build_cache(args, cache_model_name)
        dataset = build_dataset(args, image_feature_cache)
        evaluator = InteractiveEvaluator(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            device=args.device,
            max_clicks=args.max_clicks,
            thresholds=thresholds,
            num_workers=args.num_workers,
            deterministic_clicks=args.deterministic_clicks,
            use_prev_mask=use_prev_mask,
            dtype=dtype,
        )
        result = evaluator.eval(val_iters=args.val_iters)
        all_results[model_name] = result

        print(f"\n{model_name} results:")
        print(json.dumps({"NoC": result["NoC"], "IoU": result["IoU"], "num_samples": result["num_samples"]}, indent=2))

    payload = {
        "args": vars(args),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "results": all_results,
    }
    result_path = outdir / f"interactive_eval_{args.dataset}_{args.dataset_split}_{payload['timestamp']}.json"
    with open(result_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results to {result_path}")


if __name__ == "__main__":
    main()
