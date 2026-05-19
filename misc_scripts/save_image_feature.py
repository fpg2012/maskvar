import argparse
import math
import os
import sys
from itertools import islice
from pathlib import Path
import time
import json
import numpy as np
import gc

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maskvar.maskseg_build_everything import builder_map
from maskvar.utils import preprocess_image

torch.set_float32_matmul_precision('high')

def my_collate_fn(batch, image_size_encoder, device, dtype):
    return torch.stack([preprocess_image(image, image_size_encoder, device, dtype) for image, _, _, _ in batch])

def cache_image_features_dry_run(dataset, image_encoder, image_size_encoder, dtype=torch.float32, batch_size=16, shard_size=512, device='cpu'):
    collate_fn = lambda batch: my_collate_fn(batch, image_size_encoder, device, dtype)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    encode_times = []
    feature = None
    for index, batch in enumerate(islice(dataloader, 5)):
        start_time = time.time()
        feature = image_encoder(batch).to(dtype=dtype)
        end_time = time.time()
        if index > 0:
            encode_times.append(end_time - start_time)

    if feature is None:
        raise ValueError("Cannot cache features for an empty dataset.")

    avg_encode_time = sum(encode_times) / max(len(encode_times), 1)
    num_images = len(dataset)
    num_shards = math.ceil(num_images / shard_size)
    feature_size_in_bytes = feature[0].numel() * feature[0].element_size() * shard_size
    feature_shape = list(feature.shape)
    feature_shape[0] = shard_size
    print(f"num images: {num_images}, num shards: {num_shards}")
    print(f"estimate disk usage: {num_shards * feature_size_in_bytes / (1024*1024*1024):.2f} GB")
    print(f"estimate encode time: {len(dataloader) * avg_encode_time:.2f} seconds")
    print(f"feature shape: {feature_shape}")
    
    return {
        "num_images": num_images,
        "num_shards": num_shards,
        "feature_size_in_bytes": feature_size_in_bytes,
        "avg_encode_time": avg_encode_time,
        "feature_shape": feature_shape,
    }

def get_unique_image_indices(index_mapping_path: Path, subset_index_path: Path | None = None):
    index_mapping = np.load(index_mapping_path)
    if subset_index_path is not None:
        subset_indices = np.load(subset_index_path)
        index_mapping = index_mapping[subset_indices]
    image_indices = np.unique(index_mapping[:, 0].astype(np.int64))
    return image_indices

def subset_dataset_by_indices(dataset, indices):
    if indices is None:
        return dataset
    return Subset(dataset, indices.tolist())

@torch.no_grad()
def cache_image_features(save_dir, dataset, image_encoder, image_size_encoder, dtype=torch.float32, batch_size=16, shard_size=512, device='cpu'):
    collate_fn = lambda batch: my_collate_fn(batch, image_size_encoder, device, dtype)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    arr = []
    arr_count = 0
    cur_shard_index = 0

    def flush_shard(force=False):
        nonlocal arr, arr_count, cur_shard_index
        while arr_count >= shard_size or (force and arr_count > 0):
            shard = np.concatenate(arr, axis=0)
            if not force and shard.shape[0] < shard_size:
                return
            write_count = min(shard_size, shard.shape[0])
            np.save(save_dir / f"batch_{cur_shard_index:06d}.npy", shard[:write_count])
            cur_shard_index += 1
            shard = shard[write_count:]
            arr = [shard] if shard.shape[0] > 0 else []
            arr_count = shard.shape[0]
            gc.collect()

    for index, batch in enumerate(tqdm(dataloader, desc="Caching image features")):
        feature = image_encoder(batch).to(dtype=dtype)
        feature_np = feature.detach().cpu().numpy()
        arr.append(feature_np)
        arr_count += feature_np.shape[0]
        flush_shard()

    flush_shard(force=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default='data/cache')
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--shard_size", type=int, default=512)
    parser.add_argument("--skip_dry_run", action="store_true", default=False)
    parser.add_argument("--yes", action="store_true", default=False, help="Skip interactive confirmation after dry run.")
    parser.add_argument("--train_subset_index", type=str, default=None)
    parser.add_argument("--val_subset_index", type=str, default=None)
    parser.add_argument("--index_mapping_dir", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])

    args = parser.parse_args()

    model_name = args.model_name
    dataset = args.dataset
    cache_dir = Path(args.cache_dir)
    ckpt = args.ckpt
    device = args.device
    dtype = getattr(torch, args.dtype)
    batch_size = args.batch_size
    shard_size = args.shard_size
    if dtype == torch.bfloat16:
        raise ValueError("bfloat16 cannot be saved as numpy arrays here; use --dtype float16 or float32.")

    assert model_name in builder_map["image_encoder"].keys()
    image_encoder: nn.Module = builder_map["image_encoder"][model_name](ckpt)
    image_encoder.to(device)
    image_encoder = torch.compile(image_encoder)
    image_encoder.eval()

    assert dataset in builder_map["dataset"].keys()
    train_set, val_set = builder_map["dataset"][dataset]()
    train_source_indices = None
    val_source_indices = None
    if args.train_subset_index or args.val_subset_index:
        index_mapping_dir = Path(args.index_mapping_dir) if args.index_mapping_dir else Path("data/flat") / dataset
        if args.train_subset_index:
            train_source_indices = get_unique_image_indices(
                index_mapping_dir / "train_index_mapping.npy",
                Path(args.train_subset_index),
            )
            train_set = subset_dataset_by_indices(train_set, train_source_indices)
            print(f"Using train subset image cache: {len(train_source_indices)} unique images")
        if args.val_subset_index:
            val_source_indices = get_unique_image_indices(
                index_mapping_dir / "val_index_mapping.npy",
                Path(args.val_subset_index),
            )
            val_set = subset_dataset_by_indices(val_set, val_source_indices)
            print(f"Using val subset image cache: {len(val_source_indices)} unique images")

    if "train" in args.splits:
        os.makedirs(cache_dir / model_name / f"{dataset}_train", exist_ok=True)
    if "val" in args.splits:
        os.makedirs(cache_dir / model_name / f"{dataset}_val", exist_ok=True)

    image_size_encoder = 1024
    
    if not args.skip_dry_run:
        if "train" in args.splits:
            print("=== estimating... train set ===")
            train_stats = cache_image_features_dry_run(train_set, image_encoder, image_size_encoder, dtype, batch_size, shard_size, device)
        else:
            train_stats = None
        if "val" in args.splits:
            print("=== estimating... val set ===")
            val_stats = cache_image_features_dry_run(val_set, image_encoder, image_size_encoder, dtype, batch_size, shard_size, device)
        else:
            val_stats = None
    else:
        train_stats = val_stats = None
    
    if args.dry_run:
        exit(0)
    
    if not args.yes:
        print("Continue? (y/n)")
        if input() != "y":
            exit(0)

    if train_stats is None:
        if "train" in args.splits:
            print("=== estimating metadata... train set ===")
            train_stats = cache_image_features_dry_run(train_set, image_encoder, image_size_encoder, dtype, batch_size, shard_size, device)
        if "val" in args.splits:
            print("=== estimating metadata... val set ===")
            val_stats = cache_image_features_dry_run(val_set, image_encoder, image_size_encoder, dtype, batch_size, shard_size, device)

    if "train" in args.splits:
        with open(cache_dir / model_name / f"{dataset}_train_metadata.json", "w") as f:
            json.dump({
                "count": train_stats["num_shards"],
                "num_images": train_stats["num_images"],
                "resolution": image_size_encoder,
                "feature_shape": train_stats["feature_shape"],
                "feature_dim": "BCHW",
                "feature_size_in_bytes": train_stats["feature_size_in_bytes"],
                "avg_encode_time": train_stats["avg_encode_time"],
                "batch_size": shard_size,
                "dtype": str(dtype),
                "source_indices": train_source_indices.tolist() if train_source_indices is not None else None,
            }, f, indent=2)
    
    if "val" in args.splits:
        with open(cache_dir / model_name / f"{dataset}_val_metadata.json", "w") as f:
            json.dump({
                "count": val_stats["num_shards"],
                "num_images": val_stats["num_images"],
                "resolution": image_size_encoder,
                "feature_shape": val_stats["feature_shape"],
                "feature_dim": "BCHW",
                "feature_size_in_bytes": val_stats["feature_size_in_bytes"],
                "avg_encode_time": val_stats["avg_encode_time"],
                "batch_size": shard_size,
                "dtype": str(dtype),
                "source_indices": val_source_indices.tolist() if val_source_indices is not None else None,
            }, f, indent=2)
    
    if "train" in args.splits:
        print("=== caching... train set ===")
        cache_image_features(cache_dir / model_name / f"{dataset}_train", train_set, image_encoder, image_size_encoder, dtype, batch_size, shard_size, device)
    if "val" in args.splits:
        print("=== caching... val set ===")
        cache_image_features(cache_dir / model_name / f"{dataset}_val", val_set, image_encoder, image_size_encoder, dtype, batch_size, shard_size, device)
    

    
