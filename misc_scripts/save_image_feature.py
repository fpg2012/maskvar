import argparse
import os
from itertools import islice
from pathlib import Path
import time
import json
import numpy as np
import gc

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from maskvar.maskseg_build_everything import builder_map
from maskvar.utils import preprocess_image

torch.set_float32_matmul_precision('high')

def my_collate_fn(batch, image_size_encoder, device, dtype):
    return torch.stack([preprocess_image(image, image_size_encoder, device, dtype) for image, _, _, _ in batch])

def cache_image_features_dry_run(dataset, image_encoder, image_size_encoder, dtype=torch.float32, batch_size=16, shard_size=512, device='cpu'):
    collate_fn = lambda batch: my_collate_fn(batch, image_size_encoder, device, dtype)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    feature_size_in_bytes = 0
    avg_encode_time = 0.0
    for index, batch in enumerate(islice(dataloader, 5)):
        start_time = time.time()
        feature = image_encoder(batch).to(dtype=dtype)
        end_time = time.time()
        if index > 0:
            avg_encode_time += (end_time - start_time)
    feature_size_in_bytes = feature.numel() * feature.element_size()
    avg_encode_time /= 4

    total_len = (len(dataloader) + shard_size - 1) // shard_size
    feature_shape = list(feature.shape)
    feature_shape[0] = shard_size
    print(f"estimate disk usage: {total_len * feature_size_in_bytes * (shard_size // batch_size) / (1024*1024*1024)} GB")
    print(f"estimate encode time: {total_len * avg_encode_time * (shard_size // batch_size):.2f} seconds")
    print(f"feature shape: {feature_shape}")
    
    return total_len, feature_size_in_bytes, avg_encode_time, feature_shape

@torch.no_grad()
def cache_image_features(save_dir, dataset, image_encoder, image_size_encoder, dtype=torch.float32, batch_size=16, shard_size=512, device='cpu'):
    collate_fn = lambda batch: my_collate_fn(batch, image_size_encoder, device, dtype)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    arr = []
    cur_shard_index = 0
    for index, batch in enumerate(tqdm(dataloader, desc="Caching image features")):
        feature = image_encoder(batch).to(dtype=dtype)
        arr.append(feature.detach().cpu().numpy())
        if index % (shard_size // batch_size) == (shard_size // batch_size) - 1:
            np.save(save_dir / f"batch_{cur_shard_index:06d}.npy", np.concatenate(arr, axis=0))
            gc.collect()
            arr = []
            cur_shard_index += 1

    if len(arr) > 0:
        np.save(save_dir / f"batch_{cur_shard_index:06d}.npy", np.concatenate(arr, axis=0))
        arr = []
        gc.collect()

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

    args = parser.parse_args()

    model_name = args.model_name
    dataset = args.dataset
    cache_dir = Path(args.cache_dir)
    ckpt = args.ckpt
    device = args.device
    dtype = getattr(torch, args.dtype)
    batch_size = args.batch_size
    shard_size = args.shard_size

    assert model_name in builder_map["image_encoder"].keys()
    image_encoder: nn.Module = builder_map["image_encoder"][model_name](ckpt)
    image_encoder.to(device)
    image_encoder = torch.compile(image_encoder)
    image_encoder.eval()

    assert dataset in builder_map["dataset"].keys()
    train_set, val_set = builder_map["dataset"][dataset]()

    os.makedirs(cache_dir / model_name / f"{dataset}_train", exist_ok=True)
    os.makedirs(cache_dir / model_name / f"{dataset}_val", exist_ok=True)

    image_size_encoder = 1024
    
    print("=== estimating... train set ===")
    train_len, train_feature_size_in_bytes, train_avg_encode_time, train_shape = cache_image_features_dry_run(train_set, image_encoder, image_size_encoder, dtype, batch_size, shard_size, device)
    print("=== estimating... val set ===")
    val_len, val_feature_size_in_bytes, val_avg_encode_time, val_shape = cache_image_features_dry_run(val_set, image_encoder, image_size_encoder, dtype, batch_size, shard_size, device)
    
    if args.dry_run:
        exit(0)
    
    # ask for continue
    print("Continue? (y/n)")
    if input() != "y":
        exit(0)

    with open(cache_dir / model_name / f"{dataset}_train_metadata.json", "w") as f:
        json.dump({
            "count": train_len + shard_size - 1,
            "resolution": image_size_encoder,
            "feature_shape": train_shape,
            "feature_dim": "BCHW",
            "feature_size_in_bytes": train_feature_size_in_bytes,
            "avg_encode_time": train_avg_encode_time,
            "batch_size": train_shape[0],
            "dtype": str(dtype),
        }, f)
    
    with open(cache_dir / model_name / f"{dataset}_val_metadata.json", "w") as f:
        json.dump({
            "count": val_len,
            "resolution": image_size_encoder,
            "feature_shape": val_shape,
            "feature_dim": "BCHW",
            "feature_size_in_bytes": val_feature_size_in_bytes,
            "avg_encode_time": val_avg_encode_time,
            "batch_size": train_shape[0],
            "dtype": str(dtype),
        }, f)
    
    print("=== caching... train set ===")
    cache_image_features(cache_dir / model_name / f"{dataset}_train", train_set, image_encoder, image_size_encoder, dtype, batch_size, shard_size, device)
    print("=== caching... val set ===")
    cache_image_features(cache_dir / model_name / f"{dataset}_val", val_set, image_encoder, image_size_encoder, dtype, batch_size, shard_size, device)
    

    