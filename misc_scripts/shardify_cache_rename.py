import numpy as np
import argparse
import json
from pathlib import Path
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--shard_size', type=int, required=True)
    parser.add_argument('--metadata', type=str, required=True)

    args = parser.parse_args()

    with open(args.metadata, "r") as f:
        metadata = json.load(f)
    
    count = metadata['count']

    b = metadata['feature_shape'][0]
    shard_size = args.shard_size
    
    cache_dir = Path(args.dir)
    out_dir = Path(args.out)
    os.makedirs(out_dir, exist_ok=True)

    new_count = 0

    for i in tqdm(list(range(0, count, shard_size))):
        arrays = []
        for j in range(shard_size):
            if i + j >= count:
                break
            arr = np.load(cache_dir / f'batch_{i:06d}.npy', mmap_mode='r')
            arrays.append(arr)
        
        # concat arrays
        cct = np.concat(arrays, axis=0)
        
        # save stacked array
        np.save(out_dir / f'batch_{i:06d}.npy', cct)
        new_count += 1

        for j in range(shard_size):
            if i + j >= count:
                break
            os.remove(cache_dir / f'batch_{i+j:06d}.npy')
    
    metadata['count'] = new_count
    metadata['feature_shape'][0] = b
    with open(out_dir / '..' / f'{Path(args.metadata).stem}_1.json', 'w') as f:
        json.dump(metadata, f)
