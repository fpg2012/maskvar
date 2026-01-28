import argparse
from typing import List, Tuple
from pathlib import Path
import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from maskvar.maskseg_build_everything import builder_map

def build_index_mapping(dataset: Dataset) -> List[Tuple[int, int]]:
    index_mapping = []

    for i, sample in enumerate(tqdm(dataset)):
        for instance_idx in sample[2].keys():
            index_mapping.append((i, instance_idx))

    return index_mapping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    
    train_set, val_set = builder_map['dataset'][args.dataset]()

    index_mapping_train = build_index_mapping(train_set)
    index_mapping_val = build_index_mapping(val_set)

    os.makedirs(out_dir / args.dataset, exist_ok=True)

    im_np = np.array(index_mapping_train)
    np.save(out_dir / args.dataset / 'train_index_mapping.npy', im_np)

    im_np = np.array(index_mapping_val)
    np.save(out_dir / args.dataset / 'val_index_mapping.npy', im_np)

    