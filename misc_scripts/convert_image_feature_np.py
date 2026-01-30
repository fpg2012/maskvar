import numpy as np
import argparse
import torch
from pathlib import Path
import os
from tqdm import tqdm

def convert(file: Path):
    # load tensor
    t = torch.load(file, map_location="cpu", mmap=True).detach()
    np.save(file.with_suffix(".npy"), t.numpy())
    # delete original file
    os.remove(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    file_list = list(args.dir.glob("*.pt"))
    for file in tqdm(file_list):
        convert(file)