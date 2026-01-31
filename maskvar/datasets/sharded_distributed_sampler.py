import torch
import numpy as np
from torch.utils.data import Sampler

class ShardedDistributedSampler(Sampler):

    def __init__(self, dataset, rank=0, world_size=1, epoch=0, shard_size=1024, seed=42):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        
        self.shard_size = shard_size

        assert shard_size % world_size == 0 and "shard_size % world_size should be 0"

        ds_len = len(dataset)

        self.all_indices = np.arange(ds_len)

        to_pad = shard_size - ds_len % shard_size
        self.all_indices = np.concatenate((self.all_indices, self.all_indices[:to_pad]))
        # to_pad = len(self.all_indices) % world_size
        # self.all_indices = np.concatenate((self.all_indices, self.all_indices[:to_pad]))

        self.shuffled_indices = self._shuffle()

    def __len__(self):
        return len(self.shuffled_indices[self.rank])

    def _shuffle(self):
        # divide
        indices = self.all_indices.reshape(-1, self.shard_size)
        self.rng.shuffle(indices, axis=1)
        self.rng.shuffle(indices, axis=0)
        indices = self.all_indices.reshape(self.world_size, -1)
        return indices
    
    def set_epoch(self, epoch: int):
        self.rng = np.random.default_rng(self.seed + epoch)
        self.shuffled_indices = self._shuffle()

    def __iter__(self):
        yield from self.shuffled_indices[self.rank]