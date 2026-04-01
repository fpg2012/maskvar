# %%
from maskvar.datasets.coco_lvis import LvisDataset
from maskvar.datasets.hqseg44k import HQSeg44KTrainDataset, HQSeg44KTestDataset
from maskvar.datasets.mask_level_dataset import MaskLevelDatasetRandom
from maskvar.datasets import instance_info
from maskvar.datasets import CoconutHFDataset

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from itertools import islice

from torchvision.utils import make_grid

# %%
# datasets = LvisDataset(
#     dataset_path='data/coco_lvis',
#     split='val',
#     img_split='val',
#     stuff_prob=0.0,
# )
# dataset = HQSeg44KTrainDataset(
#     data_root='../data/sam-hq',
#     img_size=(512, 512)
# )
dataset = CoconutHFDataset(
    parquet_path="../data/coconut_cvpr2024/",
    image_root="../data/coco_lvis/train/images",
)

# %%
def visualize_mask(image, layers, instances_info, alpha=0.5):
    plt.imshow(image // 2)
    for l in range(layers.shape[-1]):
        masked_data = np.ma.masked_where(layers[:, :, l] == 0, layers[:, :, l])
        plt.imshow(masked_data, alpha=alpha, cmap='tab20c')
    plt.show()

# %%
data_iter = iter(dataset)

# %%
image, layers, instances_info = next(data_iter)

print(image.shape)
print(layers.shape)
print(instances_info)
visualize_mask(image, layers, instances_info, alpha=0.5)


# %%
mask = layers[:, :, instances_info[0].mapping[0]]
plt.imshow(mask==instances_info[0].mapping[1])


