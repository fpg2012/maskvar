# COCONut HuggingFace 数据集使用指南

## 概述

`CoconutHFDataset` 直接从 HuggingFace parquet 文件加载 COCONut 数据集，无需预转换格式。

## 文件位置

```
~/workspace/coconut_cvpr2024/
├── train-00000-of-00002.parquet  # 训练数据 Part 1
├── train-00001-of-00002.parquet  # 训练数据 Part 2
└── ...

# COCO 图片位置（COCONut 使用相同图片）
datasets/coco/
└── train2017/                    # COCO train2017 图片
    ├── 000000000001.jpg
    └── ...
```

## 使用方法

### 1. 直接使用 HuggingFace 格式（动态转换）

```python
from maskvar.datasets import CoconutHFDataset
from maskvar.datasets import MaskLevelDataset, ImageFeatureCache

# 创建数据集
dataset = CoconutHFDataset(
    parquet_path="~/workspace/coconut_cvpr2024/train-00000-of-00002.parquet",  # 或目录路径
    image_root="datasets/coco/train2017",  # COCO 图片根目录
    stuff_prob=0.0,  # 不保留背景
)

# 获取单个样本
image, layers, instances_info = dataset[0]
print(f"Image shape: {image.shape}")  # (H, W, 3)
print(f"Layers shape: {layers.shape}")  # (H, W, L)
print(f"Num instances: {len(instances_info)}")

# 用于训练（配合 MaskLevelDataset）
cache = ImageFeatureCache("./image_features", "train", len(dataset))
mask_level_ds = MaskLevelDataset(
    dataset=dataset,
    device="cuda",
    image_feature_cache=cache,
)
```

### 2. 批量转换项目格式（推荐用于多次训练）

```python
from maskvar.datasets import CoconutHFConverter

converter = CoconutHFConverter(
    parquet_path="~/workspace/coconut_cvpr2024/train-00000-of-00002.parquet",
    image_root="datasets/coco/train2017",
    output_dir="datasets/coconut/train",
)
converter.convert()

# 然后使用标准 CoconutDataset
dataset = CoconutDataset(dataset_path="datasets/coconut", split="train")
```

### 3. 加载多个 parquet 文件

```python
# 方式1：传入目录（自动合并所有 parquet 文件）
dataset = CoconutHFDataset(
    parquet_path="~/workspace/coconut_cvpr2024/",  # 包含多个 .parquet 的目录
    image_root="datasets/coco/train2017",
)

# 方式2：使用列表（手动指定）
import pandas as pd
from maskvar.datasets import CoconutHFDataset

# 先手动合并
df1 = pd.read_parquet("~/workspace/coconut_cvpr2024/train-00000-of-00002.parquet")
df2 = pd.read_parquet("~/workspace/coconut_cvpr2024/train-00001-of-00002.parquet")
df = pd.concat([df1, df2], ignore_index=True)

# 保存合并后的文件（可选）
df.to_parquet("~/workspace/coconut_cvpr2024/train_merged.parquet")

# 然后加载
dataset = CoconutHFDataset(
    parquet_path="~/workspace/coconut_cvpr2024/train_merged.parquet",
    image_root="datasets/coco/train2017",
)
```

## Parquet 文件结构

```python
{
    'mask': {
        'bytes': b'\x89PNG\r\n...'  # PNG 编码的全景分割 mask
    },
    'segments_info': {
        'file_name': '000000000009.png',
        'image_id': 9,
        'segments_info': [
            {
                'category_id': 189,
                'id': 12345,
                'bbox': [x, y, w, h],
                'area': 1234
            },
            ...
        ]
    },
    'image_info': {
        'coco_url': 'http://images.cocodataset.org/train2017/000000000009.jpg',
        'date_captured': '2013-11-14 11:18:45',
        'file_name': '000000000009.jpg',
        'flickr_url': '...',
        'height': 480,
        'id': 9,
        'license': 4,
        'width': 640
    }
}
```

## 处理流程

```
Parquet File
    │
    ├── mask (PNG bytes) ──► decode PNG ──► panoptic mask (H, W)
    │
    ├── segments_info ─────► category IDs ──► stuff/things 分类
    │
    └── image_info ────────► file_name ──► 加载 COCO 图片
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────┐
    │                  On-the-fly Conversion              │
    │                                                     │
    │  panoptic mask ──► split instances ──► layers (H,W,L)│
    │                                                     │
    │  segments_info ──► filter stuff ──► instances_info  │
    └─────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         (image, layers, instances_info)
```

## 性能优化

### 1. 图片缓存
```python
dataset = CoconutHFDataset(
    parquet_path="~/workspace/coconut_cvpr2024/train-00000-of-00002.parquet",
    image_root="datasets/coco/train2017",
    cache_images=True,  # 缓存解码后的图片在内存中
)
# 注意：这会占用大量内存，仅适用于小数据集
```

### 2. 预计算 Image Features
```python
from maskvar.datasets import ImageFeatureCache

# 提前提取 image features
cache = ImageFeatureCache("./cache", "train", len(dataset))
for i in range(len(dataset)):
    image, _, _ = dataset[i]
    # 使用 SAM encoder 提取 feature...
    # cache.save(i, feature)

# 然后使用缓存训练
mask_level_ds = MaskLevelDataset(
    dataset=dataset,
    device="cuda",
    image_feature_cache=cache,
)
```

### 3. 使用转换后的格式
如果需要多次训练，推荐先转换为项目格式：

```python
from maskvar.datasets import CoconutHFConverter

converter = CoconutHFConverter(
    parquet_path="~/workspace/coconut_cvpr2024/",
    image_root="datasets/coco/train2017",
    output_dir="datasets/coconut/train",
)
converter.convert()
# 转换后使用 CoconutDataset 加载更快
```

## 与 CoconutDataset 的区别

| 特性 | CoconutHFDataset | CoconutDataset |
|------|------------------|----------------|
| 输入格式 | HuggingFace Parquet | 项目 Pickle 格式 |
| 转换时机 | 实时（on-the-fly） | 预转换 |
| 训练速度 | 较慢（需要解码 PNG） | 较快 |
| 磁盘占用 | 小（复用 COCO 图片） | 大（需要存储转换后的 masks）|
| 适用场景 | 快速实验、数据检查 | 大规模训练 |

## 常见问题

### Q: 找不到图片？
确保 `image_root` 指向 COCO train2017 目录：
```python
dataset = CoconutHFDataset(
    parquet_path="~/workspace/coconut_cvpr2024/train-00000-of-00002.parquet",
    image_root="/path/to/coco/train2017",  # 确保这个目录存在且包含 .jpg 文件
)
```

### Q: 如何处理多个 parquet 文件？
方式1：传入目录路径
```python
dataset = CoconutHFDataset(parquet_path="~/workspace/coconut_cvpr2024/", ...)
```

方式2：合并后使用
```python
import pandas as pd

files = ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
df.to_parquet("train_merged.parquet")
```

### Q: stuff_prob 的作用？
- `stuff_prob=0.0`（默认）：移除所有背景（stuff）对象，只保留前景（things）
- `stuff_prob=1.0`：保留所有背景对象
- `stuff_prob=0.5`：以 50% 概率保留每个背景对象
