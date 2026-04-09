# COCONut 数据集准备指南

## 简介

COCONut 使用与 COCO 相同的图片（train2017/val2017），但提供了更高质量、人工验证的分割 mask。

## 数据格式

项目使用自定义的 pickle 格式存储层次化分割标注。目录结构如下：

```
coconut/
├── train/
│   ├── images/              # COCO train2017 图片 (.jpg)
│   ├── masks/               # 压缩的 mask 文件 (.pickle)
│   │   ├── 000000000001.pickle
│   │   └── ...
│   └── hannotation.pickle   # 层次化标注信息
└── val/
    ├── images/              # COCO val2017 图片 (.jpg)
    ├── masks/               # 压缩的 mask 文件 (.pickle)
    └── hannotation.pickle   # 层次化标注信息
```

## 数据准备步骤

### 1. 下载 COCO 图片

由于 COCONut 使用 COCO 的图片，先下载 COCO 2017：

```bash
# 下载 COCO 图片
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

unzip train2017.zip
unzip val2017.zip
```

### 2. 下载 COCONut 标注

从官方仓库下载 COCONut 标注文件：
- GitHub: https://github.com/bytedance/coconut_cvpr2024
- 或使用 HuggingFace: `datasets/xdeng77/coconut_pancap`

### 3. 转换为项目格式

需要将 COCONut 的 COCO 格式 JSON 转换为项目的 pickle 格式。

#### hannotation.pickle 格式：

```python
{
    "image_id_1": {
        "hierarchy": {
            0: {"parent": None, "children": [1, 2], "node_level": 0},  # 实例层次关系
            1: {"parent": 0, "children": [], "node_level": 1},
            2: {"parent": 0, "children": [], "node_level": 1},
            # ...
        },
        "num_instance_masks": 10  # thing 类别数量（不含 stuff）
    },
    # ...
}
```

#### mask pickle 格式（每个图片一个文件）：

```python
# (encoded_layers, objs_mapping)
(
    [compressed_layer_1, compressed_layer_2, ...],  # cv2.imencode 压缩的 mask 层
    {
        0: (layer_index, mask_id),  # 实例ID -> (层索引, mask ID)
        1: (layer_index, mask_id),
        # ...
    }
)
```

### 4. 参考转换脚本

可以使用类似以下的脚本进行转换：

```python
import json
import pickle
import cv2
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils

def convert_coconut_to_pickle(coconut_json_path, output_dir):
    """
    将 COCONut COCO 格式转换为项目 pickle 格式
    """
    with open(coconut_json_path) as f:
        coco_data = json.load(f)

    # 构建 image_id -> annotations 映射
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    hannotation = {}

    for img_id, anns in image_annotations.items():
        # 处理 mask（需要处理重叠，使用层次化编码）
        # ... 实现 mask 层打包逻辑

        # 构建层次关系（COCONut 是平铺的，可能需要重新构建）
        hierarchy = {}
        for i, ann in enumerate(anns):
            hierarchy[i] = {
                "parent": None,  # COCONut 没有显式层次关系
                "children": [],
                "node_level": 0
            }

        hannotation[str(img_id)] = {
            "hierarchy": hierarchy,
            "num_instance_masks": len(anns)
        }

    # 保存 hannotation
    with open(output_dir / "hannotation.pickle", 'wb') as f:
        pickle.dump(hannotation, f)
```

## 使用方法

```python
from maskvar.datasets import CoconutDataset
from maskvar.datasets import MaskLevelDataset
from maskvar.datasets import ImageFeatureCache

# 创建数据集
dataset = CoconutDataset(
    dataset_path='/path/to/coconut',
    split='train',
    img_split='train2017',
    stuff_prob=0.0
)

# 使用 MaskLevelDataset 进行训练
cache = ImageFeatureCache('/path/to/image_features', 'train', len(dataset))
mask_level_dataset = MaskLevelDataset(
    dataset=dataset,
    device='cuda',
    image_feature_cache=cache
)
```

## 与 LVIS 数据集的对比

| 特性 | LVIS | COCONut |
|------|------|---------|
| 图片 | COCO train2017/val2017 | 相同 |
| Mask 质量 | 原始 COCO/LVIS 标注 | 人工验证，更高质量 |
| 实例数量 | ~150万 | ~500万 |
| 层次结构 | 有（LVIS 层次） | 平铺（需适配） |
| 格式 | 项目 pickle 格式 | 需转换为 pickle 格式 |

## 注意事项

1. **图片路径**：COCONut 使用 COCO 图片，确保 `img_split` 参数正确设置为 `'train2017'` 或 `'val2017'`

2. **Mask 统计**：`DEFAULT_NUM_MASKS_SPLITS` 需要根据实际数据统计更新

3. **层次结构**：COCONut 标注是平铺的（无显式父子关系），如果项目需要层次结构，可能需要在转换时构建伪层次

4. **内存使用**：COCONut 有更多 mask，确保 `ImageFeatureCache` 有足够的存储空间
