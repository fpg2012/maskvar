# Standalone `interaction` Package

`interaction` 是一个可独立复用的小包，核心依赖只有 `numpy` 和 `torch`。它主要提供 click sampler，以及 sampler 依赖的 CPU/CUDA 扩展算子。

## Package Contents

使用本包时，目录中至少应包含以下文件：

```text
interaction/
├── __init__.py
├── backend.py
├── interaction.py
├── sampler.py
├── session.py
├── types.py
└── csrc/
    ├── cpu_distance.cpp
    ├── cuda_distance.cpp
    └── cuda_distance_kernel.cu
```

说明：

- `csrc/` 不能缺失，因为第一次调用时会现场编译这里的 C++/CUDA 源码
- `__pycache__/` 不需要保留
- 建议保留当前 `README.md`，便于后续接入

## Recommended Layout

最简单的集成方式不是单独制作 wheel，而是直接将 `src/interaction` 作为一个 Python package 放入目标项目。

例如，项目目录可以组织为：

```text
their_project/
├── interaction/
│   ├── __init__.py
│   ├── backend.py
│   ├── interaction.py
│   ├── sampler.py
│   ├── session.py
│   ├── types.py
│   └── csrc/
└── demo.py
```

此时可以直接：

```python
from interaction import ClickSampler
```

如果仍然保留 `src/interaction` 这一层目录结构，则使用：

```python
from src.interaction import ClickSampler
```

## Requirements

基础依赖：

- Python 3.10+
- `numpy`
- `torch`

如果只使用 CPU 路径，还需要：

- 可用的 C++ 编译器

如果使用 CUDA 路径，还需要：

- CUDA 版 PyTorch
- NVIDIA GPU
- 可供 PyTorch JIT 编译 `.cu` 文件的 CUDA toolchain

首次实际调用 sampler 或 backend 时，会通过 `torch.utils.cpp_extension.load(...)` 动态编译扩展，因此：

- 第一次调用会较慢
- 当前机器必须具备对应编译环境

## Public API

`__init__.py` 已经导出了这几个常用对象：

```python
from interaction import (
    CLK_NEGATIVE,
    CLK_POSITIVE,
    ClickSampler,
    Interaction,
    Session,
    create_session,
)
```

常用入口通常是：

- `ClickSampler`
- `Interaction`
- `create_session`

## Main Components

如果只需要单独使用 sampler，主要会接触两层接口：

1. 高层 Python 接口：`ClickSampler`
2. 低层独立算子：`cpu_distance_pair` 和 `cuda_sample_from_masks`

## High-Level API: `ClickSampler`

导入：

```python
from interaction import ClickSampler
```

构造：

```python
sampler = ClickSampler(seed=42, backend=None)
```

参数：

- `seed`：控制 Python RNG 和 CUDA RNG
- `backend=None`：自动选择；有 CUDA 时使用 `"cuda"`，否则使用 `"cpu"`
- `backend="cpu"`：强制 CPU
- `backend="cuda"`：强制 CUDA

主要方法：

- `compute_maps(gt, prev_output, clicks=None, ignore_mask=None)`
- `sample_click(gt, prev_output, clicks=None, ignore_mask=None, sfc_inner_k=1.7, return_maps=False)`
- `sample_clicks(gt, prev_output, num_clicks, clicks=None, ignore_mask=None, sfc_inner_k=1.7)`

输入约定：

- `gt`：前景位置必须为 `1`
- `prev_output`：内部按 `prev_output > 0` 转成预测前景
- 支持 `H x W` 或 `(1, H, W)`
- `ignore_mask`：同形状布尔 mask，`True` 表示该位置禁止采样
- `clicks`：历史点击，格式为 `[(y, x, mode), ...]`
- `mode`：`"positive"` 或 `"negative"`

返回约定：

- 单个点击：`(y, x, mode)`
- 无可采样点：`(None, None, None)`
- 如果 `return_maps=True`，还会返回一个 `ClickMaps`

### Minimal Example

```python
import numpy as np
from interaction import ClickSampler

gt = np.zeros((8, 8), dtype=np.uint8)
gt[2:6, 2:6] = 1

prev_output = np.zeros((8, 8), dtype=np.float32)

sampler = ClickSampler(seed=42, backend="cpu")
click, maps = sampler.sample_click(
    gt=gt,
    prev_output=prev_output,
    clicks=[],
    sfc_inner_k=1.7,
    return_maps=True,
)

print(click)
print(maps.mode)
print(maps.threshold)
```

## Sampling Rule

sampler 会先构造两类错误区域：

- false negative：`gt == 1` 且 `pred == 0`
- false positive：`gt == 0` 且 `pred == 1`

然后分别计算两张错误图的距离变换，并从“最大距离更大”的那一类错误区域里采点：

- false negative 更大时，返回 positive 点
- false positive 更大时，返回 negative 点

`sfc_inner_k` 用来控制候选中心区：

- `sfc_inner_k >= 1.0` 时，阈值为 `max_distance / sfc_inner_k`
- `sfc_inner_k < 0.0` 时，阈值为 `0`，等价于整块错误区域都可采
- `[0, 1)` 是非法区间

## Low-Level Operators

如果不使用 `ClickSampler`，也可以直接调用 `backend.py` 中的底层接口。

### 1. CPU Distance Operator

导入：

```python
from interaction.backend import cpu_distance_pair
```

调用：

```python
negative_dist, positive_dist = cpu_distance_pair(
    negative_mask,
    positive_mask,
)
```

作用：

- 只计算两张错误 mask 的距离图
- 不负责最终点坐标采样

输入要求：

- 输入会被转成连续的 CPU `torch.bool`
- 两个输入形状必须一致

返回：

- `negative_dist`：CPU `torch.float32`
- `positive_dist`：CPU `torch.float32`

### 2. CUDA Sampler Operator

导入：

```python
from interaction.backend import cuda_sample_from_masks
```

调用：

```python
payload = cuda_sample_from_masks(
    pred_mask,
    gt_mask,
    ignore_mask,
    sfc_inner_k=1.7,
    random_value=123,
    need_debug=True,
)
```

这是 sampler 对应的核心独立算子，内部会一次性完成：

1. 构造 `false_negative` 和 `false_positive`
2. 计算两类错误区域 bbox
3. 在裁剪区域上做距离变换
4. 决定本次采 positive 还是 negative
5. 依据 `sfc_inner_k` 构造候选区域
6. 用 `random_value % candidate_count` 选中一个点
7. 按需返回调试张量

输入要求：

- 输入会被转成连续的 CUDA `torch.bool`
- 所有输入必须是 2D
- 所有输入形状必须一致

返回的 `payload` 布局：

- `payload[0]`：`result`，形状 `(3,)`，`int32`
- `payload[1]`：`meta`，形状 `(4,)`，`float32`
- `payload[2]`：`negative_distance`
- `payload[3]`：`positive_distance`
- `payload[4]`：`candidate_mask`
- `payload[5]`：`false_negative`
- `payload[6]`：`false_positive`

`result` 含义：

- `result[0]`：采样出的 `y`
- `result[1]`：采样出的 `x`
- `result[2]`：`1` 表示 positive，`0` 表示 negative

无点可采时，`result` 为 `[-1, -1, -1]`。

`meta` 含义：

- `meta[0]`：`negative_max`
- `meta[1]`：`positive_max`
- `meta[2]`：模式标记，`1` 表示 positive，`0` 表示 negative
- `meta[3]`：本次候选阈值

### CUDA Operator Example

```python
import torch
from interaction.backend import cuda_sample_from_masks

pred_mask = torch.zeros((8, 8), device="cuda", dtype=torch.bool)
gt_mask = torch.zeros((8, 8), device="cuda", dtype=torch.bool)
gt_mask[2:6, 2:6] = True
ignore_mask = torch.zeros((8, 8), device="cuda", dtype=torch.bool)

payload = cuda_sample_from_masks(
    pred_mask=pred_mask,
    gt_mask=gt_mask,
    ignore_mask=ignore_mask,
    sfc_inner_k=1.7,
    random_value=42,
    need_debug=True,
)

result = payload[0]
meta = payload[1]
print(result.tolist())
print(meta.tolist())
```

## Full Interaction Loop

除了 sampler，本包还提供一个很薄的交互循环封装 `Interaction`。

使用时假设外部模型实现两个方法：

- `prepare(sessions)`
- `forward(sessions)`

最小接入方式：

```python
from interaction import ClickSampler, Interaction, create_session

sampler = ClickSampler(seed=42)
interaction = Interaction(model=your_model, sampler=sampler)

session = create_session(image=image, prev_output=prev_output, clicks=[])
sessions = interaction.prepare([session])
sessions = interaction.step(sessions=sessions, gts=[gt])
```

这里 `Interaction` 只负责：

- 调模型前向
- 从当前输出里继续采点击
- 把点击写回 session state

它不依赖当前仓库中的其他训练逻辑。

## Common Issues

- 缺少 `csrc/`，第一次调用就会失败
- 导入路径不正确，`interaction` 不在 Python 可搜索路径下
- 当前环境没有安装 `torch`
- 安装的是 CPU 版 `torch`，却强制指定 `backend="cuda"`
- 输入不是 2D 或 `(1, H, W)`
- `gt`、`prev_output`、`ignore_mask` 的空间尺寸不一致
