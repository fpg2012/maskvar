# MaskVQVAE 开发文档

## 概述

MaskVQVAE 是一个结合了 VQVAE 和 SAM MaskDecoder 的图像分割模型。它使用 VQVAE 的 encoder/quantizer 来编码 mask，使用类似 SAM 的 decoder 来利用图像特征重建 mask。

## 架构

```
Input: 0-1 mask (B,1,H,W) + 图像 feature (B,C_img,H_img,W_img)
    ↓
Encoder: 0-1 mask → 多尺度特征 f_BChw
    ↓
Quant: 多尺度量化 → 多尺度 token (ms_h_BChw)
    ↓
MaskDecoder: 图像feature + ms_h_BChw → 多尺度 mask logits
    ↓
Fusion: 像素空间相加 → 最终 mask (B,1,H,W)
```

## 核心组件

### 1. MaskVQVAE (`maskvar/models/mask_vqvae/mask_vqvae.py`)

主类，整合所有组件。

**关键方法：**
- `forward(mask, image_features, ret_usages=False, use_image_features=True)` - 前向传播
- `encode_to_indices(mask)` - 编码 mask 到多尺度索引
- `decode_from_indices(ms_idx_Bl, image_features, ...)` - 从索引解码 mask

**配置参数：**
- `vocab_size`: 码本大小 (默认: 4096)
- `z_channels`: 潜在空间通道数 (默认: 32)
- `v_patch_nums`: 多尺度 patch 数量 (默认: (1, 2, 4, 8, 16))
- `img_feat_dim`: 图像特征维度 (默认: 256, SAM ViT 输出)
- `transformer_dim`: Transformer 维度 (默认: 256)
- `fusion_type`: 多尺度融合策略 ('sum' 或 'weighted')

### 2. MaskDecoderModule (`maskvar/models/mask_vqvae/mask_decoder.py`)

核心 decoder 模块，复用 SAM MaskDecoder 的结构但重写 forward。

**关键方法：**
- `_decode_single_scale()` - 单尺度解码逻辑
- `forward()` - 多尺度解码并融合

**SAM 组件复用：**
- `self.sam_mask_decoder.transformer` - TwoWayTransformer
- `self.sam_mask_decoder.output_upscaling` - 4x 上采样模块

**单尺度解码流程：**
1. 下采样 image features 到当前尺度 (pn, pn)
2. 投影 tokens 和 image features 到 transformer_dim
3. Two-way cross attention (tokens ↔ image features)
4. 上采样 image features (4x SAM + 额外上采样)
5. tokens 经 MLP 后与 features 逐位置相乘得到 mask logits

### 3. 多尺度融合 (`fusion_modules` 概念)

目前融合模块内嵌在 MaskDecoderModule 中：
- `PixelSumFusion` - 直接像素相加
- `WeightedFusion` - 可学习权重相加

## 关键接口

### 输入/输出规范

**MaskVQVAE.forward()**
```python
Args:
    mask: (B, 1, H, W), 值域 [-1, 1] (归一化后的 0-1 mask)
    image_features: (B, C_img, H_img, W_img), SAM image encoder 输出
    ret_usages: bool, 是否返回 codebook 使用率
    use_image_features: bool, 是否使用图像特征解码

Returns:
    reconstructed_mask: (B, 1, H, W)
    usages: List[float] (if ret_usages=True)
    vq_loss: Tensor
```

**编码/解码流程**
```python
# 编码
ms_idx_Bl = model.encode_to_indices(mask)  # List[(B, pn*pn)]

# 解码
rec_mask = model.decode_from_indices(
    ms_idx_Bl,
    image_features,
    original_size=(H, W)
)
```

## 训练相关

### 损失函数

1. **VQ Loss** - Vector Quantization loss (来自 quantizer)
2. **Reconstruction Loss** - Mask 重建损失
   - 可以使用 Focal Loss (处理类别不平衡)
   - 或 MSE/L1 Loss

### 数据流

参考 `train_mask_vqvae.py`:
1. 加载 mask 和对应的 image features
2. Mask 归一化到 [-1, 1]
3. 前向传播计算 reconstruction 和 vq_loss
4. 反向传播更新 encoder/decoder/quantizer

### 与 VQVAE_Single 的区别

| 特性 | VQVAE_Single | MaskVQVAE |
|------|-------------|-----------|
| Decoder 类型 | 纯卷积 Decoder | SAM-style MaskDecoder |
| 使用图像特征 | 否 | 是 |
| 多尺度融合 | 残差累加 | 像素相加 (可配置) |
| 适用场景 | 纯 mask 重建 | 利用图像信息辅助分割 |

## 修改指南

### 1. 修改 Decoder 结构

**位置:** `mask_decoder.py` 中的 `MaskDecoderModule`

**场景 A: 替换 TwoWayTransformer**
```python
# 当前使用 SAM 的 transformer
self.transformer_ref = self.sam_mask_decoder.transformer

# 可以替换为自定义 transformer
self.transformer_ref = MyCustomTransformer(...)
```

**场景 B: 修改 mask logits 计算方式**
```python
# 当前: 相乘方式 (_decode_single_scale 中)
mask_logits = (upscaled_img * token_features_upscaled).sum(dim=1, keepdim=True)

# 可以改为: 卷积方式
mask_logits = self.mask_head(torch.cat([upscaled_img, token_features_upscaled], dim=1))
```

### 2. 修改多尺度融合策略

**位置:** `mask_decoder.py` 中的 `forward()` 方法

```python
# 当前: 使用 self.fusion (PixelSumFusion 或 WeightedFusion)
final_mask = self.fusion(all_mask_logits)

# 可以改为: 渐进式融合 (小尺度作为大尺度的条件)
final_mask = self.progressive_fusion(all_mask_logits)
```

### 3. 添加新的尺度配置

**位置:** `mask_vqvae.py` 中的 `__init__`

```python
# 当前: v_patch_nums=(1, 2, 4, 8, 16)
# 注意: 需要与 encoder 的 downsample 比例匹配

# 例如改为 8x8 最大尺度
v_patch_nums=(1, 2, 4, 8)
ddconfig=dict(
    ch_mult=(1, 2, 2, 4),  # 4 levels = 8x downsample
    ...
)
```

### 4. 修改图像特征处理方式

**位置:** `mask_decoder.py` 中的 `_decode_single_scale()`

```python
# 当前: 简单下采样
img_feat_scaled = F.interpolate(image_features, size=target_size, mode="area")

# 可以改为: 多尺度特征金字塔
img_feat_scaled = self.fpn(image_features, target_size)
```

## 消融实验配置

### 配置 1: 纯卷积 Decoder 基线
```python
model = MaskVQVAE(
    use_sam_mask_decoder=False,  # 使用内置 transformer
    fusion_type='sum',
)
```

### 配置 2: 不使用图像特征
```python
rec_mask = model(mask, use_image_features=False)  # 使用标准 decoder
```

### 配置 3: 加权融合
```python
model = MaskVQVAE(
    fusion_type='weighted',  # 学习各尺度权重
)
```

### 配置 4: 不同深度 Transformer
```python
model = MaskVQVAE(
    transformer_depth=4,  # 默认是 2
    transformer_num_heads=8,
)
```

## 调试技巧

### 检查中间输出形状

```python
# 在 _decode_single_scale 中添加打印
print(f"tokens: {tokens.shape}")
print(f"image_features: {image_features.shape}")
print(f"processed_tokens: {processed_tokens.shape}")
print(f"mask_logits: {mask_logits.shape}")
```

### 可视化各尺度 mask

```python
_, all_masks = model.mask_decoder(image_features, ms_tokens)
for i, m in enumerate(all_masks):
    visualize(m[0, 0])  # 可视化 batch 中第一个样本
```

### 检查梯度流

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm()}")
    else:
        print(f"{name}: no grad!")
```

## 性能优化

### 1. 梯度检查点
在 MaskDecoderModule 中添加:
```python
torch.utils.checkpoint.checkpoint(self.transformer_ref, ...)
```

### 2. 混合精度训练
```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    rec_mask, vq_loss = model(mask, image_features)
```

### 3. 编译模型
```python
model = torch.compile(model)
```

## 常见问题

### Q: mask 和 image feature 的尺寸不匹配？
A: SAM image encoder 通常 16x downsample。如果 mask 是 (256, 256)，image feature 应该是 (16, 16)。

### Q: v_patch_nums 和 encoder downsample 比例的关系？
A: v_patch_nums 的最大值应该等于 H/downsample。例如 256x256 输入，16x downsample，则最大 pn=16。

### Q: 如何加载预训练的 VQVAE 权重？
A: 可以先加载 VQVAE_Single 的 encoder/quantizer 权重到 MaskVQVAE 对应模块。

## 文件索引

| 文件 | 说明 |
|------|------|
| `maskvar/models/mask_vqvae/__init__.py` | 模块导出 |
| `maskvar/models/mask_vqvae/mask_vqvae.py` | 主类 |
| `maskvar/models/mask_vqvae/mask_decoder.py` | Decoder 实现 |
| `train_scripts/train_mask_vqvae.py` | 训练脚本 |
| `misc_scripts/test_mask_vqvae.py` | 测试脚本 |

## 参考资料

- SAM MaskDecoder: `maskvar/models/sam/mask_decoder.py`
- SAM TwoWayTransformer: `maskvar/models/sam/transformer.py`
- VQVAE_Single: `maskvar/models/vqvae_single.py`
- VectorQuantizer2: `maskvar/models/quant.py`
