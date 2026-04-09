# SimpleMaskVQVAE 技术总结

## 概述

SimpleMaskVQVAE目标是构建一个轻量、直接、能结合图像特征的 mask 自编码器。

## 设计动机

尝试这个方案主要基于以下几点考虑：

### 1. 简化两阶段任务的复杂度

原先 VQVAE + VAR 的两阶段方案中：
- **第一阶段（VQVAE）**：只学习形状信息，将 mask 编码为离散的 tokens
- **第二阶段（VAR）**：需要学习形状与 image tokens 之间的对应关系

这个任务对 VAR 来说相当复杂——只有两层的 transformer 似乎难堪大任。如果增加太多参数量，一方面相比直接使用 SAM MaskDecoder 没有优势，另一方面也更难训练。因此有必要把第二阶段的任务简化，这意味着第一阶段就不得不引入形状之外的信息（如图像特征）。

### 2. 借鉴 SAM MaskDecoder 的聚类思想

SAM 的 MaskDecoder 可以这样理解：
- 通过点击编码，提取/摒弃某个位置附近的特征，存入 query token
- 上采样后，计算相似度，把相似特征的区域找出来

相比去对应形状和 image token，这种类似聚类/分类的方法对模型来说显然更容易。既然点击信息可以用于提取我们想要的某一类特征，粗糙的 mask 信息应该也可以起到类似作用。

### 3. 关于 VAR 结合的思考

VAR 本身是一种 coarse-to-fine 的生成方法，但如何与当前架构结合还需要探索：

- **方案 A**：让 VAR 继续预测 mask token——但这似乎没有把模型面临的问题变简单
- **方案 B**：让 VAR 预测 query token——但这又没有理由用 VAR 这么复杂的结构，也不好从 coarse-to-fine 的角度讲故事

原先引入生成模型的动机在于：

1. coarse to fine提高精度
2. 多粒度分割
3. 能结合VLM？

## 模型结构

### 整体架构

```
Input: mask (B,1,H,W) + image (B,3,H,W)
    ↓
MaskEncoderLite: CNN patch embed + MLP
    ↓ (b,h,w,c)
[Quantize] - 当前训练中跳过
    ↓
SimpleMaskDecoder: 双向交叉注意力 → mask logits
    ↓
Output: mask (B,1,H,W)
```

### 核心组件

#### 1. MaskEncoderLite

轻量级 mask 编码器：
- **Patch Embedding**: `Conv2d(1, dim, kernel_size=16, stride=16)`，将 1024x1024 mask 压缩为 64x64 tokens
- **MLP**: 两层 Linear + GELU，类似 transformer 的 FFN
- **输出**: `(B, H/16, W/16, dim)` 格式的 tokens

相比原版 VQVAE encoder（4-5 层 downsample + ResBlock），这个设计大幅简化了 mask 的编码过程。

#### 2. SimpleMaskDecoder

核心解码器，借鉴 SAM 的设计使用 query token 作为中介：

**Query Token 机制**:
- 可学习的 `query_tokens` (num_queries, dim) 作为 mask tokens 和 image tokens 之间的中介
- Query tokens 先从 mask tokens 提取信息（Cross Attention）（图中1）
- 再将信息传递给 image tokens（Reverse Cross Attention）（图中2）
- Block 1 和 Block 2 重复此过程两次，增强特征交互

**输出头**:（图中虚线右侧）
- HyperNetwork: 将 query token 映射到上采样维度
- 上采样: 2x ConvTranspose2d，将 image tokens 从 64x64 上采样到 256x256
- Mask 计算: `einsum('bc,bhwc->bhw')`，点积生成最终 mask

#### 3. SimpleCrossBlock / SimpleCrossBlockReverse

自定义交叉注意力实现：
- 使用 2D RoPE（旋转位置编码）处理 spatial features
- H 轴和 W 轴分别编码，增强位置感知
- 使用 `F.scaled_dot_product_attention` 进行高效注意力计算

#### 4. SimpleVectorQuantize

向量量化器（当前训练中跳过）：
- 支持 L2 距离或余弦相似度
- 包含码本使用率统计（EMA）
- 返回 vq_loss（codebook loss + commitment loss）

**当前状态**: 训练中 `vq_loss = 0`，相当于普通 autoencoder

## 训练策略与尝试

### 损失函数探索

实现了多种损失函数并进行了对比实验：

| 损失函数 | 说明 | 适用场景 |
|---------|------|---------|
| **NFL** (NormalizedFocalLossSigmoid) | 归一化 Focal Loss，处理类别不平衡 | 默认使用，对难样本更敏感 |
| FocalLoss | 标准 Focal Loss，α=0.75, γ=2.0 | 基础分割任务 |
| DICELoss | Dice 系数损失 | 边界对齐 |
| DICEFocalLoss | Dice + Focal 组合 | 平衡两者优势 |
| DICEBCELoss | Dice + BCE 组合 | 稳定收敛 |
| DiceNFLoss | Dice + NormalizedFocal | 更强的难样本处理 |

### 训练特性

1. **分布式训练**: 完整支持 DDP，使用 `ShardedDistributedSampler` 处理大数据集
2. **混合精度**: 支持 fp16/bf16/fp32，自动使用 `torch.autocast`
3. **梯度累积**: 支持大 batch size 模拟
4. **可视化**: 丰富的 TensorBoard 记录
   - Error Map: Blue(TP), Red(FP), Green(FN) 叠加在原图上
   - Logits Heatmap: 显示模型输出的原始 logits 分布
   - IoU 指标追踪

### 冻结实验

支持冻结不同组件进行消融：
- `--freeze_image_encoder`: 冻结图像编码器（SAM/TinyViT）
- `--freeze_mask_encoder`: 冻结 mask 编码器

## 关键设计决策

1. **跳过 VQ**: 当前训练中将 quantization 损失设为 0，先验证 autoencoder 能否有效重建 mask

2. **轻量级设计**: MaskEncoderLite 只有一层卷积 + 两层 MLP，参数量极小

3. **双向交叉注意力**: Decoder 中 query tokens 和 image tokens 相互更新，增强交互

## 文件索引

| 文件 | 说明 |
|------|------|
| `maskvar/models/simple_mask_vqvae/simple_mask_vqvae.py` | 模型实现 |
| `train_scripts/train_simple_mask_vqvae.py` | 训练脚本 |
| `maskvar/maskseg_build_everything.py` | 模型构建器 |

## 后续可能方向

1. **启用 VQ**: 在 autoencoder 收敛后，逐步引入 quantization
2. **多尺度**: 类似原版的多尺度 token 设计
3. **SAM Decoder**: 可选切换回 SAM MaskDecoder 进行对比
4. **更大规模**: 尝试更大规模的 image encoder（如 SAM ViT-H）
