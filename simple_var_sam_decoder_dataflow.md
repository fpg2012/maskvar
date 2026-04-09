# SimpleVAR-SAM Decoder 数据流分析

## 概述

SimpleVAR-SAM Decoder 是一个结合了 SAM (Segment Anything Model) 架构和自回归 (autoregressive) 预测的模型，用于生成多尺度的 VQ-VAE mask tokens。本文档详细描述了训练和推理时的数据流，重点关注 image token 和 mask token 的处理流程。

## 核心架构组件

### 1. SimpleVARSamDecoder (`simple_var_sam_decoder.py`)
复用 SAM 的 MaskDecoder 架构进行自回归 mask token 预测。关键特性：
- 支持多尺度 patch 预测（coarse-to-fine）
- 使用 TwoWayAttention 机制
- 集成位置编码和层级编码
- 支持 block attention masks 用于训练

### 2. AdaptedMaskDecoder (`adapted_mask_decoder.py`)
适配的 SAM MaskDecoder，处理 mask tokens 输入：
- 扩展原始 SAM MaskDecoder 支持自回归 mask token 预测
- 添加 mask_tokens 和 mask_tokens_pe 参数
- 返回处理后的 query tokens (qs) 和 mask tokens (qm)

### 3. AdaptedTwoWayTransformer (`adapted_twt.py`)
适配的 SAM TwoWayTransformer，支持：
- 额外的 mask tokens 用于自回归预测
- 独立的 mask tokens 位置编码
- block attention masks 用于训练时 token 可见性控制

### 4. SimpleVAR (`simple_var.py`)
原始 SimpleVAR 实现，包含：
- 训练函数 (`forward` 和 `simple_var_train_pass`)
- 推理函数 (`simple_var_inference`)
- 多尺度自回归生成逻辑

---

## 组件内部数据流

### 1. AdaptedMaskDecoder 内部数据流 (`adapted_mask_decoder.py`)

AdaptedMaskDecoder 是 SAM MaskDecoder 的适配版本，专门处理自回归 mask token 预测。

#### 初始化参数
- `transformer_dim`: Transformer 通道维度
- `transformer`: AdaptedTwoWayTransformer 实例
- `num_multimask_outputs`: 输出 mask 数量 (兼容性保留)
- `iou_token`, `mask_tokens`, `sos_token`: 嵌入层
- `output_upscaling`, `output_hypernetworks_mlps`, `iou_prediction_head`: 原始 SAM 组件 (当前未使用)

#### `predict_masks()` 方法数据流

```python
def predict_masks(
    self,
    image_embeddings: torch.Tensor,      # (B, C, H, W) SAM 图像特征
    image_pe: torch.Tensor,              # (B, C, H, W) 图像位置编码
    sparse_prompt_embeddings: torch.Tensor | None,  # (B, Lp, C) 稀疏 prompt
    dense_prompt_embeddings: torch.Tensor,          # (B, C, H, W) 密集 prompt
    mask_tokens: torch.Tensor,           # (B, L, C) 自回归 mask tokens
    mask_tokens_pe: torch.Tensor,        # (B, L, C) mask tokens 位置编码
    block_mask=None                      # 注意力掩码
):
```

**步骤 1: 构建 query tokens**
```python
# 拼接 IOU token、mask tokens、SOS token
qs_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.sos_token.weight], dim=0)
qs_tokens = rearrange(qs_tokens, 'n c -> b n c', b=B)  # (1, Lqs_base, C) → (B, Lqs_base, C)

# 添加稀疏 prompt tokens (如果存在)
if sparse_prompt_embeddings is not None:
    qs_tokens = torch.cat((qs_tokens, sparse_prompt_embeddings), dim=1)  # (B, Lqs, C)
```

**步骤 2: 调用 AdaptedTwoWayTransformer**
```python
qs, src, qm = self.transformer(
    image_embedding=image_embeddings,    # (B, C, H, W)
    image_pe=image_pe,                    # (B, C, H, W)
    point_embedding=qs_tokens,           # (B, Lqs, C)
    mask_tokens=mask_tokens,             # (B, L, C)
    mask_tokens_pe=mask_tokens_pe,       # (B, L, C)
    block_mask=block_mask                # 训练/推理掩码
)
```

**步骤 3: 返回处理后的 tokens**
- `qs`: (B, Lqs, C) 处理后的 query tokens (IOU, mask, SOS, prompt)
- `qm`: (B, L, C) 处理后的 mask tokens (用于自回归预测)

**关键特性**:
- 原始 SAM 的 mask 预测逻辑被注释掉，改为返回 tokens
- 保留 `output_upscaling` 和 `output_hypernetworks_mlps` 以备将来使用
- 专注于 token 表示学习而非直接 mask 生成

### 2. AdaptedTwoWayTransformer 内部数据流 (`adapted_twt.py`)

AdaptedTwoWayTransformer 是 SAM TwoWayTransformer 的适配版本，支持自回归 mask token 预测和 block attention masks。

#### 初始化参数
- `depth`: Transformer 层数
- `embedding_dim`: 嵌入维度
- `num_heads`: 注意力头数
- `mlp_dim`: MLP 隐藏维度
- `layers`: AdaptedTwoWayAttentionBlock 列表
- `final_attn_token_to_image`: 最终 tokens → image 注意力层

#### `forward()` 方法数据流

```python
def forward(
    self,
    image_embedding: Tensor,    # (B, C, H, W)
    image_pe: Tensor,           # (B, C, H, W)
    point_embedding: Tensor,    # (B, Lqs, C) query tokens
    mask_tokens: Tensor,        # (B, Lqm, C) mask tokens
    mask_tokens_pe: Tensor,     # (B, Lqm, C) mask tokens 位置编码
    block_mask=None             # 注意力掩码
) -> Tuple[Tensor, Tensor, Tensor]:
```

**步骤 1: 图像特征展平**
```python
# (B, C, H, W) → (B, H*W, C)
image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # (B, N_image, C)
image_pe = image_pe.flatten(2).permute(0, 2, 1)                # (B, N_image, C)
```

**步骤 2: 准备 query 和 key**
```python
queries = point_embedding          # (B, Lqs, C)
keys = image_embedding             # (B, N_image, C)
query_mask_pe = torch.cat([point_embedding, mask_tokens_pe], dim=1)  # (B, Lqs+Lqm, C)
```

**步骤 3: 多层 AdaptedTwoWayAttentionBlock**
```python
for layer in self.layers:
    queries, keys, mask_tokens = layer(
        queries=queries,
        keys=keys,
        query_mask_pe=query_mask_pe,
        key_pe=image_pe,
        ar_queries=mask_tokens,
        block_mask=block_mask,
    )
```

**步骤 4: 最终注意力层 (tokens → image)**
```python
# 拼接 query tokens 和 mask tokens
full_queries = torch.cat([queries, mask_tokens], dim=1)  # (B, Lqs+Lqm, C)
full_queries_w_pe = full_queries + query_mask_pe         # 添加位置编码

# tokens 到 image 的注意力
k = keys + image_pe  # image keys 添加位置编码
attn_out = self.final_attn_token_to_image(q=full_queries_w_pe, k=k, v=keys)
full_queries = full_queries + attn_out  # 残差连接
full_queries = self.norm_final_attn(full_queries)  # 层归一化

# 分离 query tokens 和 mask tokens
qs, qm = full_queries[:, :Lqs], full_queries[:, Lqs:]
```

**步骤 5: 返回结果**
- `qs`: (B, Lqs, C) 处理后的 query tokens
- `keys`: (B, N_image, C) 处理后的 image tokens
- `qm`: (B, Lqm, C) 处理后的 mask tokens

### 3. AdaptedTwoWayAttentionBlock 内部数据流

AdaptedTwoWayAttentionBlock 是 TwoWayAttention 的核心构建块，执行四步注意力操作。

#### `forward()` 方法数据流

```python
def forward(
    self,
    queries: Tensor,        # (B, Lqs, C) query tokens
    keys: Tensor,           # (B, Lk, C) image tokens
    query_mask_pe: Tensor,  # (B, Lqs+Lqm, C) query+mask 位置编码
    key_pe: Tensor,         # (B, Lk, C) image 位置编码
    ar_queries: Tensor,     # (B, Lqm, C) mask tokens
    block_mask=None,        # 自注意力掩码
    block_mask2=None        # 交叉注意力掩码
):
```

**步骤 1: Self-attention (queries + mask tokens)**
```python
# 拼接 query tokens 和 mask tokens
full_queries = torch.cat([queries, ar_queries], dim=1)  # (B, Lqs+Lqm, C)

# 应用自注意力 (支持 block mask)
if self.skip_first_layer_pe:
    full_queries = self.self_attn(q=full_queries, k=full_queries, v=full_queries, block_mask=block_mask)
else:
    q = full_queries + query_mask_pe  # 添加位置编码
    attn_out = self.self_attn(q=q, k=q, v=full_queries, block_mask=block_mask)
    full_queries = full_queries + attn_out  # 残差连接

full_queries = self.norm1(full_queries)  # 层归一化
```

**步骤 2: Cross-attention (tokens → image)**
```python
# tokens 到 image 的注意力
q = full_queries + query_mask_pe  # tokens 添加位置编码
k = keys + key_pe                  # image 添加位置编码
attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys, block_mask=block_mask2)
full_queries = full_queries + attn_out  # 残差连接
full_queries = self.norm2(full_queries)  # 层归一化
```

**步骤 3: MLP 块**
```python
mlp_out = self.mlp(full_queries)          # MLP 变换
full_queries = full_queries + mlp_out     # 残差连接
full_queries = self.norm3(full_queries)   # 层归一化
```

**步骤 4: Cross-attention (image → query tokens only)**
```python
# 分离 query tokens 和 mask tokens
qs, qm = full_queries[:, :Lqs], full_queries[:, Lqs:]

# image 到 query tokens 的注意力 (仅 query tokens 参与)
qs_w_pe = qs + query_mask_pe[:, :Lqs]  # query tokens 添加位置编码
k = keys + key_pe                       # image 添加位置编码
attn_out = self.cross_attn_image_to_token(q=k, k=qs_w_pe, v=qs)
keys = keys + attn_out                  # 更新 image tokens
keys = self.norm4(keys)                 # 层归一化
```

**返回**:
- `qs`: (B, Lqs, C) 更新的 query tokens
- `keys`: (B, Lk, C) 更新的 image tokens
- `qm`: (B, Lqm, C) 更新的 mask tokens

### 4. AdaptedAttention 内部数据流

AdaptedAttention 使用 PyTorch 的 `flex_attention` 支持块注意力掩码。

#### `forward()` 方法数据流

```python
def forward(self, q, k, v, block_mask=None):
```

**步骤 1: 线性投影到内部维度**
```python
q = self.q_proj(q)  # (B, Lq, embed_dim) → (B, Lq, internal_dim)
k = self.k_proj(k)  # (B, Lk, embed_dim) → (B, Lk, internal_dim)
v = self.v_proj(v)  # (B, Lv, embed_dim) → (B, Lv, internal_dim)
```

**步骤 2: 多头注意力重塑**
```python
q = rearrange(q, 'B Lq (H c) -> B H Lq c', H=self.num_heads)  # (B, H, Lq, head_dim)
k = rearrange(k, 'B Lk (H c) -> B H Lk c', H=self.num_heads)  # (B, H, Lk, head_dim)
v = rearrange(v, 'B Lv (H c) -> B H Lv c', H=self.num_heads)  # (B, H, Lv, head_dim)
```

**步骤 3: 应用 flex_attention (支持 block mask)**
```python
out = flex_attention(q, k, v, block_mask=block_mask)  # (B, H, Lq, head_dim)
```

**步骤 4: 重塑并投影回原始维度**
```python
out = rearrange(out, 'B H L c -> B L (H c)')  # (B, Lq, internal_dim)
out = self.out_proj(out)                      # (B, Lq, embed_dim)
```

**关键特性**:
- 支持 `downsample_rate` 降低内部维度计算成本
- 使用 `flex_attention` 高效处理块对角掩码
- 适用于 Markovian 属性 (不同尺度 tokens 不相互关注)

### 组件连接关系

这四个组件按以下层次结构连接：

```
SimpleVARSamDecoder.forward()
    ↓
AdaptedMaskDecoder.predict_masks()
    ↓
AdaptedTwoWayTransformer.forward()
    ↓
AdaptedTwoWayAttentionBlock.forward() × depth
    ↓
AdaptedAttention.forward()  # 自注意力、交叉注意力
```

**数据流传递**:
1. **SimpleVARSamDecoder**: 处理 mask tokens 预处理、位置编码、调用 AdaptedMaskDecoder
2. **AdaptedMaskDecoder**: 构建 query tokens、调用 AdaptedTwoWayTransformer
3. **AdaptedTwoWayTransformer**: 管理多层注意力块、处理图像特征展平
4. **AdaptedTwoWayAttentionBlock**: 执行四步注意力操作 (自注意力、交叉注意力、MLP、交叉注意力)
5. **AdaptedAttention**: 实现支持 block mask 的注意力机制

**训练时**: Block mask 在 AdaptedAttention 中应用，限制不同尺度 tokens 间的注意力。
**推理时**: 无 block mask，允许全连接注意力。

---

## 训练数据流 (Teacher Forcing)

训练时使用 teacher forcing，将完整的 ground truth tokens 输入模型，仅预测下一个 token。

### 输入数据
1. **Image features**: 来自 SAM encoder 的图像特征 `(B, C, H, W)`
2. **Discrete codes**: 来自 VQ-VAE 的离散 token 序列 `[idx1, idx2, ...]`，每个 `(B, l)`
   - `l = patch_num[i] ** 2`，对应每个尺度的 patch 数量
3. **Prompt tokens** (可选): 点、框等 prompt 嵌入 `(B, Lp, C)`

### 数据流步骤

#### 步骤 1: VQ-VAE tokens 转换
```python
# 使用 VQ-VAE 的 quantize.idxBl_to_var_input 方法
x = vqvae.quantize.idxBl_to_var_input(idx)  # (B, L-1, C)
# L-1: 因为训练时预测下一个 token，输入不包括最后一个 token

# 添加噪声 (epsilon > 0 时)
if epsilon > 0:
    noise = torch.randn_like(x) * epsilon
    x = x + noise
```

#### 步骤 2: Mask tokens 预处理
```python
def preprocess(self, x: torch.Tensor):
    # 1. 线性投影到模型维度
    x = self.linear(x)  # (B, L-1, C) -> (B, L-1, dim)

    # 2. 添加 SOS (Start of Sequence) token
    sos = repeat(self.sos, 'c -> b 1 c', b=B)
    x = torch.cat([sos, x], dim=1)  # (B, L, dim)，L = (L-1) + 1

    # 3. 添加位置编码和层级编码
    pos_embed_to_add, level_embed_to_add = self.calc_embed_to_add()
    x = x + pos_embed_to_add + level_embed_to_add

    return x  # (B, L, dim)
```

**位置编码计算 (`calc_embed_to_add`)**:
- **位置编码**: 从基础位置嵌入 `(1, pn_last, pn_last, dim)` 插值到各尺度
- **层级编码**: 学习每个尺度的嵌入，通过 `level_embedding` 获取
- **组合**: `mask_tokens_pe = pos_embed_to_add + level_embed_to_add`

#### 步骤 3: Image tokens 预处理
```python
def preprocess_image_feat(self, image_feat: torch.Tensor):
    # 将 SAM 单尺度特征插值到目标尺度 (最细粒度)
    h_target = w_target = self.patch_num[-1]
    feat_down = F.interpolate(image_feat, size=(h_target, w_target), mode='bilinear')
    feat_down = rearrange(feat_down, 'B C h w -> B (h w) C')

    # 添加最细尺度的位置编码和层级编码
    feats = feat_down + pos_embed_to_add[:, -h_target*w_target:] + level_embed_to_add[:, -h_target*w_target:]
    return feats  # (B, Lf, dim)，Lf = h_target * w_target
```

#### 步骤 4: Block Mask 初始化
```python
def init_block_mask(self):
    # 创建块对角掩码，确保 tokens 只能与同一尺度的 tokens 相互关注
    def mask_mod(b, h, q_idx, k_idx):
        return self.level_map_tensor[q_idx] == self.level_map_tensor[k_idx]

    self.block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=self.max_len, KV_LEN=self.max_len, device=self.device
    )
```
**Block Mask 作用**:
- 实现 Markovian 属性：不同尺度的 tokens 不能相互关注
- 训练时可并行处理所有 tokens
- 推理时设置为 `None`，允许全连接注意力

#### 步骤 5: 前向传播 (通过 AdaptedMaskDecoder)
```python
def forward(self, x: torch.Tensor, image_tokens: torch.Tensor, prompt_tokens=None, block_mask=None):
    # 计算位置编码和层级编码
    pos_embed_to_add, level_embed_to_add = self.calc_embed_to_add()
    mask_tokens_pe = pos_embed_to_add + level_embed_to_add

    # 调用 AdaptedMaskDecoder
    qs, qm = self.adapted_mask_decoder.forward(
        image_embeddings=image_tokens,      # (B, Lf, C)
        image_pe=pos_embed_to_add,          # (1, L, C) - 仅位置编码部分
        sparse_prompt_embeddings=prompt_tokens,
        dense_prompt_embeddings=None,
        multimask_output=False,
        mask_tokens=x,                      # (B, L, C) - 预处理后的 mask tokens
        mask_tokens_pe=mask_tokens_pe,      # (1, L, C) - mask tokens 的位置编码
        block_mask=block_mask               # 训练时使用 self.block_mask
    )

    # 使用 mask tokens 的输出 (qm) 进行预测
    logits = self.cls(qm)  # (B, L, vocab_size)
    return logits
```

#### 步骤 6: 损失计算
- **输入**: `logits (B, L, vocab_size)`
- **目标**: `idx` (ground truth tokens)
- **损失函数**: Cross-entropy loss，预测下一个 token
- **注意**: 由于 SOS token 的加入，需要调整目标对齐

### 训练数据流总结
1. **输入准备**: VQ-VAE tokens → 添加噪声 → 线性投影 → 添加 SOS → 添加位置/层级编码
2. **Image tokens**: SAM 特征 → 插值到目标尺度 → 添加位置/层级编码
3. **注意力控制**: 使用 block mask 限制不同尺度 tokens 间的注意力
4. **前向传播**: 通过 AdaptedMaskDecoder 进行 TwoWayAttention
5. **预测**: 从 qm 生成 logits，计算交叉熵损失

---

## 推理数据流 (Autoregressive Generation)

推理时自回归生成 tokens，一次一个尺度，逐步细化。

### 输入数据
1. **Image features**: 来自 SAM encoder 的图像特征 `(B, C, H, W)`
2. **Prompt tokens** (可选): 点、框等 prompt 嵌入
3. **VQ-VAE 模型**: 用于 token 到特征的转换

### 数据流步骤

#### 步骤 1: 初始化
```python
B = image_feat.shape[0]
H = W = simple_var.patch_num[-1]
C = simple_var.vqvae_dim

# 预处理 image tokens
image_tokens = simple_var.preprocess_image_feat(image_feat)

# 计算位置编码和层级编码
pos_embed_to_add, level_embed_to_add = simple_var.calc_embed_to_add()

# 初始化生成序列和特征累积
id_seq = []  # 存储生成的 tokens
current_token = repeat(rearrange(simple_var.sos, 'c -> 1 1 c'), '1 1 c -> b 1 c', b=B)
f_hat = torch.zeros(B, C, H, W, dtype=torch.float, device=simple_var.device)
start_pos = 0
```

#### 步骤 2: 多尺度自回归生成
```python
for scale, pn in enumerate(simple_var.patch_num):
    # 当前尺度的位置和层级编码
    end_pos = start_pos + pn * pn
    pos_embed = pos_embed_to_add[:, start_pos:end_pos]
    level_embed = level_embed_to_add[:, start_pos:end_pos]

    # 添加编码到当前 tokens
    current_token = current_token + pos_embed + level_embed

    # 前向传播 (无 block mask)
    logits = simple_var.block_forward(current_token, image_tokens=image_tokens, block_mask=None)
    # logits: (B, pn*pn, vocab_size)

    # 采样下一个 token (top-k sampling)
    logits_flat = rearrange(logits, 'b l v -> (b l) v')
    next_tokens = simple_var.sample_with_top_k_(logits_flat, top_k=1)
    next_tokens = rearrange(next_tokens, '(b l) 1 -> b l', b=B, l=pn*pn)

    # 保存生成的 tokens
    id_seq.append(next_tokens)
```

#### 步骤 3: 特征累积和更新 (粗到细)
```python
if scale < len(simple_var.patch_num) - 1:
    # 1. 将 tokens 转换为特征
    h = rearrange(vqvae.quantize.embedding(next_tokens), 'B (h w) C -> B C h w', h=pn, w=pn)

    # 2. 上采样到目标尺寸
    h_up = F.interpolate(h, size=(H, W), mode='bicubic')

    # 3. 累积特征 (残差连接)
    t = scale / (len(simple_var.patch_num) - 1)
    f_hat.add_(vqvae.quantize.quant_resi[t](h_up))

    # 4. 准备下一尺度的输入
    pn_next = simple_var.patch_num[scale + 1]
    f_hat_down = F.interpolate(f_hat, size=(pn_next, pn_next), mode='area')
    current_token = rearrange(f_hat_down, 'B C h w -> B (h w) C')

    # 5. 线性投影到模型维度
    current_token = simple_var.linear(current_token)

start_pos = end_pos
```

#### 步骤 4: 返回结果
```python
return id_seq  # 每个尺度的 tokens 列表
```

### 推理数据流总结
1. **初始化**: SOS token + 空特征累积
2. **尺度循环**: 从粗到细逐尺度生成
3. **当前尺度生成**: 添加位置/层级编码 → 前向传播 → top-k 采样
4. **特征累积**: tokens → 特征 → 上采样 → 残差累积
5. **下一尺度准备**: 下采样累积特征 → 线性投影
6. **输出**: 所有尺度的 tokens 序列

---

## 关键差异：训练 vs 推理

### 1. Token 可见性控制
| 模式 | Block Mask | 注意力模式 | 并行性 |
|------|------------|------------|--------|
| 训练 | `self.block_mask` | 块对角，同尺度内全连接 | 全并行 |
| 推理 | `None` | 全连接 | 串行 (自回归) |

**训练**: 使用 block mask 实现 Markovian 属性，tokens 只能关注同一尺度的 tokens，支持并行训练。
**推理**: 无 mask，允许全连接注意力，但自回归生成限制为串行。

### 2. 输入序列长度
| 模式 | 输入长度 | 输出长度 | SOS token |
|------|----------|----------|-----------|
| 训练 | L-1 (ground truth) | L (预测下一个) | 预添加 |
| 推理 | 1 (当前 token) | 1 (下一 token) | 初始 token |

### 3. 特征处理
| 模式 | 特征累积 | 残差连接 | 多尺度协调 |
|------|----------|----------|------------|
| 训练 | 一次性处理所有尺度 | 通过 block mask 隐式实现 | block mask 控制 |
| 推理 | 显式累积 `f_hat` | 显式残差加法 `quant_resi` | 循环更新 |

### 4. 位置编码应用
| 模式 | 位置编码时机 | 层级编码 |
|------|--------------|----------|
| 训练 | 预处理时一次性添加所有尺度 | 固定 |
| 推理 | 每尺度生成前动态添加 | 动态 |

---

## 核心挑战和解决方案

### 挑战 1: 训练-推理一致性 (TwoWayAttention)
**问题**: SAM 的 TwoWayAttention 机制导致训练时难以并行，推理时自回归生成不一致。

**解决方案尝试** (代码注释中提出):
1. **解法一**: 不并行训练 (速度慢)
2. **解法二**: 让 image tokens 只看到 SOS，添加额外 self-attention block
3. **解法三**: 转为残差预测任务，在点击位置使用更多 tokens
4. **解法四**: 降采样 image token，让 image token 也自回归

**当前实现**: 使用 block mask 限制注意力，但 TwoWayAttention 的交叉注意力部分仍存在问题。

### 挑战 2: 多尺度协调
**问题**: 如何确保不同尺度的预测一致性和连续性。

**解决方案**:
- **训练**: block mask 强制同尺度内注意力
- **推理**: 特征累积机制 (`f_hat`) 传递信息到更细尺度
- **残差连接**: `quant_resi` 模块学习尺度间残差

### 挑战 3: 位置编码对齐
**问题**: 不同尺度的位置编码需要正确对齐插值。

**解决方案**:
- `calc_embed_to_add()` 统一计算所有尺度的位置编码
- 从最细尺度插值到较粗尺度
- 确保层级编码与位置编码匹配

---

## 数据流图示

### 训练数据流
```
Image Features (SAM)
        ↓
[Interpolate to target scale]
        ↓
[Add pos/level embeddings] → Image Tokens (B, Lf, C)
        ↓
VQ-VAE Tokens (idx) → [Convert to features] → [Add noise] → [Linear projection]
        ↓
[Add SOS token] → [Add pos/level embeddings] → Mask Tokens (B, L, C)
        ↓
AdaptedMaskDecoder (with block mask)
        ↓
TwoWayAttention:
  - Self-attention (masked)
  - Cross-attention (tokens → image)
  - MLP
  - Cross-attention (image → query tokens)
        ↓
Mask Tokens Output (qm)
        ↓
[Linear classifier] → Logits (B, L, vocab_size)
        ↓
Cross-entropy loss
```

### 推理数据流
```
Image Features (SAM)
        ↓
[Interpolate to target scale]
        ↓
[Add pos/level embeddings] → Image Tokens
        ↓
Initialize: SOS token, empty f_hat
        ↓
for each scale (coarse to fine):
    ↓
    [Add scale-specific pos/level embeddings]
    ↓
    AdaptedMaskDecoder (no mask)
    ↓
    TwoWayAttention (full)
    ↓
    [Top-k sampling] → Next tokens
    ↓
    if not last scale:
        ↓
        [Tokens → features] → [Upsample] → [Residual addition] → Update f_hat
        ↓
        [Downsample f_hat] → [Linear projection] → Next input tokens
        ↓
Return all scales tokens
```

---

## 关键函数和参数

### SimpleVARSamDecoder 核心方法
1. `calc_embed_to_add()`: 计算所有尺度的位置和层级编码
2. `preprocess()`: 预处理 mask tokens (训练用)
3. `preprocess_image_feat()`: 预处理 image tokens
4. `init_block_mask()`: 初始化训练 block mask
5. `forward()`: 训练前向传播
6. `sample_with_top_k_()`: top-k 采样

### 训练参数
- `patch_num`: 多尺度配置，如 `[1, 4, 8, 16, 32]`
- `dim`: 模型维度 (默认 256)
- `vocab_size`: VQ-VAE 词汇表大小 (默认 4096)
- `vqvae_dim`: VQ-VAE token 维度 (默认 256)
- `epsilon`: 训练噪声强度

### 推理参数
- `top_k`: 采样时考虑的 top-k tokens (默认 50)
- 无 `block_mask`: 推理时禁用注意力限制

---

## 总结

SimpleVAR-SAM Decoder 通过复用 SAM 的 TwoWayAttention 架构，实现了多尺度自回归 mask token 预测。关键创新点包括：

1. **架构复用**: 适配 SAM MaskDecoder 支持自回归输入
2. **多尺度处理**: 通过 block mask 和层级编码支持 coarse-to-fine 预测
3. **训练-推理协调**: block mask 实现训练并行化，特征累积实现推理连续性
4. **位置感知**: 统一的位置和层级编码系统

**主要挑战**: TwoWayAttention 机制导致的训练-推理不一致性仍是待解决问题，需要在后续工作中探索代码注释中提出的四种解法。

---

## 附录：文件结构

```
maskvar/models/simple_ar/
├── simple_var_sam_decoder.py     # 主模块：复用 SAM 架构
├── adapted_mask_decoder.py       # 适配的 SAM MaskDecoder
├── adapted_twt.py                # 适配的 TwoWayTransformer
├── simple_var.py                 # 原始 SimpleVAR 实现
├── common.py                     # 共享组件
└── (其他相关文件)
```

**关键依赖**:
- `einops`: 张量重排
- `torch.nn.attention.flex_attention`: 块注意力掩码
- `..sam.*`: SAM 架构组件
- `..vqvae_single`: VQ-VAE 模型