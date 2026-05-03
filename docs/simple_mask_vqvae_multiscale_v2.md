# SimpleMaskVQVAE Multiscale V2

## 目标

V2 的目标不是重新设计一套 mask autoencoder，而是尽量复用已经训练好的
`SimpleMaskVqvae`：

- `mask_encoder` 保持不变
- `image_encoder` 保持不变
- `mask_decoder` 保持不变
- 原来的 `quant.embedding.weight` 继续作为共享 codebook
- 只把 `SimpleVectorQuantize` 换成 `MultiscaleVectorQuantize`

这样可以从已有 checkpoint
`out/ddp_simple_mask_vqvae_coconut_ep5_19bugfix_retrain/checkpoints/latest.pth`
继续训练，而不是从头学习 encoder/decoder。

## 总体流程

V2 的 forward 基本等价于 `SimpleMaskVqvae.forward()`：

```text
mask_normalized
  -> mask_encoder
  -> flatten to BLC
  -> MultiscaleVectorQuantize
  -> reshape to B,H,W,C
  -> SimpleMaskDecoder(mask_tokens, image_tokens)
  -> mask logits
```

和原模型唯一核心差别在 quantizer：

```text
SimpleMaskVqvae:
  mask_tokens = SimpleVectorQuantize(mask_tokens)

SimpleMaskVqvaeMultiScaleResidual:
  mask_tokens = MultiscaleVectorQuantize(mask_tokens)
```

实现位置：

- `maskvar/models/simple_mask_vqvae/simple_mask_vqvae_multiscale.py`
- `maskvar/models/simple_mask_vqvae/quant.py`

Builder 名称：

```text
simple_mask_vqvae_multiscale_v2_dim384
```

## MultiscaleVectorQuantize

`MultiscaleVectorQuantize` 是 `SimpleVectorQuantize` 的子类，因此保留原 codebook
参数名：

```text
quant.embedding.weight
quant.ema_vocab_hit
quant.record_hit
```

这让旧 `SimpleMaskVqvae` checkpoint 中的 codebook 可以直接加载。V2 新增的参数主要是：

```text
quant.phi.*
quant.scale_gates
```

### 编码逻辑

V2 按 VAR 的 residual quantization 方式从粗到细编码：

```python
f = mask_encoder(mask)
f_rest = f.detach()
f_hat = 0

for scale in [1, 2, 4, 8, 16, 32, 64]:
    z = area_downsample(f_rest, scale)
    idx = nearest_codebook(z)
    h = codebook_lookup(idx)
    h = bicubic_upsample(h, 64)
    h = phi_k(h)
    f_hat = f_hat + h
    f_rest = f_rest - h

z_q = straight_through(f_hat, f)
```

返回的 `z_q` 仍然是 full-resolution BLC，和原 `SimpleVectorQuantize` 接口一致。

### 解码逻辑

V2 有两种解码场景。

#### 1. 训练/重建时的解码

训练 forward 里，`MultiscaleVectorQuantize.forward()` 已经把多尺度 code 重建成
full-resolution latent：

```text
z_q: (B, 64*64, C)
```

因此后续和原 `SimpleMaskVqvae` 完全一致：

```python
mask_tokens = rearrange(z_q, "b (h w) c -> b h w c", h=64, w=64)
image_tokens = image_encoder(image)
image_tokens = rearrange(image_tokens, "b c h w -> b h w c")
mask_logits = mask_decoder(mask_tokens, image_tokens)
```

也就是说，`SimpleMaskDecoder` 不知道前面发生了多尺度量化；它看到的仍然是一个
`64x64` mask-token grid。

#### 2. 从多尺度 token ids 解码

当 AR 或可视化代码已经有多尺度 token ids 时，流程是：

```python
token_ids_by_scale = [
    ids_s1,   # (B, 1*1)
    ids_s2,   # (B, 2*2)
    ids_s4,   # (B, 4*4)
    ...
    ids_s64,  # (B, 64*64)
]

z_q = quant.idxBl_to_full_tokens(token_ids_by_scale)
```

`idxBl_to_full_tokens()` 对每个尺度做：

```python
h_k = codebook_lookup(ids_k)
h_k = rearrange(h_k, "b (h w) c -> b c h w")
h_k = bicubic_upsample(h_k, 64, 64)   # 最后一层 64x64 不需要上采样
h_k = phi_k(h_k)
f_hat = f_hat + h_k
```

最终得到：

```text
f_hat: (B, 64*64, C)
```

然后仍然交给原 decoder：

```python
mask_tokens = rearrange(f_hat, "b (h w) c -> b h w c", h=64, w=64)
mask_logits = mask_decoder(mask_tokens, image_tokens)
```

这和 VAR 的解码思想一致：每个尺度的离散 codebook embedding 被投回最高分辨率 latent，
再累加成最终 `f_hat`。区别是 V2 的最终输出不是图像 decoder，而是
`SimpleMaskDecoder(mask_tokens, image_tokens)`。

### 数值稳定设计

V2 对齐了原 `maskvar/models/quant.py::VectorQuantizer2` 的几个关键细节。

1. 量化路径使用 fp32，并关闭 autocast：

```python
z_fp32 = z.float()
with torch.amp.autocast(device_type="cuda", enabled=False):
    ...
```

2. residual 从 detached feature 开始：

```python
z_no_grad = z_fp32.detach()
residual = z_no_grad.clone()
```

3. VQ loss 使用累计 full feature，而不是每层 residual token loss：

```python
vq_loss += mse(f_hat.data, f) * beta + mse(f_hat, f.detach())
```

4. `phi_k` 是 residual blend：

```python
phi_k(h) = h * (1 - quant_resi) + conv_k(h) * quant_resi
```

5. 下采样使用 `area`，上采样使用 `bicubic`。

## 从旧模型稳定初始化

直接随机初始化所有 `phi_k` 会破坏旧 decoder 的输入分布，训练中可能出现 VQ loss 爆炸。
因此 V2 初始化为“等价原单尺度 VQ”：

```text
scale_gates[1..32] = 0
scale_gates[64] = 1
phi_64 = identity
phi_1..32 conv = 0
```

初始时，V2 基本等价于旧的 `SimpleVectorQuantize`：

```text
只有 64x64 scale 生效
coarse scale 初始不改变输出
```

后续训练中，coarse scale 的 gate 和 phi 可以逐步学会解释低频 residual。

## 训练脚本

脚本：

```text
train_scripts/simple_mask_vqvae/ddp_train_multiscale_v2_coconut_hf_dino.sh
```

默认初始化 checkpoint：

```text
out/ddp_simple_mask_vqvae_coconut_ep5_19bugfix_retrain/checkpoints/latest.pth
```

默认启用 VQ：

```bash
--enable_vq
--vq_loss_weight 0.25
```

运行：

```bash
bash train_scripts/simple_mask_vqvae/ddp_train_multiscale_v2_coconut_hf_dino.sh 4 \
  out/ddp_simple_mask_vqvae_multiscale_v2_coconut_ep10
```

也可以显式传入初始化 checkpoint：

```bash
bash train_scripts/simple_mask_vqvae/ddp_train_multiscale_v2_coconut_hf_dino.sh 4 \
  out/ddp_simple_mask_vqvae_multiscale_v2_coconut_ep10 \
  29500 \
  out/ddp_simple_mask_vqvae_coconut_ep5_19bugfix_retrain/checkpoints/latest.pth
```

## Checkpoint 加载

V2 builder 使用 `strict=False` 加载 checkpoint。

从旧 `SimpleMaskVqvae` checkpoint 加载时，预期行为是：

- 已加载：`mask_encoder.*`
- 已加载：`image_encoder.*`
- 已加载：`mask_decoder.*`
- 已加载：`quant.embedding.weight`
- 新初始化：`quant.phi.*`
- 新初始化：`quant.scale_gates`

Builder 会打印 missing / unexpected keys，方便确认加载情况。

## 和 V1 Multiscale 的区别

V1 multiscale 是并联池化：

```text
mask_feature
  -> pool to each scale
  -> quantize all scales
  -> upsample all to 64x64
  -> average/sum fusion
```

实验中它退化为主要依赖 `64x64` scale，coarse scales 几乎没有贡献。

V2 是 residual quantization：

```text
coarse scale 先解释 residual
fine scale 只解释剩下的 residual
```

因此最高分辨率 token 不再天然需要承载全部信息。

## 和原 VAR VQVAE 的差异

V2 借鉴了 VAR 的 quantization 逻辑，但保留 SimpleMaskVqvae 的任务结构：

- VAR VQVAE 解码图像 latent
- V2 解码 mask-conditioned/image-conditioned segmentation logits
- VAR 使用 `quant_conv` / `post_quant_conv`
- V2 不新增这两个卷积，以便最大程度复用原 `SimpleMaskVqvae`
- VAR 每个 scale 有独立 usage 统计
- V2 当前仍使用一个共享 `ema_vocab_hit`

## 注意事项

1. 不要 resume 已经 VQ loss 爆炸的 checkpoint。
2. 新实验应从原 SimpleMaskVqvae checkpoint 重新启动。
3. 如果 VQ loss 再次快速升高，优先降低：

```bash
--vq_loss_weight 0.1
```

4. 如果希望 coarse scale 更快参与，可以后续考虑给 `scale_gates` 单独更大学习率。

## 相关文件

| 文件 | 说明 |
| --- | --- |
| `maskvar/models/simple_mask_vqvae/quant.py` | `MultiscaleVectorQuantize` 实现 |
| `maskvar/models/simple_mask_vqvae/simple_mask_vqvae_multiscale.py` | V2 模型壳，复用 SimpleMaskVqvae forward 结构 |
| `maskvar/maskseg_build_everything.py` | builder 注册与 checkpoint 加载 |
| `train_scripts/simple_mask_vqvae/ddp_train_multiscale_v2_coconut_hf_dino.sh` | V2 训练脚本 |
| `notebooks/visualize_simple_mask_vqvae_multiscale.py` | 多尺度可视化/ablation |
