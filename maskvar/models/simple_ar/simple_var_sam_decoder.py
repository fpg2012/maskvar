# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

# from ..sam.mask_decoder import MaskDecoder
from .adapted_mask_decoder import AdaptedMaskDecoder


class SimpleVARSamDecoder(nn.Module):
    def __init__(self,
                 adapted_mask_decoder: AdaptedMaskDecoder,
                 dim=256,
                 depth=2,
                 vocab_size=4096, 
                 device='cpu', 
                 patch_num=[1, 4, 8, 16, 32], 
                 num_heads=4,
                 vqvae_dim=256,
                 ):
        super().__init__()
        self.patch_num = patch_num
        self.adapted_mask_decoder = adapted_mask_decoder
        self.vocab_size = vocab_size
        self.dim = dim
        self.vqvae_dim = vqvae_dim
        # we don't need embed now, replaced with a linear
        # self.embed = nn.Embedding(self.vocab_size, dim)
        self.cls = nn.Linear(dim, self.vocab_size)

        self.device = device
        self.max_len = sum([x**2 for x in self.patch_num])
        self.block_mask = None
        
        pn_last = self.patch_num[-1]
        
        self.pos_embed = nn.Parameter(torch.randn(1, pn_last, pn_last, dim))
        self.level_embedding = nn.Embedding(len(patch_num), dim)
        
        self.sos = nn.Parameter(torch.randn(dim))

        self.level_map = []
        for i in range(len(self.patch_num)):
            self.level_map.extend([i] * (self.patch_num[i]**2))
        self.level_map_tensor = torch.tensor(self.level_map, device=self.device, dtype=torch.long)
        
        self.linear = nn.Linear(self.vqvae_dim, self.dim)
        # self.norm = nn.LayerNorm(self.dim)

    def calc_embed_to_add(self):
        # pos emb
        pos_embed_to_add = []

        pos_embed_1chw = rearrange(self.pos_embed, '1 h w c -> 1 c h w')
        for i, pn in enumerate(self.patch_num):
            if pn == self.patch_num[-1]:
                pos_embed_interpolated = pos_embed_1chw
            else:
                pos_embed_interpolated = F.interpolate(pos_embed_1chw, size=(pn, pn), mode='bilinear', align_corners=False)
            pos_embed_interpolated = rearrange(pos_embed_interpolated, '1 c h w -> 1 (h w) c')
            # print(f"pn={pn}, interpolated shape: {pos_embed_interpolated.shape}")
            pos_embed_to_add.append(pos_embed_interpolated)
        pos_embed_to_add = torch.cat(pos_embed_to_add, dim=1)

        # level embed
        level_embed_to_add = []
        for i in range(len(self.patch_num)):
            level_embed = repeat(self.level_embedding.weight[i], 'c -> l c', l=self.patch_num[i]**2)
            level_embed_to_add.append(level_embed)
        level_embed_to_add = torch.cat(level_embed_to_add, dim=0) # L c
        level_embed_to_add = repeat(level_embed_to_add, 'l c -> b l c', b=1) # 1 L c

        return pos_embed_to_add, level_embed_to_add

    def init_block_mask(self):
        # block mask
        def mask_mod(b, h, q_idx, k_idx):
            return self.level_map_tensor[q_idx] == self.level_map_tensor[k_idx]
        
        self.block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=self.max_len, KV_LEN=self.max_len, device=self.device)
    
    def preprocess(self, x: torch.Tensor):
        """
        1. project x to model dimension
        2. prepend SOS token
        3. add positional and level embeddings

        x: (B, L-1, C) output from quant.idxBl_to_var_input()
        """
        B, L_minus_1, C = x.shape
        L = L_minus_1 + 1
        pos_embed_to_add, level_embed_to_add = self.calc_embed_to_add()
        
        # map x using linear
        x = self.linear(x)
        # x = self.norm(x)

        sos = repeat(self.sos, 'c -> b 1 c', b=B)
        x = torch.cat([sos, x], dim=1)
        
        x = x + pos_embed_to_add + level_embed_to_add

        return x
    
    def preprocess_image_feat(self, image_feat: torch.Tensor):
        """
        convert single scale sam image feats to multiscale image_tokens

        image_feat: (B, C, H, W)

        returns:
            feats: (B, Lf, C)
        """
        B, C, H, W = image_feat.shape
        h_target = w_target = self.patch_num[-1]

        pos_embed_to_add, level_embed_to_add = self.calc_embed_to_add()

        # feats = []
        # for i, pn in enumerate(self.patch_num):
        #     feat_down = F.interpolate(image_feat, size=(pn, pn), mode='bilinear')
        #     feat_down = rearrange(feat_down, 'B C h w -> B (h w) C')
        #     feats.append(feat_down)

        # feats = torch.cat(feats, dim=1)

        # # add pos embed and level embed to feat
        # feats = feats + pos_embed_to_add + level_embed_to_add

        feat_down = F.interpolate(image_feat, size=(h_target, w_target), mode='bilinear')
        feat_down = rearrange(feat_down, 'B C h w -> B (h w) C')

        # add the last scale's pos emb and level emb
        feats = feat_down + pos_embed_to_add[:, -h_target*w_target:] + level_embed_to_add[:, -h_target*w_target:]

        return feats
    
    def forward(self, x: torch.Tensor, image_tokens: torch.Tensor, prompt_tokens=None, block_mask=None):
        """
        Applies the transformer blocks and outputs logits.
        When training, set block_mask to `self.block_mask`
        When inferencing, set block_mask to `None`

        x: (B, L, C)
        image_tokens: (B, Li, C)
        # prompt_tokens: (B, Lp, C)
        """
        # NOTE
        # 这里存在attention mask无法应用的问题
        # 导致高效并行训练不可行
        # 原因：由于SAM采用TwoWay Attention，如果并行训练，很难确保训推一致性
        # 
        # 解法一：不并行（训练速度会成倍减慢，实际上不可接受）
        # 解法二：让image tokens只看到sos，加入额外的一层self attention block让sos和其他mask token共享信息
        # 解法三：不并行，把任务转成残差预测任务，在点击位置使用更多token？
        # 解法四：降采样image token，让image token也自回归

        # for block in self.blocks:
        #     x = block(x, image_tokens=image_tokens, block_mask=block_mask)
        x = self.adapted_mask_decoder.forward(
            image_embeddings=image_tokens,
            image_pe=self.calc_embed_to_add(),
            sparse_prompt_embeddings=prompt_tokens,
            dense_prompt_embeddings=None,
            multimask_output=False,
            output_tokens=x,
            mask_mod=None,
        )
        logits = self.cls(x)
        return logits
    
    def sample_with_top_k_(self, logits, top_k=50):
        """
        Sample from logits using top-k sampling
        
        Args:
            logits: (B, C) - batch of token logits for the next token
            top_k: keep only top k tokens for sampling
        
        Returns:
            (B, 1) - sampled token indices
        """
        # 确保top_k不超过词汇表大小
        vocab_size = logits.shape[-1]
        top_k = min(top_k, vocab_size)
        
        if top_k <= 0:
            # 如果top_k无效，直接采样
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            return next_token
        
        # 获取top-k的值和索引
        topk_values, topk_indices = torch.topk(logits, k=top_k, dim=-1)
        
        # 创建掩码：只保留top-k位置的logits
        # 方法1: 使用scatter直接构建新logits
        batch_size = logits.shape[0]
        
        # 创建全为-inf的logits
        filtered_logits = torch.full_like(logits, float('-inf'))
        
        # 使用einops重新排列索引以便scatter操作
        # 将topk_indices从(B, k)转换为适合scatter的形状
        batch_indices = torch.arange(batch_size, device=logits.device)
        batch_indices = rearrange(batch_indices, 'b -> b 1')
        batch_indices = batch_indices.expand(-1, top_k)
        
        # 使用scatter_填充top-k值
        filtered_logits[batch_indices, topk_indices] = topk_values
        
        # 从过滤后的分布中采样
        probs = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
