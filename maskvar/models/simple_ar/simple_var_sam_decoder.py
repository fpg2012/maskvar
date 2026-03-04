# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from torch.nn.attention.flex_attention import create_block_mask

from typing import List, Tuple, Type

from maskvar.models.vqvae_single import VQVAE_Single

# from ..sam.mask_decoder import MaskDecoder
from .adapted_mask_decoder import AdaptedMaskDecoder


class SimpleVARSamDecoder(nn.Module):
    """
    SimpleVAR decoder that reuses SAM's MaskDecoder architecture for autoregressive
    mask token prediction.

    This module adapts SAM's TwoWayTransformer to predict VQ-VAE mask tokens
    in an autoregressive manner. It supports multi-scale patch prediction and
    integrates with SAM's image encoder features.

    Key components:
    1. AdaptedMaskDecoder: Modified SAM MaskDecoder that handles mask tokens
    2. Position and level embeddings: For multi-scale patch encoding
    3. Linear projection: Maps VQ-VAE token dimensions to model dimensions
    4. Classification head: Predicts next token from vocabulary

    The model supports both training (with teacher forcing) and inference
    (autoregressive sampling) modes.
    """
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
        """
        Initialize the SimpleVARSamDecoder.

        Args:
            adapted_mask_decoder: Pre-configured AdaptedMaskDecoder instance
            dim: Model dimension (default: 256)
            depth: Number of transformer layers (unused, kept for compatibility)
            vocab_size: Size of VQ-VAE vocabulary (default: 4096)
            device: Device to run on (default: 'cpu')
            patch_num: List of patch sizes for multi-scale prediction
                       Each element represents the grid size at that scale
            num_heads: Number of attention heads (unused, kept for compatibility)
            vqvae_dim: Dimension of VQ-VAE tokens (default: 256)
        """
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
        """
        Calculate positional and level embeddings for all patch scales.

        This method computes:
        1. Positional embeddings: Interpolated from the base positional embedding
           to match each patch scale
        2. Level embeddings: Learned embeddings for each patch scale (coarse to fine)

        Returns:
            pos_embed_to_add: (1, total_patches, dim) positional embeddings
            level_embed_to_add: (1, total_patches, dim) level embeddings
        """
        # Positional embeddings: interpolate base positional embedding to each scale
        pos_embed_to_add = []

        # Reshape positional embedding from (1, H, W, C) to (1, C, H, W) for interpolation
        pos_embed_1chw = rearrange(self.pos_embed, '1 h w c -> 1 c h w')
        for i, pn in enumerate(self.patch_num):
            if pn == self.patch_num[-1]:
                # For the finest scale, use the original embedding
                pos_embed_interpolated = pos_embed_1chw
            else:
                # For coarser scales, interpolate to the target size
                pos_embed_interpolated = F.interpolate(
                    pos_embed_1chw, size=(pn, pn), mode='bilinear', align_corners=False
                )
            # Reshape to (1, patches, dim)
            pos_embed_interpolated = rearrange(pos_embed_interpolated, '1 c h w -> 1 (h w) c')
            pos_embed_to_add.append(pos_embed_interpolated)
        pos_embed_to_add = torch.cat(pos_embed_to_add, dim=1)

        # Level embeddings: learned embeddings for each scale
        level_embed_to_add = []
        for i in range(len(self.patch_num)):
            # Repeat level embedding for each patch at this scale
            level_embed = repeat(self.level_embedding.weight[i], 'c -> l c', l=self.patch_num[i]**2)
            level_embed_to_add.append(level_embed)
        level_embed_to_add = torch.cat(level_embed_to_add, dim=0)  # (total_patches, dim)
        level_embed_to_add = repeat(level_embed_to_add, 'l c -> b l c', b=1)  # (1, total_patches, dim)

        return pos_embed_to_add, level_embed_to_add

    def init_block_mask(self):
        """
        Initialize the block mask for controlling attention between tokens.

        The mask ensures that tokens can only attend to other tokens at the same
        scale (level). This creates a block-diagonal attention pattern where
        each scale's tokens are independent during self-attention.

        This is used during training to enforce the Markovian property:
        tokens at finer scales can only attend to tokens at the same or coarser scales.
        """
        def mask_mod(b, h, q_idx, k_idx):
            """Mask function: returns True if query and key tokens are at the same scale."""
            return self.level_map_tensor[q_idx] == self.level_map_tensor[k_idx]

        # Create block mask using PyTorch's flex_attention utility
        self.block_mask = create_block_mask(
            mask_mod,
            B=None,  # Batch dimension (None means support any batch size)
            H=None,  # Head dimension (None means support any number of heads)
            Q_LEN=self.max_len,
            KV_LEN=self.max_len,
            device=self.device
        )
    
    def preprocess(self, x: torch.Tensor):
        """
        Preprocess input tokens for the autoregressive model.

        Steps:
        1. Project VQ-VAE tokens to model dimension using linear layer
        2. Prepend SOS (Start of Sequence) token
        3. Add positional and level embeddings

        Args:
            x: (B, L-1, C) VQ-VAE tokens from quant.idxBl_to_var_input()
               Note: L-1 because during training we predict the next token
               given previous tokens (teacher forcing)

        Returns:
            x: (B, L, C) preprocessed tokens ready for transformer input
               where L = (L-1) + 1 (SOS token)
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
        Convert single-scale SAM image features to multi-scale image tokens.

        The SAM encoder produces features at a single scale (e.g., 64x64 for 1024x1024 input).
        This method interpolates these features to the target scale (finest patch resolution)
        and adds positional embeddings for that scale only.

        Note: The original multi-scale version (commented out) would create features
        for each patch scale, but the current implementation uses only the finest scale
        for efficiency.

        Args:
            image_feat: (B, C, H, W) Image features from SAM encoder

        Returns:
            feats: (B, Lf, C) Image tokens at the finest scale with positional embeddings
                   where Lf = h_target * w_target
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
    
    def forward(self, idx, image_feat: torch.Tensor, vqvae: VQVAE_Single, epsilon=0.001):
        """
        Training pass for SimpleVAR model.

        1. Convert discrete codes to VQVAE input format
        2. Preprocess input for SimpleVAR
        3. Forward pass with block mask
        
        Args:
            idx: List of (B, l) - Input discrete codes from VQVAE
            image_feat: (B, C, H, W) - Image features from SAM
            simple_var: SimpleVAR model instance
            vqvae: VQVAE model instance

        Returns: logits (B, l, vocab_size)
        """
        assert self.block_mask is not None, "Block mask must be initialized before training"
        with torch.no_grad():
            x = vqvae.quantize.idxBl_to_var_input(idx)
            # add noise
            if epsilon > 0:
                # Add random noise to the input
                noise = torch.randn_like(x) * epsilon
                x = x + noise

        x = self.preprocess(x)
        image_tokens = self.preprocess_image_feat(image_feat)
        logits = self.block_forward(x, image_tokens=image_tokens, block_mask=self.block_mask)
        return logits

    def block_forward(self, x: torch.Tensor, image_tokens: torch.Tensor, prompt_tokens=None, block_mask=None):
        """
        Applies the transformer blocks and outputs logits.
        When training, set block_mask to `self.block_mask`
        When inferencing, set block_mask to `None`

        Args:
            x: (B, L, C) - input tokens (SOS + mask tokens) after preprocessing
            image_tokens: (B, Li, C) - image features from SAM encoder
            prompt_tokens: (B, Lp, C) - optional prompt tokens (points, boxes)
            block_mask: attention mask for controlling token visibility

        Returns:
            logits: (B, L, vocab_size) - predicted token logits
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

        # 计算位置编码和层级编码
        pos_embed_to_add, level_embed_to_add = self.calc_embed_to_add()

        # mask_tokens_pe 是位置编码和层级编码的和
        mask_tokens_pe = pos_embed_to_add + level_embed_to_add

        # 调用AdaptedMaskDecoder
        # x作为mask_tokens传入，image_pe只需要位置编码部分
        qs, qm = self.adapted_mask_decoder.forward(
            image_embeddings=image_tokens,
            image_pe=pos_embed_to_add,
            sparse_prompt_embeddings=prompt_tokens,
            dense_prompt_embeddings=None,
            mask_tokens=x,
            mask_tokens_pe=mask_tokens_pe,
            self_attn_mask=block_mask,
        )

        # 我们只需要mask tokens的输出（qm），而不是query tokens（qs）
        # qm包含了处理后的mask tokens
        logits = self.cls(qm)
        return logits
    
    def sample_with_top_k_(self, logits, top_k=50):
        """
        Sample from logits using top-k sampling for autoregressive generation.

        Top-k sampling restricts sampling to the k tokens with the highest probabilities,
        which helps maintain diversity while avoiding low-probability tokens.

        Args:
            logits: (B, vocab_size) Batch of token logits for the next token prediction
            top_k: Number of top tokens to consider for sampling (default: 50)

        Returns:
            next_token: (B, 1) Sampled token indices for each batch element
        """
        # Ensure top_k doesn't exceed vocabulary size
        vocab_size = logits.shape[-1]
        top_k = min(top_k, vocab_size)

        if top_k <= 0:
            # If top_k is invalid, sample from the full distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            return next_token

        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(logits, k=top_k, dim=-1)

        # Create mask: keep only logits at top-k positions
        batch_size = logits.shape[0]

        # Initialize filtered logits with -inf
        filtered_logits = torch.full_like(logits, float('-inf'))

        # Prepare batch indices for scatter operation
        batch_indices = torch.arange(batch_size, device=logits.device)
        batch_indices = rearrange(batch_indices, 'b -> b 1')
        batch_indices = batch_indices.expand(-1, top_k)

        # Fill top-k values into filtered logits using scatter
        filtered_logits[batch_indices, topk_indices] = topk_values

        # Sample from the filtered distribution
        probs = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

# NOTE: just reuse train pass & inference in simple_var.py

# def simple_var_train_pass(idx, image_feat: torch.Tensor, simple_var: SimpleVAR, vqvae: VQVAE_Single, epsilon=0.001):
#     pass

# @torch.no_grad()
# def simple_var_inference(image_feat: torch.Tensor, simple_var: SimpleVAR, vqvae: VQVAE_Single):
#     pass