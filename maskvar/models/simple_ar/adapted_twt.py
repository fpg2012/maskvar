# adapted from meta by nth233

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import flex_attention
from einops import rearrange

import math
from typing import Tuple, Type

from ..sam.common import MLPBlock
from ..sam.transformer import Attention
from .common import SimpleSelfAttention, SimpleCrossAttention


class AdaptedTwoWayTransformer(nn.Module):
    """
    Adapted version of SAM's TwoWayTransformer for autoregressive mask prediction.

    This transformer extends the original SAM TwoWayTransformer to support:
    1. Additional mask tokens for autoregressive prediction
    2. Separate positional encoding for mask tokens
    3. Block attention masks for controlling token visibility during training

    The transformer performs bidirectional attention between:
    - Query tokens: IOU token, mask tokens, SOS token, and prompt tokens
    - Image tokens: Features from SAM image encoder
    - Mask tokens: Autoregressive tokens being predicted
    """
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        Initialize the AdaptedTwoWayTransformer.

        Args:
          depth (int): Number of transformer layers
          embedding_dim (int): Channel dimension for input embeddings
          num_heads (int): Number of attention heads (must divide embedding_dim)
          mlp_dim (int): Hidden dimension for MLP blocks
          activation (nn.Module): Activation function for MLP blocks (default: ReLU)
          attention_downsample_rate (int): Downsample rate for attention (default: 2)
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                AdaptedTwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        mask_tokens: Tensor,
        mask_tokens_pe: Tensor,
        block_mask=None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the AdaptedTwoWayTransformer.

        Args:
          image_embedding: (B, C, H, W) Image features to attend to
          image_pe: (B, C, H, W) Positional encoding for image features
          point_embedding: (B, Lqs, C) Query tokens (IOU, mask, SOS, prompt tokens)
          mask_tokens: (B, Lqm, C) Mask tokens for autoregressive prediction
          mask_tokens_pe: (B, Lqm, C) Positional encoding for mask tokens
          block_mask: Optional attention mask for controlling token visibility
                     (e.g., for enforcing Markovian property during training)

        Returns:
          qs: (B, Lqs, C) Processed query tokens
          keys: (B, H*W, C) Processed image tokens
          qm: (B, Lqm, C) Processed mask tokens (for autoregressive prediction)
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        B, Lqs, C = point_embedding.shape
        _, Lqm, _ = mask_tokens.shape

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        query_mask_pe = torch.cat([point_embedding, mask_tokens_pe], dim=1)

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys, mask_tokens = layer(
                queries=queries,
                keys=keys,
                query_mask_pe=query_mask_pe,
                key_pe=image_pe,
                ar_queries=mask_tokens,
                block_mask=block_mask,
            )

        # Apply the final attention layer from the points to the image
        full_queries = torch.cat([queries, mask_tokens], dim=1)
        full_queries_w_pe = full_queries + query_mask_pe
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=full_queries_w_pe, k=k, v=keys)
        full_queries = full_queries + attn_out
        full_queries = self.norm_final_attn(full_queries)

        qs, qm = full_queries[:, :Lqs], full_queries[:, Lqs:]

        return qs, keys, qm


class AdaptedTwoWayAttentionBlock(nn.Module):
    """
    Adapted version of SAM's TwoWayAttentionBlock for autoregressive prediction.

    This block extends the original TwoWayAttentionBlock to handle:
    1. Combined query and mask tokens for self-attention
    2. Separate positional encodings for queries and mask tokens
    3. Block attention masks for training-time token visibility control

    The block performs four operations in sequence:
    1. Self-attention on combined queries and mask tokens
    2. Cross-attention from tokens to image features
    3. MLP on token representations
    4. Cross-attention from image features to query tokens only
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        Initialize the AdaptedTwoWayAttentionBlock.

        Args:
          embedding_dim (int): Channel dimension of embeddings
          num_heads (int): Number of attention heads
          mlp_dim (int): Hidden dimension for MLP blocks (default: 2048)
          activation (nn.Module): Activation function for MLP blocks (default: ReLU)
          attention_downsample_rate (int): Downsample rate for attention (default: 2)
          skip_first_layer_pe (bool): Whether to skip positional encoding in first layer
        """
        super().__init__()
        self.self_attn = AdaptedAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = AdaptedAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_mask_pe: Tensor,
        key_pe: Tensor,
        ar_queries: Tensor,
        block_mask=None,
        block_mask2=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the AdaptedTwoWayAttentionBlock.

        Args:
            queries: (B, Lqs, C) Query tokens (IOU, mask, SOS, prompt tokens)
            keys: (B, Lk, C) Image tokens (flattened image features)
            query_mask_pe: (B, Lqs+Lqm, C) Positional encoding for queries + mask tokens
            key_pe: (B, Lk, C) Positional encoding for image tokens
            ar_queries: (B, Lqm, C) Mask tokens for autoregressive prediction
            block_mask: Optional mask for self-attention (queries ↔ queries)
            block_mask2: Optional mask for cross-attention (queries → keys)

        Returns:
            qs: (B, Lqs, C) Updated query tokens
            keys: (B, Lk, C) Updated image tokens
            qm: (B, Lqm, C) Updated mask tokens
        """
        B, Lqs, C = queries.shape
        _, Lqm, _ = ar_queries.shape
        _, Lk, _ = keys.shape

        # Self attention block
        # NOTE: MARKOVIAN mask or CAUSAL mask should be applied here
        full_queries = torch.cat([queries, ar_queries], dim=1)
        if self.skip_first_layer_pe:
            full_queries = self.self_attn(q=full_queries, k=full_queries, v=full_queries, block_mask=block_mask)
        else:
            q = full_queries + query_mask_pe
            attn_out = self.self_attn(q=q, k=q, v=full_queries, block_mask=block_mask)
            full_queries = full_queries + attn_out
        full_queries = self.norm1(full_queries)

        # Cross attention block, tokens attending to image embedding
        # no need for block mask in principle
        q = full_queries + query_mask_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys, block_mask=block_mask2)
        full_queries = full_queries + attn_out
        full_queries = self.norm2(full_queries)

        # MLP block
        mlp_out = self.mlp(full_queries)
        full_queries = full_queries + mlp_out
        full_queries = self.norm3(full_queries)

        # Cross attention block, image embedding attending to tokens
        # only qs is involved
        qs, qm = full_queries[:, :Lqs], full_queries[:, Lqs:]
        qs_w_pe = qs + query_mask_pe[:, :Lqs]
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=qs_w_pe, v=qs)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return qs, keys, qm


class AdaptedAttention(nn.Module):
    """
    Adapted attention module that supports block attention masks.

    This attention module uses PyTorch's flex_attention to efficiently handle
    block-diagonal attention masks, which are useful for enforcing constraints
    like Markovian property (tokens at different scales cannot attend to each other).

    The module supports optional downsampling of the internal dimension to
    reduce computational cost.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, downsample_rate: int = 1):
        """
        Initialize the AdaptedAttention module.

        Args:
            embed_dim: Input embedding dimension (default: 256)
            num_heads: Number of attention heads (default: 4)
            downsample_rate: Factor to downsample internal dimension (default: 1)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.internal_dim = embed_dim // downsample_rate
        assert self.internal_dim % self.num_heads == 0, "internal_dim must be divisible by num_heads"

        self.head_dim = self.internal_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, self.internal_dim)
        self.k_proj = nn.Linear(embed_dim, self.internal_dim)
        self.v_proj = nn.Linear(embed_dim, self.internal_dim)

        self.out_proj = nn.Linear(self.internal_dim, embed_dim)
    
    def forward(self, q, k, v, block_mask=None):
        """
        Forward pass of the AdaptedAttention module.

        Args:
            q: (B, Lq, C) Query tensor
            k: (B, Lk, C) Key tensor
            v: (B, Lv, C) Value tensor
            block_mask: Optional block attention mask for flex_attention

        Returns:
            out: (B, Lq, C) Attention output
        """
        B, Lq, C = q.shape
        _, Lk, _ = k.shape
        _, Lv, _ = v.shape

        # Project inputs to internal dimension
        q = self.q_proj(q)  # (B, Lq, internal_dim)
        k = self.k_proj(k)  # (B, Lk, internal_dim)
        v = self.v_proj(v)  # (B, Lk, internal_dim)

        # Reshape for multi-head attention
        q = rearrange(q, 'B Lq (H c) -> B H Lq c', H=self.num_heads)
        k = rearrange(k, 'B Lk (H c) -> B H Lk c', H=self.num_heads)
        v = rearrange(v, 'B Lv (H c) -> B H Lv c', H=self.num_heads)

        # Apply attention with optional block mask
        out = flex_attention(q, k, v, block_mask=block_mask)
        out = rearrange(out, 'B H L c -> B L (H c)')  # (B, Lq, internal_dim)

        # Project back to original dimension
        out = self.out_proj(out)
        return out