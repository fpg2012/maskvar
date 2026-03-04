# adapted from meta by nth233

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from ..sam.common import MLPBlock
from ..sam.transformer import Attention
from .common import SimpleSelfAttention, SimpleCrossAttention


class AdaptedTwoWayTransformer(nn.Module):
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
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
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
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
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
                ar_queires=mask_tokens,
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
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
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
        block_mask=None, # block mask for self attn: q <-> q
        block_mask2=None, # block mask for q -> k
    ) -> Tuple[Tensor, Tensor]:
        """
        queries: B Lqs C
        keys: B Lk C
        ar_queires: B Lqm C
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

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, downsample_rate: int = 1):
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
        B, Lq, C = q.shape
        _, Lk, _ = k.shape
        _, Lv, _ = v.shape

        q = self.q_proj(q) # (B, Lq, C)
        k = self.k_proj(k) # (B, Lk, C)
        v = self.v_proj(v) # (B, Lk, C)
        
        q = rearrange(q, 'B Lq (H c) -> B H Lq c', H=self.num_heads)
        k = rearrange(k, 'B Lk (H c) -> B H Lk c', H=self.num_heads)
        v = rearrange(v, 'B Lv (H c) -> B H Lv c', H=self.num_heads)
        
        out = flex_attention(q, k, v, block_mask=block_mask)
        out = rearrange(out, 'B H L c -> B L (H c)')

        out = self.out_proj(out)
        return out