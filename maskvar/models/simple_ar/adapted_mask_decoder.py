# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

from typing import List, Tuple, Type

from maskvar.models.simple_ar.adapted_twt import AdaptedTwoWayTransformer

from ..sam.common import LayerNorm2d


class AdaptedMaskDecoder(nn.Module):
    """
    Adapted version of SAM's MaskDecoder for autoregressive mask token prediction.

    This decoder extends the original SAM MaskDecoder to:
    1. Support autoregressive mask token prediction instead of direct mask output
    2. Handle additional mask tokens with separate positional encoding
    3. Return processed tokens for classification rather than mask logits

    The decoder maintains the original SAM architecture components but
    repurposes them for token prediction tasks.
    """
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: AdaptedTwoWayTransformer,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Initialize the AdaptedMaskDecoder.

        Args:
          transformer_dim (int): Channel dimension of the transformer
          transformer (AdaptedTwoWayTransformer): Transformer for mask token prediction
          num_multimask_outputs (int): Number of mask outputs (kept for compatibility)
          activation (nn.Module): Activation function for upscaling layers
          iou_head_depth (int): Depth of the IoU prediction MLP
          iou_head_hidden_dim (int): Hidden dimension of the IoU prediction MLP
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # self.sos_token = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        mask_tokens: torch.Tensor=None,
        mask_tokens_pe: torch.Tensor=None,
        self_attn_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          mask_tokens: (B, L, C) - mask tokens for autoregressive prediction
          mask_tokens_pe: (B, L, C) - positional encoding for mask tokens
          block_mask: attention mask for controlling token visibility

        Returns:
          torch.Tensor: query tokens (qs) - processed query tokens (iou, mask, prompt)
          torch.Tensor: mask tokens (qm) - processed mask tokens for autoregressive prediction (including iou token)
        """
        return self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            mask_tokens=mask_tokens,
            mask_tokens_pe=mask_tokens_pe,
            self_attn_mask=self_attn_mask,
        )

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor | None,
        dense_prompt_embeddings: torch.Tensor,
        mask_tokens: torch.Tensor,
        mask_tokens_pe: torch.Tensor,
        self_attn_mask=None,
    ):
        """
        Predicts masks. See 'forward' for more details.

        Args:
            image_embeddings: (B, C, H, W) image embeddings from SAM encoder
            image_pe: (B, C, H, W) positional encoding for image embeddings
            sparse_prompt_embeddings: (B, Lp, C) sparse prompt embeddings (points, boxes)
            dense_prompt_embeddings: (B, C, H, W) dense prompt embeddings (mask inputs)
            mask_tokens: (B, L, C) mask tokens for autoregressive prediction
            mask_tokens_pe: (1, L, C) positional encoding for mask tokens
            block_mask: attention mask for controlling token visibility

        Returns:
            qs: (B, Lqs, C) processed query tokens (iou, mask, sos, prompt tokens)
            qm: (B, Lqm, C) processed mask tokens for autoregressive prediction
        """
        B = image_embeddings.shape[0]
        # Concatenate output tokens
        
        qs_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        qs_tokens = repeat(qs_tokens, 'n c -> b n c', b=B)

        mask_tokens_pe = repeat(mask_tokens_pe, '1 n c -> b n c', b=B)

        if sparse_prompt_embeddings is not None:
            qs_tokens = torch.cat((qs_tokens, sparse_prompt_embeddings), dim=1)

        Lqs = qs_tokens.shape[1]
        Lqm = mask_tokens.shape[1]

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, qs_tokens.shape[0], dim=0)
        # # src = src + dense_prompt_embeddings
        # pos_src = torch.repeat_interleave(image_pe, qs_tokens.shape[0], dim=0)
        # print('image_embeddings.shape', image_embeddings.shape)
        b, hw, c = image_embeddings.shape
        image_embeddings = rearrange(image_embeddings, 'b (h w) c -> b c h w', h=int(hw**0.5), w=int(hw**0.5))

        # Run the transformer
        qs, src, qm = self.transformer(
            image_embedding=image_embeddings,
            image_pe=image_pe,
            point_embedding=qs_tokens,
            mask_tokens=mask_tokens,
            mask_tokens_pe=mask_tokens_pe,
            self_attn_mask=self_attn_mask,
        )

        # Upscale mask embeddings and predict masks using the mask tokens
        # src = src.transpose(1, 2).view(b, c, h, w)
        # upscaled_embedding = self.output_upscaling(src)
        # hyper_in_list: List[torch.Tensor] = []
        # for i in range(self.num_mask_tokens):
        #     hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        # hyper_in = torch.stack(hyper_in_list, dim=1)

        return qs, qm


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
