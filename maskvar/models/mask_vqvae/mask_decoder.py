"""
Mask Decoder Module for MaskVQVAE.

This module uses SAM's MaskDecoder structure but with a custom forward pass
to handle batch image embeddings and multi-scale token decoding.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..sam.mask_decoder import MaskDecoder
from ..sam.transformer import TwoWayTransformer


class PixelSumFusion(nn.Module):
    """
    Simple fusion module that sums mask logits from all scales in pixel space.
    """
    def forward(self, masks: List[Tensor]) -> Tensor:
        """
        Args:
            masks: List of mask logits, each of shape (B, 1, H, W)
        Returns:
            Fused mask logits of shape (B, 1, H, W)
        """
        return sum(masks)


class WeightedFusion(nn.Module):
    """
    Learnable weighted fusion for multi-scale masks.
    """
    def __init__(self, num_scales: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_scales))

    def forward(self, masks: List[Tensor]) -> Tensor:
        """
        Args:
            masks: List of mask logits, each of shape (B, 1, H, W)
        Returns:
            Fused mask logits of shape (B, 1, H, W)
        """
        weights = F.softmax(self.weights, dim=0)
        return sum(w * m for w, m in zip(weights, masks))


class MaskDecoderModule(nn.Module):
    """
    Mask decoder module that uses SAM's MaskDecoder structure with custom forward logic.

    This module decodes multi-scale quantized tokens into mask logits using
    two-way cross attention with image features, similar to SAM's mask decoder.

    Args:
        cvae_dim: Dimension of VQVAE latent space (Cvae)
        img_feat_dim: Dimension of input image features (e.g., 256 for SAM ViT)
        transformer_dim: Dimension for transformer (default: 256)
        transformer_depth: Number of transformer layers (default: 2)
        transformer_num_heads: Number of attention heads (default: 8)
        transformer_mlp_dim: MLP hidden dimension (default: 2048)
        v_patch_nums: Tuple of patch numbers for each scale
        fusion_type: Type of fusion for multi-scale masks ('sum' or 'weighted')
        use_sam_weights: Whether to reuse SAM MaskDecoder weights (if available)
    """

    def __init__(
        self,
        cvae_dim: int = 32,
        img_feat_dim: int = 256,
        transformer_dim: int = 256,
        transformer_depth: int = 2,
        transformer_num_heads: int = 8,
        transformer_mlp_dim: int = 2048,
        v_patch_nums: Tuple[int, ...] = (1, 2, 4, 8, 16),
        fusion_type: str = "sum",
        use_sam_mask_decoder: bool = True,
    ):
        super().__init__()

        self.cvae_dim = cvae_dim
        self.img_feat_dim = img_feat_dim
        self.transformer_dim = transformer_dim
        self.v_patch_nums = v_patch_nums
        self.num_scales = len(v_patch_nums)

        # Projection layers for tokens and image features
        self.token_proj = nn.Conv2d(cvae_dim, transformer_dim, kernel_size=1)
        self.img_feat_proj = nn.Conv2d(img_feat_dim, transformer_dim, kernel_size=1)

        # SAM MaskDecoder as submodule - we reuse its structure but rewrite forward
        if use_sam_mask_decoder:
            self.sam_mask_decoder = MaskDecoder(
                transformer_dim=transformer_dim,
                transformer=TwoWayTransformer(
                    depth=transformer_depth,
                    embedding_dim=transformer_dim,
                    num_heads=transformer_num_heads,
                    mlp_dim=transformer_mlp_dim,
                ),
                num_multimask_outputs=3,  # Not used but required by SAM
            )
        else:
            # If not using SAM, create our own transformer
            self.sam_mask_decoder = None
            self.transformer = TwoWayTransformer(
                depth=transformer_depth,
                embedding_dim=transformer_dim,
                num_heads=transformer_num_heads,
                mlp_dim=transformer_mlp_dim,
            )

        # Get transformer reference (either from SAM or our own)
        self.transformer_ref = (
            self.sam_mask_decoder.transformer
            if self.sam_mask_decoder
            else self.transformer
        )

        # Learnable positional encoding for image features
        # Shape: (1, transformer_dim, max_patch_num, max_patch_num)
        max_pn = max(v_patch_nums)
        self.image_pe = nn.Parameter(
            torch.zeros(1, transformer_dim, max_pn, max_pn)
        )
        nn.init.normal_(self.image_pe, std=0.01)

        # MLP to map tokens to mask prediction space (similar to SAM's hypernetwork)
        # SAM uses multiple MLPs for different mask tokens, we use a shared one
        self.mask_token_mlp = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim // 8),  # Match SAM's output_upscaling output channels
        )

        # Output upsampling to match original image size
        # SAM's output_upscaling does 4x upsampling: 256 -> 64 -> 32
        if self.sam_mask_decoder:
            self.output_upscaling = self.sam_mask_decoder.output_upscaling
        else:
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(
                    transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
                ),
                nn.GroupNorm(1, transformer_dim // 4),
                nn.GELU(),
                nn.ConvTranspose2d(
                    transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
                ),
                nn.GELU(),
            )

        # Additional upsampling if needed (4x from SAM may not be enough)
        # We apply this dynamically based on input/output size ratio
        self.extra_upsample = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim // 8, transformer_dim // 8, kernel_size=2, stride=2
            ),
            nn.GELU(),
        )

        # Fusion module for multi-scale masks
        if fusion_type == "sum":
            self.fusion = PixelSumFusion()
        elif fusion_type == "weighted":
            self.fusion = WeightedFusion(self.num_scales)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def _get_positional_encoding(self, size: Tuple[int, int]) -> Tensor:
        """
        Get positional encoding for given spatial size.

        Args:
            size: (h, w) spatial size
        Returns:
            Positional encoding of shape (1, transformer_dim, h, w)
        """
        h, w = size
        return self.image_pe[:, :, :h, :w]

    def _decode_single_scale(
        self,
        image_features: Tensor,
        tokens: Tensor,
        target_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> Tensor:
        """
        Decode a single scale's tokens into mask logits.

        Args:
            image_features: (B, img_feat_dim, H_img, W_img) image features from external encoder
            tokens: (B, cvae_dim, pn, pn) quantized tokens for this scale
            target_size: (pn, pn) target spatial size for this scale (number of patches)
            original_size: (H, W) original image size to upsample to

        Returns:
            mask_logits: (B, 1, H, W) mask logits for this scale
        """
        B = tokens.shape[0]
        pn = target_size[0]  # Assuming h = w = pn

        # Step 1: Downsample image features to target patch size (pn, pn)
        # image_features: (B, img_feat_dim, H_img, W_img)
        img_feat_scaled = F.interpolate(
            image_features, size=target_size, mode="area"
        )  # (B, img_feat_dim, pn, pn)

        # Step 2: Project to transformer dimension
        tokens_proj = self.token_proj(tokens)  # (B, transformer_dim, pn, pn)
        img_feat_proj = self.img_feat_proj(img_feat_scaled)  # (B, transformer_dim, pn, pn)

        # Step 3: Prepare inputs for TwoWayTransformer
        # SAM transformer expects 4D inputs (B, C, H, W) and does flattening internally
        # tokens_proj: (B, transformer_dim, pn, pn) - but transformer expects point_embedding as (B, N, C)
        # So we need to flatten tokens but keep img_feat as 4D

        # Get positional encoding
        img_pe = self._get_positional_encoding(target_size)  # (1, transformer_dim, pn, pn)

        # Flatten tokens for point_embedding: (B, transformer_dim, pn, pn) -> (B, pn*pn, transformer_dim)
        tokens_flat = tokens_proj.flatten(2).permute(0, 2, 1)  # (B, pn*pn, transformer_dim)

        # Step 4: Two-way cross attention using SAM's transformer
        # SAM transformer expects:
        #   image_embedding: (B, C, H, W) - 4D
        #   image_pe: (B, C, H, W) - 4D positional encoding
        #   point_embedding: (B, N, C) - 3D (already flattened queries)
        processed_tokens, processed_img_feat = self.transformer_ref(
            image_embedding=img_feat_proj,  # (B, transformer_dim, pn, pn) - 4D
            image_pe=img_pe.expand(B, -1, -1, -1),  # (B, transformer_dim, pn, pn) - 4D
            point_embedding=tokens_flat,  # (B, pn*pn, transformer_dim) - 3D
        )
        # processed_tokens: (B, pn*pn, transformer_dim)
        # processed_img_feat: (B, pn*pn, transformer_dim)

        # Step 5: Reshape processed image features back to spatial
        processed_img_spatial = (
            processed_img_feat.permute(0, 2, 1).view(B, self.transformer_dim, pn, pn)
        )  # (B, transformer_dim, pn, pn)

        # Step 6: Upsample image features using SAM's output_upscaling (4x)
        upscaled_img = self.output_upscaling(processed_img_spatial)
        # (B, transformer_dim//8, pn*4, pn*4)

        # Additional upsampling if needed
        current_size = upscaled_img.shape[2:]
        if current_size[0] < original_size[0] or current_size[1] < original_size[1]:
            upscaled_img = self.extra_upsample(upscaled_img)
        # Ensure exact size match
        if upscaled_img.shape[2:] != original_size:
            upscaled_img = F.interpolate(
                upscaled_img, size=original_size, mode="bilinear", align_corners=False
            )
        # (B, transformer_dim//8, H, W)

        # Step 7: Compute mask logits using "multiplication" style (SAM style)
        # tokens -> MLP -> (B, pn*pn, transformer_dim//8)
        token_features = self.mask_token_mlp(processed_tokens)  # (B, pn*pn, transformer_dim//8)

        # Since tokens correspond to spatial locations, we can directly multiply
        # Reshape tokens to spatial grid
        token_features_spatial = token_features.permute(0, 2, 1).view(
            B, -1, pn, pn
        )  # (B, transformer_dim//8, pn, pn)

        # Upsample token features to match image feature size
        token_features_upscaled = F.interpolate(
            token_features_spatial, size=original_size, mode="bilinear", align_corners=False
        )  # (B, transformer_dim//8, H, W)

        # Element-wise multiplication and sum over channels (like dot product per spatial location)
        # (B, C, H, W) * (B, C, H, W) -> (B, C, H, W) then sum over C
        mask_logits = (upscaled_img * token_features_upscaled).sum(dim=1, keepdim=True)
        # (B, 1, H, W)

        return mask_logits

    def forward(
        self,
        image_features: Tensor,
        ms_tokens: List[Tensor],
        v_patch_nums: Optional[Tuple[int, ...]] = None,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Forward pass decoding multi-scale tokens into mask logits.

        Args:
            image_features: (B, img_feat_dim, H_img, W_img) image features
            ms_tokens: List of quantized tokens, each (B, cvae_dim, pn, pn)
            v_patch_nums: Tuple of patch numbers for each scale (default: self.v_patch_nums)
            original_size: (H, W) original image size to upsample to
                          If None, inferred from image_features shape

        Returns:
            If return_all_scales=True: List of mask logits for each scale
            Otherwise: Fused mask logits (B, 1, H, W)
        """
        if v_patch_nums is None:
            v_patch_nums = self.v_patch_nums

        B = image_features.shape[0]

        # Determine original image size
        if original_size is None:
            # Assume image_features are 16x downsampled from original
            # This is true for SAM image encoder
            H_img, W_img = image_features.shape[2:]
            original_size = (H_img * 16, W_img * 16)

        # Decode each scale
        all_mask_logits = []
        for si, pn in enumerate(v_patch_nums):
            tokens = ms_tokens[si]  # (B, cvae_dim, pn, pn)
            mask_logits = self._decode_single_scale(
                image_features=image_features,
                tokens=tokens,
                target_size=(pn, pn),
                original_size=original_size,
            )
            all_mask_logits.append(mask_logits)

        # Fuse multi-scale masks
        fused_mask = self.fusion(all_mask_logits)

        return fused_mask, all_mask_logits

    def decode_single_scale(
        self,
        image_features: Tensor,
        tokens: Tensor,
        scale_idx: int,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Decode a single scale (convenience method for inference).

        Args:
            image_features: (B, img_feat_dim, H_img, W_img)
            tokens: (B, cvae_dim, pn, pn) tokens for a single scale
            scale_idx: Index of the scale (to determine pn from v_patch_nums)
            original_size: (H, W) original image size

        Returns:
            mask_logits: (B, 1, H, W)
        """
        pn = self.v_patch_nums[scale_idx]

        if original_size is None:
            H_img, W_img = image_features.shape[2:]
            original_size = (H_img * 16, W_img * 16)

        return self._decode_single_scale(
            image_features=image_features,
            tokens=tokens,
            target_size=(pn, pn),
            original_size=original_size,
        )
