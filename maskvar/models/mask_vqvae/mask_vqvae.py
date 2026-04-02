"""
MaskVQVAE: VQVAE variant that uses image features for mask decoding.

This model extends the standard VQVAE by incorporating external image features
(e.g., from SAM image encoder) into the mask decoding process via a custom
decoder that uses two-way cross attention.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..basic_vae import Decoder, Encoder
from ..quant import VectorQuantizer2
from .mask_decoder import MaskDecoderModule


class MaskVQVAE(nn.Module):
    """
    MaskVQVAE for mask generation with image feature guidance.

    Architecture:
        1. Encoder: 0-1 mask -> multi-scale features
        2. Quantizer: features -> quantized tokens
        3. MaskDecoder: image_features + tokens -> reconstructed mask

    Args:
        vocab_size: Codebook size (number of discrete tokens)
        z_channels: Number of channels in latent space
        ch: Base number of channels for network layers
        dropout: Dropout rate
        beta: Commitment loss weight
        using_znorm: Whether to normalize when computing nearest neighbors
        quant_conv_ks: Kernel size for quantization convolution
        quant_resi: Residual connection ratio
        share_quant_resi: Number of shared phi layers across scales
        default_qresi_counts: Default number of quantization residual layers
        v_patch_nums: Number of patches per scale (h=w=v_patch_nums[k])
        img_feat_dim: Dimension of external image features (e.g., 256 for SAM)
        transformer_dim: Dimension for transformer
        transformer_depth: Number of transformer layers
        transformer_num_heads: Number of attention heads
        transformer_mlp_dim: MLP hidden dimension
        fusion_type: Multi-scale fusion type ('sum' or 'weighted')
        use_sam_mask_decoder: Whether to use SAM MaskDecoder structure
        test_mode: Whether in test mode (freeze parameters)
        ddconfig: Encoder/decoder configuration dict
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        z_channels: int = 32,
        ch: int = 128,
        dropout: float = 0.0,
        beta: float = 0.25,
        using_znorm: bool = False,
        quant_conv_ks: int = 3,
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
        default_qresi_counts: int = 0,
        v_patch_nums: Tuple[int, ...] = (1, 2, 4, 8, 16),
        img_feat_dim: int = 256,
        transformer_dim: int = 256,
        transformer_depth: int = 2,
        transformer_num_heads: int = 8,
        transformer_mlp_dim: int = 2048,
        fusion_type: str = "sum",
        use_sam_mask_decoder: bool = True,
        test_mode: bool = True,
        ddconfig: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.test_mode = test_mode
        self.V = vocab_size
        self.Cvae = z_channels
        self.v_patch_nums = v_patch_nums
        self.img_feat_dim = img_feat_dim

        # Encoder/Decoder configuration
        if ddconfig is None:
            ddconfig = dict(
                dropout=dropout,
                ch=ch,
                z_channels=z_channels,
                in_channels=1,  # Single-channel mask input
                ch_mult=(1, 1, 2, 2, 4),
                num_res_blocks=2,
                using_sa=True,
                using_mid_sa=True,
            )
        else:
            ddconfig = dict(
                dropout=dropout,
                ch=ch,
                z_channels=z_channels,
                **ddconfig,
            )
        ddconfig.pop("double_z", None)

        # Initialize encoder and decoder
        # Note: We still use the basic decoder for reconstruction baseline
        # but the main decoding is done by MaskDecoderModule
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)  # For fallback/comparison

        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig["ch_mult"]) - 1)

        # Vector quantizer
        self.quantize = VectorQuantizer2(
            vocab_size=vocab_size,
            Cvae=self.Cvae,
            using_znorm=using_znorm,
            beta=beta,
            default_qresi_counts=default_qresi_counts,
            v_patch_nums=v_patch_nums,
            quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
        )

        # Convolution layers before and after quantization
        self.quant_conv = nn.Conv2d(
            self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2
        )
        self.post_quant_conv = nn.Conv2d(
            self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2
        )

        # Mask decoder that uses image features
        self.mask_decoder = MaskDecoderModule(
            cvae_dim=z_channels,
            img_feat_dim=img_feat_dim,
            transformer_dim=transformer_dim,
            transformer_depth=transformer_depth,
            transformer_num_heads=transformer_num_heads,
            transformer_mlp_dim=transformer_mlp_dim,
            v_patch_nums=v_patch_nums,
            fusion_type=fusion_type,
            use_sam_mask_decoder=use_sam_mask_decoder,
        )

        # Freeze parameters if in test mode
        if self.test_mode:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def encode(self, inp: Tensor, ret_usages: bool = False) -> Tuple[Tensor, Optional[List[float]], Tensor]:
        """
        Encode mask to quantized features.

        Args:
            inp: Input single-channel mask, shape (B, 1, H, W), values in [-1, 1]
            ret_usages: Whether to return codebook usage statistics

        Returns:
            f_hat: Quantized features (B, Cvae, H//downsample, W//downsample)
            usages: Codebook usage per scale (if ret_usages=True)
            vq_loss: Vector quantization loss
        """
        f = self.quant_conv(self.encoder(inp))
        f_hat, usages, vq_loss = self.quantize(f, ret_usages=ret_usages)
        return f_hat, usages, vq_loss

    def encode_to_indices(self, inp: Tensor) -> List[Tensor]:
        """
        Encode mask to multi-scale indices.

        Args:
            inp: Input mask (B, 1, H, W)

        Returns:
            List of indices for each scale, each (B, pn*pn)
        """
        f = self.quant_conv(self.encoder(inp))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=None)

    def decode_from_indices(
        self,
        ms_idx_Bl: List[Tensor],
        image_features: Tensor,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Decode multi-scale indices to mask using image features.

        Args:
            ms_idx_Bl: List of indices for each scale, each (B, pn*pn)
            image_features: (B, img_feat_dim, H_img, W_img) external image features
            v_patch_nums: Patch numbers for each scale (default: self.v_patch_nums)
            original_size: (H, W) original image size

        Returns:
            Reconstructed mask (B, 1, H, W)
        """
        if v_patch_nums is None:
            v_patch_nums = self.v_patch_nums

        B = ms_idx_Bl[0].shape[0]

        # Convert indices to embeddings (multi-scale tokens)
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            h_BChw = (
                self.quantize.embedding(idx_Bl)
                .transpose(1, 2)
                .view(B, self.Cvae, pn, pn)
            )
            ms_h_BChw.append(h_BChw)

        return self.decode_from_tokens(
            ms_h_BChw, image_features, v_patch_nums, original_size
        )

    def decode_from_tokens(
        self,
        ms_h_BChw: List[Tensor],
        image_features: Tensor,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Decode multi-scale token features to mask using image features.

        Args:
            ms_h_BChw: List of token features for each scale, each (B, Cvae, pn, pn)
            image_features: (B, img_feat_dim, H_img, W_img) external image features
            v_patch_nums: Patch numbers for each scale
            original_size: (H, W) original image size

        Returns:
            Reconstructed mask (B, 1, H, W)
        """
        if v_patch_nums is None:
            v_patch_nums = self.v_patch_nums

        # Use mask decoder with image features
        fused_mask, _ = self.mask_decoder(
            image_features=image_features,
            ms_tokens=ms_h_BChw,
            v_patch_nums=v_patch_nums,
            original_size=original_size,
        )

        return fused_mask

    def forward(
        self,
        inp: Tensor,
        image_features: Optional[Tensor] = None,
        ret_usages: bool = False,
        use_image_features: bool = True,
    ) -> Union[Tuple[Tensor, Optional[List[float]], Tensor], Tensor]:
        """
        Forward pass for training.

        Args:
            inp: Input single-channel mask (B, 1, H, W), values in [-1, 1]
            image_features: Optional external image features (B, img_feat_dim, H_img, W_img)
                           If None and use_image_features=True, raises error
            ret_usages: Whether to return codebook usage
            use_image_features: Whether to use image-guided decoder (default: True)
                               If False, falls back to standard VAE decoder

        Returns:
            If use_image_features and ret_usages:
                (reconstructed_mask, usages, vq_loss)
            If use_image_features and not ret_usages:
                (reconstructed_mask, vq_loss)
            If not use_image_features:
                reconstructed_mask (standard VAE output)
        """
        # Encode and quantize
        f_hat, usages, vq_loss = self.encode(inp, ret_usages=ret_usages)

        if use_image_features:
            if image_features is None:
                raise ValueError(
                    "image_features must be provided when use_image_features=True"
                )

            # Get multi-scale tokens from quantized features
            # f_hat: (B, Cvae, H, W) where H=W=downsampled_size
            B, _, H, W = f_hat.shape

            # Convert to multi-scale indices then back to tokens
            # This follows the standard VQVAE flow
            ms_idx_Bl = self.quantize.f_to_idxBl_or_fhat(
                f_hat, to_fhat=False, v_patch_nums=None
            )

            # Decode using image features
            rec_mask = self.decode_from_indices(
                ms_idx_Bl, image_features, original_size=(inp.shape[2], inp.shape[3])
            )

            if ret_usages:
                return rec_mask, usages, vq_loss
            else:
                return rec_mask, vq_loss
        else:
            # Fallback to standard decoder
            rec = self.decoder(self.post_quant_conv(f_hat))
            return rec

    def img_to_idxBl(
        self,
        inp_img_no_grad: Tensor,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    ) -> List[torch.LongTensor]:
        """
        Convert mask image to multi-scale index list.

        Args:
            inp_img_no_grad: Input mask (no gradient), shape (B, 1, H, W)
            v_patch_nums: Patch numbers per scale

        Returns:
            List of indices for each scale
        """
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def idxBl_to_img(
        self,
        ms_idx_Bl: List[Tensor],
        image_features: Tensor,
        same_shape: bool = True,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Convert multi-scale index list to mask image using image features.

        Args:
            ms_idx_Bl: Multi-scale index list
            image_features: (B, img_feat_dim, H_img, W_img) external image features
            same_shape: Whether to interpolate features to max scale (default: True)
            original_size: (H, W) original image size

        Returns:
            Reconstructed mask (B, 1, H, W)
        """
        return self.decode_from_indices(
            ms_idx_Bl, image_features, original_size=original_size
        )

    def idxBl_to_img_or_reconstructed_img(
        self,
        ms_idx_Bl: List[Tensor],
        image_features: Optional[Tensor] = None,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        same_shape: bool = True,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Convert indices to reconstructed image.

        This is a convenience method that works both with and without image features.

        Args:
            ms_idx_Bl: Multi-scale index list
            image_features: Optional external image features
            v_patch_nums: Patch numbers per scale
            same_shape: Whether to interpolate to same shape
            original_size: Original image size

        Returns:
            Reconstructed mask
        """
        if image_features is not None:
            return self.idxBl_to_img(
                ms_idx_Bl, image_features, same_shape, original_size
            )
        else:
            # Fallback to standard embed_to_img logic
            B = ms_idx_Bl[0].shape[0]
            ms_h_BChw = []
            for idx_Bl in ms_idx_Bl:
                l = idx_Bl.shape[1]
                pn = round(l ** 0.5)
                h_BChw = (
                    self.quantize.embedding(idx_Bl)
                    .transpose(1, 2)
                    .view(B, self.Cvae, pn, pn)
                )
                ms_h_BChw.append(h_BChw)

            # Use embed_to_fhat and standard decoder
            f_hat = self.quantize.embed_to_fhat(
                ms_h_BChw, all_to_max_scale=same_shape, last_one=True
            )
            return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)

    def img_to_reconstructed_img(
        self,
        x: Tensor,
        image_features: Optional[Tensor] = None,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        last_one: bool = False,
    ) -> Union[List[Tensor], Tensor]:
        """
        Convert mask image to reconstructed image.

        Args:
            x: Input mask (B, 1, H, W)
            image_features: Optional external image features
            v_patch_nums: Patch numbers per scale
            last_one: Whether to return only the last scale's reconstruction

        Returns:
            Reconstructed mask(s)
        """
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(
            f, to_fhat=True, v_patch_nums=v_patch_nums
        )

        if image_features is not None:
            # Use mask decoder with image features
            original_size = (x.shape[2], x.shape[3])
            all_masks = []
            for si, f_hat in enumerate(ls_f_hat_BChw):
                pn = round(f_hat.shape[2])
                mask = self.mask_decoder.decode_single_scale(
                    image_features, f_hat, si, original_size
                )
                all_masks.append(mask)

            if last_one:
                return all_masks[-1]
            else:
                return all_masks
        else:
            # Use standard decoder
            if last_one:
                return (
                    self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1]))
                    .clamp_(-1, 1)
                )
            else:
                return [
                    self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
                    for f_hat in ls_f_hat_BChw
                ]

    def load_state_dict(
        self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False
    ):
        """
        Load state dict with handling for quantizer ema_vocab_hit shape mismatches.
        """
        if (
            "quantize.ema_vocab_hit_SV" in state_dict
            and state_dict["quantize.ema_vocab_hit_SV"].shape[0]
            != self.quantize.ema_vocab_hit_SV.shape[0]
        ):
            state_dict["quantize.ema_vocab_hit_SV"] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
