import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from .mask_decoder import SimpleMaskDecoder
from .quant import SimpleVectorQuantize, MultiscaleVectorQuantize
from ..rope2d import RotaryPositionEmbedding2D


class SimpleMaskVqvaeMultiScale(nn.Module):
    """
    Multi-scale SimpleMaskVqvae tokenizer.

    The mask encoder still produces the proven 64x64 mask feature map. Lower
    scales are obtained by adaptive average pooling from that feature map, then
    every scale is optionally quantized with a shared codebook. Quantized
    features are upsampled back to 64x64, fused, and decoded with the existing
    SimpleMaskDecoder.
    """

    def __init__(
        self,
        image_encoder,
        mask_encoder,
        dim=384,
        vocab_size=4096,
        beta=0.25,
        scales=(1, 2, 4, 8, 16, 32, 64),
        h=64,
        w=64,
        enable_vq=True,
        device="cuda",
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.beta = beta
        self.scales = tuple(scales)
        self.h = h
        self.w = w
        self.enable_vq = enable_vq
        self.device = device

        if self.scales[-1] != h or h != w:
            raise ValueError(f"Expected last scale to match square decoder grid {h}x{w}, got scales={self.scales}")

        self.image_encoder = image_encoder
        self.mask_encoder = mask_encoder
        self.rope = RotaryPositionEmbedding2D(h=h, w=w)
        self.mask_decoder = SimpleMaskDecoder(rope=self.rope, dim=dim)
        self.quant = SimpleVectorQuantize(dim=dim, vocab_size=vocab_size, beta=beta, using_znorm=False)

        self.scale_embed = nn.Parameter(torch.zeros(len(self.scales), dim))
        self.scale_projs = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=1) for _ in self.scales])
        self.fuse_norm = nn.LayerNorm(dim)

    def _extract_multiscale_tokens(self, mask_normalized: torch.Tensor):
        """
        Encode a normalized mask into one token sequence per scale.

        Args:
            mask_normalized: Float mask tensor of shape [B, 1, H, W].

        Returns:
            list[Tensor]: For each scale S in self.scales, a tensor of shape
                [B, S*S, C], where C == self.dim. The scale embedding has
                already been added to each token.
        """
        mask_feature = self.mask_encoder(mask_normalized)
        tokens_by_scale = []
        for scale_idx, scale in enumerate(self.scales):
            feat = F.adaptive_avg_pool2d(mask_feature, output_size=(scale, scale))
            tokens = rearrange(feat, "b c h w -> b (h w) c")
            tokens = tokens + self.scale_embed[scale_idx].view(1, 1, self.dim)
            tokens_by_scale.append(tokens)
        return tokens_by_scale

    def _quantize_multiscale(self, tokens_by_scale, return_usage=False):
        """
        Quantize all scale tokens with the shared VQ codebook.

        Args:
            tokens_by_scale: list of tensors with shapes [B, S_i*S_i, C].
            return_usage: Whether to return codebook usage statistics.

        Returns:
            tokens_q_by_scale: list of quantized tensors matching the input
                shapes [B, S_i*S_i, C].
            vq_loss: Scalar tensor from the vector quantizer.
            vq_usage: Usage tensor/statistic when requested, otherwise None.
        """
        lengths = [tokens.shape[1] for tokens in tokens_by_scale]
        all_tokens = torch.cat(tokens_by_scale, dim=1)
        if return_usage:
            all_tokens_q, vq_loss, vq_usage = self.quant(all_tokens, return_usage=True)
        else:
            all_tokens_q, vq_loss = self.quant(all_tokens)
            vq_usage = None
        tokens_q_by_scale = list(all_tokens_q.split(lengths, dim=1))
        return tokens_q_by_scale, vq_loss, vq_usage

    def _fuse_multiscale_tokens(self, tokens_by_scale):
        """
        Upsample scale token maps to the decoder grid and fuse them.

        Args:
            tokens_by_scale: list of tensors with shapes [B, S_i*S_i, C],
                one per scale in self.scales.

        Returns:
            Tensor of shape [B, self.h, self.w, C]. This is the fused mask
            token grid consumed by SimpleMaskDecoder.
        """
        fused = None
        for scale_idx, (scale, tokens) in enumerate(zip(self.scales, tokens_by_scale)):
            feat = rearrange(tokens, "b (h w) c -> b c h w", h=scale, w=scale)
            if scale != self.h:
                feat = F.interpolate(feat, size=(self.h, self.w), mode="bilinear", align_corners=False)
            feat = self.scale_projs[scale_idx](feat)
            fused = feat if fused is None else fused + feat
        fused = fused / len(tokens_by_scale)
        fused = rearrange(fused, "b c h w -> b h w c")
        return self.fuse_norm(fused)

    def _zero_tokens_like(self, tokens_by_scale, scale_idx):
        return torch.zeros_like(tokens_by_scale[scale_idx])

    def _decode_tokens_by_scale(self, tokens_by_scale, image_tokens, output_size=None):
        mask_tokens = self._fuse_multiscale_tokens(tokens_by_scale)
        mask = self.mask_decoder(mask_tokens, image_tokens)
        if output_size is not None and mask.shape[-2:] != output_size:
            mask = F.interpolate(mask, size=output_size, mode="bilinear", align_corners=False)
        return mask

    def _image_tokens_to_grid(self, image_tokens):
        if image_tokens.dim() == 4 and image_tokens.shape[1] == self.dim:
            return rearrange(image_tokens, "b c h w -> b h w c")
        if image_tokens.dim() == 3:
            h = w = int(image_tokens.shape[1] ** 0.5)
            return rearrange(image_tokens, "b (h w) c -> b h w c", h=h, w=w)
        return image_tokens

    def encode_to_multiscale_token_ids(self, mask_normalized: torch.Tensor):
        """
        Convert a mask into discrete token ids for every scale.

        Args:
            mask_normalized: Float mask tensor of shape [B, 1, H, W].

        Returns:
            list[Tensor]: For each scale S, token ids of shape [B, S*S].
        """
        tokens_by_scale = self._extract_multiscale_tokens(mask_normalized)
        token_ids_by_scale = [self.quant.x_to_idx(tokens.float()) for tokens in tokens_by_scale]
        return token_ids_by_scale

    def decode_from_multiscale_token_ids(self, token_ids_by_scale, image=None, image_tokens=None, output_size=None):
        """
        Decode multiscale token ids back into mask logits.

        Args:
            token_ids_by_scale: list of integer tensors with shapes [B, S_i*S_i],
                one per scale in self.scales.
            image: Optional image tensor passed to image_encoder, typically
                [B, 3, H_img, W_img]. Required when image_tokens is None.
            image_tokens: Optional encoded image features. Accepted shapes are
                [B, C, H_img', W_img'] or [B, H_img'*W_img', C].
            output_size: Optional target spatial size (H_out, W_out).

        Returns:
            Tensor of mask logits with shape [B, 1, H_out, W_out] when
            output_size is provided, otherwise the decoder's native mask size.
        """
        if len(token_ids_by_scale) == 0:
            return []
        if len(token_ids_by_scale) > len(self.scales):
            raise ValueError(f"Expected at most {len(self.scales)} scales, got {len(token_ids_by_scale)}")
        tokens_by_scale = [self.quant.idx_to_x(token_ids) for token_ids in token_ids_by_scale]
        mask_tokens = self._fuse_multiscale_tokens(tokens_by_scale)

        if image_tokens is None:
            if image is None:
                raise ValueError("Either image or image_tokens must be provided.")
            image_tokens = self.image_encoder(image)
        if image_tokens.dim() == 4 and image_tokens.shape[1] == self.dim:
            image_tokens = rearrange(image_tokens, "b c h w -> b h w c")
        elif image_tokens.dim() == 3:
            h = w = int(image_tokens.shape[1] ** 0.5)
            image_tokens = rearrange(image_tokens, "b (h w) c -> b h w c", h=h, w=w)

        mask = self.mask_decoder(mask_tokens, image_tokens)
        if output_size is not None and mask.shape[-2:] != output_size:
            mask = F.interpolate(mask, size=output_size, mode="bilinear", align_corners=False)
        return mask

    def forward(self, mask_normalized: torch.Tensor, image: torch.Tensor, return_usage: bool = False):
        """
        Train/evaluate the multiscale mask tokenizer end to end.

        Args:
            mask_normalized: Float mask tensor of shape [B, 1, H, W].
            image: Image tensor consumed by image_encoder, typically
                [B, 3, H_img, W_img].
            return_usage: Whether to return codebook usage statistics.

        Returns:
            If return_usage is False:
                (mask, vq_loss), where mask has shape [B, 1, H, W] and
                vq_loss is a scalar tensor.
            If return_usage is True:
                (mask, vq_loss, vq_usage), with the same mask shape and the
                quantizer usage statistic as the third value.
        """
        _, _, H, W = mask_normalized.shape
        tokens_by_scale = self._extract_multiscale_tokens(mask_normalized)
        image_tokens = self.image_encoder(image)
        image_tokens = rearrange(image_tokens, "b c h w -> b h w c")

        if self.enable_vq:
            tokens_by_scale, vq_loss, vq_usage = self._quantize_multiscale(tokens_by_scale, return_usage=return_usage)
        else:
            vq_loss = torch.tensor(0.0, device=mask_normalized.device, dtype=torch.float32)
            vq_usage = torch.tensor(0.0, device=mask_normalized.device, dtype=torch.float32)

        mask_tokens = self._fuse_multiscale_tokens(tokens_by_scale)
        mask = self.mask_decoder(mask_tokens, image_tokens)
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)

        if return_usage:
            return mask, vq_loss, vq_usage
        return mask, vq_loss


class SimpleMaskVqvaeMultiScaleResidual(SimpleMaskVqvaeMultiScale):
    """
    VAR-style residual multi-scale quantizer for SimpleMaskVqvae.

    This follows the residual quantization recipe used by VAR:

        f = E(mask)
        for scale in scales:
            z_k = Q(interpolate(f, scale))
            f = f - phi_k(upsample(z_k, full_resolution))

    At decode time, the full 64x64 latent is reconstructed by summing the same
    projected quantized features. Compared with the v1 parallel pooling path,
    the finest scale now quantizes only the residual that earlier scales failed
    to explain.
    """

    def __init__(
        self,
        image_encoder,
        mask_encoder,
        dim=384,
        vocab_size=4096,
        beta=0.25,
        scales=(1, 2, 4, 8, 16, 32, 64),
        h=64,
        w=64,
        enable_vq=True,
        device="cuda",
    ):
        nn.Module.__init__(self)
        self.dim = dim
        self.vocab_size = vocab_size
        self.beta = beta
        self.scales = tuple(scales)
        self.h = h
        self.w = w
        self.enable_vq = enable_vq

        self.image_encoder = image_encoder
        self.mask_encoder = mask_encoder
        self.rope = RotaryPositionEmbedding2D(h=h, w=w)
        self.mask_decoder = SimpleMaskDecoder(rope=self.rope, dim=dim)
        self.quant = MultiscaleVectorQuantize(
            dim=dim,
            vocab_size=vocab_size,
            beta=beta,
            scales=scales,
            h=h,
            w=w,
            using_znorm=False,
        )
        self.device = device

    def forward(self, mask_normalized: torch.Tensor, image: torch.Tensor, return_usage: bool = False):
        """
        Same data flow as SimpleMaskVqvae.forward(), with only the quantizer
        swapped from SimpleVectorQuantize to MultiscaleVectorQuantize.
        """
        _, _, H, W = mask_normalized.shape

        mask_tokens = self.mask_encoder(mask_normalized)
        image_tokens = self.image_encoder(image)

        _, _, h, w = mask_tokens.shape
        mask_tokens = rearrange(mask_tokens, "b c h w -> b (h w) c")
        image_tokens = rearrange(image_tokens, "b c h w -> b h w c")

        if self.enable_vq:
            if return_usage:
                mask_tokens, vq_loss, vq_usage = self.quant(mask_tokens, return_usage=True)
            else:
                mask_tokens, vq_loss = self.quant(mask_tokens)
                vq_usage = None
        else:
            vq_loss = torch.tensor(0.0, device=mask_normalized.device, dtype=torch.float32)
            vq_usage = torch.tensor(0.0, device=mask_normalized.device, dtype=torch.float32)

        mask_tokens = rearrange(mask_tokens, "b (h w) c -> b h w c", h=h, w=w)
        mask = self.mask_decoder(mask_tokens, image_tokens)
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)

        if return_usage:
            return mask, vq_loss, vq_usage
        return mask, vq_loss

    def encode_to_multiscale_token_ids(self, mask_normalized: torch.Tensor):
        mask_tokens = self.mask_encoder(mask_normalized)
        _, _, h, w = mask_tokens.shape
        if h != self.h or w != self.w:
            raise ValueError(f"Expected mask encoder grid {self.h}x{self.w}, got {h}x{w}")
        mask_tokens = rearrange(mask_tokens, "b c h w -> b (h w) c")
        return self.quant.x_to_idx_multiscale(mask_tokens.float())

    def to_var_input(self, token_ids_by_scale):
        """
        Convert multiscale V2 token ids to SimpleMaskVAR teacher-forcing inputs.

        This mirrors VAR's idxBl_to_var_input: after each known scale is
        projected into the cumulative full-resolution f_hat, f_hat is area
        downsampled to the next scale and appended as BLC. With all GT scales
        this returns inputs for scales 1..K-1, concatenated along L.
        """
        if len(token_ids_by_scale) > len(self.scales):
            raise ValueError(f"Expected at most {len(self.scales)} scales, got {len(token_ids_by_scale)}")
        if len(token_ids_by_scale) == 0:
            return None

        B = token_ids_by_scale[0].shape[0]
        device = token_ids_by_scale[0].device
        f_hat = self.quant.embedding.weight.new_zeros(B, self.h * self.w, self.dim).to(device=device)
        next_inputs = []

        for scale_idx, token_ids in enumerate(token_ids_by_scale):
            if scale_idx >= len(self.scales) - 1:
                break

            tokens = self.quant.idx_to_x(token_ids)
            projected = self.quant._project_tokens_to_full(tokens, scale_idx)
            f_hat = f_hat + rearrange(projected, "b c h w -> b (h w) c")

            next_scale = self.scales[scale_idx + 1]
            next_input = rearrange(f_hat, "b (h w) c -> b c h w", h=self.h, w=self.w)
            if next_scale != self.h or next_scale != self.w:
                next_input = F.interpolate(next_input, size=(next_scale, next_scale), mode="area")
            next_inputs.append(rearrange(next_input, "b c h w -> b (h w) c"))

        return torch.cat(next_inputs, dim=1) if next_inputs else None

    def decode_from_multiscale_token_ids(self, token_ids_by_scale, image=None, image_tokens=None, output_size=None):
        mask_tokens = self.quant.idxBl_to_full_tokens(token_ids_by_scale)
        mask_tokens = rearrange(mask_tokens, "b (h w) c -> b h w c", h=self.h, w=self.w)

        if image_tokens is None:
            if image is None:
                raise ValueError("Either image or image_tokens must be provided.")
            image_tokens = self.image_encoder(image)
        if image_tokens.dim() == 4 and image_tokens.shape[1] == self.dim:
            image_tokens = rearrange(image_tokens, "b c h w -> b h w c")
        elif image_tokens.dim() == 3:
            h = w = int(image_tokens.shape[1] ** 0.5)
            image_tokens = rearrange(image_tokens, "b (h w) c -> b h w c", h=h, w=w)

        mask = self.mask_decoder(mask_tokens, image_tokens)
        if output_size is not None and mask.shape[-2:] != output_size:
            mask = F.interpolate(mask, size=output_size, mode="bilinear", align_corners=False)
        return mask
