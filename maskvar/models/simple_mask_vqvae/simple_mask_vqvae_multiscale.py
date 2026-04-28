import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from .mask_decoder import SimpleMaskDecoder
from .quant import SimpleVectorQuantize
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
        mask_feature = self.mask_encoder(mask_normalized)
        tokens_by_scale = []
        for scale_idx, scale in enumerate(self.scales):
            feat = F.adaptive_avg_pool2d(mask_feature, output_size=(scale, scale))
            tokens = rearrange(feat, "b c h w -> b (h w) c")
            tokens = tokens + self.scale_embed[scale_idx].view(1, 1, self.dim)
            tokens_by_scale.append(tokens)
        return tokens_by_scale

    def _quantize_multiscale(self, tokens_by_scale, return_usage=False):
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

    def encode_to_multiscale_token_ids(self, mask_normalized: torch.Tensor):
        tokens_by_scale = self._extract_multiscale_tokens(mask_normalized)
        token_ids_by_scale = [self.quant.x_to_idx(tokens.float()) for tokens in tokens_by_scale]
        return token_ids_by_scale

    def decode_from_multiscale_token_ids(self, token_ids_by_scale, image=None, image_tokens=None, output_size=None):
        if len(token_ids_by_scale) != len(self.scales):
            raise ValueError(f"Expected {len(self.scales)} scales, got {len(token_ids_by_scale)}")
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
        _, _, H, W = mask_normalized.shape
        tokens_by_scale = self._extract_multiscale_tokens(mask_normalized)
        image_tokens = self.image_encoder(image)
        image_tokens = rearrange(image_tokens, "b c h w -> b h w c")

        if self.enable_vq:
            tokens_by_scale, vq_loss, vq_usage = self._quantize_multiscale(tokens_by_scale, return_usage=return_usage)
        else:
            vq_loss = torch.tensor(0.0, device=mask_normalized.device, dtype=mask_normalized.dtype)
            vq_usage = torch.tensor(0.0, device=mask_normalized.device, dtype=mask_normalized.dtype)

        mask_tokens = self._fuse_multiscale_tokens(tokens_by_scale)
        mask = self.mask_decoder(mask_tokens, image_tokens)
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)

        if return_usage:
            return mask, vq_loss, vq_usage
        return mask, vq_loss
