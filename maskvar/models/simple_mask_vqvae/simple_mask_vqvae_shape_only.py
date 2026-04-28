import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, repeat

from .basic import LayerNorm2d, SimpleCrossBlockReverse
from .quant import SimpleVectorQuantize
from .simple_mask_vqvae import MaskFeatureCompactor
from ..rope2d import RotaryPositionEmbedding2D


class ShapeOnlyMaskDecoder(nn.Module):

    def __init__(
        self,
        rope: RotaryPositionEmbedding2D,
        dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.rope = rope
        self.dim = dim

        self.spatial_tokens = nn.Parameter(torch.randn(1, rope.h, rope.w, dim))
        self.cross_blocks = nn.ModuleList([
            SimpleCrossBlockReverse(rope=rope, dim=dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.output_upscaling = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(dim // 4),
            nn.GELU(),
            nn.PixelShuffle(2),
            nn.Conv2d(dim // 16, dim // 16, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1),
        )

    def forward(self, query_tokens: torch.Tensor):
        spatial_tokens = repeat(self.spatial_tokens, '1 h w c -> b h w c', b=query_tokens.shape[0])
        for blk in self.cross_blocks:
            spatial_tokens = blk(spatial_tokens, query_tokens, pe_type='rope')

        feature_map = rearrange(spatial_tokens, 'b h w c -> b c h w')
        return self.output_upscaling(feature_map).contiguous()


class SimpleMaskVqvaeShapeOnly(nn.Module):

    def __init__(
        self,
        mask_encoder,
        dim: int = 256,
        num_queries: int = 8,
        vocab_size: int = 4096,
        beta: float = 0.25,
        h: int = 64,
        w: int = 64,
        enable_vq: bool = True,
        num_heads: int = 4,
        decoder_depth: int = 2,
        device: str = 'cuda',
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.beta = beta
        self.enable_vq = enable_vq
        self.device = device
        # Keep a no-op image encoder attribute so existing trainers can reuse
        # the same freeze/introspection logic without special-casing.
        self.image_encoder = nn.Identity()
        self.mask_encoder = mask_encoder
        self.rope = RotaryPositionEmbedding2D(h=h, w=w)
        self.mask_feature_compactor = MaskFeatureCompactor(
            rope=self.rope,
            dim=dim,
            num_queries=num_queries,
            num_heads=num_heads,
            depth=1,
        )
        self.mask_decoder = ShapeOnlyMaskDecoder(
            rope=self.rope,
            dim=dim,
            num_heads=num_heads,
            num_layers=decoder_depth,
        )
        self.quant = SimpleVectorQuantize(
            dim=dim,
            vocab_size=vocab_size,
            beta=beta,
            using_znorm=False,
        )

    def encode(self, mask_normalized: torch.Tensor):
        mask_tokens = self.mask_encoder(mask_normalized)
        mask_tokens = rearrange(mask_tokens, 'b c h w -> b h w c')
        return self.mask_feature_compactor(mask_tokens)

    def decode(self, query_tokens: torch.Tensor, output_size=None):
        mask = self.mask_decoder(query_tokens)
        if output_size is not None and mask.shape[-2:] != output_size:
            mask = F.interpolate(mask, size=output_size, mode='bilinear', align_corners=False)
        return mask

    def forward(
        self,
        mask_normalized: torch.Tensor,
        image: torch.Tensor = None,
        return_usage: bool = False,
    ):
        _, _, H, W = mask_normalized.shape

        query_tokens = self.encode(mask_normalized)

        if self.enable_vq:
            if return_usage:
                query_tokens, vq_loss, vq_usage = self.quant(query_tokens, return_usage=True)
            else:
                query_tokens, vq_loss = self.quant(query_tokens)
                vq_usage = None
        else:
            vq_loss = query_tokens.new_tensor(0.0)
            vq_usage = query_tokens.new_tensor(0.0)

        mask = self.decode(query_tokens, output_size=(H, W))

        if return_usage:
            return mask, vq_loss, vq_usage
        return mask, vq_loss

    @torch.no_grad()
    def encode_mask_to_token_ids(self, mask_normalized: torch.Tensor, image: torch.Tensor = None):
        query_tokens = self.encode(mask_normalized)
        return self.quant.x_to_idx(query_tokens.float())

    @torch.no_grad()
    def decode_token_ids_to_mask_logits(self, token_ids: torch.Tensor, output_size=None):
        query_tokens = self.quant.idx_to_x(token_ids)
        return self.decode(query_tokens, output_size=output_size)
