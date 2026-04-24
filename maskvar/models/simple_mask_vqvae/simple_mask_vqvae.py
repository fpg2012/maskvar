import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as tdist

import timm
from einops import rearrange, repeat
from .basic import SimpleCrossBlock
from .mask_decoder import SimpleMaskDecoder, SimpleMaskDecoderV2
from .quant import SimpleVectorQuantize
from ..rope2d import RotaryPositionEmbedding2D

class SimpleMaskVqvae(nn.Module):

    def __init__(self, image_encoder, mask_encoder, dim=256, vocab_size=4096, beta=0.25, h=64, w=64, enable_vq=True, device='cuda'):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.beta = beta
        self.enable_vq = enable_vq

        self.image_encoder = image_encoder
        self.mask_encoder = mask_encoder

        self.rope = RotaryPositionEmbedding2D(h=h, w=w)
        
        self.mask_decoder = SimpleMaskDecoder(rope=self.rope, dim=dim)
        self.quant = SimpleVectorQuantize(
            dim=dim,
            vocab_size=vocab_size,
            beta=beta,
            using_znorm=False,
        )
        self.device = device
    
    def forward(self, mask_normalized: torch.Tensor, image: torch.Tensor, return_usage: bool = False):
        """
        for training only
        Args:
            mask_normalized: (B, 1, H, W) normalized mask
            image: (B, 3, H, W) image
            return_usage: if True, return vocabulary usage percentage
        """
        B, _, H, W = mask_normalized.shape

        # 1. encoder mask and image with corresponding encoders
        # mask_normalized_3 = repeat(mask_normalized, 'b 1 h w -> b 3 h w')
        mask_tokens = self.mask_encoder(mask_normalized)
        image_tokens = self.image_encoder(image)

        # Convert from (b, c, h, w) to (b, h, w, c) for decoder
        _, _, h, w = mask_tokens.shape
        mask_tokens = rearrange(mask_tokens, 'b c h w -> b (h w) c')
        image_tokens = rearrange(image_tokens, 'b c h w -> b h w c')

        # 2. quantize mask_tokens
        if self.enable_vq:
            if return_usage:
                mask_tokens, vq_loss, vq_usage = self.quant(mask_tokens, return_usage=True)
            else:
                mask_tokens, vq_loss = self.quant(mask_tokens)
                vq_usage = None
        else:
            # skip quant
            vq_loss = torch.tensor(0.0)
            vq_usage = torch.tensor(0.0)
        
        mask_tokens = rearrange(mask_tokens, 'b (h w) c -> b h w c', h=h, w=w)

        # 3. decode
        mask = self.mask_decoder(mask_tokens, image_tokens)

        # print(f'mask shape: {mask.shape}')

        # 4. resize mask to original size (decoder outputs 256x256, input may be 1024x1024)
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)

        if return_usage:
            return mask, vq_loss, vq_usage
        return mask, vq_loss


class MaskFeatureCompactor(nn.Module):

    def __init__(self, rope: RotaryPositionEmbedding2D, dim=256, num_queries=8, num_heads=4, depth=1):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, dim))
        self.cross_blocks = nn.ModuleList([
           SimpleCrossBlock(rope=rope, dim=dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.depth = depth

    def forward(self, mask_tokens, pe_type='rope', image_pe=None):
        """
        mask_tokens: (b, h, w, c)
        """
        query_tokens = repeat(self.query_tokens, '1 l c -> b l c', b=mask_tokens.shape[0])
        for blk in self.cross_blocks:
            query_tokens = blk(query_tokens, mask_tokens, pe_type=pe_type, image_pe=image_pe)
        
        return query_tokens


class SimpleMaskVqvaeV2(nn.Module):

    def __init__(self, image_encoder, mask_encoder, dim=256, num_queries=8, vocab_size=4096, beta=0.25, h=64, w=64, enable_vq=True, device='cuda'):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.beta = beta
        self.enable_vq = enable_vq
        self.rope = RotaryPositionEmbedding2D(h=h, w=w)

        self.image_encoder = image_encoder
        self.mask_encoder = mask_encoder
        
        self.mask_decoder = SimpleMaskDecoderV2(rope=self.rope, dim=dim, num_heads=4, num_queries=num_queries)
        self.mask_feature_compactor: MaskFeatureCompactor = MaskFeatureCompactor(
            rope=self.rope,
            dim=dim,
            num_queries=num_queries,
            num_heads=4,
            depth=1,
        )

        self.quant = SimpleVectorQuantize(
            dim=dim,
            vocab_size=vocab_size,
            beta=beta,
            using_znorm=False,
        )
        self.device = device
    
    def forward(self, mask_normalized: torch.Tensor, image: torch.Tensor, return_usage: bool = False):
        """
        for training only
        Args:
            mask_normalized: (B, 1, H, W) normalized mask
            image: (B, 3, H, W) image
            return_usage: if True, return vocabulary usage percentage
        """
        B, _, H, W = mask_normalized.shape

        # 1. encoder mask and image with corresponding encoders
        # mask_normalized_3 = repeat(mask_normalized, 'b 1 h w -> b 3 h w')
        mask_tokens = self.mask_encoder(mask_normalized)
        image_tokens = self.image_encoder(image)

        # Convert from (b, c, h, w) to (b, h, w, c) for decoder
        mask_tokens = rearrange(mask_tokens, 'b c h w -> b h w c')
        image_tokens = rearrange(image_tokens, 'b c h w -> b h w c')

        queries_tokens = self.mask_feature_compactor(mask_tokens)

        # 2. quantize mask_tokens
        if self.enable_vq:
            if return_usage:
                queries_tokens, vq_loss, vq_usage = self.quant(queries_tokens, return_usage=True)
            else:
                queries_tokens, vq_loss = self.quant(queries_tokens)
                vq_usage = None
        else:
            # skip quant
            vq_loss = torch.tensor(0.0)
            vq_usage = torch.tensor(0.0)

        # 3. decode
        mask = self.mask_decoder(queries_tokens, image_tokens)

        # print(f'mask shape: {mask.shape}')

        # 4. resize mask to original size (decoder outputs 256x256, input may be 1024x1024)
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)

        if return_usage:
            return mask, vq_loss, vq_usage
        return mask, vq_loss 