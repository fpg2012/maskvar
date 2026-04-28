import torch
from torch import nn

from einops import rearrange, repeat
from ..sam import MaskDecoder as SamMaskDecoder
from .basic import LayerNorm2d, TwoWayBlock, SimpleTwoWayBlock, MLP, SimpleTwoWayBlock
from ..rope2d import RotaryPositionEmbedding2D

class SimpleMaskDecoder(nn.Module):

    def __init__(self, rope: RotaryPositionEmbedding2D, dim, num_heads=4, num_queries=4, num_two_way_blocks=2):
        super().__init__()
        self.dim = dim
        self.num_two_way_blocks = num_two_way_blocks
        self.rope = rope

        self.two_way_blocks = nn.ModuleList([
            TwoWayBlock(rope, dim, num_heads) for _ in range(num_two_way_blocks)
        ])
        
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, dim))
        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(dim, dim // 4, kernel_size=4, stride=4),
        #     LayerNorm2d(dim // 4),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=4, stride=4),
        #     nn.GELU(),
        # )
        self.output_upscaling = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(dim//4, dim//4, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(dim // 4),
            nn.GELU(),
            nn.PixelShuffle(2),
            nn.Conv2d(dim//16, dim//16, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        # self.output_upscaling2 = nn.Sequential(
        #     nn.ConvTranspose2d(dim, dim // 4, kernel_size=2, stride=2),
        #     LayerNorm2d(dim // 4),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
        #     nn.GELU(),
        # )
        self.hyper_in = MLP(dim, dim, dim // 16, 3)

        # self.layer_norm_pre_image = nn.LayerNorm(dim)
        # self.layer_norm_pre_query = nn.LayerNorm(dim)
        # self.layer_norm_pre_mask = nn.LayerNorm(dim)
        self.layer_norm_post_image = nn.LayerNorm(dim // 16)
        self.layer_norm_post_query = nn.LayerNorm(dim // 16)
        # self.layer_norm_post_mask = nn.LayerNorm(dim // 16)
    
    def forward(self, mask_tokens: torch.Tensor, image_tokens: torch.Tensor):
        # print(f"[DEBUG] Decoder forward - mask_tokens: {mask_tokens.shape}, image_tokens: {image_tokens.shape}")
        
        # mask_tokens = self.layer_norm_image(mask_tokens)
        # image_tokens = self.layer_norm_mask(image_tokens)

        query_tokens = repeat(self.query_tokens, '1 l c -> b l c', b=mask_tokens.shape[0])
        for blk in self.two_way_blocks:
            query_tokens, mask_tokens, image_tokens = blk(query_tokens, mask_tokens, image_tokens, pe_type='rope')

        image_feature_map = rearrange(image_tokens, 'b h w c -> b c h w')
        mask_feature_map = rearrange(mask_tokens, 'b h w c -> b c h w')
        up_query_token = self.hyper_in(query_tokens[:, 0, :])
        up_image_map = self.output_upscaling(image_feature_map)
        # up_mask_map = self.output_upscaling2(mask_feature_map)

        up_query_token = self.layer_norm_post_query(up_query_token)
        up_image_map = self.layer_norm_post_image(rearrange(up_image_map, 'b c h w -> b h w c'))
        # up_mask_map = self.layer_norm_post_mask(rearrange(up_mask_map, 'b c h w -> b h w c'))

        masks = torch.einsum('bc,bhwc->bhw', up_query_token, up_image_map).unsqueeze(1)
        # masks = torch.einsum('bhwc,bhwc->bhw', up_mask_map, up_image_map)
        # try addition
        # masks = rearrange(up_query_token, 'b c -> b c 1 1') + up_image_map
        # masks = torch.mean(masks, dim=1, keepdim=True)

        # Ensure final output is contiguous
        return masks.contiguous()


class SimpleMaskDecoderV2(nn.Module):

    def __init__(self, rope: RotaryPositionEmbedding2D, dim, num_heads=4, num_queries=4, num_two_way_blocks=2):
        super().__init__()
        self.rope = rope
        self.dim = dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_two_way_blocks = num_two_way_blocks
        
        self.output_upscaling = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(dim//4, dim//4, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(dim // 4),
            nn.GELU(),
            nn.PixelShuffle(2),
            nn.Conv2d(dim//16, dim//16, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )

        self.hyper_in = MLP(dim, dim, dim // 16, 3)
        self.layer_norm_post_image = nn.LayerNorm(dim // 16)
        self.layer_norm_post_query = nn.LayerNorm(dim // 16)

        self.two_way_blocks = nn.ModuleList([
            SimpleTwoWayBlock(rope=self.rope, dim=dim, num_heads=num_heads) for _ in range(num_two_way_blocks)
        ])

    def forward(self, query_tokens: torch.Tensor, image_tokens: torch.Tensor):
        for blk in self.two_way_blocks:
            query_tokens, image_tokens = blk(query_tokens, image_tokens)

        image_feature_map = rearrange(image_tokens, 'b h w c -> b c h w')
        up_query_token = self.hyper_in(query_tokens[:, 0, :])
        up_image_map = self.output_upscaling(image_feature_map)

        up_query_token = self.layer_norm_post_query(up_query_token)
        up_image_map = self.layer_norm_post_image(rearrange(up_image_map, 'b c h w -> b h w c'))

        masks = torch.einsum('bc,bhwc->bhw', up_query_token, up_image_map).unsqueeze(1)

        return masks.contiguous()


class SimpleMaskDecoderV4(SimpleMaskDecoderV2):
    """
    V2 decoder variant where every query token contributes to mask logits.

    SimpleMaskDecoderV2 only uses query_tokens[:, 0] after the two-way blocks.
    This variant runs the same hyper-network for all query tokens and sums the
    resulting per-token logits.
    """

    def forward(self, query_tokens: torch.Tensor, image_tokens: torch.Tensor):
        for blk in self.two_way_blocks:
            query_tokens, image_tokens = blk(query_tokens, image_tokens)

        image_feature_map = rearrange(image_tokens, 'b h w c -> b c h w')
        up_query_tokens = self.hyper_in(query_tokens)
        up_image_map = self.output_upscaling(image_feature_map)

        up_query_tokens = self.layer_norm_post_query(up_query_tokens)
        up_image_map = self.layer_norm_post_image(rearrange(up_image_map, 'b c h w -> b h w c'))

        masks = torch.einsum('blc,bhwc->blhw', up_query_tokens, up_image_map).sum(dim=1, keepdim=True)

        return masks.contiguous()


class MaskDecoderWithSAM(SimpleMaskDecoder):
    
    def __init__(self, sam_mask_decoder: SamMaskDecoder):
        super().__init__()
        self.sam_mask_decoder = sam_mask_decoder

    def forward(self, mask_tokens: torch.Tensor, image_tokens: torch.Tensor):
        """
        mask_tokens: (b, h, w, c)
        image_tokens: (b, h, w, c)

        return: mask
        """
        mask_tokens = self.apply_2d_rope(mask_tokens)
        image_tokens = self.apply_2d_rope(image_tokens)

        # flatten mask tokens
        mask_tokens = rearrange(mask_tokens, 'b h w c -> b (h w) c')
        
        # we don't use sam pe, so just create a zero tensor
        fake_image_pe = torch.zeros_like(image_tokens)
        mask_tokens, image_tokens = self.sam_mask_decoder.transformer(
            image_tokens, fake_image_pe, mask_tokens
        )
        image_features = rearrange(image_tokens, 'b h w c -> b c h w')
        mask_token_map = rearrange(mask_tokens, 'b h w c -> b c h w')

        upscaled_embedding = self.sam_mask_decoder.output_upscaling(image_features) # (b, C, H, W)
        upscaled_token_map = self.sam_mask_decoder.output_upscaling(mask_token_map) # (b, C, H, W)

        masks = torch.einsum('bchw,bchw->bhw', upscaled_embedding, upscaled_token_map)
        
        return masks
