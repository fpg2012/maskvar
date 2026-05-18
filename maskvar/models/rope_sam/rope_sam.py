import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from ..rope2d import RotaryPositionEmbedding2D
from ..simple_mask_vqvae.mask_decoder import SimpleMaskDecoderV2
from .click_encoder import RopeClickEncoder


class RopeSAM(nn.Module):
    """A non-generative click-conditioned segmentation model."""

    def __init__(
        self,
        image_encoder: nn.Module,
        dim: int = 384,
        h: int = 64,
        w: int = 64,
        num_heads: int = 4,
        max_clicks: int = 10,
        num_two_way_blocks: int = 2,
        device: str = "cuda",
    ):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.max_clicks = max_clicks
        self.image_encoder = image_encoder
        self.click_encoder = RopeClickEncoder(dim=dim, h=h, w=w)
        self.seg_token = nn.Parameter(torch.randn(1, 1, dim))
        self.prev_mask_encoder = nn.Sequential(
            nn.Conv2d(1, dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
        )
        self.rope = RotaryPositionEmbedding2D(h=h, w=w)
        self.mask_decoder = SimpleMaskDecoderV2(
            rope=self.rope,
            dim=dim,
            num_heads=num_heads,
            num_queries=max_clicks + 1,
            num_two_way_blocks=num_two_way_blocks,
        )
        self.device = device

    def get_device(self):
        return next(self.parameters()).device

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_encoder(image)
        if image_tokens.dim() != 4:
            raise ValueError(f"Expected image encoder output (B, C, H, W), got {tuple(image_tokens.shape)}")
        return rearrange(image_tokens, "b c h w -> b h w c")

    def encode_prev_mask(self, prev_mask_logits: torch.Tensor | None, spatial_shape: tuple[int, int]) -> torch.Tensor | None:
        if prev_mask_logits is None:
            return None
        if prev_mask_logits.dim() != 4 or prev_mask_logits.shape[1] != 1:
            raise ValueError(f"Expected prev_mask_logits shape (B, 1, H, W), got {tuple(prev_mask_logits.shape)}")
        prev_mask_logits = prev_mask_logits.float()
        if prev_mask_logits.shape[-2:] != spatial_shape:
            prev_mask_logits = F.interpolate(prev_mask_logits, size=spatial_shape, mode="bilinear", align_corners=False)
        prev_tokens = self.prev_mask_encoder(prev_mask_logits)
        return rearrange(prev_tokens, "b c h w -> b h w c")

    def encode_clicks(self, click_coords: torch.Tensor, click_labels: torch.Tensor) -> torch.Tensor:
        click_tokens = self.click_encoder(click_coords, click_labels)
        seg_token = self.seg_token.expand(click_tokens.shape[0], -1, -1)
        return torch.cat([seg_token, click_tokens], dim=1)

    def forward(
        self,
        image: torch.Tensor,
        click_coords: torch.Tensor,
        click_labels: torch.Tensor,
        prev_mask_logits: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W)
            click_coords: (B, N, 2), row/col coordinates in the token grid.
            click_labels: (B, N), 1 positive, 0 negative, -1 padding.
            prev_mask_logits: Optional previous-step mask logits used as dense prompt.
            output_size: Optional mask-logit output size.
        """
        image_tokens = self.encode_image(image)
        prev_mask_tokens = self.encode_prev_mask(prev_mask_logits, image_tokens.shape[1:3])
        if prev_mask_tokens is not None:
            image_tokens = image_tokens + prev_mask_tokens.to(image_tokens.dtype)
        query_tokens = self.encode_clicks(click_coords, click_labels)
        mask_logits = self.mask_decoder(query_tokens, image_tokens)

        if output_size is None:
            output_size = image.shape[-2:]
        if mask_logits.shape[-2:] != output_size:
            mask_logits = F.interpolate(mask_logits, size=output_size, mode="bilinear", align_corners=False)
        return mask_logits
