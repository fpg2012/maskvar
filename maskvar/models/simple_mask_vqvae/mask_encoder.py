import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class MaskEncoderLite(nn.Module):

    def __init__(self, dim, patch_size=16, image_size=1024):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.image_size = image_size

        assert image_size % patch_size == 0

        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, mask):
        """
        mask: (B, 1, H, W)
        """
        # reshuffle
        mask_tokens = rearrange(self.patch_embed(mask), 'b c h w -> b h w c')
        mask_tokens = mask_tokens + self.mlp(self.layer_norm(mask_tokens))
        
        return rearrange(mask_tokens, 'b h w c -> b c h w')