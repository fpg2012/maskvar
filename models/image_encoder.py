import torch
from torch import nn
from .sam_image_encoder import ImageEncoderViT as SamImageEncoder

class ImageEncoder(nn.Module):
    def __init__(self, sam_embed_dim: int, embed_dim: int, sam_encoder: SamImageEncoder, freeze_sam_encoder: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.sam_encoder = sam_encoder
        self.adapt_conv = nn.Conv2d(sam_embed_dim, embed_dim, kernel_size=2, stride=2)
        if freeze_sam_encoder:
            for param in self.sam_encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sam_image_embedding = self.sam_encoder(x) # (B, sam_embed_dim, H, W) H=W=64
        image_embedding = self.adapt_conv(sam_image_embedding) # (B, embed_dim, H/2, W/2) H=W=64
        image_embedding = image_embedding.permute(0, 2, 3, 1) # (B, H/2, W/2, embed_dim)
        return image_embedding
        