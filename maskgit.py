import torch
from torch import nn
import torch.nn.functional as F
from models.vqvae_single import VQVAE_Single
from transformer import TransformerSimple, TransformerCross

class MaskGIT(nn.Module):

    def __init__(self, vqvae: VQVAE_Single, image_size=256, dim=256, num_heads=8, num_blocks=6):
        super().__init__()
        self.image_size = image_size
        self.vqvae = vqvae
        self.dim = dim

        self.pos_embedding = nn.Embedding(image_size // 16 * image_size // 16, dim)
        self.blank_embedding = nn.Embedding(1, dim)
        self.transformer = TransformerCross(dim=dim, num_heads=num_heads, num_blocks=num_blocks, cross_layers=[0, 2, 4])
    
    def forward(self, x, image_embedding):
        """
        x: (B, L, C)
        image_embedding: (B, Lk, C)
        """
        x = self.transformer(x, image_embedding)
        return x