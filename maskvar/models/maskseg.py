import torch
from torch import nn
import torch.nn.functional as F

from .vqvae_single import VQVAE_Single
from .maskgit import MaskGIT
from .prompt_encoder import PromptEncoder
from .image_encoder import ImageEncoder

class MaskSeg(nn.Module):

    def __init__(self, maskgit: MaskGIT, prompt_encoder: PromptEncoder, image_encoder: ImageEncoder, 
                 freeze_prompt_encoder: bool = True):
        super().__init__()
        self.maskgit = maskgit
        self.prompt_encoder = prompt_encoder
        self.image_encoder = image_encoder
        # self.prompt_enc_adapter = nn.Linear(prompt_encoder.embed_dim, maskgit.dim)
        
        if freeze_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x, image, prompt):
        """
        x: (B, L)
        image: (B, H, W, 3)
        prompt: (B, L, 2)
        """
        image_embed = self.image_encoder(image) # (B, h_i, w_i, C)
        prompt_embed = self.prompt_encoder(prompt) # (B, L, C)
        logits = self.maskgit.forward(x, image_embed, prompt_embed)
        return logits