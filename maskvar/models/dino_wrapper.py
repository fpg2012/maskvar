import torch
from torch import nn
from einops import rearrange

class DinoV3Wrapper(nn.Module):

    def __init__(self, dino_v3_model):
        super().__init__()
        self.model = dino_v3_model
    
    def forward(self, images):
        b, c, h, w = images.shape
        output = self.model.forward_features(images)

        patch_tokens = rearrange(output[:, 5:], 'b (h w) c -> b c h w', h=h//16, w=w//16)
        return patch_tokens
