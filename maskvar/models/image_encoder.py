import torch
from torch import nn
import torch.nn.functional as F
from typing import List


from .sam_image_encoder import ImageEncoderViT as SamImageEncoder

class ImageEncoder(nn.Module):
    def __init__(self, sam_embed_dim: int, embed_dim: int, sam_encoder: SamImageEncoder, freeze_sam_encoder: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.sam_encoder = sam_encoder
        # self.adapt_conv = nn.Conv2d(sam_embed_dim, embed_dim, kernel_size=2, stride=2)
        if freeze_sam_encoder:
            for param in self.sam_encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sam_image_embedding = self.sam_encoder(x) # (B, sam_embed_dim, H, W) H=W=64
        image_embedding = sam_image_embedding
        # image_embedding = self.adapt_conv(sam_image_embedding) # (B, embed_dim, H/2, W/2) H=W=64
        image_embedding = image_embedding.permute(0, 2, 3, 1) # (B, H/2, W/2, embed_dim)
        return image_embedding

class NeckFPN(nn.Module):

    def __init__(self, embed_dim, in_dim, in_size=(64, 64), real_size=(256, 256), patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32)):
        """
        !NOTE: in_size must be (64, 64), real_size must be (256, 256) now
        """
        super().__init__()
        self.patch_nums = patch_nums
        # self.adapt_convs = nn.ModuleList([
        #     # nn.Conv2d(in_dim, embed_dim, kernel_size=(in_size[0]//pn, in_size[1]//pn), stride=(in_size[0]//pn, in_size[1]//pn))
        #     nn.Conv2d(in_dim, embed_dim, kernel_size=(3, 3), padding=1)
        #     for pn in patch_nums
        # ])
        self.adapt_conv = nn.Conv2d(in_dim, embed_dim, kernel_size=(3, 3), padding=1)
        self.in_size = in_size
        self.real_size = real_size
    
    def forward(self, x: torch.Tensor, pe_grids: List) -> List[torch.Tensor]:
        """
        x: (B, H, W, C)

        return: list of multiscale features (B, L, embed_dim)
        """
        B, H, W, C = x.shape
        multiscale_feats = []
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)
        x1 = self.adapt_conv(x) + x
        
        for i, pn in enumerate(self.patch_nums):
            x1_interpolated = F.interpolate(x1, size=(pn, pn), mode='bilinear', align_corners=False)
            # x1_reverse = F.interpolate(x1_interpolated, size=(H, W), mode='bilinear', align_corners=False)
            
            x1_interpolated = x1_interpolated.permute(0, 2, 3, 1) + pe_grids[i].unsqueeze(0).to(x1_interpolated.device)
            x1_interpolated = x1_interpolated.view(B, -1, C).contiguous()
            multiscale_feats.append(x1_interpolated)
            
            # residual
            # x1_resi = x1 - x1_reverse
            # x1 = x1_resi
        
        # for i, conv in enumerate(self.adapt_convs):
        #     pn = self.patch_nums[i]
        #     x1 = conv(x)
        #     x1_interpolated = F.interpolate(x1, size=(pn, pn), mode='bilinear', align_corners=False)

        #     # print(f'x1_interpolated.shape: {x1_interpolated.shape}, pe_grid[{i}].shape: {pe_grids[i].shape}')

        #     x1_interpolated = x1_interpolated.permute(0, 2, 3, 1) + pe_grids[i].unsqueeze(0).to(x1_interpolated.device)
        #     x1_interpolated = x1_interpolated.view(B, -1, C)
        #     multiscale_feats.append(x1_interpolated)
        return multiscale_feats

class VarImageEncoder(nn.Module):
    def __init__(self, neck_fpn: NeckFPN):
        super().__init__()
        self.neck_fpn = neck_fpn
    
    def forward(self, sam_image_embedding: torch.Tensor, pe_grids: List) -> torch.Tensor:
        """
        sam_image_embedding: (B, C, H, W)
        pe_grids: list of (B, h, w, C)

        return: (B, L, embed_dim)
        """
        image_embedding = sam_image_embedding.permute(0, 2, 3, 1)
        multiscale_feats = self.neck_fpn(image_embedding, pe_grids=pe_grids) # list of (B, L, embed_dim)

        multiscale_feats = torch.cat(multiscale_feats, dim=1)
        return multiscale_feats