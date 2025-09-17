import torch
import torch.nn.functional as F

def resize_longest_side(image, target_length, mode='bilinear'):
    scale = target_length * 1.0 / max(image.shape[-2], image.shape[-1])
    newh, neww = image.shape[-2] * scale, image.shape[-1] * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)

    if mode == 'bilinear':
        return F.interpolate(
            image, (newh, neww), mode=mode, align_corners=False, antialias=True
        )
    else:
        return F.interpolate(
            image, (newh, neww), mode=mode,
        )

def divide_image(x_0: torch.Tensor, division):
    B, C, H, W = x_0.shape
    x = x_0.unfold(2, H//division, H//division).unfold(3, W//division, W//division)
    x = x.contiguous().view(B, C, -1, H//division, W//division)
    # B, C, 4, H//division, W//division => (4, B, C, H//division, W//division) => (4*B, C, H//division, W//division)
    x = x.permute(2, 0, 1, 3, 4).contiguous().view(-1, C, H//division, W//division)
    return x

def merge_image(x_0, division):
    B_, C, h, w = x_0.shape
    B = B_ // (division**2)
    H, W = h * division, w * division
    x = x_0.view(division**2, B, C, h, w).permute(1, 2, 0, 3, 4).contiguous() # (B, C, 4, h, w)
    x = x.view(B, C, 2, 2, h, w).permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W) # (B, C, H, W)
    return x