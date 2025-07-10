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