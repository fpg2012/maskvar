import torch

def calc_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    pred: (B, 1, H, W)
    target: (B, 1, H, W)

    return: (B,)
    """
    pred = pred.squeeze(1) > 0
    target = target.squeeze(1) > 0
    intersection = (pred & target).sum(dim=(1, 2))
    union = (pred | target).sum(dim=(1, 2))
    iou = intersection / union
    return iou

