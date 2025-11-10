import torch

def calc_iou(pred: torch.Tensor, target: torch.Tensor, return_nan_count=False) -> float:
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
    nan_count = torch.isnan(iou).sum()
    iou.nan_to_num_(nan=1.0)
    if return_nan_count:
        return iou, nan_count
    return iou