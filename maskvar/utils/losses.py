import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)
    return dims


class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(
        self,
        axis=-1,
        alpha=0.25,
        gamma=2,
        max_mult=-1,
        eps=1e-12,
        from_sigmoid=False,
        detach_delimeter=True,
        batch_axis=0,
        weight=None,
        size_average=True,
        ignore_label=-1,
    ):
        super().__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis
        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma
        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()
                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax.mean().item()

        one = torch.ones(1, dtype=torch.float, device=pt.device)
        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, one))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.float()
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = torch.sigmoid(pred)
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DICELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DICEFocalLoss(nn.Module):
    def __init__(self, smooth=1.0, alpha=0.25, gamma=2.0, weight_dice=2.0):
        super().__init__()
        self.dice_loss = DICELoss(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        return self.weight_dice * self.dice_loss(pred, target) + self.focal_loss(pred, target)


class DICEBCELoss(nn.Module):
    def __init__(self, smooth=1.0, weight_dice=2.0):
        super().__init__()
        self.dice_loss = DICELoss(smooth=smooth)
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
        return self.weight_dice * self.dice_loss(pred, target) + bce_loss


class DiceNFLoss(nn.Module):
    def __init__(self, smooth=1.0, weight_dice=1.0):
        super().__init__()
        self.dice_loss = DICELoss(smooth=smooth)
        self.focal_loss = NormalizedFocalLossSigmoid()
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        return self.weight_dice * self.dice_loss(pred, target) + self.focal_loss(pred, target)
