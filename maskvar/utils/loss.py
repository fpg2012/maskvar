import torch
from torch import nn
import torch.nn.functional as F

class NormalizedFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, eps=torch.finfo(torch.float).eps):
        """
        归一化的Focal Loss，用于解决梯度消失问题
        
        Args:
            alpha (float): 平衡正负样本的权重
            gamma (float): 聚焦参数，用于降低易分类样本的权重
            eps (float): 数值稳定性参数
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值，范围在[-1,1]之间
            target: 目标值，范围在[-1,1]之间
        """
        # 对预测值应用sigmoid，将范围转换到[0,1]
        pred = torch.sigmoid(pred)
        
        # 将目标值二值化（0为阈值）
        target = (target > 0).float()
        
        # 计算alpha权重
        alpha = torch.where(target > 0, self.alpha, (1 - self.alpha))
        
        # 计算pt（预测正确的概率）
        pt = 1.0 - (pred - target).abs()
        
        # 计算beta（难易样本权重）
        beta = (1.0 - pt) ** self.gamma
        
        # 计算归一化因子
        scale = target.numel() / (beta.sum() + self.eps)
        scale = scale.detach()  # 阻止梯度传播
        
        # 计算最终的loss
        beta = scale * beta
        loss = -alpha * beta * (pt + self.eps).log()
        
        return loss.mean()

class NormalizedFocalLoss2(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, eps=torch.finfo(torch.float).eps):
        """
        归一化的Focal Loss，用于解决梯度消失问题
        
        Args:
            alpha (float): 平衡正负样本的权重
            gamma (float): 聚焦参数，用于降低易分类样本的权重
            eps (float): 数值稳定性参数
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值，范围在[-1,1]之间
            target: 目标值，范围在[-1,1]之间
        """
        # 对预测值应用sigmoid，将范围转换到[0,1]
        pred = torch.sigmoid(pred)
        
        # 将目标值二值化（0为阈值）
        target = (target > 0).float()
        
        # 计算alpha权重
        alpha = torch.where(target > 0, self.alpha, (1 - self.alpha))
        
        # 计算pt（预测正确的概率）
        pt = 1.0 - (pred - target).abs()
        
        # 计算beta（难易样本权重）
        beta = (1.0 - pt) ** self.gamma
        
        # 计算归一化因子
        scale = target.numel() / (beta.sum() + self.eps)
        scale = scale.detach()  # 阻止梯度传播
        
        # 计算最终的loss
        beta = beta
        loss = -alpha * beta * (pt + self.eps).log()
        
        return loss.mean(), scale

class FocalLossGeneral(nn.Module):
    
    def __init__(self, alpha=0.5, gamma=2.0, eps=torch.finfo(torch.float).eps, label_smooth=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.label_smooth = label_smooth
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.reduction = reduction    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f'Invalid reduction: {self.reduction}')

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, eps=torch.finfo(torch.float).eps):
        """
        归一化的Focal Loss，用于解决梯度消失问题
        
        Args:
            alpha (float): 平衡正负样本的权重
            gamma (float): 聚焦参数，用于降低易分类样本的权重
            eps (float): 数值稳定性参数
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值，范围在[-1,1]之间
            target: 目标值，范围在[-1,1]之间
        """
        # 对预测值应用sigmoid，将范围转换到[0,1]
        pred = torch.sigmoid(pred)
        
        # 将目标值二值化（0为阈值）
        target = (target > 0).float()
        
        # 计算alpha权重
        alpha = torch.where(target > 0, self.alpha, (1 - self.alpha))
        
        # 计算pt（预测正确的概率）
        pt = 1.0 - (pred - target).abs()
        
        # 计算beta（难易样本权重）
        beta = (1.0 - pt) ** self.gamma
        
        # 计算最终的loss
        loss = -alpha * beta * (pt + self.eps).log()
        
        return loss.mean()