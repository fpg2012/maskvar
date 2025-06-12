from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

# import dist


# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2',]


class VectorQuantizer2(nn.Module):
    """向量量化器2 (Vector Quantizer 2)
    
    这是一个改进的向量量化器，用于将连续特征映射到离散的码本空间。
    主要特点：
    1. 支持多尺度特征量化
    2. 使用残差连接处理量化后的特征
    3. 支持不同的量化策略（归一化/非归一化）
    4. 支持码本使用统计
    
    Args:
        vocab_size: 码本大小，即离散潜在空间中的token数量
        Cvae: 潜在空间的通道数
        using_znorm: 是否在计算最近邻时进行归一化
        beta: commitment loss的权重，用于控制编码器输出与量化向量之间的差异
        default_qresi_counts: 默认的量化残差层数量
        v_patch_nums: 每个尺度的patch数量
        quant_resi: 残差连接的比例
        share_quant_resi: 在不同尺度间共享的φ层数量
    """
    def __init__(
        self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
        default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.vocab_size: int = vocab_size  # 码本大小
        self.Cvae: int = Cvae  # 潜在空间通道数
        self.using_znorm: bool = using_znorm  # 是否使用归一化
        self.v_patch_nums: Tuple[int] = v_patch_nums  # 每个尺度的patch数量
        
        self.quant_resi_ratio = quant_resi  # 残差连接比例
        # 根据share_quant_resi选择不同的残差连接策略
        if share_quant_resi == 0:   # 非共享：每个尺度使用独立的φ层
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # 完全共享：所有尺度使用同一个φ层
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # 部分共享：使用多个φ层在不同尺度间共享
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        # 记录每个尺度的码本使用情况
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        self.beta: float = beta  # commitment loss权重
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)  # 码本嵌入层
        
        # 渐进式训练相关（目前不支持）
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1
    
    def eini(self, eini):
        """初始化码本嵌入层
        
        Args:
            eini: 初始化参数
                > 0: 使用截断正态分布初始化
                < 0: 使用均匀分布初始化
        """
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)
    
    def extra_repr(self) -> str:
        """返回模型的额外信息字符串"""
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    # ===================== `forward` 仅用于VAE训练 =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """前向传播函数，用于训练过程
        
        Args:
            f_BChw: 输入特征，形状为(B, C, H, W)
            ret_usages: 是否返回码本使用情况
            
        Returns:
            f_hat: 量化后的特征
            usages: 码本使用情况（如果ret_usages=True）
            mean_vq_loss: 向量量化损失
        """
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        
        f_rest = f_no_grad.clone()  # 剩余特征
        f_hat = torch.zeros_like(f_rest)  # 量化后的特征
        
        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums): # 从小尺度到大尺度
                # 找到最近的码本向量
                if self.using_znorm:
                    # 使用归一化的特征计算最近邻
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    # 使用欧氏距离计算最近邻
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)
                
                # 统计码本使用情况
                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                if self.training:
                    if tdist.is_initialized(): handler = tdist.all_reduce(hit_V, async_op=True)
                
                # 计算损失
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                if SN - 1 > 0:
                    h_BChw = self.quant_resi[si/(SN-1)](h_BChw)  # 应用残差连接
                f_hat = f_hat + h_BChw  # 累加量化后的特征
                f_rest -= h_BChw  # 更新剩余特征
                
                # 更新码本使用统计
                if self.training and tdist.is_initialized():
                    handler.wait()
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                # mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta)
            
            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
        
        # 计算码本使用率
        world_size = tdist.get_world_size() if tdist.is_initialized() else 1
        margin = world_size * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        if ret_usages: usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)]
        # if ret_usages: usages = [(self.ema_vocab_hit_SV[si]).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)]
        else: usages = None
        return f_hat, usages, mean_vq_loss
    # ===================== `forward` 仅用于VAE训练 =====================
    
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        """将多尺度的特征转换为量化后的特征
        
        Args:
            ms_h_BChw: 多尺度的特征列表
            all_to_max_scale: 是否将所有尺度的特征转换到最大尺度
            last_one: 是否只返回最后一个尺度的结果
            
        Returns:
            量化后的特征或特征列表
        """
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            # 将所有尺度转换到最大尺度
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # 从小尺度到大尺度
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                if SN - 1 > 0:
                    h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            # 保持原始尺度（仅用于实验目的）
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # 从小尺度到大尺度
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                if SN - 1 > 0:
                    h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                    f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:
        """将特征转换为索引或量化后的特征
        
        Args:
            f_BChw: 输入特征
            to_fhat: 是否返回量化后的特征
            v_patch_nums: 每个尺度的patch数量
            
        Returns:
            索引列表或量化后的特征列表
        """
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # 从小尺度到大尺度
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # 从小尺度到大尺度
            if 0 <= self.prog_si < si: break    # 渐进式训练（目前不支持）
            # 找到最近的码本向量
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            if SN - 1 > 0:
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph*pw))
        
        return f_hat_or_idx_Bl
    
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """将多尺度的索引转换为VAR模型的输入
        
        Args:
            gt_ms_idx_Bl: 多尺度的索引列表
            
        Returns:
            VAR模型的输入特征
        """
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN-1):
            if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break   # 渐进式训练（目前不支持）
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            if SN - 1 > 0:
                f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            pn_next = self.v_patch_nums[si+1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # 将BlCs连接成BLC，应该是float32类型
    
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """获取自回归模型的下一个输入
        
        Args:
            si: 当前尺度索引
            SN: 总尺度数
            f_hat: 当前的量化特征
            h_BChw: 当前尺度的特征
            
        Returns:
            下一个尺度的输入特征
        """
        HW = self.v_patch_nums[-1]
        if si != SN-1:
            if SN - 1 > 0:
                h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))     # 上采样后应用卷积
                f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area')
        else:
            if SN - 1 > 0:
                h = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h)
            return f_hat, f_hat


class Phi(nn.Conv2d):
    """残差连接卷积层
    
    用于处理量化后的特征，通过残差连接保持原始信息。
    
    Args:
        embed_dim: 嵌入维度
        quant_resi: 残差连接比例
    """
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        """前向传播
        
        Args:
            h_BChw: 输入特征
            
        Returns:
            残差连接后的特征
        """
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    """共享的残差连接层
    
    所有尺度共享同一个残差连接层。
    """
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    """部分共享的残差连接层
    
    使用多个残差连接层在不同尺度间共享。
    
    Args:
        qresi_ls: 残差连接层列表
    """
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        """根据尺度位置获取对应的残差连接层
        
        Args:
            at_from_0_to_1: 尺度位置（0到1之间）
            
        Returns:
            对应的残差连接层
        """
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    """非共享的残差连接层
    
    每个尺度使用独立的残差连接层。
    
    Args:
        qresi: 残差连接层列表
    """
    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        """根据尺度位置获取对应的残差连接层
        
        Args:
            at_from_0_to_1: 尺度位置（0到1之间）
            
        Returns:
            对应的残差连接层
        """
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
