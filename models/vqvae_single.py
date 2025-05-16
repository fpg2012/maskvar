"""
单通道图像的向量量化变分自编码器 (Vector Quantized Variational AutoEncoder for Single Channel Images)

这个模块提供了一个专门用于处理单通道图像（如灰度图、深度图等）的VQVAE实现。
主要特点：
1. 输入输出都是单通道图像
2. 保持了原始VQVAE的所有功能
3. 针对单通道图像进行了优化
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .basic_vae import Decoder, Encoder
from .quant import VectorQuantizer2


class VQVAE_Single(nn.Module):
    """单通道图像的向量量化变分自编码器
    
    这是一个专门用于处理单通道图像的VQVAE实现。主要特点：
    1. 使用编码器将单通道图像压缩到潜在空间
    2. 使用向量量化器将连续特征映射到离散码本
    3. 使用解码器将量化后的特征重建为单通道图像
    
    Args:
        vocab_size: 码本大小，即离散潜在空间中的token数量
        z_channels: 潜在空间的通道数
        ch: 基础通道数，用于构建网络层
        dropout: dropout比率，用于防止过拟合
        beta: commitment loss的权重，用于控制编码器输出与量化向量之间的差异
        using_znorm: 是否在计算最近邻时进行归一化
        quant_conv_ks: 量化卷积层的核大小
        quant_resi: 残差连接的比例，0.5表示\phi(x) = 0.5*conv(x) + (1-0.5)*x
        share_quant_resi: 在不同尺度间共享的\phi层数量
        default_qresi_counts: 默认的量化残差层数量，0表示自动设置为v_patch_nums的长度
        v_patch_nums: 每个尺度的patch数量，h_{1到K} = w_{1到K} = v_patch_nums[k]
        test_mode: 是否处于测试模式
    """
    def __init__(
        self, 
        vocab_size=4096,        # 码本大小，即离散潜在空间中的token数量
        z_channels=32,          # 潜在空间的通道数
        ch=128,                 # 基础通道数，用于构建网络层
        dropout=0.0,            # dropout比率，用于防止过拟合
        beta=0.25,              # commitment loss的权重，用于控制编码器输出与量化向量之间的差异
        using_znorm=False,      # 是否在计算最近邻时进行归一化
        quant_conv_ks=3,        # 量化卷积层的核大小
        quant_resi=0.5,         # 残差连接的比例，0.5表示\phi(x) = 0.5*conv(x) + (1-0.5)*x
        share_quant_resi=4,     # 在不同尺度间共享的\phi层数量
        default_qresi_counts=0, # 默认的量化残差层数量，0表示自动设置为v_patch_nums的长度
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 每个尺度的patch数量，h_{1到K} = w_{1到K} = v_patch_nums[k]
        test_mode=True,         # 是否处于测试模式
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                using_sa=True, using_mid_sa=True,),
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels  # V: 码本大小, Cvae: 潜在空间通道数
        
        # 网络配置，基于CompVis的vq-f16配置，但使用单通道输入
        if ddconfig is None:
            ddconfig = dict(
                dropout=dropout, ch=ch, z_channels=z_channels,
                in_channels=1, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                using_sa=True, using_mid_sa=True,                           # 是否使用自注意力机制
            )
        else:
            ddconfig = dict(
                dropout=dropout, ch=ch, z_channels=z_channels,
                **ddconfig,                         # 是否使用自注意力机制
            )
        ddconfig.pop('double_z', None)  # 移除double_z参数，因为只有KL-VAE需要
        
        # 初始化编码器和解码器
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)  # 下采样率
        
        # 初始化向量量化器
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        
        # 量化前后的卷积层
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        
        # 如果是测试模式，将模型设置为评估状态并冻结参数
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    def forward(self, inp, ret_usages=False):   # -> rec_B1HW, idx_N, loss
        """前向传播函数，用于训练过程
        
        Args:
            inp: 输入单通道图像，形状为[B, 1, H, W]
            ret_usages: 是否返回码本使用情况
            
        Returns:
            rec_B1HW: 重建的单通道图像，形状为[B, 1, H, W]
            usages: 码本使用情况（如果ret_usages=True）
            vq_loss: 向量量化损失
        """
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        return self.decoder(self.post_quant_conv(f_hat)), usages, vq_loss
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        """将量化后的特征转换为单通道图像
        
        Args:
            f_hat: 量化后的特征张量
            
        Returns:
            重建的单通道图像，值域被裁剪到[-1, 1]
        """
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:
        """将单通道图像转换为多尺度的索引列表
        
        Args:
            inp_img_no_grad: 输入单通道图像（不需要梯度），形状为[B, 1, H, W]
            v_patch_nums: 每个尺度的patch数量
            
        Returns:
            List[Bl]: 多尺度的索引列表，每个元素是一个批次大小为B，长度为l的索引张量
        """
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)
    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        """将多尺度的索引列表转换为单通道图像
        
        Args:
            ms_idx_Bl: 多尺度的索引列表
            same_shape: 是否将所有尺度的特征转换到最大尺度
            last_one: 是否只返回最后一个尺度的重建结果
            
        Returns:
            重建的单通道图像或图像列表，形状为[B, 1, H, W]
        """
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        """将多尺度的特征转换为单通道图像
        
        Args:
            ms_h_BChw: 多尺度的特征列表
            all_to_max_scale: 是否将所有尺度的特征转换到最大尺度
            last_one: 是否只返回最后一个尺度的重建结果
            
        Returns:
            重建的单通道图像或图像列表，形状为[B, 1, H, W]
        """
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        """将单通道图像转换为重建图像
        
        Args:
            x: 输入单通道图像，形状为[B, 1, H, W]
            v_patch_nums: 每个尺度的patch数量
            last_one: 是否只返回最后一个尺度的重建结果
            
        Returns:
            重建的单通道图像或图像列表，形状为[B, 1, H, W]
        """
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        """加载模型状态字典
        
        Args:
            state_dict: 状态字典
            strict: 是否严格加载
            assign: 是否直接赋值而不是复制
            
        Returns:
            加载后的模型
        """
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign) 