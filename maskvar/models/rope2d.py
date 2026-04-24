import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange


class RotaryPositionEmbedding2D(nn.Module):

    def __init__(self, h, w):
        super().__init__()
        self.h = h
        self.w = w

    def apply_2d_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        assert C % 2 == 0, "C must be even for 2D RoPE"

        scale_h, scale_w = self.h / H, self.w / W
        
        # 分成两半：前一半用 H 位置，后一半用 W 位置
        x_h, x_w = x.chunk(2, dim=-1)          # (B, H, W, C//2) each
        
        # 频率（推荐对视觉稍作调整）
        dim = C // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
        
        # H 轴旋转
        h_pos = 0.5*scale_h + torch.arange(H, device=x.device, dtype=torch.float32) * scale_h
        h_freqs = torch.outer(h_pos, inv_freq)          # (H, dim//2)
        h_cos = h_freqs.cos().unsqueeze(0).unsqueeze(2) # (1, H, 1, dim//2)
        h_sin = h_freqs.sin().unsqueeze(0).unsqueeze(2)
        
        x_h = self._apply_rotary(x_h, h_cos, h_sin)     # 见下面辅助函数
        
        # W 轴旋转
        w_pos = 0.5*scale_w + torch.arange(W, device=x.device, dtype=torch.float32) * scale_w
        w_freqs = torch.outer(w_pos, inv_freq)
        w_cos = w_freqs.cos().unsqueeze(0).unsqueeze(1) # (1, 1, W, dim//2)
        w_sin = w_freqs.sin().unsqueeze(0).unsqueeze(1)
        
        x_w = self._apply_rotary(x_w, w_cos, w_sin)
        
        return torch.cat([x_h, x_w], dim=-1)

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x: (B, H, W, D)   D = C//2
        # cos, sin: broadcastable to (..., D//2)
        x_ = rearrange(x, 'b h w (d2 c) -> b h w d2 c', d2=2)  # real, imag
        x_real, x_imag = x_[..., 0, :], x_[..., 1, :]
        
        # 广播 cos/sin
        cos = cos.unsqueeze(-1) if cos.dim() < x.dim() else cos
        sin = sin.unsqueeze(-1) if sin.dim() < x.dim() else sin
        
        rotated_real = cos * x_real - sin * x_imag
        rotated_imag = sin * x_real + cos * x_imag
        
        rotated = torch.stack([rotated_real, rotated_imag], dim=-2)
        return rearrange(rotated, 'b h w d2 c -> b h w (d2 c)')