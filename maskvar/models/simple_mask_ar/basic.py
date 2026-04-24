import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention
import torch.nn.functional as F
from einops import repeat, rearrange


class RotaryPositionEmbedding(nn.Module):

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

    def apply_2d_rope_with_coords(self, x: torch.Tensor, coords: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Apply 2D RoPE with explicit coordinates.

        Args:
            x: (B, L, C) - input tensor in sequence format
            coords: (L, 2) - (row, col) coordinates for each position, row-major order (0-indexed)
            h: actual height (number of rows)
            w: actual width (number of cols)
        """
        B, L, C = x.shape
        assert C % 2 == 0, "C must be even for 2D RoPE"
        assert coords.shape == (L, 2), f"coords shape {coords.shape} != ({L}, 2)"

        scale_h, scale_w = self.h / h, self.w / w

        # 分成两半：前一半用 H 位置，后一半用 W 位置
        x_h, x_w = x.chunk(2, dim=-1)  # (B, L, C//2) each

        # 频率
        dim = C // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))

        # 使用 patch 中心坐标: pos * scale + 0.5 * scale = (pos + 0.5) * scale
        rows = (coords[:, 0].float() + 0.5) * scale_h  # (L,)
        cols = (coords[:, 1].float() + 0.5) * scale_w  # (L,)

        # H 轴旋转 (rows)
        h_freqs = torch.outer(rows, inv_freq)  # (L, dim//2)
        h_cos = h_freqs.cos().unsqueeze(0)  # (1, L, dim//2)
        h_sin = h_freqs.sin().unsqueeze(0)

        x_h = self._apply_rotary_seq(x_h, h_cos, h_sin)

        # W 轴旋转 (cols)
        w_freqs = torch.outer(cols, inv_freq)  # (L, dim//2)
        w_cos = w_freqs.cos().unsqueeze(0)  # (1, L, dim//2)
        w_sin = w_freqs.sin().unsqueeze(0)

        x_w = self._apply_rotary_seq(x_w, w_cos, w_sin)

        return torch.cat([x_h, x_w], dim=-1)

    def _apply_rotary_seq(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Apply rotary to sequence format (B, L, D)"""
        # x: (B, L, D)   D = C//2
        # cos, sin: (1, L, D//2)
        x_ = rearrange(x, 'b l (d2 c) -> b l d2 c', d2=2)  # real, imag
        x_real, x_imag = x_[..., 0, :], x_[..., 1, :]

        rotated_real = cos * x_real - sin * x_imag
        rotated_imag = sin * x_real + cos * x_imag

        rotated = torch.stack([rotated_real, rotated_imag], dim=-2)
        return rearrange(rotated, 'b l d2 c -> b l (d2 c)')

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


class SimpleCrossBlock(nn.Module):

    def __init__(self, rope: RotaryPositionEmbedding, dim, num_heads):
        super().__init__()
        self.rope: RotaryPositionEmbedding = rope
        self.dim = dim
        self.linear_q = nn.Linear(dim, dim)
        self.linear_kv = nn.Linear(dim, dim*2)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.layernorm = nn.LayerNorm(dim)

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim_head = dim // num_heads

    
    def forward(self, q, kv):
        """
        q: (b, h, w, c)
        k: (b, h, w, c)
        """
        b, h, w, c = kv.shape
        # print(f"[DEBUG] CrossBlock input - q: {q.shape}, kv: {kv.shape}, expected dim: {self.dim}")

        q_input = q

        q = self.linear_q(q)
        # print(f"[DEBUG] After linear_q - q: {q.shape}")
        kv = self.linear_kv(kv)
        # print(f"[DEBUG] After linear_kv - kv: {kv.shape}")
        k, v = kv.chunk(2, dim=-1)
        # print(f"[DEBUG] After chunk - k: {k.shape}, v: {v.shape}")
        
        q = rearrange(q, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        k = rearrange(k, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b h w (nh c_head) -> b nh (h w) c_head', nh=self.num_heads, c_head=self.dim_head)

        # apply RoPE
        k = rearrange(k, 'b nh h w c_head -> (b nh) h w c_head')
        q = rearrange(q, 'b nh h w c_head -> (b nh) h w c_head')

        k = self.rope.apply_2d_rope(k)
        q = self.rope.apply_2d_rope(q)

        k = rearrange(k, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)
        q = rearrange(q, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)

        # out = flex_attention(q, k, v, attn_mask=None)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = q_input + self.out_proj(out)

        out = out + self.ffn(self.layernorm(out))

        return out

    def forward_seq(self, q_seq, kv, q_coords):
        """
        Forward for inference with sequence format.

        Args:
            q_seq: (b, l, c) - query tokens in sequence format
            kv: (b, h, w, c) - key/value tokens in spatial format
            q_coords: (l, 2) - (row, col) coordinates for each query position
        """
        b, l, c = q_seq.shape
        _, h, w, _ = kv.shape

        q_input = q_seq

        q = self.linear_q(q_seq)
        kv = self.linear_kv(kv)
        k, v = kv.chunk(2, dim=-1)

        # Reshape k, v for multi-head attention
        k = rearrange(k, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b h w (nh c_head) -> b nh (h w) c_head', nh=self.num_heads, c_head=self.dim_head)

        # Reshape q for multi-head
        q = rearrange(q, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)

        # Apply RoPE with coordinates for q (sequence)
        q = rearrange(q, 'b nh l c_head -> (b nh) l c_head')
        q = self.rope.apply_2d_rope_with_coords(q, q_coords, h, w)
        q = rearrange(q, '(b nh) l c_head -> b nh l c_head', b=b, nh=self.num_heads)

        # Apply RoPE for k (spatial)
        k = rearrange(k, 'b nh h w c_head -> (b nh) h w c_head')
        k = self.rope.apply_2d_rope(k)
        k = rearrange(k, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)

        # Attention
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = q_input + self.out_proj(out)

        out = out + self.ffn(self.layernorm(out))

        return out


class SimpleSelfBlock(nn.Module):

    def __init__(self, rope: RotaryPositionEmbedding, dim, num_heads):
        super().__init__()
        self.rope: RotaryPositionEmbedding = rope
        self.dim = dim
        self.linear_qkv = nn.Linear(dim, dim*3)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.layernorm = nn.LayerNorm(dim)

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim_head = dim // num_heads

    def forward(self, x, attn_mask=None):
        """
        x: (b, h, w, c)
        """
        b, h, w, c = x.shape

        qkv_input = x

        qkv = self.linear_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # split heads
        q = rearrange(q, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        k = rearrange(k, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b h w (nh c_head) -> b nh (h w) c_head', nh=self.num_heads, c_head=self.dim_head)

        # apply RoPE
        k = rearrange(k, 'b nh h w c_head -> (b nh) h w c_head')
        q = rearrange(q, 'b nh h w c_head -> (b nh) h w c_head')

        k = self.rope.apply_2d_rope(k)
        q = self.rope.apply_2d_rope(q)

        k = rearrange(k, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)
        q = rearrange(q, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)

        out = flex_attention(q, k, v, block_mask=attn_mask)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = qkv_input + self.out_proj(out)

        out = out + self.ffn(self.layernorm(out))

        return out

    def forward_seq(self, x_seq, coords, block_mask=None):
        """
        Forward for inference with sequence format.

        Args:
            x_seq: (b, l, c) - input tokens in sequence format
            coords: (l, 2) - (row, col) coordinates for each position
            block_mask: optional block mask for flex_attention
        """
        b, l, c = x_seq.shape
        # Infer h, w from coords
        h = coords[:, 0].max().item() + 1
        w = coords[:, 1].max().item() + 1

        qkv_input = x_seq

        qkv = self.linear_qkv(x_seq)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split heads
        q = rearrange(q, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)
        k = rearrange(k, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)

        # Apply RoPE with coordinates
        q = rearrange(q, 'b nh l c_head -> (b nh) l c_head')
        k = rearrange(k, 'b nh l c_head -> (b nh) l c_head')

        q = self.rope.apply_2d_rope_with_coords(q, coords, h, w)
        k = self.rope.apply_2d_rope_with_coords(k, coords, h, w)

        q = rearrange(q, '(b nh) l c_head -> b nh l c_head', b=b, nh=self.num_heads)
        k = rearrange(k, '(b nh) l c_head -> b nh l c_head', b=b, nh=self.num_heads)
        v = rearrange(v, 'b nh l c_head -> b nh l c_head')

        # Attention
        out = flex_attention(q, k, v, block_mask=block_mask)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = qkv_input + self.out_proj(out)

        out = out + self.ffn(self.layernorm(out))

        return out