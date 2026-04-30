import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


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

    def apply_2d_rope_with_batched_coords(self, x: torch.Tensor, coords: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Apply 2D RoPE with per-sample coordinates.

        Args:
            x: (B, L, C)
            coords: (B, L, 2), row/col coordinates. Float coordinates are allowed.
            h: full height used for RoPE scaling
            w: full width used for RoPE scaling
        """
        B, L, C = x.shape
        assert C % 2 == 0, "C must be even for 2D RoPE"
        assert coords.shape == (B, L, 2), f"coords shape {coords.shape} != ({B}, {L}, 2)"

        scale_h, scale_w = self.h / h, self.w / w

        x_h, x_w = x.chunk(2, dim=-1)
        dim = C // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))

        rows = (coords[..., 0].float() + 0.5) * scale_h
        cols = (coords[..., 1].float() + 0.5) * scale_w

        h_freqs = rows.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        w_freqs = cols.unsqueeze(-1) * inv_freq.view(1, 1, -1)

        x_h = self._apply_rotary_seq(x_h, h_freqs.cos(), h_freqs.sin())
        x_w = self._apply_rotary_seq(x_w, w_freqs.cos(), w_freqs.sin())

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
        q_input = q

        q = self.linear_q(q)
        kv = self.linear_kv(kv)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        k = rearrange(k, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b h w (nh c_head) -> b nh (h w) c_head', nh=self.num_heads, c_head=self.dim_head)

        k = rearrange(k, 'b nh h w c_head -> (b nh) h w c_head')
        q = rearrange(q, 'b nh h w c_head -> (b nh) h w c_head')

        k = self.rope.apply_2d_rope(k)
        q = self.rope.apply_2d_rope(q)
        k = k.to(v.dtype)
        q = q.to(v.dtype)

        k = rearrange(k, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)
        q = rearrange(q, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = rearrange(out, 'b (h w) c -> b h w c', h=h, w=w)
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
        q = q.to(v.dtype)
        q = rearrange(q, '(b nh) l c_head -> b nh l c_head', b=b, nh=self.num_heads)

        # Apply RoPE for k (spatial)
        k = rearrange(k, 'b nh h w c_head -> (b nh) h w c_head')
        k = self.rope.apply_2d_rope(k)
        k = k.to(v.dtype)
        k = rearrange(k, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)

        # Attention
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = q_input + self.out_proj(out)

        out = out + self.ffn(self.layernorm(out))

        return out

    def precompute_kv(self, kv):
        """
        Precompute static image K/V for autoregressive inference.

        Args:
            kv: (b, h, w, c) - image tokens in spatial format
        Returns:
            cached_k, cached_v, h, w
        """
        b, h, w, _ = kv.shape
        kv = self.linear_kv(kv)
        k, v = kv.chunk(2, dim=-1)

        k = rearrange(k, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b h w (nh c_head) -> b nh (h w) c_head', nh=self.num_heads, c_head=self.dim_head)

        k = rearrange(k, 'b nh h w c_head -> (b nh) h w c_head')
        k = self.rope.apply_2d_rope(k)
        k = k.to(v.dtype)
        k = rearrange(k, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)

        return k, v, h, w

    def forward_step(self, q_step, cached_k, cached_v, coord, h, w):
        """
        Incremental cross-attention for a single autoregressive step.

        Args:
            q_step: (b, 1, c)
            cached_k: (b, nh, hw, c_head)
            cached_v: (b, nh, hw, c_head)
            coord: (1, 2)
            h, w: full spatial shape for RoPE scaling
        """
        b, _, _ = q_step.shape
        q_input = q_step

        q = self.linear_q(q_step)
        q = rearrange(q, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)
        q = rearrange(q, 'b nh l c_head -> (b nh) l c_head')
        q = self.rope.apply_2d_rope_with_coords(q, coord, h, w)
        q = q.to(cached_v.dtype)
        q = rearrange(q, '(b nh) l c_head -> b nh l c_head', b=b, nh=self.num_heads)

        out = F.scaled_dot_product_attention(q, cached_k, cached_v)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = q_input + self.out_proj(out)
        out = out + self.ffn(self.layernorm(out))

        return out


class SimpleClickCrossBlock(nn.Module):

    def __init__(self, rope: RotaryPositionEmbedding, dim, num_heads):
        super().__init__()
        self.rope: RotaryPositionEmbedding = rope
        self.dim = dim
        self.linear_q = nn.Linear(dim, dim)
        self.linear_kv = nn.Linear(dim, dim * 2)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.layernorm = nn.LayerNorm(dim)

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim_head = dim // num_heads

    def _apply_click_rope(self, k, click_coords, click_labels, h, w):
        b = click_coords.shape[0]
        k_seq = rearrange(k, 'b nh n c_head -> (b nh) n c_head')
        batched_coords = click_coords[:, None].expand(-1, self.num_heads, -1, -1)
        batched_coords = rearrange(batched_coords, 'b nh n xy -> (b nh) n xy')
        k_rope = self.rope.apply_2d_rope_with_batched_coords(k_seq, batched_coords, h, w)

        valid = (click_labels[:, None, :, None] == 1).expand(-1, self.num_heads, -1, self.dim_head)
        valid = rearrange(valid, 'b nh n c_head -> (b nh) n c_head')
        k_seq = torch.where(valid, k_rope, k_seq)
        return rearrange(k_seq, '(b nh) n c_head -> b nh n c_head', b=b, nh=self.num_heads)

    def forward(self, q, click_tokens, click_coords, click_labels):
        """
        q: (b, h, w, c)
        click_tokens: (b, n, c)
        click_coords: (b, n, 2), row/col coordinates in token-grid units
        click_labels: (b, n), 1 for positive clicks and -1 for padding
        """
        b, h, w, _ = q.shape
        q_input = q

        q = self.linear_q(q)
        kv = self.linear_kv(click_tokens)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        k = rearrange(k, 'b n (nh c_head) -> b nh n c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b n (nh c_head) -> b nh n c_head', nh=self.num_heads, c_head=self.dim_head)

        q = rearrange(q, 'b nh h w c_head -> (b nh) h w c_head')
        q = self.rope.apply_2d_rope(q)
        q = q.to(v.dtype)
        q = rearrange(q, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)

        k = self._apply_click_rope(k, click_coords, click_labels, h, w).to(v.dtype)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = rearrange(out, 'b (h w) c -> b h w c', h=h, w=w)
        out = q_input + self.out_proj(out)
        out = out + self.ffn(self.layernorm(out))

        return out

    def forward_seq(self, q_seq, click_tokens, click_coords, click_labels, q_coords, spatial_shape):
        b, _, _ = q_seq.shape
        h, w = spatial_shape
        q_input = q_seq

        q = self.linear_q(q_seq)
        kv = self.linear_kv(click_tokens)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)
        k = rearrange(k, 'b n (nh c_head) -> b nh n c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b n (nh c_head) -> b nh n c_head', nh=self.num_heads, c_head=self.dim_head)

        q = rearrange(q, 'b nh l c_head -> (b nh) l c_head')
        q = self.rope.apply_2d_rope_with_coords(q, q_coords, h, w)
        q = q.to(v.dtype)
        q = rearrange(q, '(b nh) l c_head -> b nh l c_head', b=b, nh=self.num_heads)

        k = self._apply_click_rope(k, click_coords, click_labels, h, w).to(v.dtype)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = q_input + self.out_proj(out)
        out = out + self.ffn(self.layernorm(out))

        return out

    def precompute_kv(self, click_tokens, click_coords, click_labels, h, w):
        kv = self.linear_kv(click_tokens)
        k, v = kv.chunk(2, dim=-1)
        k = rearrange(k, 'b n (nh c_head) -> b nh n c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b n (nh c_head) -> b nh n c_head', nh=self.num_heads, c_head=self.dim_head)
        k = self._apply_click_rope(k, click_coords, click_labels, h, w).to(v.dtype)
        return k, v, h, w

    def forward_step(self, q_step, cached_k, cached_v, coord, h, w):
        b, _, _ = q_step.shape
        q_input = q_step

        q = self.linear_q(q_step)
        q = rearrange(q, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)
        q = rearrange(q, 'b nh l c_head -> (b nh) l c_head')
        q = self.rope.apply_2d_rope_with_coords(q, coord, h, w)
        q = q.to(cached_v.dtype)
        q = rearrange(q, '(b nh) l c_head -> b nh l c_head', b=b, nh=self.num_heads)

        out = F.scaled_dot_product_attention(q, cached_k, cached_v)
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
        k = k.to(v.dtype)
        q = q.to(v.dtype)

        k = rearrange(k, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)
        q = rearrange(q, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = rearrange(out, 'b (h w) c -> b h w c', h=h, w=w)
        out = qkv_input + self.out_proj(out)

        out = out + self.ffn(self.layernorm(out))

        return out

    def forward_seq(self, x_seq, coords, block_mask=None, spatial_shape=None):
        """
        Forward for inference with sequence format.

        Args:
            x_seq: (b, l, c) - input tokens in sequence format
            coords: (l, 2) - (row, col) coordinates for each position
            block_mask: optional block mask for flex_attention
            spatial_shape: optional full spatial shape (h, w) used for RoPE scaling
        """
        b, l, c = x_seq.shape
        if spatial_shape is None:
            h = coords[:, 0].max().item() + 1
            w = coords[:, 1].max().item() + 1
        else:
            h, w = spatial_shape

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
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        q = rearrange(q, '(b nh) l c_head -> b nh l c_head', b=b, nh=self.num_heads)
        k = rearrange(k, '(b nh) l c_head -> b nh l c_head', b=b, nh=self.num_heads)
        v = rearrange(v, 'b nh l c_head -> b nh l c_head')

        # Attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = qkv_input + self.out_proj(out)

        out = out + self.ffn(self.layernorm(out))

        return out

    def forward_step(self, x_step, coord, cache_k=None, cache_v=None, spatial_shape=None):
        """
        Incremental causal self-attention for a single autoregressive step.

        Args:
            x_step: (b, 1, c)
            coord: (1, 2)
            cache_k: optional cached keys, (b, nh, t, c_head)
            cache_v: optional cached values, (b, nh, t, c_head)
            spatial_shape: full spatial shape (h, w) for RoPE scaling
        """
        b, _, _ = x_step.shape
        if spatial_shape is None:
            h = coord[:, 0].max().item() + 1
            w = coord[:, 1].max().item() + 1
        else:
            h, w = spatial_shape

        qkv_input = x_step
        qkv = self.linear_qkv(x_step)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)
        k = rearrange(k, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)

        q = rearrange(q, 'b nh l c_head -> (b nh) l c_head')
        k = rearrange(k, 'b nh l c_head -> (b nh) l c_head')
        q = self.rope.apply_2d_rope_with_coords(q, coord, h, w)
        k = self.rope.apply_2d_rope_with_coords(k, coord, h, w)
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        q = rearrange(q, '(b nh) l c_head -> b nh l c_head', b=b, nh=self.num_heads)
        k = rearrange(k, '(b nh) l c_head -> b nh l c_head', b=b, nh=self.num_heads)

        if cache_k is not None:
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = qkv_input + self.out_proj(out)
        out = out + self.ffn(self.layernorm(out))

        return out, k, v
