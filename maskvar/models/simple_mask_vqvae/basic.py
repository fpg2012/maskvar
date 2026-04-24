import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange


from ..rope2d import RotaryPositionEmbedding2D

class SimpleCrossBlock(nn.Module):

    def __init__(self, rope: RotaryPositionEmbedding2D, dim, num_heads):
        super().__init__()
        self.rope = rope
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
    
    def forward(self, q, kv, pe_type='rope', image_pe=None):
        """
        q: (b, l, c)
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
        
        q = rearrange(q, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)
        k = rearrange(k, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b h w (nh c_head) -> b nh (h w) c_head', nh=self.num_heads, c_head=self.dim_head)

        # q = self.apply_2d_rope(q)
        if pe_type == 'rope':
            k = rearrange(k, 'b nh h w c_head -> (b nh) h w c_head')
            k = self.rope.apply_2d_rope(k)
            k = rearrange(k, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)
        elif pe_type == 'sam':
            k = k + image_pe
        else:
            raise ValueError(f"Invalid pe_type: {pe_type}")

        # out = flex_attention(q, k, v, attn_mask=None)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = q_input + self.out_proj(out)

        out = out + self.ffn(self.layernorm(out))
        
        return out


class SimpleCrossBlockReverse(SimpleCrossBlock):

    def __init__(self, rope: RotaryPositionEmbedding2D, dim, num_heads):
        super().__init__(rope, dim, num_heads)

    def forward(self, q, kv, pe_type='rope', image_pe=None):
        """
        q: (b, h, w, c)
        k: (b, l, c)
        """
        b, h, w, c = q.shape
        # print(f"[DEBUG] CrossBlock input - q: {q.shape}, kv: {kv.shape}, expected dim: {self.dim}")

        q_input = q

        q = self.linear_q(q)
        # print(f"[DEBUG] After linear_q - q: {q.shape}")
        kv = self.linear_kv(kv)
        # print(f"[DEBUG] After linear_kv - kv: {kv.shape}")
        k, v = kv.chunk(2, dim=-1)
        # print(f"[DEBUG] After chunk - k: {k.shape}, v: {v.shape}")
        
        q = rearrange(q, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        k = rearrange(k, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)

        if pe_type == 'rope':
            q = rearrange(q, 'b nh h w c_head -> (b nh) h w c_head')
            q = self.rope.apply_2d_rope(q)
            q = rearrange(q, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)
        elif pe_type == 'sam':
            q = q + image_pe
        else:
            raise ValueError(f"Invalid pe_type: {pe_type}")

        # out = flex_attention(q, k, v, attn_mask=None)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b nh (h w) c_head -> b h w (nh c_head)', h=h, w=w)
        out = q_input + self.out_proj(out)

        out = out + self.ffn(self.layernorm(out))
        
        return out

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class TwoWayBlock(nn.Module):

    def __init__(self, rope: RotaryPositionEmbedding2D, dim, num_heads):
        super().__init__()
        self.dim = dim

        self.block1 = SimpleCrossBlock(rope, dim, num_heads)
        self.reverse_block1 = SimpleCrossBlockReverse(rope, dim, num_heads)
        self.block2 = SimpleCrossBlock(rope, dim, num_heads)
        self.reverse_block2 = SimpleCrossBlockReverse(rope, dim, num_heads)

    def forward(self, query_tokens, mask_tokens, image_tokens, pe_type='rope', image_pe=None):
        query_tokens = self.block1(query_tokens, mask_tokens, pe_type=pe_type, image_pe=image_pe)
        image_tokens = self.reverse_block1(image_tokens, query_tokens, pe_type=pe_type, image_pe=image_pe)
        query_tokens = self.block2(query_tokens, image_tokens, pe_type=pe_type, image_pe=image_pe)
        mask_tokens = self.reverse_block2(mask_tokens, query_tokens, pe_type=pe_type, image_pe=image_pe)
        return query_tokens, mask_tokens, image_tokens


class SimpleTwoWayBlock(nn.Module):

    def __init__(self, rope: RotaryPositionEmbedding2D, dim, num_heads):
        super().__init__()
        self.dim = dim

        self.block1 = SimpleCrossBlock(rope, dim, num_heads)
        self.reverse_block1 = SimpleCrossBlockReverse(rope, dim, num_heads)

    def forward(self, query_tokens, image_tokens, pe_type='rope', image_pe=None):
        query_tokens = self.block1(query_tokens, image_tokens, pe_type=pe_type, image_pe=image_pe)
        image_tokens = self.reverse_block1(image_tokens, query_tokens, pe_type=pe_type, image_pe=image_pe)
        return query_tokens, image_tokens


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x