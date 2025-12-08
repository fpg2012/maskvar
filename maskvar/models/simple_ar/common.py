import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention
from einops import rearrange

from ..basic_var import SelfAttention_v2


class MLP(nn.Module):

    def __init__(self, embed_dim=256, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * mlp_ratio)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(embed_dim * mlp_ratio, embed_dim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class SimpleSelfAttention(nn.Module):

    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.proj = nn.Linear(embed_dim, embed_dim * 3)
    
    def forward(self, x, block_mask):
        B, L, C = x.shape
        qkv = self.proj(x) # B L 3*C
        qkv = rearrange(qkv, 'B L (QKV H c) -> QKV B H L c', QKV=3, H=self.num_heads, c=self.head_dim)
        q, k, v = qkv.unbind(dim=0) # B H L c
        y = flex_attention(q, k, v, block_mask=block_mask) # B H L c
        return rearrange(y, 'B H L c -> B L (H c)')

class TransformerBlock(nn.Module):
    
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.ffn = MLP(embed_dim=dim, mlp_ratio=4)
        # self.attn = SelfAttention_v2(
        #     block_idx=0,
        #     embed_dim=dim,
        #     num_heads=4,
        #     proj_drop=0.1,
        #     attn_l2_norm=False,
        # )
        self.attn = SimpleSelfAttention(embed_dim=dim, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, cond=None, prompt_cond=None, block_mask=None):
        """
        x: [B, L, C]
        """
        x = x + self.attn(self.layer_norm1(x), block_mask=block_mask)
        x = x + self.ffn(self.layer_norm2(x))
        return x