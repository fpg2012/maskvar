import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import flex_attention
import warnings

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xformers not found, using PyTorch scaled_dot_product_attention")

class Attention(nn.Module):
    """
    Self-attention with attention mask
    """
    
    def __init__(self, dim, num_heads, max_seq_len=512):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x, block_mask=None):
        B, L, C = x.shape
        # 生成QKV (B, L, 3, nH, head_dim)
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        
        q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)
        y = flex_attention(q, k, v, block_mask=block_mask)

        y = y.contiguous().view(B, L, C)
        # 调整维度并合并
        y = self.out_proj(y)
        return y

class CrossAttention(nn.Module):
    """
    Cross-attention with attention mask
    """
    
    def __init__(self, dim, num_heads):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, block_mask=None):
        B, Lq, C = query.shape
        _, Lk, _ = key.shape
        
        assert query.shape[-1] == key.shape[-1], "query and key must have the same feature dimension"
        
        # 生成 Q: (B, Lq, nH, head_dim)
        q = self.q(query).view(B, Lq, self.num_heads, self.head_dim)
        # 生成 K,V: (B, Lk, 2, nH, head_dim)
        kv = self.kv(key).view(B, Lk, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 1, 3, 4).unbind(0)
        
        y = flex_attention(q, k, v, block_mask=block_mask)
        
        # y: (B, Lq, nH, head_dim) -> (B, Lq, C)
        y = y.contiguous().view(B, Lq, C)
        y = self.out_proj(y)
        return y

class BlockSimple(nn.Module):
    
    def __init__(self, dim, num_heads, dropout=0.1, block_mask=None):
        super(BlockSimple, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn = Attention(dim=self.dim, num_heads=self.num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.SiLU(),
            nn.Linear(4*dim, dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.block_mask = block_mask
    
    def forward(self, x):
        x = self.attn(x, block_mask=self.block_mask) + x
        x = self.dropout1(x)
        x = self.layer_norm1(x)
        x = self.ffn(x) + x
        x = self.dropout2(x)
        x = self.layer_norm2(x)
        return x

class BlockCross(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.1, block_mask=None):
        super(BlockCross, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn = CrossAttention(dim=self.dim, num_heads=self.num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.SiLU(),
            nn.Linear(4*dim, dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.block_mask = block_mask

    def forward(self, x, key):
        x = self.attn(x, key, block_mask=self.block_mask) + x
        x = self.dropout1(x)
        x = self.layer_norm1(x)
        x = self.ffn(x) + x
        x = self.dropout2(x)
        x = self.layer_norm2(x)
        return x

class TransformerSimple(nn.Module):
    
    def __init__(self, dim=256, num_heads=8, num_blocks=2):
        super(TransformerSimple, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            block = BlockSimple(dim=dim, num_heads=self.num_heads)
            self.blocks.append(block)
    
    def forward(self, x):
        """
        x: (B, L, C)
        """
        for block in self.blocks:
            x = block(x)
        return x

class TransformerCross(nn.Module):

    def __init__(self, dim=256, num_heads=8, num_blocks=2, cross_layers=[0, 2, 4]):
        super(TransformerCross, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.cross_layers = cross_layers

        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            if i in self.cross_layers:
                block = BlockCross(dim=dim, num_heads=self.num_heads)
            else:
                block = BlockSimple(dim=dim, num_heads=self.num_heads)
            self.blocks.append(block)
    
    def forward(self, x, key):
        """
        x: (B, L, C)
        key: (B, Lk, C)
        """
        for i, block in enumerate(self.blocks):
            if i in self.cross_layers:
                x = block(x, key)
            else:
                x = block(x)
        return x

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # 测试 CrossAttention
    def test_cross_attention():
        dim = 64
        num_heads = 4
        batch_size = 2
        seq_len_q = 10
        seq_len_k = 20

        # 初始化 CrossAttention
        cross_attn = CrossAttention(dim=dim, num_heads=num_heads)

        # 生成随机输入
        query = torch.randn(batch_size, seq_len_q, dim)
        key = torch.randn(batch_size, seq_len_k, dim)

        # 前向传播
        output = cross_attn(query, key)

        # 检查输出维度
        assert output.shape == (batch_size, seq_len_q, dim), "Output shape mismatch"

        # 检查梯度
        output.sum().backward()
        for param in cross_attn.parameters():
            assert param.grad is not None, "Gradient not propagated"

        print("CrossAttention test passed!")

    # 测试 BlockCross
    def test_block_cross():
        dim = 64
        num_heads = 4
        batch_size = 2
        seq_len_q = 10
        seq_len_k = 20

        # 初始化 BlockCross
        block_cross = BlockCross(dim=dim, num_heads=num_heads)

        # 生成随机输入
        x = torch.randn(batch_size, seq_len_q, dim)
        key = torch.randn(batch_size, seq_len_k, dim)

        # 前向传播
        output = block_cross(x, key)

        # 检查输出维度
        assert output.shape == (batch_size, seq_len_q, dim), "Output shape mismatch"

        # 检查梯度
        output.sum().backward()
        for param in block_cross.parameters():
            assert param.grad is not None, "Gradient not propagated"

        print("BlockCross test passed!")

    def test_transformer_cross():
        dim = 64
        num_heads = 4
        num_blocks = 5
        cross_layers = [0, 2, 4]
        batch_size = 2
        seq_len_q = 10
        seq_len_k = 20

        transformer_cross = TransformerCross(dim=dim, num_heads=num_heads, num_blocks=num_blocks, cross_layers=cross_layers)
        x = torch.randn(batch_size, seq_len_q, dim)
        key = torch.randn(batch_size, seq_len_k, dim)

        # 前向传播
        output = transformer_cross(x, key)

        # 检查输出维度
        assert output.shape == (batch_size, seq_len_q, dim), "Output shape mismatch"

        # 检查梯度
        output.sum().backward()
        for param in transformer_cross.parameters():
            assert param.grad is not None, "Gradient not propagated"

        print("TransformerCross test passed!")

    # 运行测试
    test_cross_attention()
    test_block_cross()
    test_transformer_cross()