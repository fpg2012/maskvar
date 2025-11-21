import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from einops import rearrange, repeat

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
        self.attn = SelfAttention_v2(
            block_idx=0,
            embed_dim=dim,
            num_heads=4,
            proj_drop=0.1,
            attn_l2_norm=False,
        )
        # self.attn = DummyAttention(embed_dim=dim, num_heads=num_heads)
        self.attn = SimpleSelfAttention(embed_dim=dim, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        # self.layer_norm = nn.Identity()
    
    def forward(self, x: torch.Tensor, cond=None, prompt_cond=None, block_mask=None):
        """
        x: [B, L, C]
        """
        x = x + self.attn(self.layer_norm1(x), block_mask=block_mask)
        x = x + self.ffn(self.layer_norm2(x))
        return x

class SimpleAR(nn.Module):
    def __init__(self, dim=256, depth=2, vocab_size=4096, device='cpu', patch_num=[1, 4, 8, 16, 32], num_heads=4):
        super().__init__()
        self.patch_num = patch_num
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(self.vocab_size, dim)
        self.cls = nn.Linear(dim, self.vocab_size)

        self.device = device
        self.max_len = sum([x**2 for x in self.patch_num])
        self.block_mask = None
        self.pos_emb = nn.Embedding(self.max_len, dim)

        self.sos = nn.Parameter(torch.randn(dim))

    def init_block_mask(self, length=None):
        # causal mask
        def mask_mod(b, h, q_idx, k_idx) -> bool:
            return (q_idx >= k_idx)
        
        self.block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=self.max_len+1, KV_LEN=self.max_len+1, device=self.device)
    
    def preprocess(self, x: torch.Tensor):
        """
        x: (B, L)

        prepend <sos> token and add position embedding
        """
        B, L = x.shape
        pos_emb = repeat(self.pos_emb.weight, 'l c -> b l c', b=B)
        x_embed = self.embed(x)
        x_embed = x_embed + pos_emb

        sos = repeat(self.sos, 'c -> b 1 c', b=B)
        x = torch.cat([sos, x_embed], dim=1)
        return x
    
    def forward(self, x: torch.Tensor):
        """
        x: (B, L)
        """

        if self.block_mask is None:
            self.init_block_mask()
        
        x = self.preprocess(x)

        for block in self.blocks:
            x = block(x, cond=None, prompt_cond=None, block_mask=self.block_mask)
        logits = self.cls(x)
        return logits
    
    @torch.no_grad()
    def autogressive_infer(self, x):
        """
        x: (B, L)
        """
        B, L = x.shape
        x = self.preprocess(x)

        for block in self.blocks:
            x = block(x, cond=None, prompt_cond=None, block_mask=None)
        logits = self.cls(x)
        return logits
