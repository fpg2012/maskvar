import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from einops import rearrange, repeat

from .common import TransformerBlock


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