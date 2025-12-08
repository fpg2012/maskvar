import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from einops import rearrange, repeat

from ..basic_var import SelfAttention_v2
from ..vqvae_single import VQVAE_Single

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

class SimpleVAR(nn.Module):
    def __init__(self,
                 dim=256,
                 depth=2,
                 vocab_size=4096, 
                 device='cpu', 
                 patch_num=[1, 4, 8, 16, 32], 
                 num_heads=4,
                 vqvae_dim=256,
                 ):
        super().__init__()
        self.patch_num = patch_num
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.vocab_size = vocab_size
        self.dim = dim
        self.vqvae_dim = vqvae_dim
        # we don't need embed now, replaced with a linear
        # self.embed = nn.Embedding(self.vocab_size, dim)
        self.cls = nn.Linear(dim, self.vocab_size)

        self.device = device
        self.max_len = sum([x**2 for x in self.patch_num])
        self.block_mask = None
        
        pn_last = self.patch_num[-1]
        
        self.pos_embed = nn.Parameter(torch.randn(1, pn_last, pn_last, dim))
        self.level_embedding = nn.Embedding(len(patch_num), dim)
        
        self.sos = nn.Parameter(torch.randn(dim))

        self.level_map = []
        for i in range(len(self.patch_num)):
            self.level_map.extend([i] * (self.patch_num[i]**2))
        self.level_map_tensor = torch.tensor(self.level_map, device=self.device, dtype=torch.long)
        
        self.linear = nn.Linear(self.vqvae_dim, self.dim)
        # self.norm = nn.LayerNorm(self.dim)

    def calc_embed_to_add(self):
        # pos emb
        pos_embed_to_add = []

        pos_embed_1chw = rearrange(self.pos_embed, '1 h w c -> 1 c h w')
        for i, pn in enumerate(self.patch_num):
            if pn == self.patch_num[-1]:
                pos_embed_interpolated = pos_embed_1chw
            else:
                pos_embed_interpolated = F.interpolate(pos_embed_1chw, size=(pn, pn), mode='bilinear', align_corners=False)
            pos_embed_interpolated = rearrange(pos_embed_interpolated, '1 c h w -> 1 (h w) c')
            # print(f"pn={pn}, interpolated shape: {pos_embed_interpolated.shape}")
            pos_embed_to_add.append(pos_embed_interpolated)
        pos_embed_to_add = torch.cat(pos_embed_to_add, dim=1)

        # level embed
        level_embed_to_add = []
        for i in range(len(self.patch_num)):
            level_embed = repeat(self.level_embedding.weight[i], 'c -> l c', l=self.patch_num[i]**2)
            level_embed_to_add.append(level_embed)
        level_embed_to_add = torch.cat(level_embed_to_add, dim=0) # L c
        level_embed_to_add = repeat(level_embed_to_add, 'l c -> b l c', b=1) # 1 L c

        return pos_embed_to_add, level_embed_to_add

    def init_block_mask(self):
        # block mask
        def mask_mod(b, h, q_idx, k_idx):
            return self.level_map_tensor[q_idx] == self.level_map_tensor[k_idx]
        
        self.block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=self.max_len, KV_LEN=self.max_len, device=self.device)
    
    def preprocess(self, x: torch.Tensor):
        """
        1. project x to model dimension
        2. prepend SOS token
        3. add positional and level embeddings

        x: (B, L-1, C) output from quant.idxBl_to_var_input()
        """
        B, L_minus_1, C = x.shape
        L = L_minus_1 + 1
        pos_embed_to_add, level_embed_to_add = self.calc_embed_to_add()
        
        # map x using linear
        x = self.linear(x)
        # x = self.norm(x)

        sos = repeat(self.sos, 'c -> b 1 c', b=B)
        x = torch.cat([sos, x], dim=1)
        
        x = x + pos_embed_to_add + level_embed_to_add

        return x
    
    def forward(self, x: torch.Tensor, block_mask=None):
        """
        Applies the transformer blocks and outputs logits.
        When training, set block_mask to `self.block_mask`
        When inferencing, set block_mask to `None`

        x: (B, L, C)
        """
        for block in self.blocks:
            x = block(x, cond=None, prompt_cond=None, block_mask=block_mask)
        logits = self.cls(x)
        return logits
    
    def sample_with_top_k_(self, logits, top_k=50):
        """
        Sample from logits using top-k sampling
        
        Args:
            logits: (B, C) - batch of token logits for the next token
            top_k: keep only top k tokens for sampling
        
        Returns:
            (B, 1) - sampled token indices
        """
        # 确保top_k不超过词汇表大小
        vocab_size = logits.shape[-1]
        top_k = min(top_k, vocab_size)
        
        if top_k <= 0:
            # 如果top_k无效，直接采样
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            return next_token
        
        # 获取top-k的值和索引
        topk_values, topk_indices = torch.topk(logits, k=top_k, dim=-1)
        
        # 创建掩码：只保留top-k位置的logits
        # 方法1: 使用scatter直接构建新logits
        batch_size = logits.shape[0]
        
        # 创建全为-inf的logits
        filtered_logits = torch.full_like(logits, float('-inf'))
        
        # 使用einops重新排列索引以便scatter操作
        # 将topk_indices从(B, k)转换为适合scatter的形状
        batch_indices = torch.arange(batch_size, device=logits.device)
        batch_indices = rearrange(batch_indices, 'b -> b 1')
        batch_indices = batch_indices.expand(-1, top_k)
        
        # 使用scatter_填充top-k值
        filtered_logits[batch_indices, topk_indices] = topk_values
        
        # 从过滤后的分布中采样
        probs = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token


def simple_var_train_pass(idx, simple_var: SimpleVAR, vqvae: VQVAE_Single, epsilon=0.001):
    """
    Training pass for SimpleVAR model.

    1. Convert discrete codes to VQVAE input format
    2. Preprocess input for SimpleVAR
    3. Forward pass with block mask
    
    Args:
        idx: List of (B, l) - Input discrete codes from VQVAE
        simple_var: SimpleVAR model instance
        vqvae: VQVAE model instance

    Returns: logits (B, l, vocab_size)
    """
    assert simple_var.block_mask is not None, "Block mask must be initialized before training"
    with torch.no_grad():
        x = vqvae.quantize.idxBl_to_var_input(idx)
        # add noise
        if epsilon > 0:
            # Add random noise to the input
            noise = torch.randn_like(x) * epsilon
            x = x + noise

    x = simple_var.preprocess(x)
    logits = simple_var(x, block_mask=simple_var.block_mask)
    return logits

@torch.no_grad()
def simple_var_inference(simple_var: SimpleVAR, vqvae: VQVAE_Single, batch_size: int):
    """
    Autoregressive inference using top-k/top-p sampling
    
    Returns: sampled token sequences of shape (B, l)
    """
    B = batch_size
    H = W = simple_var.patch_num[-1]
    C = simple_var.vqvae_dim

    pos_embed_to_add, level_embed_to_add = simple_var.calc_embed_to_add() # (1, L, C), (1, L, C)

    id_seq = []
    current_token = repeat(
        rearrange(simple_var.sos, 'c -> 1 1 c'), 
        '1 1 c -> b 1 c',
        b=B
    ) # (B, 1, C)
    start_pos = 0

    f_hat = torch.zeros(B, C, H, W, dtype=torch.float, device=simple_var.device)

    print(f"simple_var.patch_num: {simple_var.patch_num}")

    for scale, pn in enumerate(simple_var.patch_num):
        print(f"scale, pn: {scale}, {pn}")
        print(f"current_token shape: {current_token.shape}")
        # add pos and level embeddings
        end_pos = start_pos + pn * pn

        pos_embed = pos_embed_to_add[:, start_pos:end_pos]
        level_embed = level_embed_to_add[:, start_pos:end_pos]
        current_token = current_token + pos_embed + level_embed

        # Forward pass to get logits. No need for block masking during inference
        logits = simple_var.forward(current_token, block_mask=None) # (B, pn*pn, vocab_size)
        # Sample next token
        logits_flat = rearrange(logits, 'b l v -> (b l) v')
        print(f"logits_flat shape: {logits_flat.shape}")
        next_tokens = simple_var.sample_with_top_k_(logits_flat, top_k=1)
        print(f"next_tokens shape: {next_tokens.shape}")
        next_tokens = rearrange(next_tokens, '(b l) 1 -> b l', b=B, l=pn*pn)
        print(f"next_tokens shape: {next_tokens.shape}")

        # Append prediction to sequence
        id_seq.append(next_tokens)

        if scale < len(simple_var.patch_num) - 1:
            # Convert prediction to feature for next step
            h = rearrange(vqvae.quantize.embedding(next_tokens), 'B (h w) C -> B C h w', h=pn, w=pn) # B, C, pn, pn
            h_up = F.interpolate(h, size=(H, W), mode='bicubic')
            
            # Update f_hat for next iteration
            t = scale / (len(simple_var.patch_num) - 1)
            f_hat.add_(vqvae.quantize.quant_resi[t](h_up))

            pn_next = simple_var.patch_num[scale + 1]

            f_hat_down = F.interpolate(f_hat, size=(pn_next, pn_next), mode='area')
            current_token = rearrange(f_hat_down, 'B C h w -> B (h w) C')
            # linear projection
            current_token = simple_var.linear(current_token)
            # current_token = simple_var.norm(current_token)
        
        start_pos = end_pos
        
    return id_seq
