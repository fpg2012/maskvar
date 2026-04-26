import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


class QueryTokenCrossBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.linear_q = nn.Linear(dim, dim)
        self.linear_kv = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.layernorm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.num_heads = num_heads
        self.dim_head = dim // num_heads

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        q_input = q
        q = self.linear_q(q)
        kv = self.linear_kv(kv)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, 'b l (nh dh) -> b nh l dh', nh=self.num_heads, dh=self.dim_head)
        k = rearrange(k, 'b l (nh dh) -> b nh l dh', nh=self.num_heads, dh=self.dim_head)
        v = rearrange(v, 'b l (nh dh) -> b nh l dh', nh=self.num_heads, dh=self.dim_head)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b nh l dh -> b l (nh dh)')
        out = q_input + self.out_proj(out)
        return out + self.ffn(self.layernorm(out))

    def precompute_kv(self, kv: torch.Tensor):
        kv = self.linear_kv(kv)
        k, v = kv.chunk(2, dim=-1)
        k = rearrange(k, 'b l (nh dh) -> b nh l dh', nh=self.num_heads, dh=self.dim_head)
        v = rearrange(v, 'b l (nh dh) -> b nh l dh', nh=self.num_heads, dh=self.dim_head)
        return k, v

    def forward_step(self, q_step: torch.Tensor, cached_k: torch.Tensor, cached_v: torch.Tensor):
        q_input = q_step
        q = self.linear_q(q_step)
        q = rearrange(q, 'b l (nh dh) -> b nh l dh', nh=self.num_heads, dh=self.dim_head)
        out = F.scaled_dot_product_attention(q, cached_k, cached_v)
        out = rearrange(out, 'b nh l dh -> b l (nh dh)')
        out = q_input + self.out_proj(out)
        return out + self.ffn(self.layernorm(out))


class QueryTokenSelfBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.linear_qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.layernorm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.num_heads = num_heads
        self.dim_head = dim // num_heads

    def forward(self, x: torch.Tensor):
        x_input = x
        qkv = self.linear_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, 'b l (nh dh) -> b nh l dh', nh=self.num_heads, dh=self.dim_head)
        k = rearrange(k, 'b l (nh dh) -> b nh l dh', nh=self.num_heads, dh=self.dim_head)
        v = rearrange(v, 'b l (nh dh) -> b nh l dh', nh=self.num_heads, dh=self.dim_head)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = rearrange(out, 'b nh l dh -> b l (nh dh)')
        out = x_input + self.out_proj(out)
        return out + self.ffn(self.layernorm(out))

    def forward_step(self, x_step: torch.Tensor, cache_k=None, cache_v=None):
        x_input = x_step
        qkv = self.linear_qkv(x_step)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, 'b l (nh dh) -> b nh l dh', nh=self.num_heads, dh=self.dim_head)
        k = rearrange(k, 'b l (nh dh) -> b nh l dh', nh=self.num_heads, dh=self.dim_head)
        v = rearrange(v, 'b l (nh dh) -> b nh l dh', nh=self.num_heads, dh=self.dim_head)

        if cache_k is not None:
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = rearrange(out, 'b nh l dh -> b l (nh dh)')
        out = x_input + self.out_proj(out)
        out = out + self.ffn(self.layernorm(out))
        return out, k, v


class SimpleQueryTokenARBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.cross_block = QueryTokenCrossBlock(dim=dim, num_heads=num_heads)
        self.self_block = QueryTokenSelfBlock(dim=dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor, image_tokens: torch.Tensor):
        x = self.cross_block(x, image_tokens)
        return self.self_block(x)

    def precompute_cross_kv(self, image_tokens: torch.Tensor):
        return self.cross_block.precompute_kv(image_tokens)

    def forward_step(self, x_step: torch.Tensor, cross_cache, self_cache):
        cached_k, cached_v = cross_cache
        cache_k, cache_v = (self_cache if self_cache is not None else (None, None))
        x_step = self.cross_block.forward_step(x_step, cached_k, cached_v)
        x_step, new_k, new_v = self.self_block.forward_step(x_step, cache_k=cache_k, cache_v=cache_v)
        return x_step, (new_k, new_v)


class SimpleQueryTokenAR(nn.Module):

    def __init__(self, dim=256, depth=2, vocab_size=4096, num_queries=8, image_token_len=64 * 64, num_heads=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_queries = num_queries
        self.image_token_len = image_token_len
        self.num_heads = num_heads

        self.embed = nn.Embedding(vocab_size, dim)
        self.cls = nn.Linear(dim, vocab_size)
        self.sos = nn.Parameter(torch.randn(dim))

        self.blocks = nn.ModuleList([
            SimpleQueryTokenARBlock(dim=dim, num_heads=num_heads) for _ in range(depth)
        ])

    def preprocess(self, x: torch.Tensor):
        B, L = x.shape
        if L != self.num_queries:
            raise ValueError(f"Expected query length {self.num_queries}, got {L}")
        x = self.embed(x)
        sos = self.sos.view(1, 1, self.dim).expand(B, 1, -1)
        return torch.cat([sos, x[:, :-1, :]], dim=1)

    def forward(self, x: torch.Tensor, image_tokens: torch.Tensor):
        x = self.preprocess(x)
        for block in self.blocks:
            x = block(x, image_tokens)
        return self.cls(x)

    def encode_mask_to_token_ids(self, vqvae_model, mask_normalized: torch.Tensor, image: torch.Tensor):
        with torch.no_grad():
            mask_tokens = vqvae_model.mask_encoder(mask_normalized)
            image_tokens = vqvae_model.image_encoder(image)

            mask_tokens = rearrange(mask_tokens, 'b c h w -> b h w c')
            image_tokens = rearrange(image_tokens, 'b c h w -> b (h w) c')
            query_tokens = vqvae_model.mask_feature_compactor(mask_tokens)
            token_ids = vqvae_model.quant.x_to_idx(query_tokens.float())

        return token_ids, image_tokens

    @torch.no_grad()
    def decode_token_ids_to_mask_logits(self, vqvae_model, token_ids, image_tokens, output_size):
        query_tokens = vqvae_model.quant.idx_to_x(token_ids)
        image_tokens = rearrange(image_tokens, 'b (h w) c -> b h w c', h=vqvae_model.rope.h, w=vqvae_model.rope.w)
        mask_logits = vqvae_model.mask_decoder(query_tokens, image_tokens)
        if mask_logits.shape[-2:] != output_size:
            mask_logits = nn.functional.interpolate(
                mask_logits,
                size=output_size,
                mode='bilinear',
                align_corners=False,
            )
        return mask_logits

    @torch.no_grad()
    def autoregressive_infer(self, image_tokens: torch.Tensor, temperature=1.0, top_k=None, num_samples: int = 1):
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        B, _, _ = image_tokens.shape
        batch_size = B * num_samples

        cross_caches = []
        for block in self.blocks:
            cached_k, cached_v = block.precompute_cross_kv(image_tokens)
            if num_samples > 1:
                cached_k = cached_k.repeat_interleave(num_samples, dim=0)
                cached_v = cached_v.repeat_interleave(num_samples, dim=0)
            cross_caches.append((cached_k, cached_v))

        self_caches = [None for _ in self.blocks]
        x_step = self.sos.view(1, 1, self.dim).expand(batch_size, 1, -1)
        generated_ids = []

        for i in range(self.num_queries):
            x_out = x_step
            for block_idx, block in enumerate(self.blocks):
                x_out, self_caches[block_idx] = block.forward_step(
                    x_out,
                    cross_caches[block_idx],
                    self_caches[block_idx],
                )

            logits = self.cls(x_out[:, 0, :])
            if temperature == 0:
                next_token = logits.argmax(dim=-1)
            else:
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            generated_ids.append(next_token)
            if i < self.num_queries - 1:
                x_step = self.embed(next_token).unsqueeze(1)

        generated = torch.stack(generated_ids, dim=1)
        if num_samples == 1:
            return generated
        return generated.view(B, num_samples, self.num_queries)
