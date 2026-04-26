import torch
from torch import nn
from einops import rearrange

from .basic import RotaryPositionEmbedding, SimpleCrossBlock, SimpleSelfBlock


class SimpleMaskARBlock(nn.Module):

    def __init__(self, rope: RotaryPositionEmbedding, dim, num_heads):
        super().__init__()
        self.cross_block = SimpleCrossBlock(rope, dim, num_heads)
        self.self_block = SimpleSelfBlock(rope, dim, num_heads)

    def forward(self, x, image_tokens, block_mask=None):
        """
        Training forward with spatial format.

        x: (b, h, w, c) - spatial format
        image_tokens: (b, h, w, c) - spatial format
        """
        x = self.cross_block(x, kv=image_tokens)
        x = self.self_block(x, attn_mask=block_mask)
        return x

    def forward_seq(self, x_seq, image_tokens, coords):
        """
        Inference forward with sequence format.

        Args:
            x_seq: (b, l, c) - sequence format
            image_tokens: (b, h, w, c) - spatial format
            coords: (l, 2) - coordinates for each position in sequence
        """
        _, h, w, _ = image_tokens.shape
        x_seq = self.cross_block.forward_seq(x_seq, image_tokens, coords)
        x_seq = self.self_block.forward_seq(x_seq, coords, block_mask=None, spatial_shape=(h, w))
        return x_seq

    def precompute_cross_kv(self, image_tokens):
        return self.cross_block.precompute_kv(image_tokens)

    def forward_step(self, x_step, cross_cache, self_cache, coord):
        """
        Incremental autoregressive step for a single token.

        Args:
            x_step: (b, 1, c)
            cross_cache: tuple(cached_k, cached_v, h, w)
            self_cache: optional tuple(cache_k, cache_v)
            coord: (1, 2)
        """
        cached_k, cached_v, h, w = cross_cache
        cache_k, cache_v = (self_cache if self_cache is not None else (None, None))

        x_step = self.cross_block.forward_step(x_step, cached_k, cached_v, coord, h, w)
        x_step, new_k, new_v = self.self_block.forward_step(
            x_step,
            coord,
            cache_k=cache_k,
            cache_v=cache_v,
            spatial_shape=(h, w),
        )
        return x_step, (new_k, new_v)


class SimpleMaskAR(nn.Module):

    def __init__(self, dim=256, depth=2, vocab_size=4096, h=64, w=64, num_heads=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.h = h
        self.w = w
        self.num_heads = num_heads
        self.max_len = h * w

        self.embed = nn.Embedding(self.vocab_size, dim)
        self.cls = nn.Linear(dim, self.vocab_size)

        self.rope = RotaryPositionEmbedding(h=h, w=w)

        self.sos = nn.Parameter(torch.randn(dim))

        self.blocks = nn.ModuleList([
            SimpleMaskARBlock(rope=self.rope, dim=dim, num_heads=num_heads) for _ in range(depth)
        ])

    def get_device(self):
        return next(self.parameters()).device

    def preprocess(self, x: torch.Tensor):
        """
        Training preprocess.
        x: (B, H, W) - spatial token indices, the last token will be dropped
        Returns: (B, H, W, C) - spatial embeddings with sos at (0,0)

        Processing:
        - Flatten to sequence (row-major): [t00, t01, ..., t0(w-1), t10, ..., t(h-1)(w-1)]
        - Drop the last token
        - Prepend sos: [sos, t00, t01, ..., up to H*W-1 tokens]
        - Reshape back to (B, H, W, C)

        So position (0,0) has sos, and original token at (0,0) goes to (0,1), etc.
        The token at (h-1, w-1) is dropped.
        """
        B, H, W = x.shape
        # Embed tokens
        x_embed = self.embed(x)  # (B, H, W, C)

        # Flatten to sequence
        x_seq = rearrange(x_embed, 'b h w c -> b (h w) c')  # (B, H*W, C)

        # Drop last token and prepend sos
        x_seq = x_seq[:, :-1, :]  # (B, H*W-1, C) - drop last
        sos_seq = self.sos.view(1, 1, self.dim).expand(B, 1, -1)  # (B, 1, C)
        x_seq = torch.cat([sos_seq, x_seq], dim=1)  # (B, H*W, C)

        # Reshape back to spatial
        x_spatial = x_seq.view(B, H, W, self.dim)  # (B, H, W, C)

        return x_spatial

    def forward(self, x: torch.Tensor, image_tokens: torch.Tensor):
        """
        Training forward.
        x: (B, H, W) - spatial token indices in row-major order
        image_tokens: (B, H, W, C)
        Returns: logits (B, H, W, vocab_size) for next token prediction

        Note: The last position's prediction is not used in loss (no ground truth).
        """
        x = self.preprocess(x)  # (B, H, W, C)

        for block in self.blocks:
            x = block(x, image_tokens, block_mask=None)

        # Classify to get logits
        logits = self.cls(x)  # (B, H, W, vocab_size)
        return logits

    @torch.no_grad()
    def autoregressive_infer(
        self,
        image_tokens: torch.Tensor,
        temperature=1.0,
        top_k=None,
        num_samples: int = 1,
    ):
        """
        Autoregressive inference in row-major order.

        Args:
            image_tokens: (B, H, W, C)
            temperature: sampling temperature (0 for greedy)
            top_k: top-k sampling (None to disable)
            num_samples: number of sampled sequences per input image

        Returns:
            generated_ids:
                - (B, H, W) when num_samples == 1
                - (B, num_samples, H, W) when num_samples > 1
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        B, H, W, C = image_tokens.shape
        device = image_tokens.device
        batch_size = B * num_samples

        # Pre-generate coordinates for all positions in row-major order
        coords = torch.tensor(
            [[h, w] for h in range(H) for w in range(W)],
            device=device, dtype=torch.long
        )  # (H*W, 2)

        # Initialize step input with sos embedding and per-block caches.
        x_step = self.sos.view(1, 1, self.dim).expand(batch_size, 1, -1)  # (B * num_samples, 1, C)
        cross_caches = []
        for block in self.blocks:
            cached_k, cached_v, h, w = block.precompute_cross_kv(image_tokens)
            if num_samples > 1:
                cached_k = cached_k.repeat_interleave(num_samples, dim=0)
                cached_v = cached_v.repeat_interleave(num_samples, dim=0)
            cross_caches.append((cached_k, cached_v, h, w))
        self_caches = [None for _ in self.blocks]

        generated_ids = []

        for i in range(H * W):
            curr_coord = coords[i:i+1]  # (1, 2)

            x_out = x_step
            for block_idx, block in enumerate(self.blocks):
                x_out, self_caches[block_idx] = block.forward_step(
                    x_out,
                    cross_caches[block_idx],
                    self_caches[block_idx],
                    curr_coord,
                )

            # Get logits for the current (last) position
            logits = self.cls(x_out[:, 0, :])  # (B * num_samples, vocab_size)

            # Sample next token
            if temperature == 0:
                next_token = logits.argmax(dim=-1)  # (B * num_samples,)
            else:
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B * num_samples,)

            generated_ids.append(next_token)

            # Append next token embedding to sequence (for next iteration)
            if i < H * W - 1:
                x_step = self.embed(next_token).unsqueeze(1)  # (B * num_samples, 1, C)

        # Stack generated tokens and reshape to spatial
        generated = torch.stack(generated_ids, dim=1)  # (B * num_samples, H*W)
        if num_samples == 1:
            return generated.view(B, H, W)

        generated = generated.view(B, num_samples, H * W)
        return generated.view(B, num_samples, H, W)
