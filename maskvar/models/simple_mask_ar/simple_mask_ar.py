import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
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
        x_seq = self.cross_block.forward_seq(x_seq, image_tokens, coords)
        x_seq = self.self_block.forward_seq(x_seq, coords, block_mask=None)
        return x_seq


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

        self.block_mask = None
        self.rope = RotaryPositionEmbedding(h=h, w=w)

        self.sos = nn.Parameter(torch.randn(dim))

        self.blocks = nn.ModuleList([
            SimpleMaskARBlock(rope=self.rope, dim=dim, num_heads=num_heads) for _ in range(depth)
        ])

    def get_device(self):
        return next(self.parameters()).device

    def init_block_mask(self):
        """Initialize causal block mask for training."""
        def mask_mod(b, h, q_idx, k_idx) -> bool:
            return (q_idx >= k_idx)

        device = self.get_device()
        # L = H*W (including sos at position 0)
        L = self.max_len
        self.block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=device)

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
        if self.block_mask is None:
            self.init_block_mask()

        x = self.preprocess(x)  # (B, H, W, C)

        for block in self.blocks:
            x = block(x, image_tokens, block_mask=self.block_mask)

        # Classify to get logits
        logits = self.cls(x)  # (B, H, W, vocab_size)
        return logits

    @torch.no_grad()
    def autoregressive_infer(self, image_tokens: torch.Tensor, temperature=1.0, top_k=None):
        """
        Autoregressive inference in row-major order.

        Args:
            image_tokens: (B, H, W, C)
            temperature: sampling temperature (0 for greedy)
            top_k: top-k sampling (None to disable)

        Returns:
            generated_ids: (B, H, W) - generated token indices in spatial format
        """
        B, H, W, C = image_tokens.shape
        device = image_tokens.device

        # Pre-generate coordinates for all positions in row-major order
        coords = torch.tensor(
            [[h, w] for h in range(H) for w in range(W)],
            device=device, dtype=torch.long
        )  # (H*W, 2)

        # Initialize sequence with sos token embedding
        sos_embed = self.sos.view(1, 1, self.dim).expand(B, 1, -1)  # (B, 1, C)
        x_seq = sos_embed  # (B, 1, C)

        generated_ids = []

        for i in range(H * W):
            # Current position coordinates (up to position i, which is i+1 tokens)
            curr_coords = coords[:i+1]  # (i+1, 2)

            # Forward pass through all blocks
            x_out = x_seq
            for block in self.blocks:
                x_out = block.forward_seq(x_out, image_tokens, curr_coords)

            # Get logits for the current (last) position
            logits = self.cls(x_out[:, -1, :])  # (B, vocab_size)

            # Sample next token
            if temperature == 0:
                next_token = logits.argmax(dim=-1)  # (B,)
            else:
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

            generated_ids.append(next_token)

            # Append next token embedding to sequence (for next iteration)
            if i < H * W - 1:
                next_embed = self.embed(next_token).unsqueeze(1)  # (B, 1, C)
                x_seq = torch.cat([x_seq, next_embed], dim=1)  # (B, i+2, C)

        # Stack generated tokens and reshape to spatial
        generated = torch.stack(generated_ids, dim=1)  # (B, H*W)
        generated_spatial = generated.view(B, H, W)  # (B, H, W)

        return generated_spatial
