import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from einops import rearrange

from .basic import RotaryPositionEmbedding, SimpleCrossBlock


class SimpleScaleBlockedSelfBlock(nn.Module):
    """Self-attention over a concatenated multiscale sequence with scale blocks."""

    def __init__(self, rope: RotaryPositionEmbedding, dim: int, num_heads: int):
        super().__init__()
        self.rope = rope
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.linear_qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.layernorm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x_seq, coords, spatial_shape, block_mask=None):
        b, _, _ = x_seq.shape
        h, w = spatial_shape
        residual = x_seq

        qkv = self.linear_qkv(x_seq)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b l (nh c) -> b nh l c", nh=self.num_heads, c=self.dim_head)
        k = rearrange(k, "b l (nh c) -> b nh l c", nh=self.num_heads, c=self.dim_head)
        v = rearrange(v, "b l (nh c) -> b nh l c", nh=self.num_heads, c=self.dim_head)

        q = rearrange(q, "b nh l c -> (b nh) l c")
        k = rearrange(k, "b nh l c -> (b nh) l c")
        q = self.rope.apply_2d_rope_with_coords(q, coords, h, w)
        k = self.rope.apply_2d_rope_with_coords(k, coords, h, w)
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        q = rearrange(q, "(b nh) l c -> b nh l c", b=b, nh=self.num_heads)
        k = rearrange(k, "(b nh) l c -> b nh l c", b=b, nh=self.num_heads)

        if block_mask is None:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = flex_attention(q, k, v, block_mask=block_mask)
        out = rearrange(out, "b nh l c -> b l (nh c)")
        out = residual + self.out_proj(out)
        out = out + self.ffn(self.layernorm(out))
        return out


class SimpleMaskVARBlock(nn.Module):
    def __init__(self, rope: RotaryPositionEmbedding, dim: int, num_heads: int):
        super().__init__()
        self.cross_block = SimpleCrossBlock(rope, dim, num_heads)
        self.self_block = SimpleScaleBlockedSelfBlock(rope, dim, num_heads)

    def forward(self, x_seq, image_tokens, coords, spatial_shape, block_mask=None):
        x_seq = self.cross_block.forward_seq(x_seq, image_tokens, coords)
        x_seq = self.self_block(
            x_seq,
            coords=coords,
            spatial_shape=spatial_shape,
            block_mask=block_mask,
        )
        return x_seq


class SimpleMaskVAR(nn.Module):
    """
    SimpleMaskVAR aligned with VAR's training input convention.

    The frozen multiscale V2 VQVAE converts GT residual token ids into one
    concatenated BLC sequence: cumulative f_hat after scale k, downsampled to
    scale k+1. This model prepends learned inputs for the first scale and
    predicts all scales in one transformer pass. Self-attention is block
    diagonal by scale, so tokens never attend across scales.
    """

    def __init__(
        self,
        dim=384,
        depth=2,
        vocab_size=4096,
        scales=(1, 2, 4, 8, 16, 32, 64),
        h=64,
        w=64,
        num_heads=4,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.vocab_size = vocab_size
        self.scales = tuple(scales)
        self.h = h
        self.w = w
        self.num_heads = num_heads
        self.max_len = h * w
        self.seq_len = sum(scale * scale for scale in self.scales)
        self.first_len = self.scales[0] * self.scales[0]

        self.input_proj = nn.Linear(dim, dim)
        self.first_scale_tokens = nn.Parameter(torch.randn(self.first_len, dim) * 0.02)
        self.scale_embed = nn.Embedding(len(self.scales), dim)
        self.cls = nn.Linear(dim, vocab_size)
        self.rope = RotaryPositionEmbedding(h=h, w=w)
        self.blocks = nn.ModuleList([
            SimpleMaskVARBlock(self.rope, dim=dim, num_heads=num_heads)
            for _ in range(depth)
        ])

        coords, scale_ids = self._build_sequence_metadata()
        self.register_buffer("coords", coords, persistent=False)
        self.register_buffer("scale_ids", scale_ids, persistent=False)
        self._block_mask_cache = {}

    def _build_sequence_metadata(self):
        coords = []
        scale_ids = []
        for scale_idx, scale in enumerate(self.scales):
            coord_scale = self.h / scale
            for row in range(scale):
                for col in range(scale):
                    coords.append(((row + 0.5) * coord_scale - 0.5, (col + 0.5) * coord_scale - 0.5))
                    scale_ids.append(scale_idx)
        coords = torch.tensor(coords, dtype=torch.float32)
        scale_ids = torch.tensor(scale_ids, dtype=torch.long)
        return coords, scale_ids

    def _get_block_mask(self, seq_len: int, device: torch.device):
        key = (seq_len, device.type, device.index)
        if key not in self._block_mask_cache:
            scale_ids = self.scale_ids[:seq_len].to(device=device)

            def mask_mod(b, h, q_idx, k_idx):
                return scale_ids[q_idx] == scale_ids[k_idx]

            self._block_mask_cache[key] = create_block_mask(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
            )
        return self._block_mask_cache[key]

    def get_device(self):
        return next(self.parameters()).device

    def _image_tokens_to_spatial(self, image_tokens: torch.Tensor):
        if image_tokens.dim() == 4:
            return image_tokens
        if image_tokens.dim() != 3:
            raise ValueError(f"Expected image_tokens to have shape (B, L, C), got {tuple(image_tokens.shape)}")
        b, l, c = image_tokens.shape
        if l != self.max_len:
            raise ValueError(f"Expected image token length {self.max_len}, got {l}")
        return image_tokens.view(b, self.h, self.w, c)

    def _prepare_sequence(self, var_input: torch.Tensor | None, image_tokens: torch.Tensor):
        """
        Build the full teacher-forcing sequence used by training forward.

        `var_input` follows the original VAR convention: it contains only
        scales 1..K-1 as one concatenated BLC tensor, while scale 0 is predicted
        from learned start tokens. The returned sequence is therefore
        `[first_scale_tokens, input_proj(var_input)]` plus scale embeddings.

        Returns:
            x: (B, L, C) input sequence for the transformer.
            seq_len: actual sequence length, useful for prefix inference.
        """
        b = image_tokens.shape[0]
        first = self.first_scale_tokens.view(1, self.first_len, self.dim).expand(b, -1, -1)
        if var_input is None:
            x = first
        else:
            max_len = self.seq_len - self.first_len
            if var_input.shape[1] > max_len:
                raise ValueError(f"Expected VAR input length at most {max_len}, got {var_input.shape[1]}")
            x = torch.cat([first, self.input_proj(var_input)], dim=1)
        seq_len = x.shape[1]
        scale_ids = self.scale_ids[:seq_len].to(device=x.device)
        x = x + self.scale_embed(scale_ids).view(1, seq_len, self.dim)
        return x, seq_len

    def _prepare_scale_sequence(self, scale_idx: int, var_input: torch.Tensor | None, image_tokens: torch.Tensor):
        b = image_tokens.shape[0]
        scale = self.scales[scale_idx]
        start = sum(s * s for s in self.scales[:scale_idx])
        length = scale * scale
        if scale_idx == 0:
            x = self.first_scale_tokens.view(1, self.first_len, self.dim).expand(b, -1, -1)
        else:
            if var_input is None:
                raise ValueError("var_input is required for scale_idx > 0")
            x = self.input_proj(var_input[:, -length:, :])
        scale_ids = self.scale_ids[start:start + length].to(device=image_tokens.device)
        coords = self.coords[start:start + length].to(device=image_tokens.device)
        x = x + self.scale_embed(scale_ids).view(1, length, self.dim)
        return x, coords

    def _split_logits_by_scale(self, logits):
        logits_by_scale = []
        start = 0
        for scale in self.scales:
            length = scale * scale
            if start + length > logits.shape[1]:
                break
            logits_by_scale.append(logits[:, start:start + length])
            start += length
        return logits_by_scale

    def forward(self, var_input: torch.Tensor | None, image_tokens: torch.Tensor, click_coords=None, click_labels=None):
        """
        var_input: concatenated BLC from vqvae.to_var_input(), containing scales 1..K-1.
        image_tokens: (B, 64*64, C)
        returns: list of logits, one (B, scale*scale, vocab_size) tensor per scale.
        """
        del click_coords, click_labels
        image_tokens_spatial = self._image_tokens_to_spatial(image_tokens)
        x, seq_len = self._prepare_sequence(var_input, image_tokens)

        coords = self.coords[:seq_len].to(device=x.device)
        block_mask = self._get_block_mask(seq_len, x.device)
        for block in self.blocks:
            x = block(
                x,
                image_tokens_spatial,
                coords=coords,
                spatial_shape=(self.h, self.w),
                block_mask=block_mask,
            )

        return self._split_logits_by_scale(self.cls(x))

    def forward_scale(self, scale_idx: int, var_input: torch.Tensor | None, image_tokens: torch.Tensor):
        image_tokens_spatial = self._image_tokens_to_spatial(image_tokens)
        x, coords = self._prepare_scale_sequence(scale_idx, var_input, image_tokens)
        for block in self.blocks:
            x = block(
                x,
                image_tokens_spatial,
                coords=coords,
                spatial_shape=(self.h, self.w),
                block_mask=None,
            )
        return self.cls(x)

    def set_vqvae_model(self, vqvae_model):
        object.__setattr__(self, "_vqvae_model", vqvae_model)

    @torch.no_grad()
    def autoregressive_infer(self, image_tokens: torch.Tensor, temperature=1.0, top_k=None, num_samples: int = 1, **kwargs):
        del kwargs
        if num_samples != 1:
            raise ValueError("SimpleMaskVAR currently supports num_samples=1")

        b = image_tokens.shape[0]
        generated = []
        for scale_idx, scale in enumerate(self.scales):
            var_input = self._vqvae_model.to_var_input(generated)
            logits = self.forward_scale(scale_idx, var_input, image_tokens)

            if temperature == 0:
                next_ids = logits.argmax(dim=-1)
            else:
                sample_logits = logits
                if top_k is not None:
                    values, _ = torch.topk(sample_logits, min(top_k, sample_logits.shape[-1]), dim=-1)
                    sample_logits = sample_logits.masked_fill(sample_logits < values[..., [-1]], -float("inf"))
                probs = torch.softmax(sample_logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs.flatten(0, 1), num_samples=1).view(b, scale * scale)
            generated.append(next_ids)

        return generated
