import torch
from torch import nn
from einops import rearrange

from .basic import RotaryPositionEmbedding, SimpleClickCrossBlock, SimpleCrossBlock, SimpleSelfBlock


def drop_click_condition(click_labels: torch.Tensor | None, drop_prob: float, training: bool):
    if click_labels is None or drop_prob <= 0 or not training:
        return click_labels
    B = click_labels.shape[0]
    drop = torch.rand(B, device=click_labels.device) < drop_prob
    return torch.where(drop[:, None], torch.full_like(click_labels, -1), click_labels)


def drop_image_condition(image_tokens: torch.Tensor, drop_prob: float, training: bool):
    if drop_prob <= 0 or not training:
        return image_tokens
    B = image_tokens.shape[0]
    drop = torch.rand(B, device=image_tokens.device) < drop_prob
    view_shape = (B,) + (1,) * (image_tokens.dim() - 1)
    return torch.where(drop.view(view_shape), torch.zeros_like(image_tokens), image_tokens)


def make_uncond_click_labels(click_labels: torch.Tensor | None):
    if click_labels is None:
        return None
    return torch.full_like(click_labels, -1)


def sample_from_logits(logits: torch.Tensor, temperature=1.0, top_k=None, min_p: float | None = None):
    if temperature == 0:
        return logits.argmax(dim=-1)
    if min_p is not None and not 0 <= min_p <= 1:
        raise ValueError(f"min_p must be in [0, 1], got {min_p}")
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float('inf'))
    probs = torch.softmax(logits / temperature, dim=-1)
    if min_p is not None and min_p > 0:
        max_probs = probs.max(dim=-1, keepdim=True).values
        probs = probs.masked_fill(probs < max_probs * min_p, 0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class SimpleMaskARBlock(nn.Module):

    def __init__(self, rope: RotaryPositionEmbedding, dim, num_heads, enable_click: bool = False):
        super().__init__()
        self.enable_click = enable_click
        if enable_click:
            self.click_cross_block = SimpleClickCrossBlock(rope, dim, num_heads)
        self.cross_block = SimpleCrossBlock(rope, dim, num_heads)
        self.self_block = SimpleSelfBlock(rope, dim, num_heads)

    def forward(self, x, image_tokens, click_tokens=None, click_coords=None, click_labels=None, block_mask=None):
        """
        Training forward with spatial format.

        x: (b, h, w, c) - spatial format
        image_tokens: (b, h, w, c) - spatial format
        """
        if self.enable_click and click_tokens is not None:
            x = self.click_cross_block(x, click_tokens, click_coords, click_labels)
        x = self.cross_block(x, kv=image_tokens)
        x = self.self_block(x, attn_mask=block_mask)
        return x

    def forward_seq(self, x_seq, image_tokens, coords, click_tokens=None, click_coords=None, click_labels=None):
        """
        Inference forward with sequence format.

        Args:
            x_seq: (b, l, c) - sequence format
            image_tokens: (b, h, w, c) - spatial format
            coords: (l, 2) - coordinates for each position in sequence
        """
        _, h, w, _ = image_tokens.shape
        if self.enable_click and click_tokens is not None:
            x_seq = self.click_cross_block.forward_seq(
                x_seq,
                click_tokens,
                click_coords,
                click_labels,
                coords,
                spatial_shape=(h, w),
            )
        x_seq = self.cross_block.forward_seq(x_seq, image_tokens, coords)
        x_seq = self.self_block.forward_seq(x_seq, coords, block_mask=None, spatial_shape=(h, w))
        return x_seq

    def precompute_cross_kv(self, image_tokens):
        return self.cross_block.precompute_kv(image_tokens)

    def precompute_click_kv(self, click_tokens, click_coords, click_labels, h, w):
        if not self.enable_click or click_tokens is None:
            return None
        return self.click_cross_block.precompute_kv(click_tokens, click_coords, click_labels, h, w)

    def forward_step(self, x_step, cross_cache, self_cache, coord, click_cache=None):
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

        if self.enable_click and click_cache is not None:
            click_k, click_v, click_h, click_w = click_cache
            x_step = self.click_cross_block.forward_step(x_step, click_k, click_v, coord, click_h, click_w)
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

    def __init__(self, dim=256, depth=2, vocab_size=4096, h=64, w=64, num_heads=8, enable_click: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.h = h
        self.w = w
        self.num_heads = num_heads
        self.max_len = h * w
        self.enable_click = enable_click

        self.embed = nn.Embedding(self.vocab_size, dim)
        self.cls = nn.Linear(dim, self.vocab_size)

        self.rope = RotaryPositionEmbedding(h=h, w=w)

        self.sos = nn.Parameter(torch.randn(dim))
        if enable_click:
            self.positive_click = nn.Parameter(torch.randn(dim))
            self.padding_click = nn.Parameter(torch.randn(dim))

        self.blocks = nn.ModuleList([
            SimpleMaskARBlock(rope=self.rope, dim=dim, num_heads=num_heads, enable_click=enable_click) for _ in range(depth)
        ])

    def get_device(self):
        return next(self.parameters()).device

    def preprocess(self, x: torch.Tensor):
        """
        Training preprocess.
        x: (B, L) - token indices in row-major order, the last token will be dropped
        Returns: (B, H, W, C) - spatial embeddings with sos at (0,0)

        Processing:
        - Flatten to sequence (row-major): [t00, t01, ..., t0(w-1), t10, ..., t(h-1)(w-1)]
        - Drop the last token
        - Prepend sos: [sos, t00, t01, ..., up to H*W-1 tokens]
        - Reshape back to (B, H, W, C)

        So position (0,0) has sos, and original token at (0,0) goes to (0,1), etc.
        The token at (h-1, w-1) is dropped.
        """
        B, L = x.shape
        if L != self.max_len:
            raise ValueError(f"Expected token length {self.max_len}, got {L}")

        x_seq = self.embed(x)  # (B, H*W, C)

        # Drop last token and prepend sos
        x_seq = x_seq[:, :-1, :]  # (B, H*W-1, C) - drop last
        sos_seq = self.sos.view(1, 1, self.dim).expand(B, 1, -1)  # (B, 1, C)
        x_seq = torch.cat([sos_seq, x_seq], dim=1)  # (B, H*W, C)

        # Reshape back to spatial
        x_spatial = x_seq.view(B, self.h, self.w, self.dim)  # (B, H, W, C)

        return x_spatial

    def _image_tokens_to_spatial(self, image_tokens: torch.Tensor):
        if image_tokens.dim() != 3:
            raise ValueError(f"Expected image_tokens to have shape (B, L, C), got {tuple(image_tokens.shape)}")
        B, L, C = image_tokens.shape
        if L != self.max_len:
            raise ValueError(f"Expected image token length {self.max_len}, got {L}")
        return image_tokens.view(B, self.h, self.w, C)

    def encode_clicks(self, click_coords: torch.Tensor | None, click_labels: torch.Tensor | None):
        """
        Build click condition tokens.

        click_coords: (B, N, 2), row/col coordinates in AR token-grid units
        click_labels: (B, N), 1 for positive clicks and -1 for padding
        """
        if not self.enable_click:
            return None, None, None
        if click_coords is None or click_labels is None:
            raise ValueError("click_coords and click_labels are required when enable_click=True")

        click_coords = click_coords.to(device=self.positive_click.device, dtype=torch.float32)
        click_labels = click_labels.to(device=self.positive_click.device)
        B, N, _ = click_coords.shape
        pos = self.positive_click.view(1, 1, self.dim).expand(B, N, -1)
        pad = self.padding_click.view(1, 1, self.dim).expand(B, N, -1)
        click_tokens = torch.where((click_labels == 1).unsqueeze(-1), pos, pad)

        pos_rope = self.rope.apply_2d_rope_with_batched_coords(click_tokens, click_coords, self.h, self.w)
        click_tokens = torch.where((click_labels == 1).unsqueeze(-1), pos_rope, click_tokens)
        return click_tokens, click_coords, click_labels

    def forward(
        self,
        x: torch.Tensor,
        image_tokens: torch.Tensor,
        click_coords: torch.Tensor | None = None,
        click_labels: torch.Tensor | None = None,
        cfg_drop_click_prob: float = 0.0,
        cfg_drop_image_prob: float = 0.0,
    ):
        """
        Training forward.
        x: (B, L) - token indices in row-major order
        image_tokens: (B, L_img, C)
        Returns: logits (B, L, vocab_size) for next token prediction

        Note: The last position's prediction is not used in loss (no ground truth).
        """
        x = self.preprocess(x)  # (B, H, W, C)
        click_labels = drop_click_condition(click_labels, cfg_drop_click_prob, self.training)
        image_tokens = drop_image_condition(image_tokens, cfg_drop_image_prob, self.training)
        image_tokens = self._image_tokens_to_spatial(image_tokens)
        click_tokens, click_coords, click_labels = self.encode_clicks(click_coords, click_labels)

        for block in self.blocks:
            x = block(
                x,
                image_tokens,
                click_tokens=click_tokens,
                click_coords=click_coords,
                click_labels=click_labels,
                block_mask=None,
            )

        # Classify to get logits
        logits = self.cls(x)  # (B, H, W, vocab_size)
        return rearrange(logits, 'b h w vocab -> b (h w) vocab')

    def encode_mask_to_token_ids(self, vqvae_model, mask_normalized: torch.Tensor, image: torch.Tensor):
        with torch.no_grad():
            mask_tokens = vqvae_model.mask_encoder(mask_normalized)  # (B, C, h, w)
            image_tokens = vqvae_model.image_encoder(image)  # (B, C, h, w)

            B, C, h, w = mask_tokens.shape
            if h * w != self.max_len:
                raise ValueError(f"VQ token length {h*w} does not match AR max_len {self.max_len}")

            mask_tokens_blc = rearrange(mask_tokens, 'b c h w -> b (h w) c')
            token_ids = vqvae_model.quant.x_to_idx(mask_tokens_blc.float())  # (B, L)
            image_tokens_blc = rearrange(image_tokens, 'b c h w -> b (h w) c')

        return token_ids, image_tokens_blc

    @torch.no_grad()
    def decode_token_ids_to_mask_logits(self, vqvae_model, token_ids, image_tokens, output_size):
        B, L = token_ids.shape
        if L != self.max_len:
            raise ValueError(f"Expected token length {self.max_len}, got {L}")

        mask_tokens = vqvae_model.quant.idx_to_x(token_ids)
        mask_tokens = rearrange(mask_tokens, 'b (h w) c -> b h w c', h=self.h, w=self.w)
        image_tokens_spatial = self._image_tokens_to_spatial(image_tokens)
        mask_logits = vqvae_model.mask_decoder(mask_tokens, image_tokens_spatial)

        if mask_logits.shape[-2:] != output_size:
            mask_logits = nn.functional.interpolate(
                mask_logits,
                size=output_size,
                mode='bilinear',
                align_corners=False,
            )

        return mask_logits

    @torch.no_grad()
    def autoregressive_infer(
        self,
        image_tokens: torch.Tensor,
        click_coords: torch.Tensor | None = None,
        click_labels: torch.Tensor | None = None,
        temperature=1.0,
        top_k=None,
        min_p: float | None = None,
        num_samples: int = 1,
        cfg_guidance_scale: float = 1.0,
        cfg_drop_click: bool = True,
        cfg_drop_image: bool = False,
    ):
        """
        Autoregressive inference in row-major order.

        Args:
            image_tokens: (B, H, W, C)
            temperature: sampling temperature (0 for greedy)
            top_k: top-k sampling (None to disable)
            min_p: min-p sampling threshold relative to the most likely token (None to disable)
            num_samples: number of sampled sequences per input image

        Returns:
            generated_ids:
                - (B, L) when num_samples == 1
                - (B, num_samples, L) when num_samples > 1
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        use_cfg = cfg_guidance_scale != 1.0 and (cfg_drop_click or cfg_drop_image)
        image_tokens_spatial = self._image_tokens_to_spatial(image_tokens)
        uncond_image_tokens_spatial = (
            torch.zeros_like(image_tokens_spatial) if cfg_drop_image else image_tokens_spatial
        )
        click_tokens, click_coords, click_labels = self.encode_clicks(click_coords, click_labels)
        uncond_click_tokens = None
        uncond_click_labels = make_uncond_click_labels(click_labels) if cfg_drop_click else click_labels
        if use_cfg and self.enable_click and click_coords is not None and uncond_click_labels is not None:
            uncond_click_tokens, _, uncond_click_labels = self.encode_clicks(click_coords, uncond_click_labels)
        B, _, C = image_tokens.shape
        H, W = self.h, self.w
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
        click_caches = []
        uncond_cross_caches = []
        uncond_click_caches = []
        for block in self.blocks:
            cached_k, cached_v, h, w = block.precompute_cross_kv(image_tokens_spatial)
            if num_samples > 1:
                cached_k = cached_k.repeat_interleave(num_samples, dim=0)
                cached_v = cached_v.repeat_interleave(num_samples, dim=0)
            cross_caches.append((cached_k, cached_v, h, w))
            click_cache = block.precompute_click_kv(click_tokens, click_coords, click_labels, H, W)
            if click_cache is not None and num_samples > 1:
                click_k, click_v, click_h, click_w = click_cache
                click_cache = (
                    click_k.repeat_interleave(num_samples, dim=0),
                    click_v.repeat_interleave(num_samples, dim=0),
                    click_h,
                    click_w,
                )
            click_caches.append(click_cache)
            if use_cfg:
                uncached_k, uncached_v, uh, uw = block.precompute_cross_kv(uncond_image_tokens_spatial)
                if num_samples > 1:
                    uncached_k = uncached_k.repeat_interleave(num_samples, dim=0)
                    uncached_v = uncached_v.repeat_interleave(num_samples, dim=0)
                uncond_cross_caches.append((uncached_k, uncached_v, uh, uw))
                uncond_click_cache = block.precompute_click_kv(uncond_click_tokens, click_coords, uncond_click_labels, H, W)
                if uncond_click_cache is not None and num_samples > 1:
                    click_k, click_v, click_h, click_w = uncond_click_cache
                    uncond_click_cache = (
                        click_k.repeat_interleave(num_samples, dim=0),
                        click_v.repeat_interleave(num_samples, dim=0),
                        click_h,
                        click_w,
                    )
                uncond_click_caches.append(uncond_click_cache)
        self_caches = [None for _ in self.blocks]
        uncond_self_caches = [None for _ in self.blocks] if use_cfg else None

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
                    click_cache=click_caches[block_idx],
                )

            # Get logits for the current (last) position
            logits = self.cls(x_out[:, 0, :])  # (B * num_samples, vocab_size)
            if use_cfg:
                x_uncond = x_step
                for block_idx, block in enumerate(self.blocks):
                    x_uncond, uncond_self_caches[block_idx] = block.forward_step(
                        x_uncond,
                        uncond_cross_caches[block_idx],
                        uncond_self_caches[block_idx],
                        curr_coord,
                        click_cache=uncond_click_caches[block_idx],
                    )
                uncond_logits = self.cls(x_uncond[:, 0, :])
                logits = uncond_logits + cfg_guidance_scale * (logits - uncond_logits)

            # Sample next token
            next_token = sample_from_logits(logits, temperature=temperature, top_k=top_k, min_p=min_p)

            generated_ids.append(next_token)

            # Append next token embedding to sequence (for next iteration)
            if i < H * W - 1:
                x_step = self.embed(next_token).unsqueeze(1)  # (B * num_samples, 1, C)

        # Stack generated tokens and reshape to spatial
        generated = torch.stack(generated_ids, dim=1)  # (B * num_samples, H*W)
        if num_samples == 1:
            return generated

        return generated.view(B, num_samples, H * W)
