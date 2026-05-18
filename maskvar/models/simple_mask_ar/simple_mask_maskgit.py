import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from .simple_mask_ar import (
    SimpleMaskARBlock,
    drop_click_condition,
    drop_image_condition,
    make_uncond_click_labels,
    sample_from_logits,
)
from .basic import RotaryPositionEmbedding


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal scalar embedding followed by an MLP projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=x.device, dtype=torch.float32) / half
        )
        args = x.float().view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class SimpleMaskMaskGIT(nn.Module):
    """
    Non-autoregressive masked-token model for SimpleMaskVqvae token ids.

    It uses the same frozen SimpleMaskVqvae mask/image token interface as
    SimpleMaskAR, but predicts a randomly masked subset in parallel.
    """

    def __init__(self, dim=384, depth=2, vocab_size=4096, h=64, w=64, num_heads=4, enable_click: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size
        self.dim = dim
        self.h = h
        self.w = w
        self.num_heads = num_heads
        self.max_len = h * w
        self.enable_click = enable_click

        self.embed = nn.Embedding(vocab_size + 1, dim)
        self.mask_ratio_emb = SinusoidalEmbedding(dim)
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, vocab_size)
        self.rope = RotaryPositionEmbedding(h=h, w=w)

        if enable_click:
            self.positive_click = nn.Parameter(torch.randn(dim))
            self.padding_click = nn.Parameter(torch.randn(dim))

        self.blocks = nn.ModuleList([
            SimpleMaskARBlock(rope=self.rope, dim=dim, num_heads=num_heads, enable_click=enable_click)
            for _ in range(depth)
        ])

    def get_device(self):
        return next(self.parameters()).device

    def _image_tokens_to_spatial(self, image_tokens: torch.Tensor):
        if image_tokens.dim() != 3:
            raise ValueError(f"Expected image_tokens to have shape (B, L, C), got {tuple(image_tokens.shape)}")
        B, L, C = image_tokens.shape
        if L != self.max_len:
            raise ValueError(f"Expected image token length {self.max_len}, got {L}")
        return image_tokens.view(B, self.h, self.w, C)

    def encode_clicks(self, click_coords: torch.Tensor | None, click_labels: torch.Tensor | None):
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
        mask_positions: torch.Tensor | None = None,
        mask_ratio: torch.Tensor | None = None,
        click_coords: torch.Tensor | None = None,
        click_labels: torch.Tensor | None = None,
        cfg_drop_click_prob: float = 0.0,
        cfg_drop_image_prob: float = 0.0,
    ):
        if x.shape[1] != self.max_len:
            raise ValueError(f"Expected token length {self.max_len}, got {x.shape[1]}")
        if mask_ratio is None:
            if mask_positions is None:
                mask_ratio = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
            else:
                mask_ratio = mask_positions.float().mean(dim=1)
        if mask_positions is not None:
            x = torch.where(mask_positions, torch.full_like(x, self.mask_token_id), x)

        x = self.embed(x).view(x.shape[0], self.h, self.w, self.dim)
        x = x + self.mask_ratio_emb(mask_ratio).view(x.shape[0], 1, 1, self.dim).to(x.dtype)
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

        logits = self.cls(self.norm(x))
        return rearrange(logits, "b h w vocab -> b (h w) vocab")

    def encode_mask_to_token_ids(self, vqvae_model, mask_normalized: torch.Tensor, image: torch.Tensor):
        with torch.no_grad():
            mask_tokens = vqvae_model.mask_encoder(mask_normalized)
            image_tokens = vqvae_model.image_encoder(image)

            B, C, h, w = mask_tokens.shape
            if h * w != self.max_len:
                raise ValueError(f"VQ token length {h*w} does not match MaskGIT max_len {self.max_len}")

            mask_tokens_blc = rearrange(mask_tokens, "b c h w -> b (h w) c")
            token_ids = vqvae_model.quant.x_to_idx(mask_tokens_blc.float())
            image_tokens_blc = rearrange(image_tokens, "b c h w -> b (h w) c")

        return token_ids, image_tokens_blc

    @torch.no_grad()
    def decode_token_ids_to_mask_logits(self, vqvae_model, token_ids, image_tokens, output_size):
        B, L = token_ids.shape
        if L != self.max_len:
            raise ValueError(f"Expected token length {self.max_len}, got {L}")

        mask_tokens = vqvae_model.quant.idx_to_x(token_ids)
        mask_tokens = rearrange(mask_tokens, "b (h w) c -> b h w c", h=self.h, w=self.w)
        image_tokens_spatial = self._image_tokens_to_spatial(image_tokens)
        mask_logits = vqvae_model.mask_decoder(mask_tokens, image_tokens_spatial)

        if mask_logits.shape[-2:] != output_size:
            mask_logits = nn.functional.interpolate(
                mask_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return mask_logits

    def _guided_logits(
        self,
        token_ids,
        image_tokens,
        masked,
        mask_ratio,
        click_coords,
        click_labels,
        cfg_guidance_scale,
        cfg_drop_click,
        cfg_drop_image,
    ):
        cond_logits = self(
            token_ids,
            image_tokens,
            mask_positions=masked,
            mask_ratio=mask_ratio,
            click_coords=click_coords,
            click_labels=click_labels,
        )
        if cfg_guidance_scale == 1.0 or not (cfg_drop_click or cfg_drop_image):
            return cond_logits

        uncond_image_tokens = torch.zeros_like(image_tokens) if cfg_drop_image else image_tokens
        uncond_click_labels = make_uncond_click_labels(click_labels) if cfg_drop_click else click_labels
        uncond_logits = self(
            token_ids,
            uncond_image_tokens,
            mask_positions=masked,
            mask_ratio=mask_ratio,
            click_coords=click_coords,
            click_labels=uncond_click_labels,
        )
        return uncond_logits + cfg_guidance_scale * (cond_logits - uncond_logits)

    def _click_expand_keep_mask(
        self,
        confidence_full: torch.Tensor,
        masked: torch.Tensor,
        next_mask_count: torch.Tensor,
        click_coords: torch.Tensor | None,
        click_labels: torch.Tensor | None,
        step: int,
        num_steps: int,
    ):
        if click_coords is None or click_labels is None:
            return None
        valid_clicks = click_labels == 1
        if not valid_clicks.any(dim=1).all():
            return None

        B = confidence_full.shape[0]
        ys = torch.arange(self.h, device=confidence_full.device, dtype=torch.float32)
        xs = torch.arange(self.w, device=confidence_full.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_y, grid_x], dim=-1).view(1, self.max_len, 1, 2)
        clicks = click_coords.to(device=confidence_full.device, dtype=torch.float32).view(B, 1, -1, 2)
        dist = (grid - clicks).abs().sum(dim=-1)
        dist = dist.masked_fill(~valid_clicks[:, None, :], float("inf")).min(dim=-1).values

        max_dist = dist.masked_fill(~torch.isfinite(dist), 0).amax(dim=1).clamp_min(1.0)
        allowed_radius = max_dist * float(step + 1) / float(num_steps)
        locked_future = dist > allowed_radius[:, None]
        rank_score = confidence_full
        rank_score = rank_score.masked_fill(locked_future, -float("inf"))
        rank_score = rank_score.masked_fill(~masked, float("inf"))

        next_mask_count = torch.maximum(next_mask_count, locked_future.sum(dim=1))
        if next_mask_count.max().item() == 0:
            return torch.zeros_like(masked)
        order = rank_score.argsort(dim=1)
        rank = order.argsort(dim=1)
        return rank < next_mask_count[:, None]

    @torch.no_grad()
    def maskgit_infer(
        self,
        image_tokens: torch.Tensor,
        click_coords: torch.Tensor | None = None,
        click_labels: torch.Tensor | None = None,
        num_steps: int = 12,
        temperature: float = 1.0,
        top_k=None,
        min_p: float | None = None,
        cfg_guidance_scale: float = 1.0,
        cfg_drop_click: bool = True,
        cfg_drop_image: bool = False,
        sampling_mode: str = "confidence",
    ):
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if sampling_mode not in {"confidence", "click_expand"}:
            raise ValueError(f"Unknown MaskGIT sampling_mode: {sampling_mode}")

        B = image_tokens.shape[0]
        device = image_tokens.device
        token_ids = torch.full((B, self.max_len), self.mask_token_id, dtype=torch.long, device=device)
        masked = torch.ones((B, self.max_len), dtype=torch.bool, device=device)

        for step in range(num_steps):
            mask_ratio = masked.float().mean(dim=1)
            logits = self._guided_logits(
                token_ids,
                image_tokens,
                masked,
                mask_ratio,
                click_coords,
                click_labels,
                cfg_guidance_scale,
                cfg_drop_click,
                cfg_drop_image,
            )
            masked_logits = logits[masked]
            sampled = sample_from_logits(masked_logits, temperature=temperature, top_k=top_k, min_p=min_p)
            probs = torch.softmax(masked_logits / max(temperature, 1e-6), dim=-1)
            confidence = probs.gather(1, sampled[:, None]).squeeze(1)
            token_ids[masked] = sampled

            if step == num_steps - 1:
                break

            keep_ratio = math.cos(0.5 * math.pi * (step + 1) / num_steps)
            next_mask_count = int(self.max_len * keep_ratio)
            conf_full = torch.full((B, self.max_len), float("inf"), device=device)
            conf_full[masked] = confidence
            current_mask_count = masked.sum(dim=1)
            next_mask_count = torch.full((B,), next_mask_count, device=device, dtype=torch.long)
            next_mask_count = torch.minimum(next_mask_count, current_mask_count - 1).clamp(min=0)

            next_masked = None
            if sampling_mode == "click_expand":
                next_masked = self._click_expand_keep_mask(
                    conf_full,
                    masked,
                    next_mask_count,
                    click_coords,
                    click_labels,
                    step,
                    num_steps,
                )

            if next_masked is not None:
                masked = next_masked
                token_ids[masked] = self.mask_token_id
            elif next_mask_count.max().item() == 0:
                masked = torch.zeros_like(masked)
            else:
                order = conf_full.argsort(dim=1)
                rank = order.argsort(dim=1)
                masked = rank < next_mask_count[:, None]
                token_ids[masked] = self.mask_token_id

        return token_ids.clamp_max(self.vocab_size - 1)
