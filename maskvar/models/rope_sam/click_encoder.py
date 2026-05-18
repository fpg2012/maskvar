import torch
from torch import nn
from einops import rearrange


class RopeClickEncoder(nn.Module):
    """Encode point prompts as RoPE-positioned query tokens."""

    def __init__(self, dim: int = 384, h: int = 64, w: int = 64):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w

        self.positive_click = nn.Parameter(torch.randn(dim))
        self.negative_click = nn.Parameter(torch.randn(dim))
        self.padding_click = nn.Parameter(torch.randn(dim))

    def forward(self, click_coords: torch.Tensor, click_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            click_coords: (B, N, 2), row/col coordinates in token-grid units.
            click_labels: (B, N), 1 positive, 0 negative, -1 padding.

        Returns:
            click_tokens: (B, N, C)
        """
        if click_coords is None or click_labels is None:
            raise ValueError("click_coords and click_labels are required")
        if click_coords.dim() != 3 or click_coords.shape[-1] != 2:
            raise ValueError(f"Expected click_coords shape (B, N, 2), got {tuple(click_coords.shape)}")
        if click_labels.shape != click_coords.shape[:2]:
            raise ValueError(
                f"Expected click_labels shape {tuple(click_coords.shape[:2])}, got {tuple(click_labels.shape)}"
            )

        click_coords = click_coords.to(device=self.positive_click.device, dtype=torch.float32)
        click_labels = click_labels.to(device=self.positive_click.device)
        b, n, _ = click_coords.shape

        pos = self.positive_click.view(1, 1, self.dim).expand(b, n, -1)
        neg = self.negative_click.view(1, 1, self.dim).expand(b, n, -1)
        pad = self.padding_click.view(1, 1, self.dim).expand(b, n, -1)

        click_tokens = torch.where((click_labels == 1).unsqueeze(-1), pos, pad)
        click_tokens = torch.where((click_labels == 0).unsqueeze(-1), neg, click_tokens)

        roped = self._apply_click_rope(click_tokens, click_coords)
        valid = (click_labels == 1) | (click_labels == 0)
        return torch.where(valid.unsqueeze(-1), roped, click_tokens)

    def _apply_click_rope(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Apply 2D RoPE to continuous click coordinates without center offset."""
        b, n, c = x.shape
        if c % 2 != 0:
            raise ValueError("Click token dim must be even for 2D RoPE")

        x_h, x_w = x.chunk(2, dim=-1)
        dim = c // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))

        rows = coords[..., 0].float()
        cols = coords[..., 1].float()
        h_freqs = rows.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        w_freqs = cols.unsqueeze(-1) * inv_freq.view(1, 1, -1)

        x_h = self._apply_rotary_seq(x_h, h_freqs.cos(), h_freqs.sin())
        x_w = self._apply_rotary_seq(x_w, w_freqs.cos(), w_freqs.sin())
        return torch.cat([x_h, x_w], dim=-1).view(b, n, c)

    @staticmethod
    def _apply_rotary_seq(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_ = rearrange(x, "b n (d2 c) -> b n d2 c", d2=2)
        x_real, x_imag = x_[..., 0, :], x_[..., 1, :]
        rotated_real = cos * x_real - sin * x_imag
        rotated_imag = sin * x_real + cos * x_imag
        rotated = torch.stack([rotated_real, rotated_imag], dim=-2)
        return rearrange(rotated, "b n d2 c -> b n (d2 c)")
