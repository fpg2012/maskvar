import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class LightweightDiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        t = time_emb[:, None, :]
        x_norm = self.norm_self(x + t)
        x = x + self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + self.cross_attn(self.norm_cross(x + t), cond, cond, need_weights=False)[0]
        x = x + self.mlp(self.norm_mlp(x + t))
        return x


class SimpleMaskLatentDiT(nn.Module):
    """
    Lightweight conditional DiT for SimpleMaskVAEV2 query latents.

    It predicts noise in the VAE latent space. Image tokens are compressed to a
    small condition grid with adaptive average pooling to keep the model close
    to the current query-token AR parameter scale.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        dim: int = 384,
        depth: int = 2,
        num_heads: int = 4,
        num_queries: int = 8,
        image_dim: int = 384,
        cond_grid_size: int = 8,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dim = dim
        self.num_queries = num_queries
        self.cond_grid_size = cond_grid_size
        self.num_train_timesteps = num_train_timesteps

        self.latent_in = nn.Linear(latent_dim, dim)
        self.latent_out = nn.Linear(dim, latent_dim)
        self.query_pos = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.image_proj = nn.Linear(image_dim, dim)
        self.time_embed = SinusoidalTimeEmbedding(dim)
        self.blocks = nn.ModuleList([
            LightweightDiTBlock(dim=dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _image_tokens_to_condition(self, image_tokens: torch.Tensor) -> torch.Tensor:
        if image_tokens.dim() == 4:
            if image_tokens.shape[1] == self.image_proj.in_features:
                x = image_tokens
            else:
                x = rearrange(image_tokens, "b h w c -> b c h w")
        elif image_tokens.dim() == 3:
            b, l, _ = image_tokens.shape
            h = w = int(math.sqrt(l))
            if h * w != l:
                raise ValueError(f"Cannot infer square image token grid from length {l}")
            x = rearrange(image_tokens, "b (h w) c -> b c h w", h=h, w=w)
        else:
            raise ValueError(f"Unsupported image token shape: {tuple(image_tokens.shape)}")

        x = F.adaptive_avg_pool2d(x, (self.cond_grid_size, self.cond_grid_size))
        x = rearrange(x, "b c h w -> b (h w) c")
        return self.image_proj(x)

    def q_sample(self, z0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1).to(z0.dtype)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1).to(z0.dtype)
        return sqrt_alpha * z0 + sqrt_one_minus * noise, noise

    def forward(self, z_t: torch.Tensor, timesteps: torch.Tensor, image_tokens: torch.Tensor):
        x = self.latent_in(z_t) + self.query_pos[:, : z_t.shape[1], :]
        cond = self._image_tokens_to_condition(image_tokens)
        time_emb = self.time_embed(timesteps)
        for block in self.blocks:
            x = block(x, cond, time_emb)
        return self.latent_out(self.final_norm(x))

    def diffusion_loss(self, z0: torch.Tensor, image_tokens: torch.Tensor, timesteps: torch.Tensor | None = None):
        b = z0.shape[0]
        if timesteps is None:
            timesteps = torch.randint(0, self.num_train_timesteps, (b,), device=z0.device)
        z_t, noise = self.q_sample(z0, timesteps)
        pred_noise = self(z_t, timesteps, image_tokens)
        return F.mse_loss(pred_noise.float(), noise.float()), pred_noise, noise, timesteps

    @torch.no_grad()
    def sample(self, image_tokens: torch.Tensor, shape: tuple[int, int, int] | None = None, num_steps: int = 50):
        b = image_tokens.shape[0]
        if shape is None:
            shape = (b, self.num_queries, self.latent_dim)
        z = torch.randn(shape, device=image_tokens.device, dtype=image_tokens.dtype)
        step_indices = torch.linspace(self.num_train_timesteps - 1, 0, num_steps, device=image_tokens.device).long()

        for i, t in enumerate(step_indices):
            t_batch = torch.full((b,), int(t.item()), device=image_tokens.device, dtype=torch.long)
            pred_noise = self(z, t_batch, image_tokens)
            alpha_t = self.alphas[t].to(z.dtype)
            alpha_bar_t = self.alphas_cumprod[t].to(z.dtype)
            beta_t = self.betas[t].to(z.dtype)
            z = (z - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t)
            if i < len(step_indices) - 1:
                next_t = step_indices[i + 1]
                sigma = torch.sqrt(self.betas[next_t]).to(z.dtype)
                z = z + sigma * torch.randn_like(z)
        return z
