import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from maskvar.models.rope2d import RotaryPositionEmbedding2D
from maskvar.models.simple_mask_vqvae.mask_decoder import SimpleMaskDecoderV2
from maskvar.models.simple_mask_vqvae.simple_mask_vqvae import MaskFeatureCompactor


def diagonal_gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl = 0.5 * (mu.float().pow(2) + logvar.float().exp() - 1.0 - logvar.float())
    return kl.mean()


class SimpleMaskVAEV2(nn.Module):
    """
    Query-token mask VAE.

    The encoder follows SimpleMaskVqvaeV2: mask features are compacted into a
    short query-token sequence. Instead of VQ, each query token is mapped to a
    continuous Gaussian latent. The decoder reuses the V2 image-conditioned
    mask decoder.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        mask_encoder: nn.Module,
        dim: int = 384,
        latent_dim: int = 128,
        num_queries: int = 8,
        beta_kl: float = 1e-4,
        h: int = 64,
        w: int = 64,
        num_heads: int = 4,
        device: str = "cuda",
    ):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.num_queries = num_queries
        self.beta_kl = beta_kl
        self.device = device

        self.rope = RotaryPositionEmbedding2D(h=h, w=w)
        self.image_encoder = image_encoder
        self.mask_encoder = mask_encoder
        self.mask_feature_compactor = MaskFeatureCompactor(
            rope=self.rope,
            dim=dim,
            num_queries=num_queries,
            num_heads=num_heads,
            depth=1,
        )
        self.to_mu = nn.Linear(dim, latent_dim)
        self.to_logvar = nn.Linear(dim, latent_dim)
        self.latent_to_query = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, dim),
        )
        self.mask_decoder = SimpleMaskDecoderV2(
            rope=self.rope,
            dim=dim,
            num_heads=num_heads,
            num_queries=num_queries,
        )

    def encode(self, mask_normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_tokens = self.mask_encoder(mask_normalized)
        mask_tokens = rearrange(mask_tokens, "b c h w -> b h w c")
        query_tokens = self.mask_feature_compactor(mask_tokens)
        return self.to_mu(query_tokens), self.to_logvar(query_tokens).clamp(-20.0, 10.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(
        self,
        z: torch.Tensor,
        image: torch.Tensor | None = None,
        image_tokens: torch.Tensor | None = None,
        output_size=None,
    ) -> torch.Tensor:
        if image_tokens is None:
            if image is None:
                raise ValueError("Either image or image_tokens must be provided.")
            image_tokens = self.image_encoder(image)

        if image_tokens.dim() == 3:
            h = w = int(math.sqrt(image_tokens.shape[1]))
            if h * w != image_tokens.shape[1]:
                raise ValueError(f"Cannot infer square image token grid from length {image_tokens.shape[1]}")
            image_tokens = rearrange(image_tokens, "b (h w) c -> b h w c", h=h, w=w)
        elif image_tokens.dim() == 4 and image_tokens.shape[1] == self.dim:
            image_tokens = rearrange(image_tokens, "b c h w -> b h w c")

        query_tokens = self.latent_to_query(z)
        mask_logits = self.mask_decoder(query_tokens, image_tokens)

        if output_size is not None and mask_logits.shape[-2:] != output_size:
            mask_logits = F.interpolate(mask_logits, size=output_size, mode="bilinear", align_corners=False)
        return mask_logits

    def forward(self, mask_normalized: torch.Tensor, image: torch.Tensor, sample: bool | None = None):
        if sample is None:
            sample = self.training

        output_size = mask_normalized.shape[-2:]
        image_tokens = self.image_encoder(image)
        mu, logvar = self.encode(mask_normalized)
        z = self.reparameterize(mu, logvar, sample=sample)
        mask_logits = self.decode(z, image_tokens=image_tokens, output_size=output_size)
        kl_loss = diagonal_gaussian_kl(mu, logvar)
        return {
            "mask_logits": mask_logits,
            "kl_loss": kl_loss,
            "vae_loss": self.beta_kl * kl_loss,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "image_tokens": image_tokens,
        }

    @torch.no_grad()
    def encode_mask_to_latents(self, mask_normalized: torch.Tensor, sample: bool = False):
        mu, logvar = self.encode(mask_normalized)
        z = self.reparameterize(mu, logvar, sample=sample)
        return z, mu, logvar
