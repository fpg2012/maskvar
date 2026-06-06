import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from ..rope2d import RotaryPositionEmbedding2D
from ..simple_mask_vqvae.mask_decoder import SimpleMaskDecoderV2
from ..simple_mask_vqvae.basic import MLP
from .click_encoder import RopeClickEncoder


class RopeSAM(nn.Module):
    """A non-generative click-conditioned segmentation model."""

    def __init__(
        self,
        image_encoder: nn.Module,
        dim: int = 384,
        h: int = 64,
        w: int = 64,
        num_heads: int = 4,
        max_clicks: int = 10,
        num_two_way_blocks: int = 2,
        device: str = "cuda",
    ):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.max_clicks = max_clicks
        self.image_encoder = image_encoder
        self.click_encoder = RopeClickEncoder(dim=dim, h=h, w=w)
        self.seg_token = nn.Parameter(torch.randn(1, 1, dim))
        self.prev_mask_encoder = nn.Sequential(
            nn.Conv2d(1, dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
        )
        self.rope = RotaryPositionEmbedding2D(h=h, w=w)
        self.mask_decoder = SimpleMaskDecoderV2(
            rope=self.rope,
            dim=dim,
            num_heads=num_heads,
            num_queries=max_clicks + 1,
            num_two_way_blocks=num_two_way_blocks,
        )
        self.device = device

    def get_device(self):
        return next(self.parameters()).device

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_encoder(image)
        if image_tokens.dim() != 4:
            raise ValueError(f"Expected image encoder output (B, C, H, W), got {tuple(image_tokens.shape)}")
        return rearrange(image_tokens, "b c h w -> b h w c")

    def encode_image_embedding(self, image_embedding: torch.Tensor) -> torch.Tensor:
        if image_embedding.dim() != 4:
            raise ValueError(f"Expected cached image embedding (B, C, H, W), got {tuple(image_embedding.shape)}")
        return rearrange(image_embedding, "b c h w -> b h w c")

    def encode_prev_mask(self, prev_mask_logits: torch.Tensor | None, spatial_shape: tuple[int, int]) -> torch.Tensor | None:
        if prev_mask_logits is None:
            return None
        if prev_mask_logits.dim() != 4 or prev_mask_logits.shape[1] != 1:
            raise ValueError(f"Expected prev_mask_logits shape (B, 1, H, W), got {tuple(prev_mask_logits.shape)}")
        prev_mask_logits = prev_mask_logits.float()
        if prev_mask_logits.shape[-2:] != spatial_shape:
            prev_mask_logits = F.interpolate(prev_mask_logits, size=spatial_shape, mode="bilinear", align_corners=False)
        prev_tokens = self.prev_mask_encoder(prev_mask_logits)
        return rearrange(prev_tokens, "b c h w -> b h w c")

    def encode_clicks(self, click_coords: torch.Tensor, click_labels: torch.Tensor) -> torch.Tensor:
        click_tokens = self.click_encoder(click_coords, click_labels)
        seg_token = self.seg_token.expand(click_tokens.shape[0], -1, -1)
        return torch.cat([seg_token, click_tokens], dim=1)

    def forward(
        self,
        image: torch.Tensor,
        click_coords: torch.Tensor,
        click_labels: torch.Tensor,
        prev_mask_logits: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
        image_embedding: torch.Tensor | None = None,
        point_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W)
            image_embedding: Optional cached image encoder output, (B, C, H, W).
            click_coords: (B, N, 2), row/col coordinates in the token grid.
            click_labels: (B, N), 1 positive, 0 negative, -1 padding.
            prev_mask_logits: Optional previous-step mask logits used as dense prompt.
            output_size: Optional mask-logit output size.
        """
        if image_embedding is None:
            image_tokens = self.encode_image(image)
        else:
            image_tokens = self.encode_image_embedding(image_embedding)
        prev_mask_tokens = self.encode_prev_mask(prev_mask_logits, image_tokens.shape[1:3])
        if prev_mask_tokens is not None:
            image_tokens = image_tokens + prev_mask_tokens.to(image_tokens.dtype)
        query_tokens = self.encode_clicks(click_coords, click_labels)
        mask_logits = self.mask_decoder(query_tokens, image_tokens)

        if output_size is None:
            output_size = image.shape[-2:]
        if mask_logits.shape[-2:] != output_size:
            mask_logits = F.interpolate(mask_logits, size=output_size, mode="bilinear", align_corners=False)
        return mask_logits


class PointCrossBlock(nn.Module):
    """Cross-attention from sequence queries to point tokens with continuous RoPE on point keys."""

    def __init__(self, dim: int, num_heads: int, rope_coord_offset: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.rope_coord_offset = rope_coord_offset
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.linear_q = nn.Linear(dim, dim)
        self.linear_kv = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.layernorm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_coords: torch.Tensor) -> torch.Tensor:
        q_input = q
        q = self.linear_q(q)
        kv = self.linear_kv(kv)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, "b l (nh c) -> b nh l c", nh=self.num_heads, c=self.dim_head)
        k = rearrange(k, "b n (nh c) -> b nh n c", nh=self.num_heads, c=self.dim_head)
        v = rearrange(v, "b n (nh c) -> b nh n c", nh=self.num_heads, c=self.dim_head)
        k = apply_point_rope(k, kv_coords, coord_offset=self.rope_coord_offset)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b nh l c -> b l (nh c)")
        out = q_input + self.out_proj(out)
        out = out + self.ffn(self.layernorm(out))
        return out


class PointCrossBlockReverse(nn.Module):
    """Cross-attention from point tokens to sequence tokens with continuous RoPE on point queries."""

    def __init__(self, dim: int, num_heads: int, rope_coord_offset: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.rope_coord_offset = rope_coord_offset
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.linear_q = nn.Linear(dim, dim)
        self.linear_kv = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.layernorm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, q: torch.Tensor, q_coords: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_input = q
        q = self.linear_q(q)
        kv = self.linear_kv(kv)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, "b n (nh c) -> b nh n c", nh=self.num_heads, c=self.dim_head)
        k = rearrange(k, "b l (nh c) -> b nh l c", nh=self.num_heads, c=self.dim_head)
        v = rearrange(v, "b l (nh c) -> b nh l c", nh=self.num_heads, c=self.dim_head)
        q = apply_point_rope(q, q_coords, coord_offset=self.rope_coord_offset)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b nh n c -> b n (nh c)")
        out = q_input + self.out_proj(out)
        out = out + self.ffn(self.layernorm(out))
        return out


class PointTwoWayBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_coord_offset: float = 0.5):
        super().__init__()
        self.query_to_points = PointCrossBlock(dim=dim, num_heads=num_heads, rope_coord_offset=rope_coord_offset)
        self.points_to_query = PointCrossBlockReverse(dim=dim, num_heads=num_heads, rope_coord_offset=rope_coord_offset)

    def forward(
        self,
        query_tokens: torch.Tensor,
        point_tokens: torch.Tensor,
        point_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_tokens = self.query_to_points(query_tokens, point_tokens, point_coords)
        point_tokens = self.points_to_query(point_tokens, point_coords, query_tokens)
        return query_tokens, point_tokens


class PointMaskDecoder(nn.Module):
    """Decode sparse image point tokens into per-point logits."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_two_way_blocks: int = 2,
        use_point_head: bool = False,
        point_head_dim: int = 64,
        rope_coord_offset: float = 0.5,
    ):
        super().__init__()
        self.use_point_head = use_point_head
        self.point_head_dim = point_head_dim
        self.two_way_blocks = nn.ModuleList([
            PointTwoWayBlock(dim=dim, num_heads=num_heads, rope_coord_offset=rope_coord_offset)
            for _ in range(num_two_way_blocks)
        ])
        if use_point_head:
            self.mask_head = MLP(dim, dim, point_head_dim, 3)
            self.point_head = MLP(dim, dim, point_head_dim, 3)
            self.query_norm = nn.LayerNorm(point_head_dim)
            self.point_norm = nn.LayerNorm(point_head_dim)

    def forward(
        self,
        query_tokens: torch.Tensor,
        point_tokens: torch.Tensor,
        point_coords: torch.Tensor,
    ) -> torch.Tensor:
        for blk in self.two_way_blocks:
            query_tokens, point_tokens = blk(query_tokens, point_tokens, point_coords)

        mask_token = query_tokens[:, 0]
        if self.use_point_head:
            mask_token = self.query_norm(self.mask_head(mask_token))
            point_tokens = self.point_norm(self.point_head(point_tokens))
        return torch.einsum("bc,bnc->bn", mask_token, point_tokens)


class PointRopeSAM(RopeSAM):
    """RopeSAM variant that replaces dense image embeddings with edge-sampled point tokens."""

    def __init__(
        self,
        image_encoder: nn.Module,
        dim: int = 384,
        h: int = 64,
        w: int = 64,
        num_heads: int = 4,
        max_clicks: int = 10,
        num_two_way_blocks: int = 2,
        num_points: int | None = None,
        density_floor: float = 0.05,
        sampling_strategy: str = "uniform",
        interpolation_k: int = 8,
        interpolation_power: float = 2.0,
        interpolation_chunk_size: int = 1024,
        use_point_head: bool = False,
        point_head_dim: int = 64,
        rope_coord_offset: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__(
            image_encoder=image_encoder,
            dim=dim,
            h=h,
            w=w,
            num_heads=num_heads,
            max_clicks=max_clicks,
            num_two_way_blocks=num_two_way_blocks,
            device=device,
        )
        self.num_points = num_points or h * w
        self.density_floor = density_floor
        self.sampling_strategy = sampling_strategy
        self.interpolation_k = interpolation_k
        self.interpolation_power = interpolation_power
        self.interpolation_chunk_size = interpolation_chunk_size
        self.use_point_head = use_point_head
        self.point_head_dim = point_head_dim
        self.rope_coord_offset = rope_coord_offset
        self.mask_decoder = PointMaskDecoder(
            dim=dim,
            num_heads=num_heads,
            num_two_way_blocks=num_two_way_blocks,
            use_point_head=use_point_head,
            point_head_dim=point_head_dim,
            rope_coord_offset=rope_coord_offset,
        )
        self.register_buffer("gaussian_kernel", _make_gaussian_kernel(), persistent=False)
        self.register_buffer("sobel_x", _make_sobel_kernel("x"), persistent=False)
        self.register_buffer("sobel_y", _make_sobel_kernel("y"), persistent=False)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_encoder(image)
        if image_tokens.dim() != 4:
            raise ValueError(f"Expected image encoder output (B, C, H, W), got {tuple(image_tokens.shape)}")
        return image_tokens

    def encode_image_embedding(self, image_embedding: torch.Tensor) -> torch.Tensor:
        if image_embedding.dim() != 4:
            raise ValueError(f"Expected cached image embedding (B, C, H, W), got {tuple(image_embedding.shape)}")
        return image_embedding

    def encode_prev_mask(self, prev_mask_logits: torch.Tensor | None, spatial_shape: tuple[int, int]) -> torch.Tensor | None:
        if prev_mask_logits is None:
            return None
        if prev_mask_logits.dim() != 4 or prev_mask_logits.shape[1] != 1:
            raise ValueError(f"Expected prev_mask_logits shape (B, 1, H, W), got {tuple(prev_mask_logits.shape)}")
        prev_mask_logits = prev_mask_logits.float()
        if prev_mask_logits.shape[-2:] != spatial_shape:
            prev_mask_logits = F.interpolate(prev_mask_logits, size=spatial_shape, mode="bilinear", align_corners=False)
        return self.prev_mask_encoder(prev_mask_logits)

    def forward(
        self,
        image: torch.Tensor,
        click_coords: torch.Tensor,
        click_labels: torch.Tensor,
        prev_mask_logits: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
        image_embedding: torch.Tensor | None = None,
        point_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_embedding is None:
            image_features = self.encode_image(image)
        else:
            image_features = self.encode_image_embedding(image_embedding)

        if point_coords is None:
            point_coords = self.sample_point_coords(image)
        else:
            point_coords = point_coords.to(device=image_features.device, dtype=torch.float32)
        point_tokens = sample_feature_points(image_features, point_coords, self.h, self.w)

        prev_mask_features = self.encode_prev_mask(prev_mask_logits, image_features.shape[-2:])
        if prev_mask_features is not None:
            point_tokens = point_tokens + sample_feature_points(prev_mask_features, point_coords, self.h, self.w).to(point_tokens.dtype)

        query_tokens = self.encode_clicks(click_coords, click_labels)
        point_logits = self.mask_decoder(query_tokens, point_tokens, point_coords)

        if output_size is None:
            output_size = image.shape[-2:]
        return knn_interpolate_point_logits(
            point_logits=point_logits,
            point_coords=point_coords,
            output_size=output_size,
            coord_h=self.h,
            coord_w=self.w,
            k=self.interpolation_k,
            power=self.interpolation_power,
            chunk_size=self.interpolation_chunk_size,
        )

    @torch.no_grad()
    def sample_point_coords(self, image: torch.Tensor) -> torch.Tensor:
        if self.sampling_strategy == "uniform":
            return self.uniform_point_coords(image.shape[0], image.device)
        if self.sampling_strategy != "edge":
            raise ValueError(f"Unknown point sampling strategy: {self.sampling_strategy}")
        density = self.edge_density(image)
        b, _, h, w = density.shape
        probs = density.flatten(1)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        sample_count = min(self.num_points, probs.shape[1])
        indices = torch.multinomial(probs, num_samples=sample_count, replacement=False)

        rows = (indices // w).to(torch.float32)
        cols = (indices % w).to(torch.float32)
        if self.training:
            rows = rows + torch.rand_like(rows) - 0.5
            cols = cols + torch.rand_like(cols) - 0.5
        rows = rows.clamp(0, h - 1) * ((self.h - 1) / max(h - 1, 1))
        cols = cols.clamp(0, w - 1) * ((self.w - 1) / max(w - 1, 1))
        return torch.stack([rows, cols], dim=-1)

    @torch.no_grad()
    def uniform_point_coords(self, batch_size: int, device: torch.device) -> torch.Tensor:
        rows = torch.arange(self.h, device=device, dtype=torch.float32)
        cols = torch.arange(self.w, device=device, dtype=torch.float32)
        grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([grid_rows.reshape(-1), grid_cols.reshape(-1)], dim=-1)
        if coords.shape[0] > self.num_points:
            coords = coords[: self.num_points]
        return coords.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    @torch.no_grad()
    def edge_density(self, image: torch.Tensor) -> torch.Tensor:
        gray = image.float().mean(dim=1, keepdim=True)
        gray = self._blur(gray)
        grad_x = F.conv2d(gray, self.sobel_x.to(gray), padding=1)
        grad_y = F.conv2d(gray, self.sobel_y.to(gray), padding=1)
        density = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        density = self._blur(density)
        density = density - density.flatten(1).amin(dim=1).view(-1, 1, 1, 1)
        max_val = density.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        density = density / max_val.clamp_min(1e-6)
        return density + self.density_floor

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.gaussian_kernel.to(device=x.device, dtype=x.dtype)
        return F.conv2d(x, kernel, padding=2)


def apply_point_rope(x: torch.Tensor, coords: torch.Tensor, coord_offset: float = 0.5) -> torch.Tensor:
    """Apply 2D RoPE to point-token tensors shaped (B, heads, N, C_head)."""
    b, nh, n, c = x.shape
    if c % 2 != 0:
        raise ValueError("RoPE head dim must be even")

    x_h, x_w = x.chunk(2, dim=-1)
    dim = c // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    rows = coords[..., 0].to(device=x.device, dtype=torch.float32) + coord_offset
    cols = coords[..., 1].to(device=x.device, dtype=torch.float32) + coord_offset
    h_freqs = rows.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    w_freqs = cols.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    x_h = _apply_rotary_points(x_h, h_freqs.cos(), h_freqs.sin())
    x_w = _apply_rotary_points(x_w, w_freqs.cos(), w_freqs.sin())
    return torch.cat([x_h, x_w], dim=-1).view(b, nh, n, c)


def _apply_rotary_points(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_ = rearrange(x, "b nh n (d2 c) -> b nh n d2 c", d2=2)
    x_real, x_imag = x_[..., 0, :], x_[..., 1, :]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotated_real = cos * x_real - sin * x_imag
    rotated_imag = sin * x_real + cos * x_imag
    rotated = torch.stack([rotated_real, rotated_imag], dim=-2)
    return rearrange(rotated, "b nh n d2 c -> b nh n (d2 c)")


def sample_feature_points(features: torch.Tensor, coords: torch.Tensor, coord_h: int, coord_w: int) -> torch.Tensor:
    rows = coords[..., 0]
    cols = coords[..., 1]
    grid_y = rows / max(coord_h - 1, 1) * 2 - 1
    grid_x = cols / max(coord_w - 1, 1) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return rearrange(sampled.squeeze(-1), "b c n -> b n c")


def knn_interpolate_point_logits(
    point_logits: torch.Tensor,
    point_coords: torch.Tensor,
    output_size: tuple[int, int],
    coord_h: int,
    coord_w: int,
    k: int = 8,
    power: float = 2.0,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Interpolate point logits to every output pixel by inverse-distance kNN weighting."""
    b, n = point_logits.shape
    out_h, out_w = output_size
    k = min(k, n)
    rows = (torch.arange(out_h, device=point_logits.device, dtype=torch.float32) + 0.5) * (coord_h / out_h) - 0.5
    cols = (torch.arange(out_w, device=point_logits.device, dtype=torch.float32) + 0.5) * (coord_w / out_w) - 0.5
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
    pixel_coords = torch.stack([grid_rows.reshape(-1), grid_cols.reshape(-1)], dim=-1)

    point_coords = point_coords.to(device=point_logits.device, dtype=torch.float32)
    point_logits_float = point_logits.float()
    output_chunks = []
    for start in range(0, pixel_coords.shape[0], chunk_size):
        pixel_chunk = pixel_coords[start:start + chunk_size].view(1, -1, 1, 2)
        delta = pixel_chunk - point_coords.view(b, 1, n, 2)
        dist2 = delta.square().sum(dim=-1)
        knn_dist2, knn_idx = dist2.topk(k=k, dim=-1, largest=False)
        knn_logits = point_logits_float.gather(1, knn_idx.reshape(b, -1)).view(b, -1, k)
        weights = (knn_dist2 + 1e-6).pow(-0.5 * power)
        chunk_logits = (knn_logits * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-6)
        output_chunks.append(chunk_logits)

    logits = torch.cat(output_chunks, dim=1).view(b, 1, out_h, out_w)
    return logits.to(dtype=point_logits.dtype).contiguous()


def _make_gaussian_kernel() -> torch.Tensor:
    kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, 5, 5)


def _make_sobel_kernel(axis: str) -> torch.Tensor:
    if axis == "x":
        kernel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    elif axis == "y":
        kernel = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    else:
        raise ValueError(f"Unknown Sobel axis: {axis}")
    return kernel.view(1, 1, 3, 3)
