import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from ..rope2d import RotaryPositionEmbedding2D
from ..simple_mask_vqvae.mask_decoder import SimpleMaskDecoderV2
from ..simple_mask_vqvae.basic import MLP, SimpleCrossBlock
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


class QuerySelfBlock(nn.Module):
    """Self-attention over query tokens with the same parameter shape as SimpleCrossBlock."""

    def __init__(self, rope: RotaryPositionEmbedding2D, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.linear_q = nn.Linear(dim, dim)
        self.linear_kv = nn.Linear(dim, dim * 2)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.layernorm = nn.LayerNorm(dim)
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim_head = dim // num_heads

    def forward(self, query_tokens: torch.Tensor) -> torch.Tensor:
        q_input = query_tokens
        q = self.linear_q(query_tokens)
        kv = self.linear_kv(query_tokens)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, "b l (nh c) -> b nh l c", nh=self.num_heads, c=self.dim_head)
        k = rearrange(k, "b l (nh c) -> b nh l c", nh=self.num_heads, c=self.dim_head)
        v = rearrange(v, "b l (nh c) -> b nh l c", nh=self.num_heads, c=self.dim_head)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b nh l c -> b l (nh c)")
        out = q_input + self.out_proj(out)
        out = out + self.ffn(self.layernorm(out))
        return out


class NoImageUpdateTwoWayBlock(nn.Module):
    """Query self-attention plus query-to-image cross-attention with static image tokens."""

    def __init__(self, rope: RotaryPositionEmbedding2D, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.query_self_attn = QuerySelfBlock(rope, dim, num_heads)
        self.block2 = SimpleCrossBlock(rope, dim, num_heads)

    def forward(self, query_tokens: torch.Tensor, image_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_tokens = self.query_self_attn(query_tokens)
        query_tokens = self.block2(query_tokens, image_tokens, pe_type="rope")
        return query_tokens, image_tokens


class NoImageUpdateMaskDecoder(SimpleMaskDecoderV2):
    """
    Dense RopeSAM decoder ablation that preserves SimpleMaskDecoderV2 parameter
    count but never updates image tokens inside the transformer stack.
    """

    def __init__(
        self,
        rope: RotaryPositionEmbedding2D,
        dim: int,
        num_heads: int = 4,
        num_queries: int = 4,
        num_two_way_blocks: int = 2,
        num_register_tokens: int = 0,
    ):
        super().__init__(
            rope=rope,
            dim=dim,
            num_heads=num_heads,
            num_queries=num_queries,
            num_two_way_blocks=num_two_way_blocks,
        )
        self.num_register_tokens = num_register_tokens
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, dim))
        else:
            self.register_parameter("register_tokens", None)
        self.two_way_blocks = nn.ModuleList([
            NoImageUpdateTwoWayBlock(rope=self.rope, dim=dim, num_heads=num_heads)
            for _ in range(num_two_way_blocks)
        ])

    def forward(self, query_tokens: torch.Tensor, image_tokens: torch.Tensor):
        original_query_count = query_tokens.shape[1]
        if self.register_tokens is not None:
            register_tokens = self.register_tokens.expand(query_tokens.shape[0], -1, -1)
            query_tokens = torch.cat([query_tokens, register_tokens], dim=1)

        for blk in self.two_way_blocks:
            query_tokens, image_tokens = blk(query_tokens, image_tokens)

        query_tokens = query_tokens[:, :original_query_count]
        image_feature_map = rearrange(image_tokens, 'b h w c -> b c h w')
        up_query_token = self.hyper_in(query_tokens[:, 0, :])
        up_image_map = self.output_upscaling(image_feature_map)

        up_query_token = self.layer_norm_post_query(up_query_token)
        up_image_map = self.layer_norm_post_image(rearrange(up_image_map, 'b c h w -> b h w c'))

        masks = torch.einsum('bc,bhwc->bhw', up_query_token, up_image_map).unsqueeze(1)

        return masks.contiguous()


class NoTwoWayRopeSAM(RopeSAM):
    """
    RopeSAM ablation where query tokens attend to image tokens repeatedly, while
    image tokens stay fixed throughout decoder attention.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        dim: int = 384,
        h: int = 64,
        w: int = 64,
        num_heads: int = 4,
        max_clicks: int = 10,
        num_two_way_blocks: int = 2,
        num_register_tokens: int = 0,
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
        self.mask_decoder = NoImageUpdateMaskDecoder(
            rope=self.rope,
            dim=dim,
            num_heads=num_heads,
            num_queries=max_clicks + 1,
            num_two_way_blocks=num_two_way_blocks,
            num_register_tokens=num_register_tokens,
        )


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
        point_rend_coarse_size: int = 16,
        point_rend_max_size: int = 256,
        point_sampling_space: str = "feature",
        click_point_radius: float = 2.0,
        click_point_grid_size: int = 5,
        ignore_padding_in_sampling: bool = True,
        interpolation_k: int = 4,
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
        self.point_rend_coarse_size = point_rend_coarse_size
        self.point_rend_max_size = point_rend_max_size
        if point_sampling_space not in {"feature", "output"}:
            raise ValueError(f"Unknown point sampling space: {point_sampling_space}")
        self.point_sampling_space = point_sampling_space
        self.click_point_radius = click_point_radius
        self.click_point_grid_size = click_point_grid_size
        self.ignore_padding_in_sampling = ignore_padding_in_sampling
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
        return_coarse_logits: bool = False,
    ) -> torch.Tensor:
        if image_embedding is None:
            image_features = self.encode_image(image)
        else:
            image_features = self.encode_image_embedding(image_embedding)

        prev_mask_features = self.encode_prev_mask(prev_mask_logits, image_features.shape[-2:])
        query_tokens = self.encode_clicks(click_coords, click_labels)

        if output_size is None:
            output_size = image.shape[-2:]
        point_coord_size = output_size if self.point_sampling_space == "output" else (self.h, self.w)
        if point_coords is None and self.sampling_strategy == "pointrend":
            if self.ignore_padding_in_sampling:
                valid_mask = torch.ones(image.shape[0], point_coord_size[0], point_coord_size[1], device=image.device, dtype=torch.bool)
            else:
                valid_mask = self.valid_coord_mask(image, point_coord_size)
            logits = self.pointrend_refine_logits(
                image_features=image_features,
                query_tokens=query_tokens,
                valid_mask=valid_mask,
                coord_size=point_coord_size,
                output_size=output_size,
                prev_mask_features=prev_mask_features,
            )
            if not return_coarse_logits:
                return logits
            coarse_logits = F.interpolate(logits.float(), size=(self.h, self.w), mode="bilinear", align_corners=False)
            return logits, coarse_logits.to(dtype=logits.dtype)

        if point_coords is None:
            with torch.no_grad():
                point_coords_for_interp = self.sample_point_coords(
                    image=image,
                    image_features=image_features,
                    query_tokens=query_tokens,
                    prev_mask_features=prev_mask_features,
                    click_coords=click_coords,
                    click_labels=click_labels,
                    coord_size=point_coord_size,
                )
        else:
            point_coords_for_interp = point_coords.to(device=image_features.device, dtype=torch.float32)
        point_coords = self.scale_point_coords(point_coords_for_interp, point_coord_size, (self.h, self.w))
        point_tokens = sample_feature_points(image_features, point_coords, self.h, self.w)

        if prev_mask_features is not None:
            point_tokens = point_tokens + sample_feature_points(prev_mask_features, point_coords, self.h, self.w).to(point_tokens.dtype)

        point_logits = self.mask_decoder(query_tokens, point_tokens, point_coords)

        logits = knn_interpolate_point_logits(
            point_logits=point_logits,
            point_coords=point_coords_for_interp,
            output_size=output_size,
            coord_h=point_coord_size[0],
            coord_w=point_coord_size[1],
            k=self.interpolation_k,
            power=self.interpolation_power,
            chunk_size=self.interpolation_chunk_size,
        )
        if not return_coarse_logits:
            return logits
        coarse_logits = knn_interpolate_point_logits(
            point_logits=point_logits,
            point_coords=point_coords_for_interp,
            output_size=(self.h, self.w),
            coord_h=point_coord_size[0],
            coord_w=point_coord_size[1],
            k=self.interpolation_k,
            power=self.interpolation_power,
            chunk_size=self.interpolation_chunk_size,
        )
        return logits, coarse_logits

    @torch.no_grad()
    def sample_point_coords(
        self,
        image: torch.Tensor,
        image_features: torch.Tensor | None = None,
        query_tokens: torch.Tensor | None = None,
        prev_mask_features: torch.Tensor | None = None,
        click_coords: torch.Tensor | None = None,
        click_labels: torch.Tensor | None = None,
        coord_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        coord_size = coord_size or (self.h, self.w)
        if self.ignore_padding_in_sampling:
            valid_mask = torch.ones(image.shape[0], coord_size[0], coord_size[1], device=image.device, dtype=torch.bool)
        else:
            valid_mask = self.valid_coord_mask(image, coord_size)
        if self.sampling_strategy == "uniform":
            return self.uniform_point_coords(valid_mask)
        if self.sampling_strategy == "pointrend":
            if image_features is None or query_tokens is None:
                return self.grid_coords_for_size(valid_mask, self.point_rend_coarse_size, max_points=self.num_points)
            return self.pointrend_point_coords(
                image_features=image_features,
                query_tokens=query_tokens,
                valid_mask=valid_mask,
                prev_mask_features=prev_mask_features,
                click_coords=click_coords,
                click_labels=click_labels,
                coord_size=coord_size,
            )
        if self.sampling_strategy != "edge":
            raise ValueError(f"Unknown point sampling strategy: {self.sampling_strategy}")
        density = self.edge_density(image)
        valid_pixel_mask = self.valid_pixel_mask(image)
        density = density * valid_pixel_mask.to(density.dtype)
        b, _, h, w = density.shape
        probs = density.flatten(1)
        fallback = probs.sum(dim=1, keepdim=True) <= 1e-6
        if fallback.any():
            probs = torch.where(fallback, valid_pixel_mask.flatten(1).to(probs.dtype), probs)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        sample_count = min(self.num_points, probs.shape[1])
        replacement = sample_count > (probs > 0).sum(dim=1).amin().item()
        indices = torch.multinomial(probs, num_samples=sample_count, replacement=replacement)

        rows = (indices // w).to(torch.float32)
        cols = (indices % w).to(torch.float32)
        if self.training:
            rows = rows + torch.rand_like(rows) - 0.5
            cols = cols + torch.rand_like(cols) - 0.5
        rows = rows.clamp(0, h - 1) * ((coord_size[0] - 1) / max(h - 1, 1))
        cols = cols.clamp(0, w - 1) * ((coord_size[1] - 1) / max(w - 1, 1))
        return torch.stack([rows, cols], dim=-1)

    @torch.no_grad()
    def uniform_point_coords(self, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.select_valid_coords(valid_mask, self.num_points, mode="uniform")

    @torch.no_grad()
    def coarse_point_coords(self, valid_mask: torch.Tensor, num_points: int) -> torch.Tensor:
        return self.select_valid_coords(valid_mask, num_points, mode="linspace")

    @torch.no_grad()
    def pointrend_point_coords(
        self,
        image_features: torch.Tensor,
        query_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        prev_mask_features: torch.Tensor | None = None,
        click_coords: torch.Tensor | None = None,
        click_labels: torch.Tensor | None = None,
        coord_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        coord_size = coord_size or (self.h, self.w)
        max_refine_size = coord_size if self.point_sampling_space == "output" else (self.point_rend_max_size, self.point_rend_max_size)
        coarse_size = (
            max(1, min(self.point_rend_coarse_size, max_refine_size[0])),
            max(1, min(self.point_rend_coarse_size, max_refine_size[1])),
        )
        coords = self.grid_coords_for_size(valid_mask, coarse_size, max_points=self.num_points)
        click_coords_to_add = self.click_neighborhood_coords(click_coords, click_labels, valid_mask, coord_size=coord_size)
        if click_coords_to_add is not None:
            coords = self.merge_point_coords(coords, click_coords_to_add)
            coords = coords[:, : self.num_points]
        current_size = coarse_size

        while coords.shape[1] < self.num_points and current_size != max_refine_size:
            feature_coords = self.scale_point_coords(coords, coord_size, (self.h, self.w))
            point_tokens = sample_feature_points(image_features, feature_coords, self.h, self.w)
            if prev_mask_features is not None:
                point_tokens = point_tokens + sample_feature_points(prev_mask_features, feature_coords, self.h, self.w).to(point_tokens.dtype)
            point_logits = self.mask_decoder(query_tokens, point_tokens, feature_coords)
            dense_logits = knn_interpolate_point_logits(
                point_logits=point_logits,
                point_coords=coords,
                output_size=coord_size,
                coord_h=coord_size[0],
                coord_w=coord_size[1],
                k=min(self.interpolation_k, coords.shape[1]),
                power=self.interpolation_power,
                chunk_size=self.interpolation_chunk_size,
            )[:, 0]

            next_size = (
                min(current_size[0] * 2, max_refine_size[0]),
                min(current_size[1] * 2, max_refine_size[1]),
            )
            candidate_coords = self.grid_coords_for_size(valid_mask, next_size)
            add_budget = min(coords.shape[1], candidate_coords.shape[1], self.num_points - coords.shape[1])
            new_coords = self.select_uncertain_new_coords(
                dense_logits=dense_logits,
                current_coords=coords,
                candidate_coords=candidate_coords,
                valid_mask=valid_mask,
                count=add_budget,
            )
            coords = self.merge_point_coords(coords, new_coords)
            current_size = next_size

        return coords[:, : self.num_points]

    def pointrend_refine_logits(
        self,
        image_features: torch.Tensor,
        query_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        coord_size: tuple[int, int],
        output_size: tuple[int, int],
        prev_mask_features: torch.Tensor | None = None,
        return_point_coords: bool = False,
    ) -> torch.Tensor:
        max_refine_size = coord_size if self.point_sampling_space == "output" else (self.point_rend_max_size, self.point_rend_max_size)
        current_size = (
            max(1, min(self.point_rend_coarse_size, max_refine_size[0])),
            max(1, min(self.point_rend_coarse_size, max_refine_size[1])),
        )
        coords = self.grid_coords_for_size(valid_mask, current_size)
        point_logits = self.decode_point_logits(
            image_features=image_features,
            query_tokens=query_tokens,
            point_coords=coords,
            coord_size=coord_size,
            prev_mask_features=prev_mask_features,
        )
        if coords.shape[1] != current_size[0] * current_size[1]:
            logits = knn_interpolate_point_logits(
                point_logits=point_logits,
                point_coords=coords,
                output_size=output_size,
                coord_h=coord_size[0],
                coord_w=coord_size[1],
                k=self.interpolation_k,
                power=self.interpolation_power,
                chunk_size=self.interpolation_chunk_size,
            )
            if return_point_coords:
                return logits, coords
            return logits
        logit_map = point_logits.view(point_logits.shape[0], 1, current_size[0], current_size[1])
        current_coords = coords
        point_count = current_coords.shape[1]

        while point_count < self.num_points and current_size != max_refine_size:
            next_size = (
                min(current_size[0] * 2, max_refine_size[0]),
                min(current_size[1] * 2, max_refine_size[1]),
            )
            next_logit_map = F.interpolate(logit_map.float(), size=next_size, mode="bilinear", align_corners=False).to(logit_map.dtype)
            candidate_coords = self.grid_coords_for_size(valid_mask, next_size)
            add_budget = min(point_count, candidate_coords.shape[1], self.num_points - point_count)
            with torch.no_grad():
                dense_logits = F.interpolate(next_logit_map.detach().float(), size=coord_size, mode="bilinear", align_corners=False)[:, 0]
                new_coords = self.select_uncertain_new_coords(
                    dense_logits=dense_logits,
                    current_coords=current_coords,
                    candidate_coords=candidate_coords,
                    valid_mask=valid_mask,
                    count=add_budget,
                )
            new_logits = self.decode_point_logits(
                image_features=image_features,
                query_tokens=query_tokens,
                point_coords=new_coords,
                coord_size=coord_size,
                prev_mask_features=prev_mask_features,
            )
            logit_map = self.scatter_point_logits_to_grid(next_logit_map, new_logits, new_coords, coord_size)
            current_coords = self.merge_point_coords(current_coords, new_coords)
            point_count = min(current_coords.shape[1], self.num_points)
            current_size = next_size

        if current_size != output_size:
            logit_map = F.interpolate(logit_map.float(), size=output_size, mode="bilinear", align_corners=False).to(logit_map.dtype)
        if return_point_coords:
            return logit_map.contiguous(), current_coords[:, :point_count]
        return logit_map.contiguous()

    def decode_point_logits(
        self,
        image_features: torch.Tensor,
        query_tokens: torch.Tensor,
        point_coords: torch.Tensor,
        coord_size: tuple[int, int],
        prev_mask_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature_coords = self.scale_point_coords(point_coords, coord_size, (self.h, self.w))
        point_tokens = sample_feature_points(image_features, feature_coords, self.h, self.w)
        if prev_mask_features is not None:
            point_tokens = point_tokens + sample_feature_points(prev_mask_features, feature_coords, self.h, self.w).to(point_tokens.dtype)
        return self.mask_decoder(query_tokens, point_tokens, feature_coords)

    def scatter_point_logits_to_grid(
        self,
        logit_map: torch.Tensor,
        point_logits: torch.Tensor,
        point_coords: torch.Tensor,
        coord_size: tuple[int, int],
    ) -> torch.Tensor:
        b, _, grid_h, grid_w = logit_map.shape
        rows = ((point_coords[..., 0] + 0.5) * (grid_h / coord_size[0]) - 0.5).round().long().clamp(0, grid_h - 1)
        cols = ((point_coords[..., 1] + 0.5) * (grid_w / coord_size[1]) - 0.5).round().long().clamp(0, grid_w - 1)
        flat_idx = rows * grid_w + cols
        flat_logits = logit_map.flatten(2).clone()
        for sample_idx in range(b):
            flat_logits[sample_idx, 0, flat_idx[sample_idx]] = point_logits[sample_idx]
        return flat_logits.view_as(logit_map)

    @torch.no_grad()
    def scale_point_coords(
        self,
        coords: torch.Tensor,
        from_size: tuple[int, int],
        to_size: tuple[int, int],
    ) -> torch.Tensor:
        if from_size == to_size:
            return coords
        scaled = coords.clone()
        scaled[..., 0] = scaled[..., 0] * ((to_size[0] - 1) / max(from_size[0] - 1, 1))
        scaled[..., 1] = scaled[..., 1] * ((to_size[1] - 1) / max(from_size[1] - 1, 1))
        return scaled

    @torch.no_grad()
    def click_neighborhood_coords(
        self,
        click_coords: torch.Tensor | None,
        click_labels: torch.Tensor | None,
        valid_mask: torch.Tensor,
        coord_size: tuple[int, int] | None = None,
    ) -> torch.Tensor | None:
        if click_coords is None or click_labels is None or self.click_point_grid_size <= 0:
            return None
        coord_size = coord_size or (self.h, self.w)
        click_coords = click_coords.to(device=valid_mask.device, dtype=torch.float32)
        click_labels = click_labels.to(device=valid_mask.device)
        click_coords = self.scale_point_coords(click_coords, (self.h, self.w), coord_size)
        radius_scale = max(coord_size[0] / self.h, coord_size[1] / self.w)
        offsets_1d = torch.linspace(
            -self.click_point_radius * radius_scale,
            self.click_point_radius * radius_scale,
            self.click_point_grid_size,
            device=valid_mask.device,
            dtype=torch.float32,
        )
        off_r, off_c = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
        offsets = torch.stack([off_r.reshape(-1), off_c.reshape(-1)], dim=-1)
        coords_by_sample = []
        for sample_clicks, sample_labels, sample_valid in zip(click_coords, click_labels, valid_mask):
            valid_clicks = sample_clicks[sample_labels >= 0]
            if valid_clicks.numel() == 0:
                coords_by_sample.append(torch.empty(0, 2, device=valid_mask.device))
                continue
            coords = valid_clicks[:, None, :] + offsets[None, :, :]
            coords = coords.reshape(-1, 2)
            coords[:, 0].clamp_(0, coord_size[0] - 1)
            coords[:, 1].clamp_(0, coord_size[1] - 1)
            row_idx = coords[:, 0].round().long().clamp(0, coord_size[0] - 1)
            col_idx = coords[:, 1].round().long().clamp(0, coord_size[1] - 1)
            coords = coords[sample_valid[row_idx, col_idx]]
            coords_by_sample.append(coords)
        max_len = max((coords.shape[0] for coords in coords_by_sample), default=0)
        if max_len == 0:
            return None
        padded = []
        for coords in coords_by_sample:
            if coords.shape[0] == 0:
                coords = coords_by_sample[0][:1].clone() if coords_by_sample[0].shape[0] > 0 else torch.zeros(1, 2, device=valid_mask.device)
            if coords.shape[0] < max_len:
                coords = coords.repeat((max_len + coords.shape[0] - 1) // coords.shape[0], 1)[:max_len]
            padded.append(coords[:max_len])
        return torch.stack(padded, dim=0)

    @torch.no_grad()
    def merge_point_coords(self, coords: torch.Tensor, new_coords: torch.Tensor, quant: float = 4.0) -> torch.Tensor:
        merged = []
        for sample_coords, sample_new in zip(coords, new_coords):
            combined = torch.cat([sample_coords, sample_new], dim=0)
            keys = torch.round(combined * quant).to(torch.long)
            seen = set()
            keep = []
            for idx, key in enumerate(keys.detach().cpu().tolist()):
                key_tuple = tuple(key)
                if key_tuple in seen:
                    continue
                seen.add(key_tuple)
                keep.append(idx)
            unique_idx = torch.tensor(keep, device=combined.device, dtype=torch.long)
            merged.append(combined[unique_idx])
        max_len = max(x.shape[0] for x in merged)
        padded = []
        for sample_coords in merged:
            if sample_coords.shape[0] < max_len:
                repeat = sample_coords.repeat((max_len + sample_coords.shape[0] - 1) // sample_coords.shape[0], 1)
                sample_coords = repeat[:max_len]
            padded.append(sample_coords)
        return torch.stack(padded, dim=0)

    @torch.no_grad()
    def grid_coords_for_size(
        self,
        valid_mask: torch.Tensor,
        grid_size: int | tuple[int, int],
        max_points: int | None = None,
    ) -> torch.Tensor:
        if isinstance(grid_size, int):
            grid_h, grid_w = grid_size, grid_size
        else:
            grid_h, grid_w = grid_size
        coord_h, coord_w = valid_mask.shape[-2:]
        row_step = coord_h / grid_h
        col_step = coord_w / grid_w
        rows = (torch.arange(grid_h, device=valid_mask.device, dtype=torch.float32) + 0.5) * row_step - 0.5
        cols = (torch.arange(grid_w, device=valid_mask.device, dtype=torch.float32) + 0.5) * col_step - 0.5
        rows = rows.clamp(0, coord_h - 1)
        cols = cols.clamp(0, coord_w - 1)
        grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
        base_coords = torch.stack([grid_rows.reshape(-1), grid_cols.reshape(-1)], dim=-1)
        grid_valid_mask = self.valid_mask_for_grid_size(valid_mask, grid_size)
        coords_by_sample = []
        for sample_valid in grid_valid_mask:
            keep = sample_valid.flatten()
            coords = base_coords[keep]
            if coords.numel() == 0:
                coords = base_coords
            if max_points is not None and coords.shape[0] > max_points:
                coords = coords[:max_points]
            coords_by_sample.append(coords)

        max_len = max(coords.shape[0] for coords in coords_by_sample)
        if max_points is not None:
            max_len = min(max_len, max_points)
        padded = []
        for coords in coords_by_sample:
            coords = coords[:max_len]
            if coords.shape[0] < max_len:
                repeat = coords.repeat((max_len + coords.shape[0] - 1) // coords.shape[0], 1)
                coords = repeat[:max_len]
            padded.append(coords)
        return torch.stack(padded, dim=0)

    @torch.no_grad()
    def valid_mask_for_grid_size(self, valid_mask: torch.Tensor, grid_size: int) -> torch.Tensor:
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        valid = valid_mask.unsqueeze(1).to(torch.float32)
        return F.interpolate(valid, size=grid_size, mode="nearest")[:, 0].bool()

    @torch.no_grad()
    def valid_coord_mask(self, image: torch.Tensor, coord_size: tuple[int, int]) -> torch.Tensor:
        valid = self.valid_pixel_mask(image).to(torch.float32)
        token_valid = F.interpolate(valid, size=coord_size, mode="area") > 0.01
        return token_valid[:, 0]

    @torch.no_grad()
    def select_uncertain_new_coords(
        self,
        dense_logits: torch.Tensor,
        current_coords: torch.Tensor,
        candidate_coords: torch.Tensor,
        valid_mask: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        selected_by_sample = []
        uncertainty = -dense_logits.abs()
        for sample_uncertainty, sample_current, sample_candidates, sample_valid in zip(
            uncertainty, current_coords, candidate_coords, valid_mask
        ):
            coord_h, coord_w = sample_valid.shape
            candidate_rows = sample_candidates[:, 0].round().long().clamp(0, coord_h - 1)
            candidate_cols = sample_candidates[:, 1].round().long().clamp(0, coord_w - 1)
            candidate_valid = sample_valid[candidate_rows, candidate_cols]

            occupied = torch.zeros(coord_h, coord_w, device=sample_uncertainty.device, dtype=torch.bool)
            current_rows = sample_current[:, 0].round().long().clamp(0, coord_h - 1)
            current_cols = sample_current[:, 1].round().long().clamp(0, coord_w - 1)
            occupied[current_rows, current_cols] = True
            candidate_new = ~occupied[candidate_rows, candidate_cols]

            scores = sample_grid_values(sample_uncertainty.unsqueeze(0).unsqueeze(0), sample_candidates.unsqueeze(0), coord_h, coord_w)[0, :, 0]
            scores = scores.masked_fill(~candidate_valid | ~candidate_new, -torch.inf)
            if not torch.isfinite(scores).any():
                scores = torch.zeros_like(scores).masked_fill(~candidate_valid, -torch.inf)
            top_idx = torch.topk(scores, k=min(count, scores.numel()), largest=True).indices
            coords = sample_candidates[top_idx]
            if coords.shape[0] < count:
                repeat = coords.repeat((count + coords.shape[0] - 1) // coords.shape[0], 1)
                coords = repeat[:count]
            selected_by_sample.append(coords)
        return torch.stack(selected_by_sample, dim=0)

    @torch.no_grad()
    def valid_pixel_mask(self, image: torch.Tensor) -> torch.Tensor:
        valid = image.float().abs().sum(dim=1, keepdim=True) > 1e-6
        if not valid.flatten(1).any(dim=1).all():
            valid = torch.ones_like(valid)
        return valid

    @torch.no_grad()
    def valid_token_mask(self, image: torch.Tensor) -> torch.Tensor:
        valid = self.valid_pixel_mask(image).to(torch.float32)
        token_valid = F.interpolate(valid, size=(self.h, self.w), mode="area") > 0.01
        return token_valid[:, 0]

    @torch.no_grad()
    def select_valid_coords(self, valid_mask: torch.Tensor, num_points: int, mode: str = "uniform") -> torch.Tensor:
        coords_by_sample = []
        coord_h, coord_w = valid_mask.shape[-2:]
        target_count = min(num_points, coord_h * coord_w)
        for sample_valid in valid_mask:
            valid_indices = sample_valid.flatten().nonzero(as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                valid_indices = torch.arange(coord_h * coord_w, device=valid_mask.device)
            if valid_indices.numel() >= target_count:
                if mode == "linspace" and target_count > 1:
                    pick = torch.linspace(0, valid_indices.numel() - 1, target_count, device=valid_mask.device).round().long()
                    selected = valid_indices[pick]
                else:
                    selected = valid_indices[:target_count]
            else:
                repeats = valid_indices.repeat((target_count + valid_indices.numel() - 1) // valid_indices.numel())
                selected = repeats[:target_count]
            rows = (selected // coord_w).to(torch.float32)
            cols = (selected % coord_w).to(torch.float32)
            coords_by_sample.append(torch.stack([rows, cols], dim=-1))
        return torch.stack(coords_by_sample, dim=0)

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
    return sample_grid_values(features, coords, coord_h, coord_w)


def sample_grid_values(features: torch.Tensor, coords: torch.Tensor, coord_h: int, coord_w: int) -> torch.Tensor:
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
    k: int = 4,
    power: float = 2.0,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Interpolate point logits to every output pixel with adaptive Gaussian kNN weighting."""
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
        sigma2 = knn_dist2[..., -1:].clamp_min(1e-6) * max(power, 1e-6)
        weights = torch.exp(-0.5 * knn_dist2 / sigma2)
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
