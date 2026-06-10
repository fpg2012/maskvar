import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from ..simple_mask_vqvae.basic import MLP
from .rope_sam import sample_grid_values


def scale_points(points_xy: torch.Tensor, from_size: tuple[int, int], to_size: tuple[int, int]) -> torch.Tensor:
    """Scale row/col point coordinates between two grids."""
    from_h, from_w = from_size
    to_h, to_w = to_size
    scaled = points_xy.to(dtype=torch.float32).clone()
    scaled[..., 0] = (scaled[..., 0] + 0.5) * (to_h / float(from_h)) - 0.5
    scaled[..., 1] = (scaled[..., 1] + 0.5) * (to_w / float(from_w)) - 0.5
    return scaled


def sample_bchw(features: torch.Tensor, points_xy: torch.Tensor, coord_size: tuple[int, int]) -> torch.Tensor:
    """Sample BCHW features at row/col coordinates. Returns BNC."""
    coord_h, coord_w = coord_size
    return sample_grid_values(features, points_xy, coord_h, coord_w)


def point_distances_to_clicks(
    point_coords: torch.Tensor,
    click_coords: torch.Tensor,
    click_labels: torch.Tensor,
    coord_size: tuple[int, int],
) -> torch.Tensor:
    """Return normalized distance to nearest positive and negative click, shape B,N,2."""
    b, n, _ = point_coords.shape
    device = point_coords.device
    diag = (coord_size[0] ** 2 + coord_size[1] ** 2) ** 0.5
    fallback = torch.full((b, n), diag, device=device, dtype=torch.float32)
    dists = []
    for label in (1, 0):
        valid = click_labels == label
        if valid.any():
            delta = point_coords[:, :, None, :] - click_coords[:, None, :, :].to(point_coords)
            dist = delta.square().sum(dim=-1).sqrt()
            dist = dist.masked_fill(~valid[:, None, :], diag)
            dist = dist.amin(dim=-1)
        else:
            dist = fallback
        dists.append((dist / max(diag, 1e-6)).clamp(0, 1))
    return torch.stack(dists, dim=-1)


class FourierPointEncoding(nn.Module):
    """Small Fourier encoding for continuous row/col point coordinates."""

    def __init__(self, num_bands: int = 8):
        super().__init__()
        self.num_bands = num_bands
        bands = 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        self.register_buffer("bands", bands, persistent=False)

    @property
    def out_dim(self) -> int:
        return 4 * self.num_bands

    def forward(self, coords: torch.Tensor, coord_size: tuple[int, int]) -> torch.Tensor:
        coord_h, coord_w = coord_size
        y = (coords[..., 0] + 0.5) / float(coord_h)
        x = (coords[..., 1] + 0.5) / float(coord_w)
        xy = rearrange(torch.stack([y, x], dim=-1), "b n xy -> b n xy 1")
        angles = xy * self.bands.to(coords) * (2.0 * torch.pi)
        return rearrange(torch.stack([angles.sin(), angles.cos()], dim=-2), "b n xy sc f -> b n (xy sc f)")


class SparsePointRefiner(nn.Module):
    """
    Lightweight variable-length point decoder.

    The image encoder/RopeSAM teacher can stay frozen. This module receives sampled
    point features plus click/coarse-mask cues and predicts per-point logits.
    """

    def __init__(
        self,
        image_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        coord_bands: int = 8,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.coord_pe = FourierPointEncoding(coord_bands)
        self.use_cross_attention = use_cross_attention
        in_dim = image_dim + self.coord_pe.out_dim + 4
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.click_proj = nn.Linear(3, hidden_dim)
        if use_cross_attention:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.point_blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            self.point_blocks = None
        self.head = MLP(hidden_dim, hidden_dim, 1, 3)

    def build_point_features(
        self,
        image_features: torch.Tensor,
        point_coords: torch.Tensor,
        coord_size: tuple[int, int],
        click_coords: torch.Tensor,
        click_labels: torch.Tensor,
        coarse_logits: torch.Tensor | None = None,
        prev_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_sample = sample_bchw(image_features, point_coords, coord_size)
        coord_pe = self.coord_pe(point_coords, coord_size)
        click_dist = point_distances_to_clicks(point_coords, click_coords, click_labels, coord_size)

        scalar_feats = [click_dist]
        for logits in (coarse_logits, prev_logits):
            if logits is None:
                scalar_feats.append(torch.zeros(*point_coords.shape[:2], 1, device=point_coords.device))
            else:
                logit_coords = scale_points(point_coords, coord_size, logits.shape[-2:])
                scalar_feats.append(sample_bchw(logits, logit_coords, logits.shape[-2:]))
        return torch.cat([image_sample, coord_pe, *scalar_feats], dim=-1)

    def forward(
        self,
        point_features: torch.Tensor,
        point_coords: torch.Tensor,
        coord_size: tuple[int, int],
        click_coords: torch.Tensor,
        click_labels: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(point_features)
        click_dist = point_distances_to_clicks(point_coords, click_coords, click_labels, coord_size)
        if valid_mask is None:
            valid_hint = torch.ones_like(click_dist[..., :1])
        else:
            valid_hint = rearrange(valid_mask.to(click_dist.dtype), "b n -> b n 1")
        click_hint = torch.cat([click_dist, valid_hint], dim=-1)
        x = x + self.click_proj(click_hint)

        if valid_mask is None:
            valid_mask = torch.ones(point_coords.shape[:2], device=point_coords.device, dtype=torch.bool)
        if self.point_blocks is not None:
            x = self.point_blocks(x, src_key_padding_mask=~valid_mask)
        logits = self.head(x).squeeze(-1)
        return logits.masked_fill(~valid_mask, 0.0)


def uniform_grid_points(size: tuple[int, int], device: torch.device | str) -> torch.Tensor:
    h, w = size
    rows = torch.arange(h, device=device, dtype=torch.float32)
    cols = torch.arange(w, device=device, dtype=torch.float32)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([rearrange(rr, "h w -> (h w)"), rearrange(cc, "h w -> (h w)")], dim=-1)


def select_uncertain_points(
    logits: torch.Tensor,
    count: int,
    coord_size: tuple[int, int],
    exclude_coords: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select top uncertain points from a regular coord grid. Returns B,K,2."""
    b = logits.shape[0]
    candidates = uniform_grid_points(coord_size, logits.device)
    sample_coords = repeat(candidates, "n c -> b n c", b=b)
    logit_coords = scale_points(sample_coords, coord_size, logits.shape[-2:])
    values = sample_bchw(logits, logit_coords, logits.shape[-2:])[..., 0]
    scores = -values.abs()
    if exclude_coords is not None:
        rows = exclude_coords[..., 0].round().long().clamp(0, coord_size[0] - 1)
        cols = exclude_coords[..., 1].round().long().clamp(0, coord_size[1] - 1)
        occupied = torch.zeros(b, coord_size[0] * coord_size[1], device=logits.device, dtype=torch.bool)
        occupied.scatter_(1, rows * coord_size[1] + cols, True)
        scores = scores.masked_fill(occupied, -torch.inf)
    idx = scores.topk(k=min(count, scores.shape[1]), dim=1).indices
    gather_idx = repeat(idx, "b k -> b k c", c=2)
    return sample_coords.gather(1, gather_idx)


def interpolate_point_logits(
    point_logits: torch.Tensor,
    point_coords: torch.Tensor,
    output_size: tuple[int, int],
    coord_size: tuple[int, int],
    k: int = 4,
) -> torch.Tensor:
    """Simple kNN interpolation from point logits to a dense mask logit map."""
    b, n = point_logits.shape
    out_h, out_w = output_size
    rows = (torch.arange(out_h, device=point_logits.device, dtype=torch.float32) + 0.5) * (coord_size[0] / out_h) - 0.5
    cols = (torch.arange(out_w, device=point_logits.device, dtype=torch.float32) + 0.5) * (coord_size[1] / out_w) - 0.5
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    pixels = torch.stack([rearrange(rr, "h w -> (h w)"), rearrange(cc, "h w -> (h w)")], dim=-1)
    pixel_coords = rearrange(pixels, "p xy -> 1 p 1 xy")
    sample_coords = rearrange(point_coords, "b n xy -> b 1 n xy")
    dist2 = (pixel_coords - sample_coords).square().sum(dim=-1)
    k = min(k, n)
    knn_dist2, knn_idx = dist2.topk(k=k, dim=-1, largest=False)
    gathered = point_logits.gather(1, rearrange(knn_idx, "b p k -> b (p k)"))
    gathered = rearrange(gathered, "b (p k) -> b p k", k=k)
    weights = 1.0 / knn_dist2.clamp_min(1e-6)
    dense = (gathered * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-6)
    return rearrange(dense, "b (h w) -> b 1 h w", h=out_h, w=out_w).contiguous()
