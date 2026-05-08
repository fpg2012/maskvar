import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as tdist

from einops import rearrange, repeat


class SimpleVectorQuantize(nn.Module):
    """
    简化的单尺度向量量化器（使用einops实现，支持词表利用率统计）。
    """

    def __init__(self, dim: int, vocab_size: int, beta: float = 0.25, using_znorm: bool = False):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.beta = beta
        self.using_znorm = using_znorm

        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.5)

        # 词表使用统计（EMA）
        self.register_buffer('ema_vocab_hit', torch.zeros(vocab_size))
        self.ema_decay = 0.99
        # Use tensor instead of int to avoid torch.compile recompilation
        self.register_buffer('record_hit', torch.tensor(0, dtype=torch.long))

    def _compute_distances(self, z: torch.Tensor):
        """
        z: (B*H*W, C)
        returns: distances (B*H*W, vocab_size)
        """
        if self.using_znorm:
            z_norm = F.normalize(z, dim=-1)
            w_norm = F.normalize(self.embedding.weight, dim=-1)
            distances = -torch.mm(z_norm, w_norm.t())
        else:
            z_sq = torch.sum(z ** 2, dim=1, keepdim=True)
            w_sq = torch.sum(self.embedding.weight ** 2, dim=1)
            distances = z_sq + w_sq - 2 * torch.mm(z, self.embedding.weight.t())
        return distances

    def forward(self, z: torch.Tensor, return_usage: bool = False):
        """
        z: (B, L, C) - always in BLC format
        return_usage: 是否返回词表利用率
        returns: z_q (same shape as input, BHWC), vq_loss, (optional) usage_percent
        """
        output_dtype = z.dtype
        if z.dtype != torch.float32:
            z = z.float()
        B, L, C = z.shape
        z_orig = z

        with torch.amp.autocast(device_type="cuda", enabled=False):
            z_flat = rearrange(z, 'b l c -> (b l) c')

            distances = self._compute_distances(z_flat)
            indices = torch.argmin(distances, dim=1)

            # 统计词表使用情况
            if self.training:
                hit_count = indices.bincount(minlength=self.vocab_size).float()
                if tdist.is_initialized():
                    tdist.all_reduce(hit_count)

                # Use a boolean flag instead of item() to avoid graph break
                is_first_hit = self.record_hit == 0
                if is_first_hit:
                    self.ema_vocab_hit.copy_(hit_count)
                else:
                    self.ema_vocab_hit.mul_(self.ema_decay).add_(hit_count, alpha=1 - self.ema_decay)
                self.record_hit.add_(1)

            z_q = self.embedding(indices)
            z_q = rearrange(z_q, '(b l) c -> b l c', b=B, l=L)

            commitment_loss = F.mse_loss(z_q.detach(), z_orig)
            codebook_loss = F.mse_loss(z_q, z_orig.detach())
            loss = codebook_loss + self.beta * commitment_loss

            z_q = z_orig + (z_q - z_orig).detach()
            z_q = z_q.to(output_dtype)

        if return_usage:
            world_size = tdist.get_world_size() if tdist.is_initialized() else 1
            total_tokens = B * L * world_size
            margin = total_tokens / self.vocab_size * 0.08
            # Return as tensor to avoid graph break with torch.compile
            # Caller should call .item() outside of compiled region
            usage_percent = (self.ema_vocab_hit >= margin).float().mean() * 100
            return z_q, loss, usage_percent

        return z_q, loss

    def x_to_idx(self, x: torch.Tensor):
        """
        x: (B, L, C) - always in BLC format
        returns: indices (B, L)
        """
        if x.dtype != torch.float32:
            x = x.float()
        B, L, C = x.shape
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_flat = rearrange(x, 'b l c -> (b l) c')

            distances = self._compute_distances(x_flat)
            indices = torch.argmin(distances, dim=1)

        return indices.view(B, L).contiguous()

    def idx_to_x(self, indices: torch.Tensor):
        """
        indices: (B, L)
        returns: x (B, L, C)
        """
        B, L = indices.shape
        x = self.embedding(indices)
        # x is (B, H, W, C), already in BHWC format
        return x


class MultiscaleVectorQuantize(SimpleVectorQuantize):
    """
    VAR-style residual multi-scale vector quantizer.

    This is a drop-in replacement for SimpleVectorQuantize at the SimpleMaskVqvae
    interface: input and output are both full-resolution BLC tensors. Internally
    it quantizes residuals from coarse to fine:

        f = input feature map
        for scale in scales:
            z_k = Q(interpolate(f, scale))
            f = f - phi_k(interpolate(z_k, full_resolution))

    The final quantized full-resolution feature is the sum of projected
    quantized codes across scales.
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        beta: float = 0.25,
        scales=(1, 2, 4, 8, 16, 32, 64),
        h: int = 64,
        w: int = 64,
        using_znorm: bool = False,
        quant_resi: float = 0.5,
    ):
        super().__init__(dim=dim, vocab_size=vocab_size, beta=beta, using_znorm=using_znorm)
        self.scales = tuple(scales)
        self.h = h
        self.w = w
        self.quant_resi_ratio = abs(quant_resi)
        self.phi = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=3, padding=1) for _ in self.scales])
        self.scale_gates = nn.Parameter(torch.zeros(len(self.scales)))
        self._init_phi_as_single_scale_identity()

    def _init_phi_as_single_scale_identity(self):
        """
        Start from the original SimpleVectorQuantize behavior.

        All coarse projections are zero, and the final full-resolution scale is
        identity. With a pretrained SimpleMaskVqvae codebook this makes the
        multiscale quantizer initially equivalent to the old single-scale
        quantizer, then coarse scales can learn useful residual projections.
        """
        for phi in self.phi:
            nn.init.zeros_(phi.weight)
            if phi.bias is not None:
                nn.init.zeros_(phi.bias)

        last_phi = self.phi[-1]
        with torch.no_grad():
            self.scale_gates.zero_()
            self.scale_gates[-1] = 1.0
            last_phi.weight.zero_()
            for channel_idx in range(min(last_phi.out_channels, last_phi.in_channels)):
                last_phi.weight[channel_idx, channel_idx, 0, 0] = 1.0
            if last_phi.bias is not None:
                last_phi.bias.zero_()

    def _tokens_to_map(self, x: torch.Tensor, h: int, w: int):
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

    def _map_to_tokens(self, x: torch.Tensor):
        return rearrange(x, "b c h w -> b (h w) c")

    def _nearest_code_indices(self, tokens: torch.Tensor):
        flat_tokens = rearrange(tokens, "b l c -> (b l) c")
        if self.using_znorm:
            flat_tokens = F.normalize(flat_tokens, dim=-1)
            codebook = F.normalize(self.embedding.weight.data, dim=-1)
            indices = torch.argmax(flat_tokens @ codebook.t(), dim=1)
        else:
            distances = (
                flat_tokens.square().sum(dim=1, keepdim=True)
                + self.embedding.weight.data.square().sum(dim=1)
            )
            distances.addmm_(flat_tokens, self.embedding.weight.data.t(), alpha=-2, beta=1)
            indices = torch.argmin(distances, dim=1)
        return indices

    def _project_tokens_to_full(self, tokens: torch.Tensor, scale_idx: int):
        scale = self.scales[scale_idx]
        feat = self._tokens_to_map(tokens, h=scale, w=scale)
        if scale != self.h or scale != self.w:
            feat = F.interpolate(feat, size=(self.h, self.w), mode="bicubic", align_corners=False)
        feat = feat.mul(1 - self.quant_resi_ratio) + self.phi[scale_idx](feat).mul(self.quant_resi_ratio)
        return feat * self.scale_gates[scale_idx].view(1, 1, 1, 1)

    def forward(self, z: torch.Tensor, return_usage: bool = False):
        """
        z: (B, L, C), where L == h * w.
        returns: quantized full-resolution z_q in BLC format.
        """
        B, L, C = z.shape
        if L != self.h * self.w:
            raise ValueError(f"Expected L={self.h * self.w} for {self.h}x{self.w}, got L={L}")

        output_dtype = z.dtype
        z_fp32 = z.float()
        z_no_grad = z_fp32.detach()
        z_map = self._tokens_to_map(z_fp32, h=self.h, w=self.w)
        residual = self._tokens_to_map(z_no_grad, h=self.h, w=self.w).clone()
        quantized_full = torch.zeros_like(residual)
        vq_loss = z_fp32.new_tensor(0.0)
        vocab_hit = torch.zeros(self.vocab_size, dtype=torch.float32, device=z.device)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            for scale_idx, scale in enumerate(self.scales):
                residual_at_scale = F.interpolate(residual, size=(scale, scale), mode="area")
                tokens = self._map_to_tokens(residual_at_scale)
                indices = self._nearest_code_indices(tokens)

                hit_count = indices.bincount(minlength=self.vocab_size).float()
                if self.training:
                    if tdist.is_initialized():
                        tdist.all_reduce(hit_count)
                    is_first_hit = self.record_hit == 0
                    if is_first_hit:
                        self.ema_vocab_hit.copy_(hit_count)
                    else:
                        self.ema_vocab_hit.mul_(self.ema_decay).add_(hit_count, alpha=1 - self.ema_decay)
                    self.record_hit.add_(1)
                vocab_hit.add_(hit_count)

                idx_Bhw = indices.view(B, scale, scale)
                tokens_q = self.embedding(idx_Bhw).view(B, scale * scale, C)
                projected = self._project_tokens_to_full(tokens_q, scale_idx)
                quantized_full = quantized_full + projected
                residual = residual - projected
                vq_loss = vq_loss + (
                    F.mse_loss(quantized_full.data, z_map).mul(self.beta)
                    + F.mse_loss(quantized_full, z_map.detach())
                )

            vq_loss = vq_loss / len(self.scales)
            z_q_map = (quantized_full.data - z_map.detach()).add(z_map)
            z_q = self._map_to_tokens(z_q_map).to(output_dtype)

        if return_usage:
            world_size = tdist.get_world_size() if tdist.is_initialized() else 1
            total_tokens = sum(B * scale * scale for scale in self.scales) * world_size
            margin = total_tokens / self.vocab_size * 0.08
            if self.training:
                vq_usage = (self.ema_vocab_hit >= margin).float().mean() * 100
            else:
                vq_usage = (vocab_hit >= margin).float().mean() * 100
            return z_q, vq_loss, vq_usage
        return z_q, vq_loss

    def x_to_idx_multiscale(self, x: torch.Tensor):
        """
        Encode full-resolution BLC features into one index tensor per scale.
        """
        B, L, C = x.shape
        if L != self.h * self.w:
            raise ValueError(f"Expected L={self.h * self.w} for {self.h}x{self.w}, got L={L}")

        residual = self._tokens_to_map(x.float().detach(), h=self.h, w=self.w).clone()
        token_ids_by_scale = []
        with torch.amp.autocast(device_type="cuda", enabled=False):
            for scale_idx, scale in enumerate(self.scales):
                residual_at_scale = F.interpolate(residual, size=(scale, scale), mode="area")
                tokens = self._map_to_tokens(residual_at_scale)
                indices = self._nearest_code_indices(tokens)
                token_ids = indices.view(x.shape[0], scale * scale)
                token_ids_by_scale.append(token_ids)

                tokens_q = self.embedding(indices.view(x.shape[0], scale, scale)).view(x.shape[0], scale * scale, x.shape[-1])
                projected = self._project_tokens_to_full(tokens_q, scale_idx)
                residual = residual - projected
        return token_ids_by_scale

    def idxBl_to_full_tokens(self, token_ids_by_scale):
        """
        Decode multi-scale token ids into full-resolution BLC features.
        """
        full = None
        for scale_idx, token_ids in enumerate(token_ids_by_scale):
            tokens = super().idx_to_x(token_ids)
            projected = self._project_tokens_to_full(tokens, scale_idx)
            full = projected if full is None else full + projected
        return self._map_to_tokens(full)
