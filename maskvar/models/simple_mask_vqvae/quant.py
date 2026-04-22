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
        B, L, C = z.shape
        z_orig = z

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
        B, L, C = x.shape
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