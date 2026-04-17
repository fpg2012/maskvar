import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as tdist

import timm
from einops import rearrange, repeat
from ..sam import MaskDecoder as SamMaskDecoder, ImageEncoderViT
from ..tinyvit import TinyViT
from ..transformer import BlockCross

class SimpleCrossBlock(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.linear_q = nn.Linear(dim, dim)
        self.linear_kv = nn.Linear(dim, dim*2)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.layernorm = nn.LayerNorm(dim)

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim_head = dim // num_heads
    
    def apply_2d_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        assert C % 2 == 0, "C must be even for 2D RoPE"
        
        # 分成两半：前一半用 H 位置，后一半用 W 位置
        x_h, x_w = x.chunk(2, dim=-1)          # (B, H, W, C//2) each
        
        # 频率（推荐对视觉稍作调整）
        dim = C // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
        
        # H 轴旋转
        h_pos = torch.arange(H, device=x.device, dtype=torch.float32)
        h_freqs = torch.outer(h_pos, inv_freq)          # (H, dim//2)
        h_cos = h_freqs.cos().unsqueeze(0).unsqueeze(2) # (1, H, 1, dim//2)
        h_sin = h_freqs.sin().unsqueeze(0).unsqueeze(2)
        
        x_h = self._apply_rotary(x_h, h_cos, h_sin)     # 见下面辅助函数
        
        # W 轴旋转
        w_pos = torch.arange(W, device=x.device, dtype=torch.float32)
        w_freqs = torch.outer(w_pos, inv_freq)
        w_cos = w_freqs.cos().unsqueeze(0).unsqueeze(1) # (1, 1, W, dim//2)
        w_sin = w_freqs.sin().unsqueeze(0).unsqueeze(1)
        
        x_w = self._apply_rotary(x_w, w_cos, w_sin)
        
        return torch.cat([x_h, x_w], dim=-1)

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x: (B, H, W, D)   D = C//2
        # cos, sin: broadcastable to (..., D//2)
        x_ = rearrange(x, 'b h w (d2 c) -> b h w d2 c', d2=2)  # real, imag
        x_real, x_imag = x_[..., 0, :], x_[..., 1, :]
        
        # 广播 cos/sin
        cos = cos.unsqueeze(-1) if cos.dim() < x.dim() else cos
        sin = sin.unsqueeze(-1) if sin.dim() < x.dim() else sin
        
        rotated_real = cos * x_real - sin * x_imag
        rotated_imag = sin * x_real + cos * x_imag
        
        rotated = torch.stack([rotated_real, rotated_imag], dim=-2)
        return rearrange(rotated, 'b h w d2 c -> b h w (d2 c)')
    
    def forward(self, q, kv, pe_type='rope', image_pe=None):
        """
        q: (b, l, c)
        k: (b, h, w, c)
        """
        b, h, w, c = kv.shape
        # print(f"[DEBUG] CrossBlock input - q: {q.shape}, kv: {kv.shape}, expected dim: {self.dim}")

        q_input = q

        q = self.linear_q(q)
        # print(f"[DEBUG] After linear_q - q: {q.shape}")
        kv = self.linear_kv(kv)
        # print(f"[DEBUG] After linear_kv - kv: {kv.shape}")
        k, v = kv.chunk(2, dim=-1)
        # print(f"[DEBUG] After chunk - k: {k.shape}, v: {v.shape}")
        
        q = rearrange(q, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)
        k = rearrange(k, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b h w (nh c_head) -> b nh (h w) c_head', nh=self.num_heads, c_head=self.dim_head)

        # q = self.apply_2d_rope(q)
        if pe_type == 'rope':
            k = rearrange(k, 'b nh h w c_head -> (b nh) h w c_head')
            k = self.apply_2d_rope(k)
            k = rearrange(k, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)
        elif pe_type == 'sam':
            k = k + image_pe
        else:
            raise ValueError(f"Invalid pe_type: {pe_type}")

        # out = flex_attention(q, k, v, attn_mask=None)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b nh l c_head -> b l (nh c_head)')
        out = q_input + self.out_proj(out)

        out = out + self.ffn(self.layernorm(q_input))
        
        return out


class SimpleCrossBlockReverse(SimpleCrossBlock):

    def __init__(self, dim, num_heads):
        super().__init__(dim, num_heads)
    
    def forward(self, q, kv, pe_type='rope', image_pe=None):
        """
        q: (b, h, w, c)
        k: (b, l, c)
        """
        b, h, w, c = q.shape
        # print(f"[DEBUG] CrossBlock input - q: {q.shape}, kv: {kv.shape}, expected dim: {self.dim}")

        q_input = q

        q = self.linear_q(q)
        # print(f"[DEBUG] After linear_q - q: {q.shape}")
        kv = self.linear_kv(kv)
        # print(f"[DEBUG] After linear_kv - kv: {kv.shape}")
        k, v = kv.chunk(2, dim=-1)
        # print(f"[DEBUG] After chunk - k: {k.shape}, v: {v.shape}")
        
        q = rearrange(q, 'b h w (nh c_head) -> b nh h w c_head', nh=self.num_heads, c_head=self.dim_head)
        k = rearrange(k, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)
        v = rearrange(v, 'b l (nh c_head) -> b nh l c_head', nh=self.num_heads, c_head=self.dim_head)

        if pe_type == 'rope':
            q = rearrange(q, 'b nh h w c_head -> (b nh) h w c_head')
            q = self.apply_2d_rope(q)
            q = rearrange(q, '(b nh) h w c_head -> b nh (h w) c_head', b=b, nh=self.num_heads)
        elif pe_type == 'sam':
            q = q + image_pe
        else:
            raise ValueError(f"Invalid pe_type: {pe_type}")

        # out = flex_attention(q, k, v, attn_mask=None)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b nh (h w) c_head -> b h w (nh c_head)', h=h, w=w)
        out = q_input + self.out_proj(out)

        out = out + self.ffn(self.layernorm(q_input))
        
        return out

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskEncoderLite(nn.Module):

    def __init__(self, dim, patch_size=16, image_size=1024):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.image_size = image_size

        assert image_size % patch_size == 0

        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, mask):
        """
        mask: (B, 1, H, W)
        """
        # reshuffle
        mask_tokens = rearrange(self.patch_embed(mask), 'b c h w -> b h w c')
        mask_tokens = mask_tokens + self.mlp(self.layer_norm(mask_tokens))
        
        return rearrange(mask_tokens, 'b h w c -> b c h w')


class TwoWayBlock(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim

        self.block1 = SimpleCrossBlock(dim, num_heads)
        self.reverse_block1 = SimpleCrossBlockReverse(dim, num_heads)
        self.block2 = SimpleCrossBlock(dim, num_heads)
        self.reverse_block2 = SimpleCrossBlockReverse(dim, num_heads)

    def forward(self, query_tokens, mask_tokens, image_tokens, pe_type='rope', image_pe=None):
        query_tokens = self.block1(query_tokens, mask_tokens, pe_type=pe_type, image_pe=image_pe)
        image_tokens = self.reverse_block1(image_tokens, query_tokens, pe_type=pe_type, image_pe=image_pe)
        query_tokens = self.block2(query_tokens, image_tokens, pe_type=pe_type, image_pe=image_pe)
        mask_tokens = self.reverse_block2(mask_tokens, query_tokens, pe_type=pe_type, image_pe=image_pe)
        return query_tokens, mask_tokens, image_tokens


class SimpleMaskDecoder(nn.Module):

    def __init__(self, dim, num_heads=4, num_queries=4, num_two_way_blocks=2):
        super().__init__()
        self.dim = dim
        self.num_two_way_blocks = num_two_way_blocks

        self.two_way_blocks = nn.ModuleList([
            TwoWayBlock(dim, num_heads) for _ in range(num_two_way_blocks)
        ])
        
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, dim))
        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(dim, dim // 4, kernel_size=4, stride=4),
        #     LayerNorm2d(dim // 4),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=4, stride=4),
        #     nn.GELU(),
        # )
        self.output_upscaling = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(dim//4, dim//4, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(dim // 4),
            nn.GELU(),
            nn.PixelShuffle(2),
            nn.Conv2d(dim//16, dim//16, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        # self.output_upscaling2 = nn.Sequential(
        #     nn.ConvTranspose2d(dim, dim // 4, kernel_size=2, stride=2),
        #     LayerNorm2d(dim // 4),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
        #     nn.GELU(),
        # )
        self.hyper_in = MLP(dim, dim, dim // 16, 3)

        # self.layer_norm_pre_image = nn.LayerNorm(dim)
        # self.layer_norm_pre_query = nn.LayerNorm(dim)
        # self.layer_norm_pre_mask = nn.LayerNorm(dim)
        self.layer_norm_post_image = nn.LayerNorm(dim // 16)
        self.layer_norm_post_query = nn.LayerNorm(dim // 16)
        # self.layer_norm_post_mask = nn.LayerNorm(dim // 16)
    
    def forward(self, mask_tokens: torch.Tensor, image_tokens: torch.Tensor):
        # print(f"[DEBUG] Decoder forward - mask_tokens: {mask_tokens.shape}, image_tokens: {image_tokens.shape}")
        
        # mask_tokens = self.layer_norm_image(mask_tokens)
        # image_tokens = self.layer_norm_mask(image_tokens)

        query_tokens = repeat(self.query_tokens, '1 l c -> b l c', b=mask_tokens.shape[0])
        for blk in self.two_way_blocks:
            query_tokens, mask_tokens, image_tokens = blk(query_tokens, mask_tokens, image_tokens, pe_type='rope')

        image_feature_map = rearrange(image_tokens, 'b h w c -> b c h w')
        mask_feature_map = rearrange(mask_tokens, 'b h w c -> b c h w')
        up_query_token = self.hyper_in(query_tokens[:, 0, :])
        up_image_map = self.output_upscaling(image_feature_map)
        # up_mask_map = self.output_upscaling2(mask_feature_map)

        up_query_token = self.layer_norm_post_query(up_query_token)
        up_image_map = self.layer_norm_post_image(rearrange(up_image_map, 'b c h w -> b h w c'))
        # up_mask_map = self.layer_norm_post_mask(rearrange(up_mask_map, 'b c h w -> b h w c'))

        masks = torch.einsum('bc,bhwc->bhw', up_query_token, up_image_map).unsqueeze(1)
        # masks = torch.einsum('bhwc,bhwc->bhw', up_mask_map, up_image_map)
        # try addition
        # masks = rearrange(up_query_token, 'b c -> b c 1 1') + up_image_map
        # masks = torch.mean(masks, dim=1, keepdim=True)

        # Ensure final output is contiguous
        return masks.contiguous()

class MaskDecoderWithSAM(SimpleMaskDecoder):
    
    def __init__(self, sam_mask_decoder: SamMaskDecoder):
        super().__init__()
        self.sam_mask_decoder = sam_mask_decoder

    def forward(self, mask_tokens: torch.Tensor, image_tokens: torch.Tensor):
        """
        mask_tokens: (b, h, w, c)
        image_tokens: (b, h, w, c)

        return: mask
        """
        mask_tokens = self.apply_2d_rope(mask_tokens)
        image_tokens = self.apply_2d_rope(image_tokens)

        # flatten mask tokens
        mask_tokens = rearrange(mask_tokens, 'b h w c -> b (h w) c')
        
        # we don't use sam pe, so just create a zero tensor
        fake_image_pe = torch.zeros_like(image_tokens)
        mask_tokens, image_tokens = self.sam_mask_decoder.transformer(
            image_tokens, fake_image_pe, mask_tokens
        )
        image_features = rearrange(image_tokens, 'b h w c -> b c h w')
        mask_token_map = rearrange(mask_tokens, 'b h w c -> b c h w')

        upscaled_embedding = self.sam_mask_decoder.output_upscaling(image_features) # (b, C, H, W)
        upscaled_token_map = self.sam_mask_decoder.output_upscaling(mask_token_map) # (b, C, H, W)

        masks = torch.einsum('bchw,bchw->bhw', upscaled_embedding, upscaled_token_map)
        
        return masks


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
        z: (B, H, W, C) - always in BHWC format
        return_usage: 是否返回词表利用率
        returns: z_q (same shape as input, BHWC), vq_loss, (optional) usage_percent
        """
        assert z.shape[-1] == self.dim, f"Expected input shape (B,H,W,{self.dim}), got {z.shape}"

        B, H, W, C = z.shape
        z_orig = z

        z_flat = rearrange(z, 'b h w c -> (b h w) c')

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
        z_q = rearrange(z_q, '(b h w) c -> b h w c', b=B, h=H, w=W)

        commitment_loss = F.mse_loss(z_q.detach(), z_orig)
        codebook_loss = F.mse_loss(z_q, z_orig.detach())
        loss = codebook_loss + self.beta * commitment_loss

        z_q = z_orig + (z_q - z_orig).detach()

        if return_usage:
            world_size = tdist.get_world_size() if tdist.is_initialized() else 1
            total_tokens = B * H * W * world_size
            margin = total_tokens / self.vocab_size * 0.08
            # Return as tensor to avoid graph break with torch.compile
            # Caller should call .item() outside of compiled region
            usage_percent = (self.ema_vocab_hit >= margin).float().mean() * 100
            return z_q, loss, usage_percent

        return z_q, loss

    def x_to_idx(self, x: torch.Tensor):
        """
        x: (B, H, W, C) - always in BHWC format
        returns: indices (B, H, W)
        """
        assert x.shape[-1] == self.dim, f"Expected input shape (B,H,W,{self.dim}), got {x.shape}"

        B, H, W, C = x.shape
        x_flat = rearrange(x, 'b h w c -> (b h w) c')

        distances = self._compute_distances(x_flat)
        indices = torch.argmin(distances, dim=1)

        return indices.view(B, H, W).contiguous()

    def idx_to_x(self, indices: torch.Tensor):
        """
        indices: (B, H, W)
        returns: x (B, H, W, C)
        """
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)

        assert indices.dim() == 3, f"Expected indices shape (B,H,W), got {indices.shape}"

        B, H, W = indices.shape
        x = self.embedding(indices)
        # x is (B, H, W, C), already in BHWC format
        return x


class SimpleMaskVqvae(nn.Module):

    def __init__(self, image_encoder, mask_encoder, dim=256, vocab_size=4096, beta=0.25, enable_vq=True, device='cuda'):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.beta = beta
        self.enable_vq = enable_vq

        self.image_encoder = image_encoder
        self.mask_encoder = mask_encoder
        
        self.mask_decoder = SimpleMaskDecoder(dim=dim)
        self.quant = SimpleVectorQuantize(
            dim=dim,
            vocab_size=vocab_size,
            beta=beta,
            using_znorm=False,
        )
        self.device = device
    
    def forward(self, mask_normalized: torch.Tensor, image: torch.Tensor, return_usage: bool = False):
        """
        for training only
        Args:
            mask_normalized: (B, 1, H, W) normalized mask
            image: (B, 3, H, W) image
            return_usage: if True, return vocabulary usage percentage
        """
        B, _, H, W = mask_normalized.shape

        # 1. encoder mask and image with corresponding encoders
        # mask_normalized_3 = repeat(mask_normalized, 'b 1 h w -> b 3 h w')
        mask_tokens = self.mask_encoder(mask_normalized)
        image_tokens = self.image_encoder(image)

        # Convert from (b, c, h, w) to (b, h, w, c) for decoder
        mask_tokens = rearrange(mask_tokens, 'b c h w -> b h w c')
        image_tokens = rearrange(image_tokens, 'b c h w -> b h w c')

        # 2. quantize mask_tokens
        if self.enable_vq:
            if return_usage:
                mask_tokens, vq_loss, vq_usage = self.quant(mask_tokens, return_usage=True)
            else:
                mask_tokens, vq_loss = self.quant(mask_tokens)
                vq_usage = None
        else:
            # skip quant
            vq_loss = torch.tensor(0.0)
            vq_usage = torch.tensor(0.0)

        # 3. decode
        mask = self.mask_decoder(mask_tokens, image_tokens)

        # print(f'mask shape: {mask.shape}')

        # 4. resize mask to original size (decoder outputs 256x256, input may be 1024x1024)
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)

        if return_usage:
            return mask, vq_loss, vq_usage
        return mask, vq_loss

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
