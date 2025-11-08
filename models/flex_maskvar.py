import math
from functools import partial
from typing import Optional, Tuple, Union, List

from models.prompt_encoder import PromptEncoder
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn, CrossAttnBlock
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from .var import SharedAdaLin
from .image_encoder import VarImageEncoder
from torch.profiler import record_function
from utils.timer import profile_timer
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint

class FlexMaskVAR(nn.Module):
    """
    FlexMaskVAR模型，用于生成图像。
    
    参数:
        vae_local: VQVAE模型，用于编码和解码图像
        num_classes: 类别数量
        depth: Transformer的层数
        embed_dim: Transformer的嵌入维度
        num_heads: Transformer的注意力头数
        mlp_ratio: MLP的比率
        drop_rate: Dropout比率
        drop_path_rate: Dropout路径比率
        norm_eps: 归一化epsilon
        shared_aln: 是否共享AdaLN
        cond_drop_rate: 条件Dropout比率
        attn_l2_norm: 是否对注意力进行L2归一化
        patch_nums: 每个patch的分辨率列表
    """
    def __init__(
        self, vae_local: VQVAE, image_encoder: VarImageEncoder, prompt_encoder: PromptEncoder,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        image_encoder_requires_grad=True,
        prompt_encoder_requires_grad=False,
        sam_image_encoder=None,
        attention_checkpoint=False,
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.sam_image_encoder = sam_image_encoder

        # 设置image_encoder和prompt_encoder的参数是否需要梯度
        for param in self.image_encoder.parameters():
            param.requires_grad = image_encoder_requires_grad
        for param in self.prompt_encoder.parameters():
            param.requires_grad = prompt_encoder_requires_grad
        
        # 0. 验证和初始化基本参数
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        # 从VAE模型获取通道数和词汇表大小
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        # 模型核心参数
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        # 条件dropout率
        self.cond_drop_rate = cond_drop_rate
        # 进行式训练阶段索引
        self.prog_si = -1   # progressive training
        
        # 1. 初始化patch相关参数
        self.patch_nums: Tuple[int] = patch_nums  # 各阶段的patch大小
        # 计算总token数（所有阶段patch的平方和）
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        # 第一阶段的token数
        self.first_l = self.patch_nums[0] ** 2
        # 计算各阶段token的起始和结束索引
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))  # (start_index, end_index) for each stage
            cur += pn ** 2
        
        # 阶段数减1（用于进行式训练）
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        # 初始化随机数生成器
        self.rng = torch.Generator(device=dist.get_device())
        
        # 2. 初始化输入embedding
        # 获取VAE的量化器
        quant: VectorQuantizer2 = vae_local.quantize
        # 初始化VAE代理和量化器代理
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        # 初始化词嵌入层，将VAE的token映射到模型的embedding维度
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. 类别嵌入
        # 计算初始化标准差
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        # 初始化均匀概率分布
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        # 初始化类别嵌入层
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        # 初始化第一阶段的位置嵌入
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. 绝对位置嵌入
        # 为每个阶段生成位置嵌入
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        # 将所有阶段的位置嵌入拼接在一起
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # 初始化层级嵌入（类似于GPT的segment embedding，用于区分不同层级的token金字塔）
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. 主干网络块
        # 初始化共享的AdaLN层
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        # 初始化规范化层
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        # 计算stochastic depth的衰减率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        # 初始化Transformer块
        self.blocks = nn.ModuleList([
            CrossAttnBlock(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
            )
            for block_idx in range(depth)
        ])
        
        # 检查是否使用fused操作
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        # 打印模型配置信息
        print(
            f'\n[constructor]  ==== (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. 注意力掩码（仅在训练时使用）
        #    推理时不会使用，因为启用了kv缓存
        # 为每个阶段创建层级索引
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1).to(self.pos_1LC.device)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        level_idx = []
        for i in range(len(self.patch_nums)):
            level_idx.extend([i]*self.patch_nums[i]**2)
        level_idx_tensor = torch.tensor(level_idx)
        self.register_buffer('level_idx', level_idx_tensor)
        # 创建注意力掩码
        self.block_mask = None # 先不做，第一次forward再做
        
        # 6. 分类器头部
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        self.pe_grids = [self.prompt_encoder.pe_layer.forward((pn, pn)).permute(1, 2, 0) for pn in self.patch_nums]
        # self.block_masks = {}
        
        self.attention_checkpoint = attention_checkpoint
    
    def init_block_mask(self, length=None):
        def mask_mod(b, h, q_idx, k_idx) -> bool:
            # return (self.level_idx[q_idx] == self.level_idx[k_idx])
            return (self.level_idx[q_idx] >= self.level_idx[k_idx])
        
        # print("level_idx.device =", self.level_idx.device)
        # if length is None and self.block_masks.get(self.L) is None:
        #     self.block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=self.L, KV_LEN=self.L, device=self.level_idx.device)
        #     self.block_masks[self.L] = self.block_mask
        # else:
        self.block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=self.L, KV_LEN=self.L, device=self.level_idx.device)
        
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        sam_image_embedding: torch.Tensor | None, points_coords, points_labels,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if sam_image_embedding is None:
            with torch.no_grad():
                with record_function("sam_image_encoder"):
                    sam_image_embedding = self.sam_image_encoder(image)
                
        with record_function("multiscale_image_encoder"):   
            image_multiscale_feats = self.image_encoder(sam_image_embedding, pe_grids=self.pe_grids)
        with record_function("prompt_encoder"):
            with torch.no_grad():
                prompt_embedding, _ = self.prompt_encoder(
                    points=(points_coords, points_labels),
                    boxes=None,
                    masks=None,
                )
        with record_function("add_lvl_pos"):
            image_multiscale_feats = image_multiscale_feats + self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC
        
        # if label_B is None:
        #     label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        # elif isinstance(label_B, int):
        #     label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        if label_B is None:
            label_B = torch.zeros(B, dtype=torch.long, device=self.lvl_1L.device)

        # sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        with record_function("class_emb"):
            sos = cond_BD = self.class_emb(label_B)

            lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
            next_token_map = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
            # print(f'next_token_map.shape: {next_token_map.shape}')
            
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks:
            b.attn.kv_caching(True)
            b.self_attn.kv_caching(True)
        
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            with record_function(f'autogressive pass {si}'):
                ratio = si / self.num_stages_minus_1
                last_L = cur_L
                cur_L += pn*pn
                # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
                cond_BD_or_gss = self.shared_ada_lin(cond_BD)
                x = next_token_map
                # print(f'x.shape: {x.shape}')
                
                for b in self.blocks:
                    with record_function(f'autogressive pass {si} block {b}'):
                        # remember! x is kv_cached in the self-attn!
                        # just try using only the cond corresponding to the current scale
                        # TODO: use the entire cond with more complicated block mask
                        x = b(x=x, cond=image_multiscale_feats[:, :cur_L, :], prompt_cond=prompt_embedding, block_mask=None)
                with record_function(f'autogressive pass {si} logits'):
                    logits_BlV = self.get_logits(x, cond_BD)
                
                # t = cfg * ratio
                # logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
                with record_function(f'autogressive pass {si} sample'):
                    idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
                    if not more_smooth: # this is the default case
                        h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
                    else:   # not used when evaluating FID/IS/Precision/Recall
                        gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                        h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                
                with record_function(f'autogressive pass {si} get next input'):
                    h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                    f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
                    if si != self.num_stages_minus_1:   # prepare for next stage
                        next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                        next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                        # next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks:
            b.attn.kv_caching(False)
            b.self_attn.kv_caching(False)
        
        with record_function("vae_proxy"):
            ret = self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
        return ret
    
    def forward(self, 
                label_B: torch.LongTensor, 
                x_BLCv_wo_first_l: torch.Tensor, 
                sam_image_embedding: torch.Tensor,
                points_coords, points_labels,
                ) -> torch.Tensor:
        """
        前向传播函数，生成logits
        
        Args:
            label_B: 类别标签，形状为(B,)，B是batch size
            x_BLCv_wo_first_l: 教师强制输入，形状为(B, L-self.first_l, Cvae)，
                            其中L是总token数，Cvae是VAE的隐变量维度
                            注意：不包含第一个token的输入
            image_multiscale_feats: 多尺度图像特征，形状为(B, Li, C)
            prompt_embedding: 提示词嵌入，形状为(B, Lp, C)
        
        Returns:
            logits_BLV: 输出logits，形状为(B, L, V)，V是词表大小
        """
        if self.block_mask is None:
            self.init_block_mask()

        B = x_BLCv_wo_first_l.shape[0]
        if label_B is None:
            label_B = torch.zeros(B, dtype=torch.long, device=self.lvl_1L.device)
        with record_function("multiscale_image_encoder"):
            image_multiscale_feats = self.image_encoder(sam_image_embedding, pe_grids=self.pe_grids)
        with record_function("prompt_encoder"):
            with torch.no_grad():
                prompt_embedding, _ = self.prompt_encoder(
                    points=(points_coords, points_labels),
                    boxes=None,
                    masks=None,
                )

        # print(f'MaskVAR: x_BLCv_wo_first_l.shape: {x_BLCv_wo_first_l.shape}')
        # print(f'MaskVAR: image_multiscale_feats.shape: {image_multiscale_feats.shape}')
        # print(f'MaskVAR: prompt_embedding.shape: {prompt_embedding.shape}')

        # 获取当前progressive training阶段的token范围
        # 如果prog_si<0，则使用完整序列(0, self.L)
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]  # batch size
        
        with record_function("class_emb"):
            # 使用FP32精度处理类别嵌入和位置编码
            with torch.amp.autocast(device_type='cuda', enabled=False):
                # 标签dropout：以cond_drop_rate概率将标签替换为num_classes（表示无类别）
                # label_B = torch.where(
                #     torch.rand(B, device=label_B.device) < self.cond_drop_rate,
                #     self.num_classes,  # 使用num_classes表示无类别
                #     label_B
                # )
                
                # 获取类别嵌入（条件嵌入）
                # sos/cond_BD: (B, D), 其中D是嵌入维度
                sos = cond_BD = self.class_emb(label_B)
                
                # 为序列开始添加位置编码
                # sos: (B, first_l, D)
                sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
                
                # 构建输入序列
                if self.prog_si == 0:  # 如果是第一个progressive阶段
                    x_BLC = sos  # 只使用开始标记
                else:
                    # 将开始标记与输入序列拼接
                    # word_embed将输入映射到与sos相同的嵌入空间
                    x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
                    # print(f'sos.shape: {sos.shape}')
                    # print(f'x_BLC_wo_first_l.shape: {x_BLCv_wo_first_l.shape}')
                    # print(f'x_BLC.shape: {x_BLC.shape}')
                
                # 添加层级嵌入和位置编码
                # lvl_embed: 不同层级的嵌入
                # pos_1LC: 位置编码
                x_BLC = x_BLC + self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
                image_multiscale_feats = image_multiscale_feats + self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
        
        # 注意力掩码（用于自回归生成）
        # attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        
        # 条件投影（用于AdaLN）
        # cond_BD_or_gss: (B, D)
        # cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # 获取混合精度训练中的主数据类型
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        # 确保所有张量使用相同的数据类型
        x_BLC = x_BLC.to(dtype=main_type)
        # 也将条件和提示嵌入转换为相同的主数据类型，避免在checkpoint重算时出现隐式类型变化
        image_multiscale_feats = image_multiscale_feats.to(dtype=main_type)
        prompt_embedding = prompt_embedding.to(dtype=main_type)
        # cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        # attn_bias = attn_bias.to(dtype=main_type)
        
        # print(f'x_BLC.shape: {x_BLC.shape}')
        
        # 通过多个transformer块
        for i, b in enumerate(self.blocks):
            with record_function(f"transformer_block_{i}"):
                # 每个block包含自注意力和前馈网络
                # 使用AdaLN（自适应层归一化）结合条件信息
                # x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
                
                # 启用梯度检查点以降低显存峰值
                # 注意：checkpoint 需要一个仅以 Tensor 为参数的可调用对象，这里通过 lambda 包装模块调用
                # 注意：torch.utils.checkpoint 仅支持 Tensor 类型作为显式输入参数。
                # self.block_mask 是 flex_attention 的 BlockMask 对象，非 Tensor，将其通过闭包捕获，避免作为显式输入传入。
                # 同时避免在重算时因非 Tensor 对象状态差异导致的 metadata mismatch。
                if self.attention_checkpoint:
                    x_BLC = checkpoint(
                        lambda x, cond, pcond: b(x=x, cond=cond, prompt_cond=pcond, block_mask=self.block_mask),
                        x_BLC,
                        image_multiscale_feats,
                        prompt_embedding,
                        use_reentrant=False,
                    )
                else:
                    x_BLC = b(x=x_BLC, cond=image_multiscale_feats, prompt_cond=prompt_embedding, block_mask=self.block_mask)
        
        # 获取最终的logits
        with record_function("get_logits"):
            x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        # 以下代码用于保持计算图完整性，确保梯度正常传播
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                # 确保word_embed的梯度被计算
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                # 对于非线性的word_embed，确保其参数梯度被计算
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        
        return x_BLC  # (B, L, V)
    
    # useless
    # def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
    #     if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
    #     print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
    #     for m in self.modules():
    #         with_weight = hasattr(m, 'weight') and m.weight is not None
    #         with_bias = hasattr(m, 'bias') and m.bias is not None
    #         if isinstance(m, nn.Linear):
    #             nn.init.trunc_normal_(m.weight.data, std=init_std)
    #             if with_bias: m.bias.data.zero_()
    #         elif isinstance(m, nn.Embedding):
    #             nn.init.trunc_normal_(m.weight.data, std=init_std)
    #             if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
    #         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
    #             if with_weight: m.weight.data.fill_(1.)
    #             if with_bias: m.bias.data.zero_()
    #         # conv: VAR has no conv, only VQVAE has conv
    #         elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
    #             if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
    #             else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
    #             if with_bias: m.bias.data.zero_()
        
    #     if init_head >= 0:
    #         if isinstance(self.head, nn.Linear):
    #             self.head.weight.data.mul_(init_head)
    #             self.head.bias.data.zero_()
    #         elif isinstance(self.head, nn.Sequential):
    #             self.head[-1].weight.data.mul_(init_head)
    #             self.head[-1].bias.data.zero_()
        
    #     if isinstance(self.head_nm, AdaLNBeforeHead):
    #         self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
    #         if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
    #             self.head_nm.ada_lin[-1].bias.data.zero_()
        
    #     depth = len(self.blocks)
    #     for block_idx, sab in enumerate(self.blocks):
    #         sab: AdaLNSelfAttn
    #         sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
    #         sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
    #         if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
    #             nn.init.ones_(sab.ffn.fcg.bias)
    #             nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
    #         if hasattr(sab, 'ada_lin'):
    #             sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
    #             sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
    #             if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
    #                 sab.ada_lin[-1].bias.data.zero_()
    #         elif hasattr(sab, 'ada_gss'):
    #             sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
    #             sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'