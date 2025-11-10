from typing import List, Tuple, Optional, Iterator, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import dist
import time
from tqdm import tqdm
import numpy as np
import gc

from maskvar.models import sam_image_encoder
from maskvar.models.maskvar import MaskVAR
from maskvar.models.flex_maskvar import FlexMaskVAR
from maskvar.models.image_encoder import ImageEncoder
from maskvar.models.sam_image_encoder import ImageEncoderViT as SamImageEncoder
from maskvar.models import VAR, VQVAE, VectorQuantizer2
from maskvar.utils.amp_sc import AmpOptimizer
from maskvar.utils.clicker import Clicker
from maskvar.utils.misc import MetricLogger, TensorboardLogger

from maskvar.utils.clicker import init_clicks, predict_next_click, to_sam_format
from maskvar.utils import resize_longest_side
from maskvar.utils.loss import FocalLossGeneral

from maskvar.datasets.mask_level_dataset import MaskLevelDataset

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

class InteractiveConfig:
    def __init__(self, num_random_clicks: int = 1, num_interactive_clicks: int = 20):
        self.num_random_clicks = num_random_clicks
        self.num_interactive_clicks = num_interactive_clicks

class MaskVarTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: MaskVAR | FlexMaskVAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float, interactive_config: InteractiveConfig,
        sam_image_encoder = None
    ):
        super(MaskVarTrainer, self).__init__()
        
        # 模型相关组件
        self.var = var  # 主模型（可能是 DDP 包装后的）
        self.vae_local = vae_local  # VAE 模型
        self.quantize_local = vae_local.quantize  # VAE 的量化器
        self.quantize_local: VectorQuantizer2  # 类型注解：向量量化器
        # 确保在验证时可以直接访问未 DDP 包装的模型实例
        self.var_wo_ddp = var_wo_ddp  # 未包装的模型实例（用于 torch.compile 优化后）
        self.var_opt = var_opt  # 优化器
        self.sam_image_encoder = sam_image_encoder
        
        # 随机数生成器
        # del self.var_wo_ddp.rng  # 删除原有的 RNG
        # self.var_wo_ddp.rng = torch.Generator(device=device)  # 创建新的 RNG 并指定设备
        
        # 损失函数配置
        self.label_smooth = label_smooth  # 标签平滑系数
        # self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')  # 训练用损失函数（带标签平滑）
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')  # 验证用损失函数（无标签平滑）
        self.train_loss = FocalLossGeneral(alpha=0.1, gamma=2.0, label_smooth=label_smooth, reduction='none')
        
        # Patch 相关配置
        self.L = sum(pn * pn for pn in patch_nums)  # 总 token 数（所有 patch 的 token 数之和）
        self.last_l = patch_nums[-1] * patch_nums[-1]  # 最后一个分辨率的 token 数
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L  # 均匀的 loss 权重
        
        # Progressive Training 配置
        self.patch_nums = patch_nums  # 各阶段的分辨率列表（如 [16, 32]）
        self.resos = resos  # 各阶段的分辨率名称/标识（如 ['16x16', '32x32']）
        self.begin_ends = []  # 存储每个阶段 token 的起始和结束索引
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))  # 记录每个阶段的 token 范围
            cur += pn * pn  # 更新当前 token 索引
        
        # 训练状态
        self.prog_it = 0  # 当前 progressive 阶段的迭代次数
        self.last_prog_si = -1  # 上一个 progressive 阶段索引
        self.first_prog = True  # 是否处于第一个 progressive 阶段

        self.interactive_config = interactive_config
    
    # !TODO: not done
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for image, image_embed_sam, single_mask_normalized, single_mask in ld_val:
            B, V = single_mask.shape[0], self.vae_local.vocab_size
            image = image.to(dist.get_device(), non_blocking=True)
            image_embed_sam = image_embed_sam.to(dist.get_device(), non_blocking=True)
            single_mask_normalized = single_mask_normalized.to(dist.get_device(), non_blocking=True)
            single_mask = single_mask.to(dist.get_device(), non_blocking=True)

            label = torch.zeros(B, device=dist.get_device(), dtype=torch.long)
            
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(single_mask_normalized)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            del gt_idx_Bl  # 立即释放

            # sample clicks
            single_mask_cpu = single_mask.cpu().numpy()
            click_lists = []
            for i in range(B):
                click_list, eroded_mask, dt = init_clicks(single_mask_cpu[i][0], num_random_clicks=self.interactive_config.num_random_clicks)
                click_lists.append(click_list)
            point_coords = []
            point_labels = []
            for i in range(B):
                coords, labels = to_sam_format(click_lists[i], pad_size=self.interactive_config.num_interactive_clicks+3)
                point_coords.append(coords)
                point_labels.append(labels)
            del single_mask_cpu, click_lists  # 清理临时变量
            
            self.var_wo_ddp.forward
            logits_BLV = self.var_wo_ddp(
                label, 
                x_BLCv_wo_first_l, 
                image_embed_sam, 
                points_coords=torch.stack(point_coords).to(device=dist.get_device(), non_blocking=True), 
                points_labels=torch.stack(point_labels).to(device=dist.get_device(), non_blocking=True)
            )
            L_mean += self.val_loss(logits_BLV.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
            
            # 清理每个批次的tensor
            del image, image_embed_sam, single_mask_normalized, single_mask, label
            del gt_BL, x_BLCv_wo_first_l, logits_BLV, point_coords, point_labels
        
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def interactive_train_step(
        self, it: int, g_it: int, stepping: bool,
        metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        gt_mask_normalized_B1HW: FTen, 
        label_B: Union[ITen, FTen], 
        prog_si: int, prog_wp_it: float,
        image_embed_sam_BCencHW: FTen,
        gt_mask_B1HW,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        """
        Interactive training step.

        Args:
            it: 当前epoch内的迭代步数
            g_it: 全局迭代步数（跨epoch累计）
            stepping: 是否执行optimizer step（梯度累积时可能需要跳过）
            metric_lg: 指标记录器
            tb_lg: TensorBoard记录器
            gt_mask_normalized_B1HW: 输入gt mask，形状(B, 1, H, W)
            label_B: 标签，形状(B, L)
            prog_si: 当前progressive training阶段索引
            prog_wp_it: 当前progressive阶段的warmup迭代数
            image_embed_sam_BCencHW: 输入的SAM特征，形状(B, C_enc, H, W)
            gt_mask_B1HW: 输入gt mask，形状(B, 1, H, W)，0-1mask，未归一化

        Returns:
            grad_norm: 梯度范数
            scale_log2: 混合精度训练的scaler的log2值
        """
        B = gt_mask_normalized_B1HW.shape[0]
        # sample init clicks for each batch
        # 使用detach()避免保留计算图，减少显存占用
        gt_mask_cpu = gt_mask_B1HW.detach().cpu().numpy()
        click_lists = []
        for i in range(B):
            click_list, eroded_mask, dt = init_clicks(gt_mask_cpu[i][0], num_random_clicks=self.interactive_config.num_random_clicks)
            click_lists.append(click_list)

        point_coords = []
        point_labels = []
        for i in range(B):
            coords, labels = to_sam_format(click_lists[i], pad_size=self.interactive_config.num_interactive_clicks+3)
            point_coords.append(coords)
            point_labels.append(labels)
        
        # 显式删除CPU副本
        del gt_mask_cpu, click_lists

        # interactive sample clicks

        # pred_masks = self.var.module.autoregressive_infer_cfg(
        #     B=B,
        #     label_B=None,
        #     sam_image_embedding=image_embed_sam_BCencHW,
        #     points_coords=torch.stack(point_coords).to(device=dist.get_device()),
        #     points_labels=torch.stack(point_labels).to(device=dist.get_device()),
        # ) # (B, 1, 256, 256)

        # # sample interact clicks for each batch
        # not_click_maps = [np.ones_like(gt_mask_B1HW[i].cpu().numpy()[0], dtype=bool) for i in range(B)]
        # num_clicks = np.random.randint(1, self.interactive_config.num_interactive_clicks+1)
        # for i in range(B):
        #     for _ in range(num_clicks):
        #         predict_next_click(
        #             gt_mask_B1HW[i].cpu().numpy()[0], 
        #             pred_mask=pred_masks[i].cpu().numpy()[0], 
        #             click_list=click_lists[i], 
        #             not_clicked_map=not_click_maps[i]
        #         )
            
        #     point_coords.clear()
        #     point_labels.clear()
        #     for i in range(B):
        #         coords, labels = to_sam_format(click_lists[i], pad_size=self.interactive_config.num_interactive_clicks+3)
        #         point_coords.append(coords)
        #         point_labels.append(labels)

        #     pred_masks = self.var.module.autoregressive_infer_cfg(
        #         B=B,
        #         label_B=None,
        #         sam_image_embedding=image_embed_sam_BCencHW,
        #         points_coords=torch.stack(point_coords).to(device=dist.get_device()),
        #         points_labels=torch.stack(point_labels).to(device=dist.get_device()),
        #     ) # (B, 1, 256, 256)
        
        # point_coords.clear()
        # point_labels.clear()
        # for i in range(B):
        #     coords, labels = to_sam_format(click_lists[i], pad_size=self.interactive_config.num_interactive_clicks+3)
        #     point_coords.append(coords)
        #     point_labels.append(labels)

        # 提前转换到GPU，避免在train_step中重复转换
        prompt_points_coords_BN2 = torch.stack(point_coords).to(device=dist.get_device(), non_blocking=True)
        prompt_points_labels_BN = torch.stack(point_labels).to(device=dist.get_device(), non_blocking=True)
        
        # 清理临时列表
        del point_coords, point_labels
        
        return self.train_step(
            it=it, g_it=g_it, stepping=stepping,
            metric_lg=metric_lg, tb_lg=tb_lg,
            gt_mask_B1HW=gt_mask_normalized_B1HW,
            label_B=None,
            prog_si=prog_si, prog_wp_it=prog_wp_it,
            image_embed_sam_BCencHW=image_embed_sam_BCencHW,
            prompt_points_coords_BN2=prompt_points_coords_BN2,
            prompt_points_labels_BN=prompt_points_labels_BN,
        )
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, 
        metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        gt_mask_B1HW: FTen, 
        label_B: Union[ITen, FTen], 
        prog_si: int, prog_wp_it: float,
        image_embed_sam_BCencHW: FTen,
        prompt_points_coords_BN2: FTen,
        prompt_points_labels_BN: ITen,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        """
        Train one step on the given input.
        
        Args:
            it: 当前epoch内的迭代步数
            g_it: 全局迭代步数（跨epoch累计）
            stepping: 是否执行optimizer step（梯度累积时可能需要跳过）
            metric_lg: 指标记录器
            tb_lg: TensorBoard记录器
            gt_mask_B1HW: 输入gt mask，形状(B, 1, H, W)
            label_B: 标签，形状(B, L)
            prog_si: 当前progressive training阶段索引
            prog_wp_it: 当前progressive阶段的warmup迭代数
            image_embed_sam_BCencHW: 输入的SAM特征，形状(B, C_enc, H, W)
            prompt_points_coords_BN2: 输入的prompt点坐标，形状(B, N, 2)
            prompt_points_labels_BN: 输入的prompt点标签，形状(B, N)
        
        Returns:
            grad_norm: 梯度范数
            scale_log2: 混合精度训练的scaler的log2值
        """
        # ==================== Progressive Training 阶段管理 ====================
        # 设置模型和量化器的progressive阶段
        # self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        
        # 如果progressive阶段变化，更新状态
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: 
                self.first_prog = False  # 标记已过第一个progressive阶段
            self.last_prog_si = prog_si  # 更新上一个阶段索引
            self.prog_it = 0  # 重置当前阶段迭代计数
        
        self.prog_it += 1
        # 计算warmup进度，范围[0.01, 1.0]
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        # 第一个progressive阶段不进行warmup（因为已经在warmup阶段完成）
        if self.first_prog: 
            prog_wp = 1
        # 如果是最后一个progressive阶段，视为不使用progressive training
        if prog_si == len(self.patch_nums) - 1: 
            prog_si = -1
        
        # ==================== 前向传播 ====================
        B, V = gt_mask_B1HW.shape[0], self.vae_local.vocab_size  # batch_size, 词表大小
        self.var.require_backward_grad_sync = stepping  # 设置DDP梯度同步
        
        # 将图像转换为token索引序列
        # print(f'gt_mask_B1HW.dtype: {gt_mask_B1HW.dtype}')
        with torch.no_grad():
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(gt_mask_B1HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)  # 拼接所有token (B, L)
            # 生成模型输入（去掉第一个token）
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            # 立即删除中间结果，释放显存
            del gt_idx_Bl
        
        # 混合精度训练上下文
        with self.var_opt.amp_ctx:
            # 前向传播
            # torch.cuda.synchronize()
            # print(f"[rank{dist.get_rank()}][MEM USAGE] before!! \n {torch.cuda.memory_summary(device=dist.get_device(), abbreviated=True)}")
            logits_BLV = self.var(label_B, x_BLCv_wo_first_l, image_embed_sam_BCencHW, prompt_points_coords_BN2, prompt_points_labels_BN)  # (B, L, V)
            # torch.cuda.synchronize()
            # print(f"[rank{dist.get_rank()}][MEM USAGE] after!! \n {torch.cuda.memory_summary(device=dist.get_device(), abbreviated=True)}")
            # 计算每个token的loss (B, L)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            
            # 应用progressive training的loss权重
            if prog_si >= 0:  # progressive training阶段
                bg, ed = self.begin_ends[prog_si]  # 当前阶段的token范围
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()  # 只取当前阶段之前的权重
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)  # 当前阶段应用warmup
            else:  # 非progressive training
                lw = self.loss_weight  # 使用完整权重
            
            # 加权平均得到最终loss
            loss = loss.mul(lw).sum(dim=-1).mean()
        # ==================== 反向传播 ====================
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # ==================== 记录指标 ====================
        # 立即 detach 所有需要的 tensor，避免保留计算图
        with torch.no_grad():
            # 使用clone()确保完全断开计算图
            logits_BLV_detached = logits_BLV.detach().clone()
            gt_BL_detached = gt_BL.detach().clone()
            pred_BL = logits_BLV_detached.argmax(dim=-1)  # 预测结果 (B, L)
            
            # 在指定迭代步记录指标
            if it == 0 or it in metric_lg.log_iters:
                # 计算平均loss和准确率
                Lmean = self.val_loss(logits_BLV_detached.view(-1, V), gt_BL_detached.view(-1)).item()
                acc_mean = (pred_BL == gt_BL_detached).float().mean().item() * 100
                
                # 在progressive training时不计算tail指标
                if prog_si >= 0:
                    Ltail = acc_tail = -1
                else:
                    # 计算最后一部分token的loss和准确率
                    Ltail = self.val_loss(
                        logits_BLV_detached[:, -self.last_l:].reshape(-1, V), 
                        gt_BL_detached[:, -self.last_l:].reshape(-1)
                    ).item()
                    acc_tail = (pred_BL[:, -self.last_l:] == gt_BL_detached[:, -self.last_l:]).float().mean().item() * 100
                
                # 更新指标记录器
                grad_norm_val = grad_norm.item()
                metric_lg.update(
                    Lm=Lmean,      # 平均loss
                    Lt=Ltail,      # 尾部loss
                    Accm=acc_mean, # 平均准确率
                    Acct=acc_tail, # 尾部准确率
                    tnm=grad_norm_val  # 梯度范数
                )
            
            # ==================== TensorBoard日志 ====================
            if g_it == 0 or (g_it + 1) % 500 == 0:
                # 计算每个token类别的使用频率
                prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
                dist.allreduce(prob_per_class_is_chosen)  # 多卡同步
                prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
                # 计算token使用率（使用频率>0.1%/V的token比例）
                cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
                
                # 只在主进程记录TensorBoard
                if dist.is_master():
                    if g_it == 0:  # 初始记录
                        tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                        tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                    
                    # 准备记录数据
                    kw = dict(z_voc_usage=cluster_usage)
                    
                    # 为每个progressive阶段记录指标
                    for si, (bg, ed) in enumerate(self.begin_ends):
                        if 0 <= prog_si < si:  # 只记录当前及之前的阶段
                            break
                        # 计算当前阶段的预测和标签
                        pred = logits_BLV_detached[:, bg:ed].reshape(-1, V)
                        tar = gt_BL_detached[:, bg:ed].reshape(-1)
                        # 计算准确率和交叉熵
                        acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                        ce = self.val_loss(pred, tar).item()
                        # 添加到记录
                        kw[f'acc_{self.resos[si]}'] = acc
                        kw[f'L_{self.resos[si]}'] = ce
                    
                    # 更新TensorBoard
                    tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                    tb_lg.update(
                        head='AR_iter_schedule',
                        prog_a_reso=self.resos[prog_si],  # 当前分辨率
                        prog_si=prog_si,                  # 阶段索引
                        prog_wp=prog_wp,                  # warmup进度
                        step=g_it
                    )
        
        # 显式删除大 tensor，释放显存
        del logits_BLV, logits_BLV_detached, gt_BL, gt_BL_detached, x_BLCv_wo_first_l, loss, pred_BL
        # 删除输入tensor的引用
        del image_embed_sam_BCencHW, prompt_points_coords_BN2, prompt_points_labels_BN
        
        # 定期清理显存碎片
        if stepping and g_it % 10 == 0:
            torch.cuda.empty_cache()
        
        # 重置progressive阶段
        # self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[MaskVarTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[MaskVarTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[MaskVarTrainer.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
