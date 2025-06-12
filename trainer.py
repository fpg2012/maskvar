import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
    ):
        super(VARTrainer, self).__init__()
        
        # 模型相关组件
        self.var = var  # 主模型（可能是 DDP 包装后的）
        self.vae_local = vae_local  # VAE 模型
        self.quantize_local = vae_local.quantize  # VAE 的量化器
        self.quantize_local: VectorQuantizer2  # 类型注解：向量量化器
        self.var_wo_ddp = var_wo_ddp  # 未包装的模型实例（用于 torch.compile 优化后）
        self.var_opt = var_opt  # 优化器
        
        # 随机数生成器
        del self.var_wo_ddp.rng  # 删除原有的 RNG
        self.var_wo_ddp.rng = torch.Generator(device=device)  # 创建新的 RNG 并指定设备
        
        # 损失函数配置
        self.label_smooth = label_smooth  # 标签平滑系数
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')  # 训练用损失函数（带标签平滑）
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')  # 验证用损失函数（无标签平滑）
        
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
    
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            self.var_wo_ddp.forward
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        """
        Train one step on the given input.
        
        Args:
            it: 当前epoch内的迭代步数
            g_it: 全局迭代步数（跨epoch累计）
            stepping: 是否执行optimizer step（梯度累积时可能需要跳过）
            metric_lg: 指标记录器
            tb_lg: TensorBoard记录器
            inp_B3HW: 输入图像，形状(B, 3, H, W)
            label_B: 标签，形状(B, L)
            prog_si: 当前progressive training阶段索引
            prog_wp_it: 当前progressive阶段的warmup迭代数
        
        Returns:
            grad_norm: 梯度范数
            scale_log2: 混合精度训练的scaler的log2值
        """
        # ==================== Progressive Training 阶段管理 ====================
        # 设置模型和量化器的progressive阶段
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        
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
        B, V = label_B.shape[0], self.vae_local.vocab_size  # batch_size, 词表大小
        self.var.require_backward_grad_sync = stepping  # 设置DDP梯度同步
        
        # 将图像转换为token索引序列
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)  # 拼接所有token (B, L)
        # 生成模型输入（去掉第一个token）
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        
        # 混合精度训练上下文
        with self.var_opt.amp_ctx:
            # 前向传播
            logits_BLV = self.var(label_B, x_BLCv_wo_first_l)  # (B, L, V)
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
        pred_BL = logits_BLV.data.argmax(dim=-1)  # 预测结果 (B, L)
        
        # 在指定迭代步记录指标
        if it == 0 or it in metric_lg.log_iters:
            # 计算平均loss和准确率
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            
            # 在progressive training时不计算tail指标
            if prog_si >= 0:
                Ltail = acc_tail = -1
            else:
                # 计算最后一部分token的loss和准确率
                Ltail = self.val_loss(
                    logits_BLV.data[:, -self.last_l:].reshape(-1, V), 
                    gt_BL[:, -self.last_l:].reshape(-1)
                ).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            
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
                    pred = logits_BLV.data[:, bg:ed].reshape(-1, V)
                    tar = gt_BL[:, bg:ed].reshape(-1)
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
        
        # 重置progressive阶段
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
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
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
