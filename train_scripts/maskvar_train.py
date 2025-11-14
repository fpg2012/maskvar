import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial
from itertools import islice, cycle


import torch
import torch.distributed as tdist
from torch.utils.data import DataLoader


import maskvar.dist as dist
from maskvar.models.vqvae_single import VQVAE_Single
from maskvar.utils import arg_util, misc
# from maskvar.utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from maskvar.utils.misc import auto_resume

from maskvar.maskseg_build_everything import (
    build_cocolvis_dataset,
    build_hqseg44k_dataset,
    build_vqvae_single,
    build_maskvar,
    build_maskvar_v2,
    build_maskvar_flex,
    build_maskvar_flex_5_stages,
    build_maskvar_flex_mobile_5_stages
)
from maskvar_trainer import InteractiveConfig, MaskVarTrainer
from maskvar.models.maskvar import MaskVAR
from maskvar.models.flex_maskvar import FlexMaskVAR
from maskvar.datasets import MaskLevelDataset, MaskLevelDatasetRandom, MaskLevelDatasetDummy

def build_everything(args: arg_util.Args):
    """
    构建所有组件

    returns:
        tb_lg: tensorboard logger
        trainer: trainer
        start_ep: starting epoch
        start_it: starting iteration
        iters_train: number of iterations in training dataset
        ld_train: training data loader
        ld_val: validation data loader
    """
    # resume
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), verbose=True)
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    
    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    # 构建模型组件
    from torch.nn.parallel import DistributedDataParallel as DDP
    from maskvar.utils.amp_sc import AmpOptimizer
    from maskvar.utils.lr_control import filter_params

    # !TODO: replace with custom model
    # 构建VAE和VAR模型
    vae_local, var_wo_ddp, sam_image_encoder = build_maskvar_flex_mobile_5_stages('out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth', 'ckpt/mobile_sam.pt', device=args.device) 
    # vae_local, var_wo_ddp, sam_image_encoder = build_maskvar_flex('ckpt/vqvae_single.pth', 'ckpt/sam_vit_b_01ec64.pth', device=args.device) 
    # vae_local, var_wo_ddp, sam_image_encoder = build_maskvar_v2('out_vqvae_4_stages_2/ckpt/vqvae_single_epoch_40.pth', 'ckpt/sam_vit_b_01ec64.pth', flash_if_available=True, device=args.device) 
    
    dist.barrier()
    # !TODO: load state dict
    
    vae_local: VQVAE_Single = args.compile_model(vae_local, args.vfast)
    vae_local.eval()
    var_wo_ddp: FlexMaskVAR = args.compile_model(var_wo_ddp, args.tfast)
    var_wo_ddp.prompt_encoder.eval()
    sam_image_encoder = torch.compile(sam_image_encoder, mode='reduce-overhead')
    sam_image_encoder.eval()

    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=True, broadcast_buffers=False)
    
    print(f'[INIT] MaskVAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE_single', vae_local), ('VAE_single.enc', vae_local.encoder), ('VAE_single.dec', vae_local.decoder), ('VAE_single.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('MaskVAR', var_wo_ddp),)]) + '\n\n')
    
    # dataset
    print(f'[INIT] Building dataset...')
    if args.local_debug:
        # for debug, overfit on one image
        train_set, _ = build_hqseg44k_dataset() # validate on train set
        train_set_masklevel = MaskLevelDatasetDummy(
            dataset=train_set,
            sam_encoder=sam_image_encoder,
            device=args.device,
            mask_filter_thresh=0.1,
            seed=42,
        )
        val_set_masklevel = MaskLevelDatasetDummy(
            dataset=train_set,
            sam_encoder=sam_image_encoder,
            device=args.device,
            mask_filter_thresh=0.1,
            seed=42,
        )
    else:
        train_set, val_set = build_cocolvis_dataset()
        train_set_masklevel = MaskLevelDatasetRandom(
            dataset=train_set,
            sam_encoder=sam_image_encoder,
            device=args.device,
            mask_filter_thresh=0.1,
            seed=42,
            infinite=True,
        )
        val_set_masklevel = MaskLevelDatasetRandom(
            dataset=val_set,
            sam_encoder=sam_image_encoder,
            device=args.device,
            mask_filter_thresh=0.1,
            seed=42,
            infinite=True,
        )
    
    print(f'[INIT] counting masks...')
    # Per-rank minimum masks across splits (ensures all ranks have at least this many masks)
    # min_masks = train_set.max_count_masks(world_size=tdist.get_world_size())
    # Convert mask count to batch iteration count
    # iters_train = min_masks // args.batch_size
    
    if args.local_debug:
        iters_train = 32
        iters_val = 16
        train_dataloader = DataLoader(train_set_masklevel, batch_size=args.batch_size, drop_last=True)
        val_dataloader = DataLoader(train_set_masklevel, batch_size=args.batch_size, drop_last=True)
        val_dataloader = islice(val_dataloader, iters_val)
    else:
        iters_train = 8000
        iters_val = 64
        # Build dataloaders; drop_last to ensure exact number of full batches
        train_dataloader = DataLoader(train_set_masklevel, batch_size=args.batch_size, drop_last=True)
        val_dataloader = DataLoader(val_set_masklevel, batch_size=args.batch_size, drop_last=True)
        val_dataloader = islice(val_dataloader, iters_val)
    
    # 构建优化器
    # 过滤参数，为不同的参数组设置不同的优化策略
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })
    
    # 选择优化器类型（Adam或AdamW）
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    
    # 设置优化器参数
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
    # 创建混合精度优化器
    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, 
        optimizer=opt_clz(params=para_groups, **opt_kw), 
        names=names, 
        paras=paras,
        grad_clip=args.tclip, 
        n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    patch_nums = var_wo_ddp.patch_nums
    assert isinstance(patch_nums, (list, tuple)) and len(patch_nums) > 0, \
        f'patch_nums must be a list or tuple with at least one element, got {type(patch_nums)} with length {len(patch_nums)}'
    
    # build trainer
    trainer = MaskVarTrainer(
        device=args.device, patch_nums=patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var,
        var_opt=var_optim, label_smooth=args.ls,
        interactive_config=InteractiveConfig(
            num_random_clicks=1,
            num_interactive_clicks=10,
        ),
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    del vae_local, var_wo_ddp, var, var_optim
    
    # if args.local_debug:
    #     raise NotImplementedError
        # rng = torch.Generator('cpu')
        # rng.manual_seed(0)
        # B = 4
        # inp = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        # label = torch.ones(B, dtype=torch.long)
        
        # me = misc.MetricLogger(delimiter='  ')
        # trainer.train_step(
        #     it=0, g_it=0, stepping=True, metric_lg=me, tb_lg=tb_lg,
        #     inp_B3HW=inp, label_B=label, prog_si=args.pg0, prog_wp_it=20,
        # )
        # trainer.load_state_dict(trainer.state_dict())
        # trainer.train_step(
        #     it=99, g_it=599, stepping=True, metric_lg=me, tb_lg=tb_lg,
        #     inp_B3HW=inp, label_B=label, prog_si=-1, prog_wp_it=20,
        # )
        # print({k: meter.global_avg for k, meter in me.meters.items()})
        
        # args.dump_log(); tb_lg.flush(); tb_lg.close()
        # if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
        #     sys.stdout.close(), sys.stderr.close()
        # exit(0)
    
    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, train_dataloader, val_dataloader
    )


# 主训练函数
def main_training():
    # 添加调试信息
    print(f"[DEBUG] Starting main_training on rank {tdist.get_rank() if tdist.is_initialized() else 0}")
    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[DEBUG] Current device: {torch.cuda.current_device()}")
        print(f"[DEBUG] Device count: {torch.cuda.device_count()}")
    
    # 初始化分布式训练和获取参数
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    
    # 如果是调试模式，启用梯度异常检测
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    
    # 构建所有训练组件
    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
    ) = build_everything(args)
    
    args.patch_nums = trainer.patch_nums
    
    # 开始训练
    start_time = time.time()
    
    # 初始化最佳指标
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    
    # 当前指标初始化
    L_mean, L_tail = -1, -1
    
    # 开始训练循环
    for ep in range(start_ep, args.ep):
        print(f"[DEBUG] Starting epoch {ep} on rank {tdist.get_rank() if tdist.is_initialized() else 0}")

        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        tb_lg.set_step(ep * iters_train)

        # 避免使用cycle()，直接使用islice限制迭代次数
        # cycle()会保留所有元素的引用，导致显存累积
        ld_train_iter = islice(ld_train, iters_train)
        
        # print(f"[rank{tdist.get_rank()}][MEM USAGE]\n{torch.cuda.memory_summary(device=None, abbreviated=False)}")
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, args, tb_lg, ld_train_iter, iters_train, trainer
        )
        
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1:
            best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail)
        # is_val_and_also_saving = (ep + 1) % 10 == 0 or (ep + 1) == args.ep
        is_val_and_also_saving = True
        if args.local_debug:
            is_val_and_also_saving = (ep % 20 == 0 and ep > 0)
            
        if is_val_and_also_saving:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)

            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail)
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')
            
            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
            dist.barrier()
        
        print(    f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        tb_lg.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        args.dump_log(); tb_lg.flush()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tb_lg.flush(); tb_lg.close()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer):
    """
    Train one epoch

    Args:
        ep (int): epoch number
        is_first_ep (bool): whether it is the first epoch
        start_it (int): start iteration
        args (arg_util.Args): arguments
        tb_lg (misc.TensorboardLogger): tensorboard logger
        ld_or_itrt (DataLoader or Iterator): dataloader or iterator
        iters_train (int): number of iterations for training
        trainer (VARTrainer): trainer

    Returns:
        dict: metrics
        time_preds: time predictions
    """
    # import heavy packages after Dataloader object creation
    from maskvar_trainer import MaskVarTrainer
    from maskvar.utils.lr_control import lr_wd_annealing
    trainer: MaskVarTrainer
    
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    # Main training loop: iterate through the training data
    for it, (image, image_embed_sam, single_mask_normalized, single_mask) in me.log_every(start_it, iters_train, ld_or_itrt, 100 if iters_train > 8000 else 5, header):
            # Calculate global iteration count (across all epochs)
            g_it = ep * iters_train + it
            if it < start_it: continue  # Skip iterations if resuming from checkpoint
            if is_first_ep and it == start_it: warnings.resetwarnings()  # Reset any warnings if this is the first epoch
            
            batch_size = single_mask_normalized.shape[0]
            label = torch.zeros(batch_size, )
            inp = single_mask_normalized

            # Move data to the specified device (GPU/CPU)
            # 使用non_blocking=True加速CPU-GPU传输
            inp = inp.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)
            image_embed_sam = image_embed_sam.to(args.device, non_blocking=True)
            single_mask = single_mask.to(args.device, non_blocking=True)
            
            # Update current iteration info for logging
            args.cur_it = f'{it+1}/{iters_train}'
            
            # Calculate warmup period in terms of iterations
            wp_it = args.wp * iters_train
            # Update learning rate and weight decay based on the schedule
            min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
                args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, 
                g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe
            )
            args.cur_lr, args.cur_wd = max_tlr, max_twd
            
            # Progressive training logic
            if args.pg:  # If progressive training is enabled
                if g_it <= wp_it: 
                    prog_si = args.pg0  # Use initial progressive stage during warmup
                elif g_it >= max_it * args.pg: 
                    prog_si = len(args.patch_nums) - 1  # Use final progressive stage
                else:
                    # Calculate current progressive stage based on training progress
                    delta = len(args.patch_nums) - 1 - args.pg0
                    progress = min(max((g_it - wp_it) / (max_it * args.pg - wp_it), 0), 1)  # Normalized progress [0,1]
                    prog_si = args.pg0 + round(progress * delta)  # Linearly interpolate stage
            else:
                prog_si = -1  # No progressive training
            
            # Determine if this is a weight update step (based on accumulation steps)
            stepping = (g_it + 1) % args.ac == 0
            step_cnt += int(stepping)
            
            # Perform a single training step
            grad_norm, scale_log2 = trainer.interactive_train_step(
                it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
                gt_mask_normalized_B1HW=inp, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
                image_embed_sam_BCencHW=image_embed_sam, gt_mask_B1HW=single_mask,
            )
            
            # 显式删除输入tensor，释放显存
            del image, image_embed_sam, single_mask_normalized, single_mask, inp, label
            
            # Update metrics and learning rate
            me.update(tlr=max_tlr)
            
            # Log training metrics to TensorBoard
            tb_lg.set_step(step=g_it)
            tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
            tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
            tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
            tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
            tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
            
            # Log gradient information if gradient clipping is enabled
            if args.tclip > 0:
                tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
                tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
            
            # 定期清理显存碎片
            if stepping and (g_it + 1) % 50 == 0:
                torch.cuda.empty_cache()
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
