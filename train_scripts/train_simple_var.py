from itertools import islice
from pathlib import Path
import json
import sys
import time
from datetime import datetime
import os

import torch
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import tqdm
from einops import rearrange, repeat

from maskvar.models.vqvae_single import VQVAE_Single
from maskvar.models.simple_ar import (
    SimpleVAR,
    simple_var_train_pass,
    simple_var_inference,
)
from maskvar.datasets import (
    MaskLevelDataset,
    MaskLevelDatasetDummy,
    MaskLevelDatasetRandom,
)
from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset
from maskvar.datasets.image_feature_cache import ImageFeatureCache
from maskvar.datasets.sharded_distributed_sampler import ShardedDistributedSampler


torch.set_float32_matmul_precision('high')

def save_train_configuration(args, outdir: Path):
    cur_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # create outdir
    outdir.mkdir(parents=True, exist_ok=True)
    
    # save train config
    with open(outdir / 'train_config.json', 'w') as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=2)
    
    # save train command
    with open(outdir / 'train_command.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# train start time: ' + cur_time + '\n')
        f.write(' '.join(sys.argv))
    
    # print train config
    print('train config: ')
    for k, v in args.__dict__.items():
        print(f'    {k}: {v}')
    
    # print train command
    print('train command: ')
    print(' '.join(sys.argv))
    
    # print start time
    print('train start time: ' + cur_time)

    time.sleep(2)


class SimpleARTrainer:

    def __init__(
        self, 
        simple_var: SimpleVAR, 
        vqvae: VQVAE_Single, 
        lr: float,
        train_set: MaskLevelDataset, 
        val_set: MaskLevelDataset,
        batch_size: int,
        accumulate_steps: int,
        device: str,
        log_dir: Path,
        checkpoint_dir: Path,
        skip_eval: bool = True,
        loss_weight_per_level=[1, 1, 1, 1, 1],
        dtype=torch.float32,
        opt_checkpoint: Path | None = None,
        dataloader_workers: int = 4,
        prefetch_factor: int = 2,
        shuffle_dataloader: bool = True,
        find_unused_parameters: bool = True,
    ):
        # models
        self.simple_var: SimpleVAR = simple_var
        self.vqvae: VQVAE_Single = vqvae

        # optimizer
        self.optimizer = torch.optim.AdamW(simple_var.parameters(), lr=lr)

        # device
        self.device = device
        self.dtype = dtype

        # dataset
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.accumulate_steps = accumulate_steps
        self.sampler = None
        self.val_sampler = None
        
        # loss
        self.loss_function = nn.CrossEntropyLoss(reduction='none')
        if opt_checkpoint is not None:
            optimizer_state_dict = torch.load(opt_checkpoint)
            self.optimizer.load_state_dict(optimizer_state_dict)

        # loss weight
        with torch.no_grad():
            patch_num = simple_var.patch_num
            loss_weight_per_token = []
            for level, pn in enumerate(patch_num):
                loss_weight_per_token.extend(
                    [loss_weight_per_level[level] / pn**2] * pn**2
                )
            # print(f'loss weight per token: {loss_weight_per_token}')
            self.loss_weight_per_token = torch.tensor(loss_weight_per_token, device=self.device)
            self.loss_weight_per_token = F.normalize(self.loss_weight_per_token, p=1, dim=-1)

        # logger
        self.logger = SummaryWriter(log_dir=str(log_dir))
        self.output_dir = checkpoint_dir
        self.log_duration = 32

        self.skip_eval = skip_eval

        # torch distributed
        if tdist.is_initialized():
            self.rank = tdist.get_rank()
            self.world_size = tdist.get_world_size()
            self.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
        
        self.find_unused_parameters = find_unused_parameters

        self.compile_model()
        self.ddp_wrap()

        # dataloader
        self.dataloader_workers = dataloader_workers
        self.prefetch_factor = prefetch_factor
        self.train_dataloader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=(self.sampler is None) and shuffle_dataloader,
            sampler=self.sampler,
            drop_last=True,
            num_workers=self.dataloader_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=True
        )
        self.val_dataloader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=(self.val_sampler is None) and shuffle_dataloader,
            drop_last=True,
            sampler=self.val_sampler,
            num_workers=self.dataloader_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=True
        )
    
    def compile_model(self):
        self.simple_var.to(self.device)
        self.vqvae.to(self.device)
        self.simple_var = torch.compile(self.simple_var)
        self.vqvae = torch.compile(self.vqvae)
    
    def ddp_wrap(self):
        if self.world_size > 1:
            self.simple_var = DDP(self.simple_var, device_ids=[self.rank], find_unused_parameters=self.find_unused_parameters)
            # self.sampler = DistributedSampler(self.train_set)
            # self.val_sampler = DistributedSampler(self.val_set)
            self.sampler = ShardedDistributedSampler(self.train_set, rank=self.rank, world_size=self.world_size, shard_size=1024)
            self.val_sampler = DistributedSampler(self.val_set)

    def train_step(self, inner_iter_count, image, image_embed_sam, single_mask_normalized, single_mask):
        image_embed_sam = image_embed_sam.to(self.device, non_blocking=True)
        single_mask_normalized = single_mask_normalized.to(self.device, non_blocking=True)

        gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized) # List of (B, l)
        gt_idx_flat = torch.cat(gt_idx, dim=1) # (B, L)
        
        with torch.autocast(self.device, dtype=self.dtype):

            logits = self.simple_var(
                idx=gt_idx,
                image_feat=image_embed_sam,
                vqvae=self.vqvae
            )

            acc = (logits.argmax(dim=-1) == gt_idx_flat).float()

            logits = rearrange(logits, 'B L C -> B C L')

            loss = self.loss_function(logits, gt_idx_flat) # (B, L)
            loss = loss * rearrange(self.loss_weight_per_token, 'L -> 1 L') # (B, L)

            loss = loss.mean()
            loss = loss / self.accumulate_steps
            loss.backward()
            
            if (inner_iter_count + 1) % self.accumulate_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return loss, acc

    def train(self, num_iters: int, outer_iter: int = 0, resume_iters: int = 0, val_iters=0):
        if num_iters <= 0:
            num_iters = len(self.train_dataloader)

        self.simple_var.train()
        
        if self.rank == 0:
            pbar = tqdm.tqdm(range(num_iters), desc="Training", total=num_iters)
        
        iters_count = 0

        while iters_count < num_iters:
            if tdist.is_initialized():
                self.sampler.set_epoch(outer_iter)
            for i, (image, image_embed_sam, single_mask_normalized, single_mask) in enumerate(self.train_dataloader):
                if iters_count >= num_iters:
                    break
                if self.rank == 0:
                    pbar.update(1)

                global_iters = iters_count + num_iters * outer_iter + resume_iters
                iters_count += 1

                loss, acc = self.train_step(
                    i,
                    image,
                    image_embed_sam,
                    single_mask_normalized,
                    single_mask,
                )

                if global_iters % self.log_duration == 0:
                    acc_mean = acc.mean()
                    acc_sos = acc[:, 0].mean()
                    if tdist.is_initialized():
                        tdist.all_reduce(acc_mean, op=tdist.ReduceOp.AVG)
                        tdist.all_reduce(acc_sos, op=tdist.ReduceOp.AVG)
                        tdist.all_reduce(loss, op=tdist.ReduceOp.AVG)
                    loss = loss.item()
                    acc_mean = acc_mean.item()
                    acc_sos = acc_sos.item()

                    if self.rank == 0:
                        # update loss and acc in progressive bar
                        pbar.set_postfix({'loss': f'{loss:.4f}', 'acc_mean': f'{acc_mean:.4f}', 'acc_sos': f'{acc_sos:.4f}'})
                        # log to tensorboard
                        self.logger.add_scalar('train/loss', loss, global_step=global_iters)
                        self.logger.add_scalar('train/acc_mean', acc_mean, global_step=global_iters)
                        self.logger.add_scalar('train/acc_sos', acc_sos, global_step=global_iters)
                
        if self.rank == 0:
            global_iters = (outer_iter + 1)*num_iters + resume_iters
            self.save_checkpoint(iters=global_iters)
        if tdist.is_initialized():
            tdist.barrier()
        self.eval(args.val_iters // world_size, global_step=global_iters)
    
    @torch.no_grad()
    def eval(self, num_iters: int, global_step: int = 0):
        if num_iters < 0:
            print(f'num_iters={num_iters}, skip val')
            return
        if num_iters == 0 or num_iters > len(self.val_dataloader):
            num_iters = len(self.val_dataloader)
        self.simple_var.eval()

        total_loss = torch.tensor(0.0, device=self.device)
        total_acc_mean = torch.tensor(0.0, device=self.device)
        total_acc_sos = torch.tensor(0.0, device=self.device)
        if self.rank == 0:
            pbar = tqdm.tqdm(range(num_iters), desc="Val: ", total=num_iters)
        
        for i, (image, image_embed_sam, single_mask_normalized, single_mask) in enumerate(self.val_dataloader):
            if self.rank == 0:
                pbar.update(1)

            if i >= num_iters:
                break
            image_embed_sam = image_embed_sam.to(self.device, non_blocking=True)
            single_mask_normalized = single_mask_normalized.to(self.device, non_blocking=True)
            
            gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized)
            gt_idx_flat = torch.cat(gt_idx, dim=1)

            with torch.autocast(self.device, dtype=self.dtype):
                logits = self.simple_var(
                    idx=gt_idx,
                    image_feat=image_embed_sam,
                    vqvae=self.vqvae
                )
                
                acc = (logits.argmax(dim=-1) == gt_idx_flat).float()

                acc_mean = acc.mean()
                acc_sos = acc[:, 0].mean()
                
                logits = rearrange(logits, 'b l c -> b c l')
                loss = self.loss_function(logits, gt_idx_flat)
                loss = loss * rearrange(self.loss_weight_per_token, 'L -> 1 L') # will be automatically broadcasted to [B, L]

                loss_mean = loss.mean()

            if self.rank == 0:
                pbar.set_postfix({'loss': f'{loss_mean:.4f}', 'acc_mean': f'{acc_mean:.4f}', 'acc_sos': f'{acc_sos:.4f}'})

            total_loss += loss_mean
            total_acc_mean += acc_mean
            total_acc_sos += acc_sos
        
        if tdist.is_initialized():
            tdist.all_reduce(total_loss, op=tdist.ReduceOp.AVG)
            tdist.all_reduce(total_acc_mean, op=tdist.ReduceOp.AVG)
            tdist.all_reduce(total_acc_sos, op=tdist.ReduceOp.AVG)

        mean_loss = total_loss / num_iters
        mean_acc_mean = total_acc_mean / num_iters
        mean_acc_sos = total_acc_sos / num_iters

        if self.rank == 0:
            self.logger.add_scalar('val/loss', mean_loss.item(), global_step=global_step)
            self.logger.add_scalar('val/acc_mean', mean_acc_mean.item(), global_step=global_step)
            self.logger.add_scalar('val/acc_sos', mean_acc_sos.item(), global_step=global_step)

        return mean_loss, mean_acc_mean, mean_acc_sos
    
    def save_checkpoint(self, iters: int):
        if tdist.is_initialized():
            torch.save(self.optimizer.state_dict(), self.output_dir / f'.optimizer.{iters}.pt')
            torch.save(self.simple_var.module.state_dict(), self.output_dir / f'.simple_var.{iters}.pt')
        else:
            torch.save(self.optimizer.state_dict(), self.output_dir / f'.optimizer.{iters}.pt')
            torch.save(self.simple_var.state_dict(), self.output_dir / f'.simple_var.{iters}.pt')

def setup_dist():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        tdist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="env://")
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def cleanup():
    tdist.destroy_process_group()

if __name__ == "__main__":
    rank, local_rank, world_size = setup_dist()

    import argparse
    from maskvar.maskseg_build_everything import (
        builder_map
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str)
    # hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--outer_iters', type=int, default=2)
    parser.add_argument('--val_iters', type=int, default=0)
    parser.add_argument('--inner_iters', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulate_steps', type=int, default=1)
    # resume
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--resume_iters', type=int, default=0)
    # dataset
    parser.add_argument('--dataset', choices=['hqseg44k', 'cocolvis'], type=str, default='hqseg44k')
    parser.add_argument('--use_dummy_dataset_for_debug', action='store_true')
    parser.add_argument('--dl_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    # configs
    parser.add_argument('--simple_var', type=str, default='simple_var')
    parser.add_argument('--simple_var_init_checkpoint', type=str, default=None)
    parser.add_argument('--image_encoder', choices=['sam_vitb', 'mobile_sam'], type=str, default='mobile_sam')
    parser.add_argument('--image_encoder_checkpoint', type=str, default='ckpt/mobile_sam.pt')
    parser.add_argument('--vqvae', choices=builder_map['vqvae'].keys(), type=str, default='vqvae_single_5_stages_v1')
    parser.add_argument('--vqvae_checkpoint', type=str, default='out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth')
    parser.add_argument('--use_sam_pe', action='store_true')
    parser.add_argument('--prompt_encoder_checkpoint', type=str, default=None)
    parser.add_argument('--disable_find_unused_parameters', action='store_true')
    # image embedding caching
    parser.add_argument('--image_feature_cache_dir', type=str, default="")
    # dtype
    parser.add_argument('--dtype', choices=['float16', 'float32', 'bfloat16'], type=str, default='float32')
    args = parser.parse_args()

    if tdist.is_initialized():
        device = f"cuda:{local_rank}"
    else:
        device = args.device
    dtype = getattr(torch, args.dtype)

    outdir = Path(args.outdir)

    if args.resume_from is not None:
        checkpoint_path = Path(args.resume_from) / "checkpoints" / f".simple_var.{args.resume_iters}.pt"
        opt_checkpoint = Path(args.resume_from) / "checkpoints" / f".optimizer.{args.resume_iters}.pt"
    else:
        checkpoint_path = Path(args.simple_var_init_checkpoint) if args.simple_var_init_checkpoint else None
        opt_checkpoint = None

    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        save_train_configuration(args, outdir)
        (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (outdir / "logs").mkdir(parents=True, exist_ok=True)

    # sam_image_encoder = build_mobile_sam_image_encoder('ckpt/mobile_sam.pt')
    if args.image_feature_cache_dir:
        print("Using image feature cache: ", args.image_feature_cache_dir)
        image_feature_cache_train = ImageFeatureCache(Path(args.image_feature_cache_dir), f"{args.dataset}_train", args.image_encoder)
        image_feature_cache_val = ImageFeatureCache(Path(args.image_feature_cache_dir), f"{args.dataset}_val", args.image_encoder)
        sam_image_encoder = None
    else:
        print("Not using image feature cache")
        raise NotImplementedError
        # image_feature_cache_train = None
        # image_feature_cache_val = None
        # sam_image_encoder = builder_map['image_encoder'][args.image_encoder](args.image_encoder_checkpoint)
        # sam_image_encoder = sam_image_encoder.to(device)
        # sam_image_encoder = torch.compile(sam_image_encoder)

    dataset_dir = 'data/sam-hq' if args.dataset == 'hqseg44k' else 'data/coco_lvis'
    index_mapping_path = f'data/flat/{args.dataset}'
    # train_set, val_set = build_hqseg44k_dataset('data/sam-hq') # validate on train set
    train_set, val_set = builder_map['dataset'][args.dataset](dataset_dir)
    if args.use_dummy_dataset_for_debug:
        train_set_masklevel = MaskLevelDatasetDummy(
            dataset=train_set,
            with_image_embed=True,
            device=device,
            mask_filter_thresh=0.1,
            seed=42,
            count=5,
            image_feature_cache=image_feature_cache_train
        )
        val_set_masklevel = MaskLevelDatasetDummy(
            dataset=train_set,
            with_image_embed=True,
            device=device,
            mask_filter_thresh=0.1,
            seed=42,
            count=5,
            image_feature_cache=image_feature_cache_train
        )
    else:
        val_set_masklevel = MaskLevelFlatDataset(
            index_mapping_path=Path(index_mapping_path) / "val_index_mapping.npy",
            dataset=val_set,
            with_image_embed=True,
            image_feature_cache=image_feature_cache_val,
            mask_filter_thresh=0.1,
            dtype=torch.float32,
        )
        train_set_masklevel = MaskLevelFlatDataset(
            index_mapping_path=Path(index_mapping_path) / "train_index_mapping.npy",
            dataset=train_set,
            with_image_embed=True,
            image_feature_cache=image_feature_cache_train,
            mask_filter_thresh=0.1,
            dtype=torch.float32,
        )

    if args.use_sam_pe:
        prompt_encoder = builder_map['prompt_encoder'](args.prompt_encoder_checkpoint)
        sam_pe = prompt_encoder.get_dense_pe() # BCHW
        del prompt_encoder
    else:
        sam_pe = None

    # simple_var = build_simple_var(simple_var_checkpoint_path=checkpoint_path, device=device)
    simple_var = builder_map['simple_var'][args.simple_var](simple_var_checkpoint_path=checkpoint_path, sam_pe=sam_pe, device=device)
    # vqvae = build_vqvae_single_5_stages_v1('out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth', require_grad=False)
    vqvae = builder_map['vqvae'][args.vqvae](vqvae_checkpoint_path=args.vqvae_checkpoint, require_grad=False).to(device)

    lr = args.lr
    batch_size = args.batch_size
    accumulate_steps = args.accumulate_steps

    local_batch_size = batch_size // world_size

    trainer = SimpleARTrainer(
        simple_var=simple_var,
        vqvae=vqvae,
        lr=lr,
        train_set=train_set_masklevel,
        val_set=val_set_masklevel,
        batch_size=local_batch_size,
        accumulate_steps=accumulate_steps,
        device=device,
        log_dir=outdir / "logs",
        checkpoint_dir=outdir / "checkpoints",
        opt_checkpoint=opt_checkpoint,
        dtype=dtype,
        dataloader_workers=args.dl_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle_dataloader=(not args.use_dummy_dataset_for_debug),
        find_unused_parameters=(not args.disable_find_unused_parameters),
    )

    outer_iters = args.outer_iters
    inner_iters = args.inner_iters
    resume_iters = args.resume_iters
    for i in range(outer_iters):
        if rank == 0:
            print(f'=== outer iter {i} ===')
        trainer.train(num_iters=inner_iters // world_size, outer_iter=i, resume_iters=resume_iters, val_iters=args.val_iters // world_size)
    
    if rank == 0:
        print(f"Training finish at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Training complete. Checkpoints saved to {outdir / 'checkpoints'}")
        print(f"Logs saved to {outdir / 'logs'}")
    
    del trainer
    
    if tdist.is_initialized():
        cleanup()
