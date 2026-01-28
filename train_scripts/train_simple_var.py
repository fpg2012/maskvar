from itertools import islice
from pathlib import Path
import json
import sys
import time
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from maskvar.datasets.image_feature_cache import ImageFeatureCache


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

        self.skip_eval = skip_eval

        self.compile_model()
    
    def compile_model(self):
        self.simple_var.to(self.device)
        self.vqvae.to(self.device)
        self.simple_var = torch.compile(self.simple_var)
        self.vqvae = torch.compile(self.vqvae)

    def train_step(self, inner_iter_count, image, image_embed_sam, single_mask_normalized, single_mask):
        image_embed_sam = image_embed_sam.to(self.device)
        single_mask_normalized = single_mask_normalized.to(self.device)

        gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized) # List of (B, l)
        gt_idx_flat = torch.cat(gt_idx, dim=1) # (B, L)
        
        with torch.autocast(self.device, dtype=self.dtype):

            logits = simple_var_train_pass(
                idx=gt_idx,
                image_feat=image_embed_sam,
                simple_var=self.simple_var, 
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

        return loss.item(), acc

    def train(self, num_iters: int, outer_iter: int = 0, resume_iters: int = 0):
        # train_dataloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, drop_last=True)
        train_dataloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=64, prefetch_factor=2, pin_memory=True, persistent_workers=True)

        if num_iters > 0:
            train_dataloader = islice(train_dataloader, num_iters)

        self.simple_var.train()
        
        pbar = tqdm.tqdm(enumerate(train_dataloader), desc="Training", total=num_iters)
        for i, (image, image_embed_sam, single_mask_normalized, single_mask) in pbar:
            global_iters = i + num_iters * outer_iter + resume_iters

            loss, acc = self.train_step(
                i,
                image,
                image_embed_sam,
                single_mask_normalized,
                single_mask,
            )
            acc_mean = acc.mean().item()
            acc_sos = acc[:, 0].mean().item()

            # update loss and acc in progressive bar
            pbar.set_postfix({'loss': f'{loss:.4f}', 'acc_mean': f'{acc_mean:.4f}', 'acc_sos': f'{acc_sos:.4f}'})

            # log to tensorboard
            self.logger.add_scalar('train/loss', loss, global_step=global_iters)
            self.logger.add_scalar('train/acc_mean', acc_mean, global_step=global_iters)
            self.logger.add_scalar('train/acc_sos', acc_sos, global_step=global_iters)
        
        global_iters = (outer_iter + 1)*num_iters + resume_iters
        self.save_checkpoint(iters=global_iters)
    
    @torch.no_grad()
    def eval(self, num_iters: int, global_step: int = 0):
        val_dataloader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        self.simple_var.eval()

        losses = []
        acc_means = []
        acc_soss = []

        pbar = tqdm.tqdm(enumerate(val_dataloader), desc="Val: ", total=num_iters)

        for i, (image, image_embed_sam, single_mask_normalized, single_mask) in pbar:
            image_embed_sam = image_embed_sam.to(self.device)
            single_mask_normalized = single_mask_normalized.to(self.device)
            
            if num_iters > 0 and i >= num_iters:
                break
            gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized)
            gt_idx_flat = torch.cat(gt_idx, dim=1)

            with torch.autocast(self.device, dtype=self.dtype):
                logits = simple_var_train_pass(
                    idx=gt_idx,
                    image_feat=image_embed_sam,
                    simple_var=self.simple_var, 
                    vqvae=self.vqvae
                )
                
                acc = (logits.argmax(dim=-1) == gt_idx_flat).float()
                acc_mean = acc.mean().item()
                acc_sos = acc[:, 0].mean().item()
                
                logits = rearrange(logits, 'b l c -> b c l')
                loss = self.loss_function(logits, gt_idx_flat)
                loss = loss * rearrange(self.loss_weight_per_token, 'L -> 1 L') # will be automatically broadcasted to [B, L]
                
                loss_mean = loss.mean().item()

            pbar.set_postfix({'loss': f'{loss_mean:.4f}', 'acc_mean': f'{acc_mean:.4f}', 'acc_sos': f'{acc_sos:.4f}'})

            losses.append(loss_mean)
            acc_means.append(acc_mean)
            acc_soss.append(acc_sos)
        
        mean_loss = float(sum(losses) / len(losses))
        mean_acc_mean = float(sum(acc_means) / len(acc_means))
        mean_acc_sos = float(sum(acc_soss) / len(acc_soss))

        self.logger.add_scalar('val/loss', mean_loss, global_step=global_step)
        self.logger.add_scalar('val/acc_mean', mean_acc_mean, global_step=global_step)
        self.logger.add_scalar('val/acc_sos', mean_acc_sos, global_step=global_step)

        return mean_loss, mean_acc_mean, mean_acc_sos
    
    def save_checkpoint(self, iters: int):
        torch.save(self.optimizer.state_dict(), self.output_dir / f'.optimizer.{iters}.pt')
        torch.save(self.simple_var.state_dict(), self.output_dir / f'.simple_var.{iters}.pt')


if __name__ == "__main__":
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)

    import argparse
    from maskvar.maskseg_build_everything import (
        builder_map
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str)
    # hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--outer_iters', type=int, default=1000)
    parser.add_argument('--val_iters', type=int, default=100)
    parser.add_argument('--inner_iters', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulate_steps', type=int, default=1)
    # resume
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--resume_iters', type=int, default=0)
    # dataset
    parser.add_argument('--dataset', choices=['hqseg44k', 'cocolvis'], type=str, default='hqseg44k')
    # configs
    parser.add_argument('--simple_var', type=str, default='simple_var')
    parser.add_argument('--image_encoder', choices=['sam_vitb', 'mobile_sam'], type=str, default='mobile_sam')
    parser.add_argument('--image_encoder_checkpoint', type=str, default='ckpt/mobile_sam.pt')
    parser.add_argument('--vqvae', choices=builder_map['vqvae'].keys(), type=str, default='vqvae_single_5_stages_v1')
    parser.add_argument('--vqvae_checkpoint', type=str, default='out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth')
    # image embedding caching
    parser.add_argument('--image_feature_cache_dir', type=str, default="")
    # dtype
    parser.add_argument('--dtype', choices=['float16', 'float32', 'bfloat16'], type=str, default='float32')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    save_train_configuration(args, outdir)

    if args.resume_from is not None:
        checkpoint_path = Path(args.resume_from) / "checkpoints" / f".simple_var.{args.resume_iters}.pt"
        opt_checkpoint = Path(args.resume_from) / "checkpoints" / f".optimizer.{args.resume_iters}.pt"
    else:
        checkpoint_path = None
        opt_checkpoint = None

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)

    device = args.device
    dtype = getattr(torch, args.dtype)

    # sam_image_encoder = build_mobile_sam_image_encoder('ckpt/mobile_sam.pt')
    if args.image_feature_cache_dir:
        print("Using image feature cache: ", args.image_feature_cache_dir)
        image_feature_cache_train = ImageFeatureCache(Path(args.image_feature_cache_dir), f"{args.dataset}_train", args.image_encoder)
        image_feature_cache_val = ImageFeatureCache(Path(args.image_feature_cache_dir), f"{args.dataset}_val", args.image_encoder)
        sam_image_encoder = None
    else:
        print("Not using image feature cache")
        image_feature_cache_train = None
        image_feature_cache_val = None
        sam_image_encoder = builder_map['image_encoder'][args.image_encoder](args.image_encoder_checkpoint)
        sam_image_encoder = sam_image_encoder.to(device)
        sam_image_encoder = torch.compile(sam_image_encoder)

    dataset_dir = 'data/sam-hq' if args.dataset == 'hqseg44k' else 'data/coco-lvis'
    # train_set, val_set = build_hqseg44k_dataset('data/sam-hq') # validate on train set
    train_set, val_set = builder_map['dataset'][args.dataset](dataset_dir)
    # train_set_masklevel = MaskLevelDatasetDummy(
    #     dataset=train_set,
    #     sam_encoder=sam_image_encoder,
    #     with_image_embed=True,
    #     device=args.device,
    #     mask_filter_thresh=0.1,
    #     seed=42,
    #     count=5,
    # )
    val_set_masklevel = MaskLevelDatasetDummy(
        dataset=val_set,
        sam_encoder=sam_image_encoder,
        with_image_embed=True,
        device=args.device,
        mask_filter_thresh=0.1,
        seed=42,
        count=16,
        image_feature_cache=image_feature_cache_val,
    )
    train_set_masklevel = MaskLevelDatasetRandom(
        dataset=train_set,
        sam_encoder=sam_image_encoder,
        with_image_embed=True,
        device=args.device,
        mask_filter_thresh=0.1,
        seed=42,
        infinite=True,
        shuffle=True,
        image_feature_cache=image_feature_cache_train,
    )

    # simple_var = build_simple_var(simple_var_checkpoint_path=checkpoint_path, device=device)
    simple_var = builder_map['simple_var'][args.simple_var](simple_var_checkpoint_path=checkpoint_path, device=device)
    # vqvae = build_vqvae_single_5_stages_v1('out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth', require_grad=False)
    vqvae = builder_map['vqvae'][args.vqvae](vqvae_checkpoint_path=args.vqvae_checkpoint, require_grad=False)

    lr = args.lr
    batch_size = args.batch_size
    accumulate_steps = args.accumulate_steps

    trainer = SimpleARTrainer(
        simple_var=simple_var,
        vqvae=vqvae,
        lr=lr,
        train_set=train_set_masklevel,
        val_set=val_set_masklevel,
        batch_size=batch_size,
        accumulate_steps=accumulate_steps,
        device=device,
        log_dir=outdir / "logs",
        checkpoint_dir=outdir / "checkpoints",
        opt_checkpoint=opt_checkpoint,
        dtype=dtype,
    )

    outer_iters = args.outer_iters
    inner_iters = args.inner_iters
    resume_iters = args.resume_iters
    for i in range(outer_iters):
        print(f'=== outer iter {i} ===')
        trainer.train(num_iters=inner_iters, outer_iter=i, resume_iters=resume_iters)
        trainer.eval(args.val_iters, global_step=(i+1)*inner_iters+resume_iters)
    print(f"Training finish at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training complete. Checkpoints saved to {outdir / 'checkpoints'}")
    print(f"Logs saved to {outdir / 'logs'}")
    