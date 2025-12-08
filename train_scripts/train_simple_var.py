from itertools import islice
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
from einops import rearrange

from maskvar.models.vqvae_single import VQVAE_Single
from maskvar.models.simple_ar import (
    SimpleVAR,
    simple_var_train_pass,
    simple_var_inference,
)

from maskvar.datasets import MaskLevelDataset, MaskLevelDatasetDummy

class SimpleARTrainer:

    def __init__(
        self, 
        simple_var: SimpleVAR, 
        vqvae: VQVAE_Single, 
        lr: float,
        train_set: MaskLevelDataset, 
        val_set: MaskLevelDataset, 
        batch_size: int, 
        device: str,
        log_dir: Path,
        checkpoint_dir: Path,
        skip_eval: bool = True,
    ):
        # models
        self.simple_var: SimpleVAR = simple_var
        self.vqvae: VQVAE_Single = vqvae

        # optimizer
        self.optimizer = torch.optim.AdamW(simple_var.parameters(), lr=lr)

        # device
        self.device = device

        # dataset
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        
        # loss
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

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

    def train_step(self, image, image_embed_sam, single_mask_normalized, single_mask):
        self.optimizer.zero_grad()
        gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized) # List of (B, l)
        gt_idx_flat = torch.cat(gt_idx, dim=1) # (B, L)

        logits = simple_var_train_pass(
            idx=gt_idx,
            simple_var=self.simple_var, 
            vqvae=self.vqvae
        )

        acc = (logits.argmax(dim=-1) == gt_idx_flat).float().mean()

        logits = rearrange(logits, 'B L C -> B C L')

        loss = self.loss_function(logits, gt_idx_flat)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc.item()

    def train(self, num_iters: int):
        train_dataloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, drop_last=True)

        if num_iters > 0:
            train_dataloader = islice(train_dataloader, num_iters)

        self.simple_var.train()
        
        pbar = tqdm.tqdm(enumerate(train_dataloader), desc="Training", total=num_iters)
        for i, (image, image_embed_sam, single_mask_normalized, single_mask) in pbar:
            loss, acc = self.train_step(image, image_embed_sam, single_mask_normalized, single_mask)
            # update loss and acc in progressive bar
            pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.4f}'})

            self.logger.add_scalar('train/loss', loss, global_step=i)
            self.logger.add_scalar('train/acc', acc, global_step=i)
        
        self.save_checkpoint(iters=num_iters)
    
    @torch.no_grad()
    def eval(self, num_iters: int):
        if num_iters > 0:
            val_dataloader = DataLoader(islice(self.val_set, num_iters), batch_size=self.batch_size, shuffle=False, drop_last=True)
        else:
            val_dataloader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        self.simple_var.eval()

        losses = []
        accs = []

        for i, (image, image_embed_sam, single_mask_normalized, single_mask) in enumerate(val_dataloader):
            gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized)
            gt_idx_flat = torch.cat(gt_idx, dim=1)

            logits = simple_var_train_pass(
                idx=gt_idx,
                simple_var=self.simple_var, 
                vqvae=self.vqvae
            )
            
            acc = (logits.argmax(dim=1) == gt_idx_flat).float().mean()
            
            logits = rearrange(logits, 'b l c -> b c l')
            loss = self.loss_function(logits, gt_idx_flat)
            
            losses.append(loss.mean().item())
            accs.append(acc.item())
        
        mean_loss = float(sum(losses) / len(losses))
        mean_acc = float(sum(accs) / len(accs))

        return mean_loss, mean_acc
    
    def save_checkpoint(self, iters: int):
        torch.save(self.optimizer.state_dict(), self.output_dir / f'.optimizer.{iters}.pt')
        torch.save(self.simple_var.state_dict(), self.output_dir / f'.simple_var.{iters}.pt')


if __name__ == "__main__":
    import argparse
    from maskvar.maskseg_build_everything import (
        build_hqseg44k_dataset,
        build_simple_var,
        build_vqvae_single_5_stages_v1,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)

    device = args.device

    train_set, _ = build_hqseg44k_dataset('data/sam-hq') # validate on train set
    train_set_masklevel = MaskLevelDatasetDummy(
        dataset=train_set,
        sam_encoder=None,
        with_image_embed=False,
        device=args.device,
        mask_filter_thresh=0.1,
        seed=42,
        # count=5,
    )
    val_set_masklevel = MaskLevelDatasetDummy(
        dataset=train_set,
        sam_encoder=None,
        with_image_embed=False,
        device=args.device,
        mask_filter_thresh=0.1,
        seed=42,
        # count=5,
    )

    simple_var = build_simple_var(device=device)
    vqvae = build_vqvae_single_5_stages_v1('out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth', require_grad=False)

    trainer = SimpleARTrainer(
        simple_var=simple_var,
        vqvae=vqvae,
        lr=1e-3,
        train_set=train_set_masklevel,
        val_set=val_set_masklevel,
        batch_size=args.batch_size,
        device=device,
        log_dir=outdir / "logs",
        checkpoint_dir=outdir / "checkpoints",
    )
    trainer.train(num_iters=args.num_iters)
    print(f"Training complete. Checkpoints saved to {outdir / 'checkpoints'}")
    print(f"Logs saved to {outdir / 'logs'}")
    