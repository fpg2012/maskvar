from itertools import islice
from pathlib import Path
from datetime import datetime
import json

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
from einops import rearrange, repeat
import matplotlib.pyplot as plt

from maskvar.maskseg_build_everything import (
    build_simple_var,
    build_simple_var_16d,
    build_simple_var_6d,
)
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
from maskvar.utils import restore_normalized_image
from maskvar.utils.metrics import (
    calc_iou
)


class SimpleVAREvaluator:

    def __init__(
        self, 
        simple_var: SimpleVAR, 
        vqvae: VQVAE_Single,
        val_set: MaskLevelDataset, 
        batch_size: int, 
        out_dir: Path,
        device: str,
        loss_weight_per_level=[1, 1, 1, 1, 1],
    ):
        # models
        self.simple_var: SimpleVAR = simple_var
        self.vqvae: VQVAE_Single = vqvae

        # device
        self.device = device

        # dataset
        self.val_set = val_set
        self.batch_size = batch_size
        
        # loss
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

        # loss weight
        with torch.no_grad():
            patch_num = simple_var.patch_num
            loss_weight_per_token = []
            for level, pn in enumerate(patch_num):
                loss_weight_per_token.extend(
                    [loss_weight_per_level[level] / pn**2] * pn**2
                )
            # print(f'loss weight per token: {loss_weight_per_token}')
            self.loss_weight_per_token = torch.tensor(loss_weight_per_token, dtype=torch.float32, device=self.device)
            self.loss_weight_per_token = F.normalize(self.loss_weight_per_token, p=1, dim=-1)

        # out_dir
        self.out_dir = out_dir

        self.compile_model()
    
    def compile_model(self):
        self.simple_var.to(self.device)
        self.vqvae.to(self.device)
        self.simple_var = torch.compile(self.simple_var)
        self.vqvae = torch.compile(self.vqvae)

        self.vqvae.eval()
        self.simple_var.eval()

    @torch.no_grad()
    def eval_with_teacher_input(self, num_iters: int):
        val_dataloader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True)

        losses = []
        acc_means = []
        acc_soss = []

        pbar = tqdm.tqdm(enumerate(val_dataloader), desc="Val: ", total=num_iters)

        for i, (image, image_embed_sam, single_mask_normalized, single_mask) in pbar:
            if num_iters > 0 and i >= num_iters:
                break
            gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized)
            gt_idx_flat = torch.cat(gt_idx, dim=1)

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

        return mean_loss, mean_acc_mean, mean_acc_sos

    @torch.no_grad()
    def eval_ar(self, num_iters: int, visualize: bool = False):
        val_dataloader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        acc_means = []
        acc_soss = []
        ious = []
        acc_means_teacher = []

        pbar = tqdm.tqdm(enumerate(val_dataloader), desc="Val: ", total=num_iters)

        for i, (image, image_embed_sam, single_mask_normalized, single_mask) in pbar:
            if num_iters > 0 and i >= num_iters:
                break
            gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized)
            gt_idx_flat = torch.cat(gt_idx, dim=1)

            result = simple_var_inference(image_embed_sam, self.simple_var, self.vqvae)
            
            flat_ids = torch.cat(result, dim=1)
            acc_mean = (flat_ids == gt_idx_flat).float().mean().item()
            acc_sos = (flat_ids[:, 1:] == gt_idx_flat[:, :-1]).float().mean().item()

            acc_means.append(acc_mean)
            acc_soss.append(acc_sos)

            # pred with teacher input
            logits = simple_var_train_pass(
                idx=gt_idx,
                image_feat=image_embed_sam,
                simple_var=self.simple_var, 
                vqvae=self.vqvae
            )
            acc = (logits.argmax(dim=-1) == gt_idx_flat).float()
            acc_mean_teacher = acc.mean().item()
            acc_means_teacher.append(acc_mean_teacher)

            id_seq_teach = logits.argmax(dim=-1)
            id_seq_teach_Bl = []
            start_pos = 0
            for pn in simple_var.patch_num:
                end_pos = start_pos + pn * pn
                id_seq_teach_Bl.append(id_seq_teach[:, start_pos:end_pos])
                start_pos = end_pos

            decoded_masks = self.vqvae_decode(result)
            decoded_masks_gt = self.vqvae_decode(gt_idx)
            decoded_masks_pred_with_teacher = self.vqvae_decode(id_seq_teach_Bl)

            iou_batch = calc_iou(decoded_masks[-1], single_mask)
            ious.append(iou_batch.mean().item())
            
            if visualize:
                for j in range(self.batch_size):
                    cur_iou = iou_batch[j].item()

                    fig, ax = plt.subplots(1, 4, figsize=(15,4))
                    ax[0].imshow(restore_normalized_image(image[j]).permute(1, 2, 0).cpu().numpy())
                    ax[0].axis('off')
                    ax[0].set_title(f'Image {i*self.batch_size + j}, IOU: {cur_iou:.3f}')

                    result_gt = [m[j].unsqueeze(0) for m in decoded_masks_gt]
                    result_mask_pred = [m[j].unsqueeze(0) for m in decoded_masks]
                    result_mask_teacher = [m[j].unsqueeze(0) for m in decoded_masks_pred_with_teacher]

                    self.visualize(result_gt, ax[1], 'gt')
                    self.visualize(result_mask_pred, ax[2], 'pred')
                    self.visualize(result_mask_teacher, ax[3], 'pred w/ teacher input')

                    plt.savefig(self.out_dir / "eval" / "vis" / f'val_step_{i*self.batch_size + j}_iou{cur_iou:.3f}.png')
                    plt.close()
        
        print(f"Average IOU: {sum(ious)/len(ious):.4f}")
        print(f"Average accuracy (no teacher): {sum(acc_means)/len(acc_means):.4f}")
        print(f"Average accuracy (with teacher): {sum(acc_means_teacher)/len(acc_means_teacher):.4f}")
        with open(self.out_dir / "eval" / f"eval_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            result_data = {
                "average_iou": sum(ious)/len(ious),
                "average_accuracy_no_teacher": sum(acc_means)/len(acc_means),
                "average_accuracy_with_teacher": sum(acc_means_teacher)/len(acc_means_teacher),
                "total_samples": len(ious)
            }
            json.dump(result_data, f, indent=2, ensure_ascii=False)

    def vqvae_decode(self, indices):
        result = self.vqvae.idxBl_to_img(indices, same_shape=True)
        return result

    def visualize(self, result, ax, name='mask'):
        result = [mask for mask in result]
        chw = torchvision.utils.make_grid(torch.cat(result, dim=0), nrow=3, padding=1, pad_value=1.0)
        chw = chw.permute(1, 2, 0).cpu().numpy()
        ax.imshow(chw[:, :, 0])
        ax.axis('off')
        ax.set_title(name)

if __name__ == "__main__":
    import argparse
    from maskvar.maskseg_build_everything import (
        build_hqseg44k_dataset,
        build_simple_var,
        build_vqvae_single_5_stages_v1,
        build_mobile_sam_image_encoder,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--val_iters', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "eval" / "vis").mkdir(parents=True, exist_ok=True)

    device = args.device

    checkpoint_path = args.checkpoint
    print(f"Using checkpoint: {checkpoint_path}")
    assert Path(checkpoint_path).exists(), f"Checkpoint not found: {checkpoint_path}"

    sam_image_encoder = build_mobile_sam_image_encoder('ckpt/mobile_sam.pt')
    sam_image_encoder = sam_image_encoder.to(device)
    sam_image_encoder = torch.compile(sam_image_encoder)

    train_set, val_set = build_hqseg44k_dataset('data/sam-hq') # validate on train set
    val_set_masklevel = MaskLevelDataset(
        dataset=val_set,
        sam_encoder=sam_image_encoder,
        with_image_embed=True,
        device=args.device,
        mask_filter_thresh=0.1,
    )
    simple_var = build_simple_var_6d(simple_var_checkpoint_path=checkpoint_path, device=device)
    vqvae = build_vqvae_single_5_stages_v1('out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth', require_grad=False)

    trainer = SimpleVAREvaluator(
        simple_var=simple_var,
        vqvae=vqvae,
        val_set=val_set_masklevel,
        batch_size=args.batch_size,
        device=device,
        out_dir=outdir,
    )

    trainer.eval_ar(args.val_iters, visualize=args.visualize)
    print(f"Evaluation complete")
    if args.visualize:
        print(f"Visualization saved to {outdir / 'eval' / 'vis'}")
    