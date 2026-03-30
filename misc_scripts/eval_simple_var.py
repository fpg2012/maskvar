from itertools import islice
from pathlib import Path
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
from einops import rearrange, repeat
import matplotlib.pyplot as plt

from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset
from maskvar.maskseg_build_everything import (
    builder_map,
)
from maskvar.models.vqvae_single import VQVAE_Single
from maskvar.models.simple_ar import (
    SimpleVAR,
    SimpleVARSamDecoder,
    simple_var_train_pass,
    simple_var_inference,
    simple_var_sd_inference,
)
from maskvar.utils.clicker import init_clicks, to_sam_format
from maskvar.datasets import (
    MaskLevelDataset,
    MaskLevelDatasetDummy,
    MaskLevelDatasetRandom,
)
from maskvar.datasets.image_feature_cache import ImageFeatureCache
from maskvar.utils import restore_normalized_image
from maskvar.utils.metrics import (
    calc_iou
)

torch.set_float32_matmul_precision('high')

class SimpleVAREvaluator:

    def __init__(
        self,
        simple_var: SimpleVAR | SimpleVARSamDecoder,
        vqvae: VQVAE_Single,
        val_set: MaskLevelDataset,
        batch_size: int,
        out_dir: Path,
        device: str,
        loss_weight_per_level=[1, 1, 1, 1, 1],
        dataset_name: str = "dataset",
        prompt_encoder=None,
        enable_clicks: bool = False,
        model_type: str = "simple_var",
    ):
        # models
        self.simple_var = simple_var
        self.vqvae: VQVAE_Single = vqvae
        self.prompt_encoder = prompt_encoder
        self.enable_clicks = enable_clicks
        self.model_type = model_type

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
        self.dataset_name = dataset_name

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
        val_dataloader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True
        )

        losses = []
        acc_means = []
        acc_soss = []

        pbar = tqdm.tqdm(enumerate(val_dataloader), desc="Val: ", total=num_iters)

        for i, (image, image_embed_sam, single_mask_normalized, single_mask) in pbar:
            image_embed_sam = image_embed_sam.to(self.device)
            single_mask_normalized = single_mask_normalized.to(self.device)
            single_mask = single_mask.to(self.device)

            if num_iters > 0 and i >= num_iters:
                break
            gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized)
            gt_idx_flat = torch.cat(gt_idx, dim=1)

            # Generate clicks if enabled
            if self.enable_clicks:
                batch_clicks = self.get_clicks_in_batch(single_mask, num_clicks=2)
                sparse_embeddings = self.clicks_to_prompt_embedding(batch_clicks)
            else:
                sparse_embeddings = None

            # Use model's forward method directly (works for both simple_var and simple_var_sd)
            logits = self.simple_var(
                idx=gt_idx,
                image_feat=image_embed_sam,
                vqvae=self.vqvae,
                sparse_embeddings=sparse_embeddings
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
        # executor = ThreadPoolExecutor(max_workers=self.batch_size * 2)

        val_dataloader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True
        )

        # Determine if we're using SimpleVARSamDecoder based on model_type
        is_sd_model = self.model_type == 'simple_var_sd'
        patch_num = self.simple_var.patch_num

        acc_means = []
        acc_soss = []
        ious = []
        acc_means_teacher = []

        pbar = tqdm.tqdm(enumerate(val_dataloader), desc="Val: ", total=num_iters)

        for i, (image, image_embed_sam, single_mask_normalized, single_mask) in pbar:
            if num_iters > 0 and i >= num_iters:
                break

            image_embed_sam = image_embed_sam.to(self.device)
            single_mask_normalized = single_mask_normalized.to(self.device)
            single_mask = single_mask.to(self.device)

            # Generate clicks if enabled
            if self.enable_clicks:
                batch_clicks = self.get_clicks_in_batch(single_mask, num_clicks=2)
                sparse_embeddings = self.clicks_to_prompt_embedding(batch_clicks)
            else:
                sparse_embeddings = None

            gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized)
            gt_idx_flat = torch.cat(gt_idx, dim=1)

            # Use appropriate inference function based on model type
            if is_sd_model:
                result = simple_var_sd_inference(
                    image_embed_sam, self.simple_var, self.vqvae, sparse_embeddings
                )
            else:
                result = simple_var_inference(
                    image_embed_sam, self.simple_var, self.vqvae, sparse_embeddings
                )

            flat_ids = torch.cat(result, dim=1)
            acc_mean = (flat_ids == gt_idx_flat).float().mean().item()
            acc_sos = (flat_ids[:, 1:] == gt_idx_flat[:, :-1]).float().mean().item()

            acc_means.append(acc_mean)
            acc_soss.append(acc_sos)

            # pred with teacher input - use model's forward method directly
            logits = self.simple_var(
                idx=gt_idx,
                image_feat=image_embed_sam,
                vqvae=self.vqvae,
                sparse_embeddings=sparse_embeddings
            )
            acc = (logits.argmax(dim=-1) == gt_idx_flat).float()
            acc_mean_teacher = acc.mean().item()
            acc_means_teacher.append(acc_mean_teacher)

            id_seq_teach = logits.argmax(dim=-1)
            id_seq_teach_Bl = []
            start_pos = 0
            for pn in patch_num:
                end_pos = start_pos + pn * pn
                id_seq_teach_Bl.append(id_seq_teach[:, start_pos:end_pos])
                start_pos = end_pos

            decoded_masks = self.vqvae_decode(result)
            decoded_masks_gt = self.vqvae_decode(gt_idx)
            decoded_masks_pred_with_teacher = self.vqvae_decode(id_seq_teach_Bl)

            iou_batch = calc_iou(decoded_masks[-1], single_mask)
            ious.append(iou_batch.mean().item())

            if visualize:
                # executor.submit(self.visualize_batch, i, image, iou_batch, decoded_masks, decoded_masks_gt, decoded_masks_pred_with_teacher)
                self.visualize_batch(i, image, iou_batch, decoded_masks, decoded_masks_gt, decoded_masks_pred_with_teacher)

        print(f"Average IOU: {sum(ious)/len(ious):.4f}")
        print(f"Average accuracy (no teacher): {sum(acc_means)/len(acc_means):.4f}")
        print(f"Average accuracy (with teacher): {sum(acc_means_teacher)/len(acc_means_teacher):.4f}")
        with open(self.out_dir / "eval" / self.dataset_name / f"eval_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            result_data = {
                "average_iou": sum(ious)/len(ious),
                "average_accuracy_no_teacher": sum(acc_means)/len(acc_means),
                "average_accuracy_with_teacher": sum(acc_means_teacher)/len(acc_means_teacher),
                "total_samples": len(ious)
            }
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        # executor.shutdown(wait=True)

    def vqvae_decode(self, indices):
        result = self.vqvae.idxBl_to_img(indices, same_shape=True)
        return result

    def get_clicks_in_batch(self, single_mask, num_clicks=2):
        """Generate initial clicks for each sample in the batch."""
        masks_np = single_mask.squeeze(1).cpu().numpy()
        batch_clicks = []
        for mask in masks_np:
            click_list, _, _ = init_clicks(
                gt_mask=mask,
                num_random_clicks=num_clicks,
                random_sample=True
            )
            batch_clicks.append(click_list)
        return batch_clicks

    def clicks_to_prompt_embedding(self, batch_clicks):
        """Convert batch of clicks to SAM prompt embeddings."""
        if self.prompt_encoder is None:
            raise ValueError("prompt_encoder is not initialized.")

        batch_size = len(batch_clicks)
        all_coords = []
        all_labels = []
        max_clicks = max(max(len(clicks) for clicks in batch_clicks), 4)

        for clicks in batch_clicks:
            coords, labels = to_sam_format(clicks, pad_size=max_clicks, device=self.device)
            all_coords.append(coords)
            all_labels.append(labels)

        coords_batch = torch.stack(all_coords, dim=0)
        labels_batch = torch.stack(all_labels, dim=0)

        with torch.no_grad():
            sparse_embeddings, _ = self.prompt_encoder(
                points=(coords_batch, labels_batch),
                boxes=None,
                masks=None
            )
        return sparse_embeddings

    def visualize_batch(self, step, image, iou_batch, decoded_masks, decoded_masks_gt, decoded_masks_pred_with_teacher):
        for j in range(self.batch_size):
            cur_iou = iou_batch[j].item()

            fig, ax = plt.subplots(1, 4, figsize=(15,4))
            ax[0].imshow(restore_normalized_image(image[j]).permute(1, 2, 0).cpu().numpy())
            ax[0].axis('off')
            ax[0].set_title(f'Image {step*self.batch_size + j}, IOU: {cur_iou:.3f}')

            result_gt = [m[j].unsqueeze(0) for m in decoded_masks_gt]
            result_mask_pred = [m[j].unsqueeze(0) for m in decoded_masks]
            result_mask_teacher = [m[j].unsqueeze(0) for m in decoded_masks_pred_with_teacher]

            self.visualize(result_gt, ax[1], 'gt')
            self.visualize(result_mask_pred, ax[2], 'pred')
            self.visualize(result_mask_teacher, ax[3], 'pred w/ teacher input')

            plt.savefig(self.out_dir / "eval" / self.dataset_name / "vis" / f'val_step_{step*self.batch_size + j}_iou{cur_iou:.3f}.png')
            plt.close()

    def visualize(self, result, ax, name='mask'):
        result = [mask for mask in result]
        chw = torchvision.utils.make_grid(torch.cat(result, dim=0), nrow=3, padding=1, pad_value=1.0)
        chw = chw.permute(1, 2, 0).cpu().numpy()
        ax.imshow(chw[:, :, 0])
        ax.axis('off')
        ax.set_title(name)

if __name__ == "__main__":
    import argparse
    from maskvar.maskseg_build_everything import builder_map

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--val_iters', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--visualize', action='store_true')
    # dataset
    parser.add_argument('--dataset', choices=['hqseg44k', 'cocolvis'], type=str, default='hqseg44k')
    parser.add_argument('--dataset_split', type=str, default='val')
    # simple var
    parser.add_argument('-c', '--simple_var', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--use_sam_pe', action='store_true')
    # image encoder
    parser.add_argument('--image_encoder', type=str, default='mobile_sam')
    parser.add_argument('--image_encoder_checkpoint', type=str, default='ckpt/mobile_sam.pt')
    # vqvae
    parser.add_argument('--vqvae', type=str, default='vqvae_single_5_stages_v1')
    parser.add_argument('--vqvae_checkpoint', type=str, default='out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth')
    # image cache dir
    parser.add_argument('--image_feature_cache_dir', type=str, default=None)
    # clicks
    parser.add_argument('--enable_clicks', action='store_true', help='Enable prompt encoder with click embeddings')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "eval" / f"{args.dataset}_{args.dataset_split}" / "vis").mkdir(parents=True, exist_ok=True)

    device = args.device

    checkpoint_path = args.checkpoint
    print(f"Using checkpoint: {checkpoint_path}")
    assert Path(checkpoint_path).exists(), f"Checkpoint not found: {checkpoint_path}"

    # sam_image_encoder = builder_map['image_encoder'][args.image_encoder](args.image_encoder_checkpoint)
    # sam_image_encoder = sam_image_encoder.to(device)
    # sam_image_encoder = torch.compile(sam_image_encoder)
    if args.image_feature_cache_dir is None:
        raise ValueError("image_feature_cache_dir is required now!")
    image_feature_cache = ImageFeatureCache(
        cache_dir=Path(args.image_feature_cache_dir),
        dataset=f"{args.dataset}_{args.dataset_split}",
        model_name=args.image_encoder,
        device=device,
    )

    train_set, val_set = builder_map['dataset'][args.dataset]() # validate on train set

    val_set_masklevel = MaskLevelFlatDataset(
        index_mapping_path=f'data/flat/{args.dataset}/{args.dataset_split}_index_mapping.npy',
        dataset=val_set if args.dataset_split != 'train' else train_set,
        with_image_embed=True,
        image_feature_cache=image_feature_cache,
        mask_filter_thresh=0.1,
        dtype=torch.float32,
    )

    # val_set_masklevel = MaskLevelDataset(
    #     dataset=val_set if args.dataset_split != 'train' else train_set,
    #     with_image_embed=True,
    #     device=args.device,
    #     mask_filter_thresh=0.1,
    #     image_feature_cache=image_feature_cache,
    # )

    if args.use_sam_pe:
        prompt_encoder = builder_map['prompt_encoder'](args.image_encoder_checkpoint).to(args.device)
        sam_pe = prompt_encoder.get_dense_pe().cpu()  # BCHW
        if args.enable_clicks:
            prompt_encoder.eval()
            for param in prompt_encoder.parameters():
                param.requires_grad = False
        else:
            del prompt_encoder
            prompt_encoder = None
    else:
        sam_pe = None
        prompt_encoder = None
        if args.enable_clicks:
            raise ValueError("--enable_clicks requires --use_sam_pe to be enabled")

    simple_var = builder_map['simple_var'][args.simple_var](simple_var_checkpoint_path=checkpoint_path, sam_pe=sam_pe, device=device, enable_prompt_tokens=args.enable_clicks)
    vqvae = builder_map['vqvae'][args.vqvae](vqvae_checkpoint_path=args.vqvae_checkpoint, require_grad=False)

    trainer = SimpleVAREvaluator(
        simple_var=simple_var,
        vqvae=vqvae,
        val_set=val_set_masklevel,
        batch_size=args.batch_size,
        device=device,
        out_dir=outdir,
        dataset_name=f'{args.dataset}_{args.dataset_split}',
        prompt_encoder=prompt_encoder,
        enable_clicks=args.enable_clicks,
        model_type=args.simple_var,
    )

    trainer.eval_ar(args.val_iters, visualize=args.visualize)
    print(f"Evaluation complete")
    if args.visualize:
        print(f"Visualization saved to {outdir / 'eval' / f'{args.dataset}_{args.dataset_split}' / 'vis'}")
    