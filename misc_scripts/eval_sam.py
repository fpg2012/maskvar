"""
SAM evaluation script.

Evaluates SAM model on mask segmentation tasks and computes metrics:
1. IoU between SAM predicted mask and ground truth mask
2. Token accuracy: SAM mask -> VQVAE encode -> compare tokens with GT tokens
3. Reconstruction IoU: SAM mask -> VQVAE encode -> VQVAE decode -> compare with GT mask
"""

from pathlib import Path
from datetime import datetime
import json

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import tqdm
from einops import rearrange
import matplotlib.pyplot as plt

from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset
from maskvar.maskseg_build_everything import builder_map
from maskvar.models.vqvae_single import VQVAE_Single
from maskvar.models.sam import ImageEncoderViT as SamImageEncoder, PromptEncoder, MaskDecoder
from maskvar.utils import restore_normalized_image
from maskvar.utils.metrics import calc_iou
from maskvar.utils.clicker import init_clicks, to_sam_format

torch.set_float32_matmul_precision('high')


class SAMEvaluator:
    """Evaluator for SAM model."""

    def __init__(
        self,
        image_encoder: SamImageEncoder,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        vqvae: VQVAE_Single,
        val_set: MaskLevelFlatDataset,
        batch_size: int,
        out_dir: Path,
        device: str,
        dataset_name: str = "dataset",
        num_clicks: int = 1,
    ):
        # models
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.vqvae = vqvae

        # device
        self.device = device

        # dataset
        self.val_set = val_set
        self.batch_size = batch_size

        # config
        self.num_clicks = num_clicks
        self.out_dir = out_dir
        self.dataset_name = dataset_name

        self.compile_model()

    def compile_model(self):
        """Move models to device and compile."""
        self.image_encoder.to(self.device)
        self.prompt_encoder.to(self.device)
        self.mask_decoder.to(self.device)
        self.vqvae.to(self.device)

        self.image_encoder = torch.compile(self.image_encoder)
        # Don't compile prompt_encoder and mask_decoder due to shape issues
        # self.prompt_encoder = torch.compile(self.prompt_encoder)
        # self.mask_decoder = torch.compile(self.mask_decoder)
        self.vqvae = torch.compile(self.vqvae)

        self.image_encoder.eval()
        self.prompt_encoder.eval()
        self.mask_decoder.eval()
        self.vqvae.eval()

    @torch.no_grad()
    def get_clicks_in_batch(self, single_mask):
        """Generate initial clicks for each sample in the batch."""
        masks_np = single_mask.squeeze(1).cpu().numpy()
        batch_clicks = []
        for mask in masks_np:
            click_list, _, _ = init_clicks(
                gt_mask=mask,
                num_random_clicks=self.num_clicks,
                random_sample=True
            )
            batch_clicks.append(click_list)
        return batch_clicks

    @torch.no_grad()
    def clicks_to_prompt_embedding(self, batch_clicks, mask_size=256, image_size=1024):
        """
        Convert batch of clicks to SAM prompt embeddings.

        Args:
            batch_clicks: list of click lists, each click is (y, x, label) in mask coordinates
            mask_size: size of the mask where clicks were generated (default 256)
            image_size: size of the image for SAM (default 1024)
        """
        batch_size = len(batch_clicks)
        all_coords = []
        all_labels = []
        max_clicks = max(max(len(clicks) for clicks in batch_clicks), 4)

        # Scale factor from mask space to image space
        scale_factor = image_size / mask_size

        for clicks in batch_clicks:
            coords, labels = to_sam_format(clicks, pad_size=max_clicks, device=self.device)
            # coords are (x, y) from to_sam_format, need to scale to image space
            coords = coords * scale_factor
            all_coords.append(coords)
            all_labels.append(labels)

        coords_batch = torch.stack(all_coords, dim=0)  # (B, N, 2)
        labels_batch = torch.stack(all_labels, dim=0)  # (B, N)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(coords_batch, labels_batch),
            boxes=None,
            masks=None
        )
        return sparse_embeddings, dense_embeddings

    @torch.no_grad()
    def sam_forward(self, image_embed_sam, batch_clicks):
        """
        Run SAM forward pass to get mask predictions.
        Process each sample individually to match original SAM behavior.
        Uses pre-computed image embeddings from cache.

        Args:
            image_embed_sam: (B, 256, 64, 64) pre-computed image embeddings
            batch_clicks: list of click lists for each sample

        Returns:
            masks: (B, 1, H, W) predicted masks
        """
        batch_size = image_embed_sam.shape[0]
        all_masks = []

        # Scale factor from mask space (256) to image space (1024)
        scale_factor = 1024 / 256

        # Process each sample individually (like SAM's original forward method)
        for i in range(batch_size):
            # Get single image embedding from cache
            image_embedding = image_embed_sam[i:i+1]  # (1, 256, 64, 64)

            # Get clicks for this sample and scale coordinates
            clicks = batch_clicks[i]
            if len(clicks) > 0:
                # clicks are (y, x, label), convert to (x, y) and scale
                coords = torch.tensor(
                    [(click[1] * scale_factor, click[0] * scale_factor) for click in clicks],
                    device=self.device, dtype=torch.float
                )
                labels = torch.tensor(
                    [click[2] for click in clicks],
                    device=self.device, dtype=torch.long
                )
            else:
                # No clicks, use dummy values
                coords = torch.zeros((1, 2), device=self.device, dtype=torch.float)
                labels = torch.tensor([-1], device=self.device, dtype=torch.long)

            # Add batch dimension for single sample
            coords = coords.unsqueeze(0)  # (1, N, 2)
            labels = labels.unsqueeze(0)  # (1, N)

            # Get prompt embeddings
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(coords, labels),
                boxes=None,
                masks=None
            )

            # Decode mask for single sample
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # Upsample to 256x256
            mask = torch.nn.functional.interpolate(
                low_res_masks,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
            all_masks.append(mask)

        # Concatenate all masks
        masks = torch.cat(all_masks, dim=0)  # (B, 1, 256, 256)
        return masks

    @torch.no_grad()
    def eval(self, num_iters: int = 0, visualize: bool = False):
        """
        Evaluate SAM on validation set.

        Metrics computed:
        1. iou_sam: IoU between SAM predicted mask and GT mask
        2. token_acc: Accuracy of VQVAE tokens from SAM mask vs GT tokens
        3. iou_reconstruct: IoU between VQVAE decoded SAM mask and GT mask
        """
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

        ious_sam = []
        token_accs = []
        ious_reconstruct = []

        if num_iters == 0:
            num_iters = len(val_dataloader)

        pbar = tqdm.tqdm(enumerate(val_dataloader), desc="Eval SAM: ", total=num_iters)

        for i, (image, image_embed_sam, single_mask_normalized, single_mask) in pbar:
            if i >= num_iters:
                break

            # Move to device
            image = image.to(self.device)
            image_embed_sam = image_embed_sam.to(self.device)
            single_mask_normalized = single_mask_normalized.to(self.device)
            single_mask = single_mask.to(self.device)

            # Generate clicks from GT masks
            batch_clicks = self.get_clicks_in_batch(single_mask)

            # SAM forward to get predicted masks (using pre-computed image embeddings)
            sam_masks = self.sam_forward(image_embed_sam, batch_clicks)  # (B, 1, 256, 256)

            # Normalize SAM masks to [0, 1] for VQVAE encoding
            sam_masks_normalized = torch.clamp(sam_masks, 0, 1)

            # Get GT tokens
            gt_idx = self.vqvae.img_to_idxBl(single_mask_normalized)  # List of (B, l)
            gt_idx_flat = torch.cat(gt_idx, dim=1)  # (B, L)

            # Encode SAM masks with VQVAE to get tokens
            sam_idx = self.vqvae.img_to_idxBl(sam_masks_normalized)  # List of (B, l)
            sam_idx_flat = torch.cat(sam_idx, dim=1)  # (B, L)

            # Metric 1: Token accuracy
            token_acc = (sam_idx_flat == gt_idx_flat).float().mean().item()
            token_accs.append(token_acc)

            # Decode SAM tokens back to images for reconstruction IoU
            sam_reconstructed = self.vqvae.idxBl_to_img(sam_idx, same_shape=True)  # List of (B, 1, H, W)

            # Metric 2: SAM IoU (direct SAM output vs GT)
            iou_sam = calc_iou(sam_masks_normalized, single_mask).mean().item()
            ious_sam.append(iou_sam)

            # Metric 3: Reconstruction IoU (VQVAE decoded SAM vs GT)
            # Use the finest scale (last element)
            iou_reconstruct = calc_iou(sam_reconstructed[-1], single_mask).mean().item()
            ious_reconstruct.append(iou_reconstruct)

            pbar.set_postfix({
                'iou_sam': f'{iou_sam:.4f}',
                'token_acc': f'{token_acc:.4f}',
                'iou_recon': f'{iou_reconstruct:.4f}'
            })

            if visualize:
                self.visualize_batch(
                    i, image, sam_masks_normalized, sam_reconstructed[-1],
                    single_mask, iou_sam, token_acc, iou_reconstruct
                )

        # Compute averages
        avg_iou_sam = sum(ious_sam) / len(ious_sam)
        avg_token_acc = sum(token_accs) / len(token_accs)
        avg_iou_reconstruct = sum(ious_reconstruct) / len(ious_reconstruct)

        print(f"\nEvaluation Results:")
        print(f"  SAM IoU: {avg_iou_sam:.4f}")
        print(f"  Token Accuracy: {avg_token_acc:.4f}")
        print(f"  Reconstruction IoU: {avg_iou_reconstruct:.4f}")

        # Save results
        result_data = {
            "sam_iou": avg_iou_sam,
            "token_accuracy": avg_token_acc,
            "reconstruction_iou": avg_iou_reconstruct,
            "total_samples": len(ious_sam),
            "num_clicks": self.num_clicks,
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
        }

        result_path = self.out_dir / "eval" / self.dataset_name / f"sam_eval_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        return result_data

    def visualize_batch(self, step, image, sam_masks, sam_reconstructed, gt_mask,
                        iou_sam, token_acc, iou_reconstruct):
        """Visualize a batch of results."""
        for j in range(min(4, self.batch_size)):  # Visualize up to 4 samples
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))

            # Original image
            ax[0].imshow(restore_normalized_image(image[j]).permute(1, 2, 0).cpu().numpy())
            ax[0].axis('off')
            ax[0].set_title(f'Image {step*self.batch_size + j}')

            # GT mask
            ax[1].imshow(gt_mask[j, 0].cpu().numpy(), cmap='gray')
            ax[1].axis('off')
            ax[1].set_title('GT Mask')

            # SAM predicted mask
            ax[2].imshow(sam_masks[j, 0].cpu().numpy(), cmap='gray')
            ax[2].axis('off')
            ax[2].set_title(f'SAM Pred (IoU: {iou_sam:.3f})')

            # VQVAE reconstructed SAM mask
            ax[3].imshow(sam_reconstructed[j, 0].cpu().numpy(), cmap='gray')
            ax[3].axis('off')
            ax[3].set_title(f'VQVAE Recon (IoU: {iou_reconstruct:.3f})')

            plt.suptitle(f'Token Acc: {token_acc:.3f}')
            plt.tight_layout()

            vis_dir = self.out_dir / "eval" / self.dataset_name / "vis"
            vis_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(vis_dir / f'sam_eval_step_{step*self.batch_size + j}.png')
            plt.close()


def build_sam_model(checkpoint_path: str, device: str):
    """Build SAM model from checkpoint with correct parameters for original SAM."""
    from maskvar.build_sam import build_sam_vit_b

    # Build complete SAM model using the official builder
    sam = build_sam_vit_b(checkpoint=checkpoint_path)

    # Extract components
    image_encoder = sam.image_encoder
    prompt_encoder = sam.prompt_encoder
    mask_decoder = sam.mask_decoder

    return image_encoder, prompt_encoder, mask_decoder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate SAM model')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--val_iters', type=int, default=0, help='Number of validation iterations (0 = all)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--visualize', action='store_true')

    # SAM checkpoint
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                        help='Path to SAM checkpoint (e.g., sam_vit_b_01ec64.pth)')

    # VQVAE
    parser.add_argument('--vqvae', type=str, default='vqvae_single_5_stages_v1')
    parser.add_argument('--vqvae_checkpoint', type=str,
                        default='out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth')

    # Dataset
    parser.add_argument('--dataset', choices=['hqseg44k', 'cocolvis'], type=str, default='hqseg44k')
    parser.add_argument('--dataset_split', type=str, default='val')

    # Image feature cache
    parser.add_argument('--image_feature_cache_dir', type=str, default=None)

    # Number of clicks for evaluation
    parser.add_argument('--num_clicks', type=int, default=1,
                        help='Number of clicks to use for SAM prediction')

    args = parser.parse_args()

    # Setup paths
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "eval" / f"{args.dataset}_{args.dataset_split}" / "vis").mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")
    print(f"SAM checkpoint: {args.sam_checkpoint}")
    print(f"VQVAE: {args.vqvae}")
    print(f"Dataset: {args.dataset}")
    print(f"Num clicks: {args.num_clicks}")

    # Check checkpoint exists
    assert Path(args.sam_checkpoint).exists(), f"SAM checkpoint not found: {args.sam_checkpoint}"
    assert Path(args.vqvae_checkpoint).exists(), f"VQVAE checkpoint not found: {args.vqvae_checkpoint}"

    # Load image feature cache if provided
    if args.image_feature_cache_dir:
        from maskvar.datasets.image_feature_cache import ImageFeatureCache
        image_feature_cache = ImageFeatureCache(
            cache_dir=Path(args.image_feature_cache_dir),
            dataset=f"{args.dataset}_{args.dataset_split}",
            model_name='sam_vitb',
            device=device,
        )
    else:
        image_feature_cache = None

    # Build dataset
    train_set, val_set = builder_map['dataset'][args.dataset]()

    val_set_masklevel = MaskLevelFlatDataset(
        index_mapping_path=f'data/flat/{args.dataset}/{args.dataset_split}_index_mapping.npy',
        dataset=val_set if args.dataset_split != 'train' else train_set,
        with_image_embed=True,
        image_feature_cache=image_feature_cache,
        mask_filter_thresh=0.1,
        dtype=torch.float32,
    )

    # Build SAM model
    print("Building SAM model...")
    image_encoder, prompt_encoder, mask_decoder = build_sam_model(args.sam_checkpoint, device)

    # Build VQVAE
    print("Building VQVAE...")
    vqvae = builder_map['vqvae'][args.vqvae](vqvae_checkpoint_path=args.vqvae_checkpoint, require_grad=False)

    # Create evaluator
    evaluator = SAMEvaluator(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        vqvae=vqvae,
        val_set=val_set_masklevel,
        batch_size=args.batch_size,
        out_dir=outdir,
        device=device,
        dataset_name=f'{args.dataset}_{args.dataset_split}',
        num_clicks=args.num_clicks,
    )

    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.eval(num_iters=args.val_iters, visualize=args.visualize)

    print(f"\nEvaluation complete!")
    if args.visualize:
        print(f"Visualizations saved to {outdir / 'eval' / f'{args.dataset}_{args.dataset_split}' / 'vis'}")
