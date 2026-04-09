"""
Test script for SimpleMaskVqvae.

This script tests the forward pass of SimpleMaskVqvae and visualizes the results.

Usage:
    python notebooks/test_simple_mask_vqvae.py \
        --sam_checkpoint_path checkpoints/mobile_sam.pt \
        --dataset hqseg44k \
        --num_samples 4

    # With a trained checkpoint
    python notebooks/test_simple_mask_vqvae.py \
        --sam_checkpoint_path checkpoints/mobile_sam.pt \
        --checkpoint_path out_simple_mask_vqvae_v0/checkpoints/latest.pth \
        --dataset hqseg44k \
        --num_samples 4
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from maskvar.maskseg_build_everything import builder_map
from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset
from maskvar.utils.metrics import calc_iou
from maskvar.utils import restore_normalized_image


def visualize_results(image, gt_mask, pred_logits, pred_mask, iou, save_path=None):
    """
    Visualize SimpleMaskVqvae results with color-coded error map.

    Args:
        image: (3, H, W) tensor, original image (normalized)
        gt_mask: (H, W) tensor, ground truth mask (binary)
        pred_logits: (H, W) tensor, predicted logits (before sigmoid)
        pred_mask: (H, W) tensor, predicted mask (binary)
        iou: float, IoU score
        save_path: optional path to save the figure
    """
    # Restore normalized image to original image
    image = restore_normalized_image(image)
    # Convert to numpy and transpose to HWC
    image = image.cpu().numpy().transpose(1, 2, 0)

    gt_mask = gt_mask.cpu().numpy() > 0
    pred_mask = pred_mask.cpu().numpy() > 0
    pred_logits = pred_logits.cpu().numpy()

    # Create figure with GridSpec for better layout
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # Row 0: Original Image and Masks
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Ground Truth Mask')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    ax3.set_title(f'Predicted Mask (IoU={iou:.3f})')
    ax3.axis('off')

    # Row 1: Color-coded Error Map
    # Define soft colors for overlay (normalized RGB)
    COLOR_TP = np.array([0.2, 0.6, 1.0])   # Soft blue for correct foreground
    COLOR_FP = np.array([1.0, 0.3, 0.3])   # Soft red for false positive
    COLOR_FN = np.array([0.3, 0.9, 0.3])   # Soft green for false negative

    # Create color-coded error map with soft colors
    error_map = np.zeros((*gt_mask.shape, 3))
    tp_mask = gt_mask & pred_mask
    fp_mask = (~gt_mask) & pred_mask
    fn_mask = gt_mask & (~pred_mask)

    error_map[tp_mask] = COLOR_TP
    error_map[fp_mask] = COLOR_FP
    error_map[fn_mask] = COLOR_FN

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(error_map)
    ax4.set_title('Error Map (Blue=TP, Red=FP, Green=FN)')
    ax4.axis('off')

    # Overlay on original image with alpha blending
    alpha = 0.35  # Transparency for error colors
    overlay = image.copy().astype(np.float32) / 255.0 if image.max() > 1 else image.copy()

    # Apply each color mask with alpha blending
    for mask, color in [(tp_mask, COLOR_TP), (fp_mask, COLOR_FP), (fn_mask, COLOR_FN)]:
        if mask.any():
            overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    overlay = np.clip(overlay, 0, 1)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(overlay)
    ax5.set_title('Error Overlay on Image')
    ax5.axis('off')

    # Logits heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    vmin, vmax = pred_logits.min(), pred_logits.max()
    vmax_abs = max(abs(vmin), abs(vmax))
    im = ax6.imshow(pred_logits, cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs)
    ax6.set_title('Logits Heatmap')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

    plt.suptitle('SimpleMaskVqvae Test Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Test SimpleMaskVqvae')

    # Model
    parser.add_argument('--sam_checkpoint_path', type=str, default=None,
                        help='Path to SAM/MobileSAM checkpoint for initializing encoders')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to trained SimpleMaskVqvae checkpoint (optional)')

    # Data
    parser.add_argument('--dataset', type=str, default='hqseg44k',
                        choices=['hqseg44k', 'cocolvis', 'coconut_hf'],
                        help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Dataset path (auto-detected if None)')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to test')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],
                        help='Dataset split to use')

    # Output
    parser.add_argument('--output_dir', type=str, default='notebooks/test_outputs',
                        help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    print("Building SimpleMaskVqvae model...")
    model = builder_map['simple_mask_vqvae']['simple_mask_vqvae'](
        simple_mask_vqvae_checkpoint_path=args.checkpoint_path,
        sam_checkpoint_path=args.sam_checkpoint_path,
        device=device,
    )
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Load dataset
    print(f"\nLoading {args.dataset} dataset ({args.split} split)...")
    dataset_path_map = {
        'hqseg44k': 'data/sam-hq',
        'cocolvis': 'data/coco_lvis',
        'coconut_hf': 'data/coconut_hf',
    }
    dataset_path = args.dataset_path or dataset_path_map[args.dataset]

    train_set_base, val_set_base = builder_map['dataset'][args.dataset](dataset_path)
    base_set = train_set_base if args.split == 'train' else val_set_base

    index_mapping_path = f'data/flat/{args.dataset}'
    split_name = 'train' if args.split == 'train' else 'val'

    dataset = MaskLevelFlatDataset(
        index_mapping_path=Path(index_mapping_path) / f"{split_name}_index_mapping.npy",
        dataset=base_set,
        with_image_embed=False,
        image_feature_cache=None,
        mask_filter_thresh=0.1,
        dtype=torch.float32,
        image_size_encoder=1024,  # Image size for encoder (must match encoder expectation)
        image_size_mask=1024,     # Mask size - must match image_size_encoder for this model
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    # Test on samples
    num_samples = min(args.num_samples, len(dataset))
    print(f"\nTesting on {num_samples} samples...\n")

    results = []
    with torch.no_grad():
        for i in range(num_samples):
            # Get sample
            image, _, single_mask_normalized, single_mask = dataset[i]

            # Add batch dimension
            image_batch = image.unsqueeze(0).to(device)
            mask_normalized_batch = single_mask_normalized.unsqueeze(0).to(device)

            # Forward pass
            rec_mask_logits, vq_loss = model(mask_normalized_batch, image_batch)

            # Calculate IoU
            rec_mask_binary = (rec_mask_logits > 0).float()
            iou = calc_iou(rec_mask_binary, single_mask.unsqueeze(0).to(device))
            iou_value = iou.item()

            print(f"Sample {i+1}/{num_samples}:")
            print(f"  VQ Loss: {vq_loss.item():.4f}")
            print(f"  IoU: {iou_value:.4f}")

            # Handle different output shapes: (B, H, W) or (B, 1, H, W)
            if rec_mask_logits.dim() == 3:
                pred_logits_vis = rec_mask_logits[0]  # (B, H, W) -> (H, W)
                pred_mask_vis = rec_mask_binary[0]    # (B, H, W) -> (H, W)
            else:
                pred_logits_vis = rec_mask_logits[0, 0]  # (B, 1, H, W) -> (H, W)
                pred_mask_vis = rec_mask_binary[0, 0]    # (B, 1, H, W) -> (H, W)

            # Visualize
            fig = visualize_results(
                image=image,
                gt_mask=single_mask[0],  # (1, H, W) -> (H, W)
                pred_logits=pred_logits_vis,
                pred_mask=pred_mask_vis,
                iou=iou_value,
                save_path=output_dir / f'sample_{i+1}_result.png'
            )

            results.append({
                'sample_idx': i,
                'vq_loss': vq_loss.item(),
                'iou': iou_value,
            })

    # Print summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    avg_vq_loss = sum(r['vq_loss'] for r in results) / len(results)
    avg_iou = sum(r['iou'] for r in results) / len(results)
    print(f"Average VQ Loss: {avg_vq_loss:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"\nVisualizations saved to: {output_dir}")

    # Show plots if running interactively
    plt.show()


if __name__ == '__main__':
    main()
