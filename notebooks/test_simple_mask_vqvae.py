"""
Test script for SimpleMaskVqvae.

This script tests the forward pass of SimpleMaskVqvae and visualizes the results.

Usage:
    # Test with MobileSAM encoder (legacy)
    python notebooks/test_simple_mask_vqvae.py \
        --image_encoder_checkpoint checkpoints/mobile_sam.pt \
        --image_encoder_config mobile_sam \
        --dataset hqseg44k \
        --num_samples 4

    # Test with DINOv3 encoder (trained with ddp_train_coconut_hf_dino.sh)
    python notebooks/test_simple_mask_vqvae.py \
        --config simple_mask_vqvae_dim384 \
        --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
        --image_encoder_config dino_v3_vits \
        --dataset coconut_hf \
        --num_samples 4

    # With a trained checkpoint
    python notebooks/test_simple_mask_vqvae.py \
        --config simple_mask_vqvae_dim384 \
        --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
        --image_encoder_config dino_v3_vits \
        --checkpoint_path out_simple_mask_vqvae_v0/checkpoints/latest.pth \
        --dataset coconut_hf \
        --num_samples 4

    # Test with random rectangle mask (no GT comparison)
    python notebooks/test_simple_mask_vqvae.py \
        --config simple_mask_vqvae_dim384 \
        --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
        --image_encoder_config dino_v3_vits \
        --checkpoint_path out_simple_mask_vqvae_v0/checkpoints/latest.pth \
        --dataset coconut_hf \
        --test_mode random_rect \
        --num_samples 4

    # Test with bounding box mask (no GT comparison)
    python notebooks/test_simple_mask_vqvae.py \
        --config simple_mask_vqvae_dim384 \
        --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
        --image_encoder_config dino_v3_vits \
        --checkpoint_path out_simple_mask_vqvae_v0/checkpoints/latest.pth \
        --dataset coconut_hf \
        --test_mode bbox \
        --num_samples 4

    # Test with circles mask (sample points from mask and create circles)
    python notebooks/test_simple_mask_vqvae.py \
        --config simple_mask_vqvae_dim384 \
        --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
        --image_encoder_config dino_v3_vits \
        --checkpoint_path out_simple_mask_vqvae_v0/checkpoints/latest.pth \
        --dataset coconut_hf \
        --test_mode circles \
        --num_samples 4

    # Test with coarse mask (downsample then upsample)
    python notebooks/test_simple_mask_vqvae.py \
        --config simple_mask_vqvae_dim384 \
        --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
        --image_encoder_config dino_v3_vits \
        --checkpoint_path out_simple_mask_vqvae_v0/checkpoints/latest.pth \
        --dataset coconut_hf \
        --test_mode coarse \
        --coarse_factor 16 \
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


def create_random_rect_mask(h, w, min_size_ratio=0.1, max_size_ratio=0.5):
    """
    Create a random rectangle mask.

    Args:
        h, w: Height and width of the mask
        min_size_ratio: Minimum size ratio relative to image dimensions
        max_size_ratio: Maximum size ratio relative to image dimensions

    Returns:
        mask: (1, h, w) binary mask tensor (0 or 1)
    """
    mask = torch.zeros(1, h, w)

    # Random rectangle size
    min_h, max_h = int(h * min_size_ratio), int(h * max_size_ratio)
    min_w, max_w = int(w * min_size_ratio), int(w * max_size_ratio)

    rect_h = np.random.randint(min_h, max_h + 1)
    rect_w = np.random.randint(min_w, max_w + 1)

    # Random position
    y1 = np.random.randint(0, h - rect_h + 1)
    x1 = np.random.randint(0, w - rect_w + 1)

    # Create rectangle mask
    mask[0, y1:y1+rect_h, x1:x1+rect_w] = 1.0

    return mask


def create_bbox_mask(mask):
    """
    Create a bounding box mask from an existing binary mask.

    Args:
        mask: (1, h, w) binary mask tensor

    Returns:
        bbox_mask: (1, h, w) binary mask with bounding box filled
    """
    mask_np = mask[0].cpu().numpy()

    # Find bounding box
    rows = np.any(mask_np > 0, axis=1)
    cols = np.any(mask_np > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        # Empty mask, return zeros
        return torch.zeros_like(mask)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Create bbox mask
    bbox_mask = torch.zeros_like(mask)
    bbox_mask[0, y_min:y_max+1, x_min:x_max+1] = 1.0

    return bbox_mask


def create_circles_mask(mask, num_points=2, radius_ratio=0.05):
    """
    Create a mask from union of circles sampled from the mask.

    Args:
        mask: (1, h, w) binary mask tensor
        num_points: Number of points to sample (1 or 2)
        radius_ratio: Radius relative to min(h, w) of the image

    Returns:
        circles_mask: (1, h, w) binary mask with union of circles
        centers: list of (y, x) tuples for sampled points
    """
    mask_np = mask[0].cpu().numpy()
    h, w = mask_np.shape

    # Find all foreground pixels
    foreground_y, foreground_x = np.where(mask_np > 0)

    if len(foreground_y) == 0:
        # Empty mask, return zeros
        return torch.zeros_like(mask), []

    # Sample points from foreground
    num_points = min(num_points, len(foreground_y))
    indices = np.random.choice(len(foreground_y), size=num_points, replace=False)
    centers = [(foreground_y[i], foreground_x[i]) for i in indices]

    # Calculate radius (relative to image size, but not too small or too large)
    radius = int(min(h, w) * radius_ratio)
    radius = max(radius, 8)   # At least 8 pixels
    radius = min(radius, 64)  # At most 64 pixels

    # Create circles mask
    circles_mask = torch.zeros_like(mask)
    y_grid, x_grid = np.ogrid[:h, :w]

    for cy, cx in centers:
        # Create circle: (y - cy)^2 + (x - cx)^2 <= radius^2
        dist_sq = (y_grid - cy) ** 2 + (x_grid - cx) ** 2
        circle_mask = dist_sq <= radius ** 2
        circles_mask[0][torch.from_numpy(circle_mask)] = 1.0

    return circles_mask, centers


def create_coarse_mask(mask, downsample_factor=8):
    """
    Create a coarse mask by downsampling then upsampling with bilinear interpolation.

    Args:
        mask: (1, h, w) binary mask tensor
        downsample_factor: Factor to downsample (e.g., 8, 16, 32)

    Returns:
        coarse_mask: (1, h, w) coarse mask tensor
    """
    _, h, w = mask.shape

    # Downsample to smaller size
    small_h = max(h // downsample_factor, 2)
    small_w = max(w // downsample_factor, 2)

    # Add batch dimension for interpolate: (1, 1, h, w)
    mask_4d = mask.unsqueeze(0)

    # Downsample
    small_mask = torch.nn.functional.interpolate(
        mask_4d, size=(small_h, small_w), mode='bilinear', align_corners=False
    )

    # Upsample back to original size
    coarse_mask = torch.nn.functional.interpolate(
        small_mask, size=(h, w), mode='bilinear', align_corners=False
    )

    # Remove batch dimension and binarize
    coarse_mask = coarse_mask.squeeze(0)
    coarse_mask = (coarse_mask > 0.5).float()

    return coarse_mask


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


def visualize_bbox_test(image, input_mask, pred_logits, pred_mask, save_path=None, title="SimpleMaskVqvae Alternative Mask Test"):
    """
    Visualize results for random rectangle test (no GT comparison).

    Args:
        image: (3, H, W) tensor, original image (normalized)
        input_mask: (H, W) tensor, input mask (random rect)
        pred_logits: (H, W) tensor, predicted logits (before sigmoid)
        pred_mask: (H, W) tensor, predicted mask (binary)
        save_path: optional path to save the figure
        title: title for the figure
    """
    # Restore normalized image to original image
    image = restore_normalized_image(image)
    # Convert to numpy and transpose to HWC
    image = image.cpu().numpy().transpose(1, 2, 0)

    input_mask = input_mask.cpu().numpy() > 0
    pred_mask = pred_mask.cpu().numpy() > 0
    pred_logits = pred_logits.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 0: Original Image and Masks
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(input_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Input Mask')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Predicted Mask')
    axes[0, 2].axis('off')

    # Row 1: Overlay and comparison
    # Overlay input mask on image
    alpha = 0.35
    overlay_input = image.copy().astype(np.float32) / 255.0 if image.max() > 1 else image.copy()
    overlay_input = overlay_input * (1 - alpha) + np.array([1.0, 0.3, 0.3]) * alpha * input_mask[:, :, None] + \
                    overlay_input * (1 - input_mask[:, :, None])
    overlay_input = np.clip(overlay_input, 0, 1)

    axes[1, 0].imshow(overlay_input)
    axes[1, 0].set_title('Input Mask Overlay (Red)')
    axes[1, 0].axis('off')

    # Overlay predicted mask on image
    overlay_pred = image.copy().astype(np.float32) / 255.0 if image.max() > 1 else image.copy()
    overlay_pred = overlay_pred * (1 - alpha) + np.array([0.2, 0.6, 1.0]) * alpha * pred_mask[:, :, None] + \
                   overlay_pred * (1 - pred_mask[:, :, None])
    overlay_pred = np.clip(overlay_pred, 0, 1)

    axes[1, 1].imshow(overlay_pred)
    axes[1, 1].set_title('Predicted Mask Overlay (Blue)')
    axes[1, 1].axis('off')

    # Logits heatmap
    vmin, vmax = pred_logits.min(), pred_logits.max()
    vmax_abs = max(abs(vmin), abs(vmax))
    im = axes[1, 2].imshow(pred_logits, cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs)
    axes[1, 2].set_title('Logits Heatmap')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close(fig)
    return fig


def visualize_with_gt(image, gt_mask, input_mask, pred_logits, pred_mask, iou, save_path=None, title="SimpleMaskVqvae Alternative Mask Test"):
    """
    Visualize results for alternative mask input with GT comparison.

    Args:
        image: (3, H, W) tensor, original image (normalized)
        gt_mask: (H, W) tensor, ground truth mask
        input_mask: (H, W) tensor, input mask (bbox, circles, or coarse)
        pred_logits: (H, W) tensor, predicted logits (before sigmoid)
        pred_mask: (H, W) tensor, predicted mask (binary)
        iou: float, IoU score
        save_path: optional path to save the figure
        title: title for the figure
    """
    # Restore normalized image to original image
    image = restore_normalized_image(image)
    image = image.cpu().numpy().transpose(1, 2, 0)

    gt_mask = gt_mask.cpu().numpy() > 0
    input_mask = input_mask.cpu().numpy() > 0
    pred_mask = pred_mask.cpu().numpy() > 0
    pred_logits = pred_logits.cpu().numpy()

    # Create figure: 2 rows x 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 0
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(input_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Input Mask (Alternative)')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 3].set_title(f'Predicted Mask (IoU={iou:.3f})')
    axes[0, 3].axis('off')

    # Row 1: Error analysis and overlays
    # Define colors
    COLOR_TP = np.array([0.2, 0.6, 1.0])
    COLOR_FP = np.array([1.0, 0.3, 0.3])
    COLOR_FN = np.array([0.3, 0.9, 0.3])

    # Error map: compare prediction with GT
    error_map = np.zeros((*gt_mask.shape, 3))
    tp_mask = gt_mask & pred_mask
    fp_mask = (~gt_mask) & pred_mask
    fn_mask = gt_mask & (~pred_mask)
    error_map[tp_mask] = COLOR_TP
    error_map[fp_mask] = COLOR_FP
    error_map[fn_mask] = COLOR_FN

    axes[1, 0].imshow(error_map)
    axes[1, 0].set_title('Error Map (vs GT)\nBlue=TP, Red=FP, Green=FN')
    axes[1, 0].axis('off')

    # Overlay input mask on image
    alpha = 0.35
    overlay_input = image.copy().astype(np.float32) / 255.0 if image.max() > 1 else image.copy()
    overlay_input = overlay_input * (1 - alpha) + np.array([1.0, 0.3, 0.3]) * alpha * input_mask[:, :, None] + \
                    overlay_input * (1 - input_mask[:, :, None])
    overlay_input = np.clip(overlay_input, 0, 1)

    axes[1, 1].imshow(overlay_input)
    axes[1, 1].set_title('Input Mask Overlay (Red)')
    axes[1, 1].axis('off')

    # Overlay predicted mask on image
    overlay_pred = image.copy().astype(np.float32) / 255.0 if image.max() > 1 else image.copy()
    overlay_pred = overlay_pred * (1 - alpha) + np.array([0.2, 0.6, 1.0]) * alpha * pred_mask[:, :, None] + \
                   overlay_pred * (1 - pred_mask[:, :, None])
    overlay_pred = np.clip(overlay_pred, 0, 1)

    axes[1, 2].imshow(overlay_pred)
    axes[1, 2].set_title('Predicted Mask Overlay (Blue)')
    axes[1, 2].axis('off')

    # Logits heatmap
    vmin, vmax = pred_logits.min(), pred_logits.max()
    vmax_abs = max(abs(vmin), abs(vmax))
    im = axes[1, 3].imshow(pred_logits, cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs)
    axes[1, 3].set_title('Logits Heatmap')
    axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description='Test SimpleMaskVqvae')

    # Model config
    parser.add_argument('--config', type=str, default='simple_mask_vqvae',
                        choices=['simple_mask_vqvae', 'simple_mask_vqvae_dim384'],
                        help='Model configuration')
    parser.add_argument('--image_encoder_config', type=str, default='dino_v3_vits',
                        choices=['mobile_sam', 'dino_v3_vits', 'dino_v3_vitb'],
                        help='Image encoder configuration')

    # Checkpoints
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to trained SimpleMaskVqvae checkpoint (optional)')
    parser.add_argument('--image_encoder_checkpoint', type=str, default=None,
                        help='Path to image encoder checkpoint (e.g., ckpt/dino_v3_vits.safetensors)')

    # Data
    parser.add_argument('--dataset', type=str, default='coconut_hf',
                        choices=['hqseg44k', 'cocolvis', 'coconut_hf'],
                        help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Dataset path (auto-detected if None)')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to test')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],
                        help='Dataset split to use')

    # Test mode
    parser.add_argument('--test_mode', type=str, default='normal',
                        choices=['normal', 'bbox', 'random_rect', 'circles', 'coarse'],
                        help='Test mode: normal (use GT mask), bbox (use GT bbox), random_rect (use random rectangle), circles (use sampled circles from mask), coarse (use downsampled then upsampled mask)')
    parser.add_argument('--coarse_factor', type=int, default=16,
                        choices=[8, 16, 32],
                        help='Downsample factor for coarse mask test (8, 16, or 32)')

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

    # Build model using builder_map (consistent with training script)
    print(f"Building SimpleMaskVqvae model...")
    print(f"  Config: {args.config}")
    print(f"  Image encoder: {args.image_encoder_config}")
    if args.image_encoder_checkpoint:
        print(f"  Image encoder checkpoint: {args.image_encoder_checkpoint}")
    if args.checkpoint_path:
        print(f"  Model checkpoint: {args.checkpoint_path}")

    model = builder_map['simple_mask_vqvae'][args.config](
        simple_mask_vqvae_checkpoint_path=args.checkpoint_path,
        image_encoder_checkpoint=args.image_encoder_checkpoint,
        image_encoder_config_name=args.image_encoder_config,
        device=device,
    )
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Load dataset using builder_map (consistent with training script)
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
    print(f"Test mode: {args.test_mode}")

    # Test on samples
    num_samples = min(args.num_samples, len(dataset))
    print(f"\nTesting on {num_samples} samples...\n")

    results = []
    with torch.no_grad():
        for i in range(num_samples):
            # Get sample
            image, _, single_mask_normalized, single_mask = dataset[i]
            # single_mask_normalized is (1, H, W), single_mask is (1, H, W)
            _, h, w = single_mask_normalized.shape

            # Add batch dimension
            image_batch = image.unsqueeze(0).to(device)

            # Prepare input mask based on test mode
            if args.test_mode == 'normal':
                # Use original mask
                mask_input = single_mask_normalized
                gt_mask_for_iou = single_mask
            elif args.test_mode == 'bbox':
                # Use bounding box mask
                bbox_mask = create_bbox_mask(single_mask)  # single_mask is (1, H, W)
                mask_input = bbox_mask  # Already (1, H, W)
                gt_mask_for_iou = single_mask
                print(f"  Sample {i+1}: Using bbox mask")
            elif args.test_mode == 'random_rect':
                # Use random rectangle mask
                rect_mask = create_random_rect_mask(h, w)  # (1, H, W)
                mask_input = rect_mask  # Already (1, H, W)
                gt_mask_for_iou = single_mask  # Still use original for reference
                print(f"  Sample {i+1}: Using random rectangle mask")
            elif args.test_mode == 'circles':
                # Use circles mask sampled from mask (radius_ratio=0.05 for smaller circles)
                circles_mask, centers = create_circles_mask(single_mask, num_points=2, radius_ratio=0.05)
                mask_input = circles_mask  # Already (1, H, W)
                gt_mask_for_iou = single_mask
                print(f"  Sample {i+1}: Using circles mask with {len(centers)} centers: {centers}")
            elif args.test_mode == 'coarse':
                # Use coarse mask (downsample then upsample)
                coarse_mask = create_coarse_mask(single_mask, downsample_factor=args.coarse_factor)
                mask_input = coarse_mask  # Already (1, H, W)
                gt_mask_for_iou = single_mask
                print(f"  Sample {i+1}: Using coarse mask with factor 1/{args.coarse_factor}")
            else:
                raise ValueError(f"Unknown test_mode: {args.test_mode}")

            mask_normalized_batch = mask_input.unsqueeze(0).to(device)  # (1, 1, H, W)

            # Forward pass
            rec_mask_logits, vq_loss = model(mask_normalized_batch, image_batch)

            # Calculate IoU (meaningful for normal, bbox, circles, and coarse modes)
            rec_mask_binary = (rec_mask_logits > 0).float()
            if args.test_mode != 'random_rect':
                iou = calc_iou(rec_mask_binary, gt_mask_for_iou.unsqueeze(0).to(device))
                iou_value = iou.item()
            else:
                iou_value = 0.0  # No GT for random rect

            print(f"Sample {i+1}/{num_samples}:")
            print(f"  VQ Loss: {vq_loss.item():.4f}")
            if args.test_mode != 'random_rect':
                print(f"  IoU: {iou_value:.4f}")

            # Handle different output shapes: (B, H, W) or (B, 1, H, W)
            if rec_mask_logits.dim() == 3:
                pred_logits_vis = rec_mask_logits[0]  # (B, H, W) -> (H, W)
                pred_mask_vis = rec_mask_binary[0]    # (B, H, W) -> (H, W)
            else:
                pred_logits_vis = rec_mask_logits[0, 0]  # (B, 1, H, W) -> (H, W)
                pred_mask_vis = rec_mask_binary[0, 0]    # (B, 1, H, W) -> (H, W)

            # Visualize based on test mode
            if args.test_mode == 'normal':
                fig = visualize_results(
                    image=image,
                    gt_mask=single_mask[0],  # (1, H, W) -> (H, W)
                    pred_logits=pred_logits_vis,
                    pred_mask=pred_mask_vis,
                    iou=iou_value,
                    save_path=output_dir / f'sample_{i+1}_result.png'
                )
                plt.close(fig)
            elif args.test_mode == 'random_rect':
                # For random_rect mode (no GT comparison)
                input_mask_vis = mask_normalized_batch[0, 0] if mask_normalized_batch.dim() == 4 else mask_normalized_batch[0]
                fig = visualize_bbox_test(
                    image=image,
                    input_mask=input_mask_vis,
                    pred_logits=pred_logits_vis,
                    pred_mask=pred_mask_vis,
                    save_path=output_dir / f'sample_{i+1}_{args.test_mode}.png',
                    title='SimpleMaskVqvae Random Rectangle Input Test',
                )
            else:
                # For bbox, circles, and coarse modes (with GT comparison)
                input_mask_vis = mask_normalized_batch[0, 0] if mask_normalized_batch.dim() == 4 else mask_normalized_batch[0]

                # Set title based on test mode
                mode_titles = {
                    'bbox': 'SimpleMaskVqvae BBox Input Test',
                    'circles': 'SimpleMaskVqvae Circles Input Test (Sampled from Mask)',
                    'coarse': f'SimpleMaskVqvae Coarse Mask Test (1/{args.coarse_factor})',
                }
                title = mode_titles.get(args.test_mode, 'SimpleMaskVqvae Alternative Mask Test')

                fig = visualize_with_gt(
                    image=image,
                    gt_mask=single_mask[0],
                    input_mask=input_mask_vis,
                    pred_logits=pred_logits_vis,
                    pred_mask=pred_mask_vis,
                    iou=iou_value,
                    save_path=output_dir / f'sample_{i+1}_{args.test_mode}.png',
                    title=title,
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
    if args.test_mode != 'random_rect':
        print(f"Average IoU: {avg_iou:.4f}")
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
