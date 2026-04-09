#!/usr/bin/env python3
"""
Initialize MaskVQVAE checkpoint from SAM and VQVAE pretrained weights.

This script:
1. Loads SAM mask decoder weights (including transformer and output_upscaling)
2. Optionally loads VQVAE encoder/quantizer weights
3. Freezes SAM-related parameters by default
4. Saves the initialized checkpoint

Usage:
    python misc_scripts/init_mask_vqvae_weights.py \
        --sam_checkpoint ckpt/sam_vit_b_01ec64.pth \
        --vqvae_checkpoint out/vqvae/vqvae_single_epoch_50.pth \
        --output ckpt/mask_vqvae_init.pth \
        --freeze_sam_parts
"""

import argparse
import torch
from pathlib import Path

from maskvar.maskseg_build_everything import build_mask_vqvae_v0


def init_mask_vqvae_from_sam(
    sam_checkpoint_path: str,
    vqvae_checkpoint_path: str = None,
    output_path: str = "ckpt/mask_vqvae_init.pth",
    freeze_sam_parts: bool = True,
    device: str = "cpu",
):
    """
    Initialize MaskVQVAE from SAM and VQVAE pretrained weights.

    Args:
        sam_checkpoint_path: Path to SAM checkpoint
        vqvae_checkpoint_path: Path to VQVAE checkpoint (optional)
        output_path: Path to save initialized checkpoint
        freeze_sam_parts: Whether to freeze SAM-related parameters
        device: Device to load model on
    """
    print("=" * 60)
    print("Initializing MaskVQVAE")
    print("=" * 60)
    print(f"SAM checkpoint: {sam_checkpoint_path}")
    print(f"VQVAE checkpoint: {vqvae_checkpoint_path}")
    print(f"Freeze SAM parts: {freeze_sam_parts}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Build MaskVQVAE model (uninitialized)
    print("\nCreating MaskVQVAE model...")
    mask_vqvae = build_mask_vqvae_v0(
        checkpoint_path=None,  # Don't load any checkpoint yet
        vqvae_init_checkpoint=vqvae_checkpoint_path,  # Load VQVAE weights if provided
        require_grad=not freeze_sam_parts,
        device=device,
    )

    total_params = sum(p.numel() for p in mask_vqvae.parameters())
    print(f"✓ Model created with {total_params:,} parameters")

    # Load SAM checkpoint and extract mask decoder weights
    print(f"\nLoading SAM checkpoint from: {sam_checkpoint_path}")
    sam_state_dict = torch.load(sam_checkpoint_path, map_location=device, weights_only=True)

    # Extract SAM mask decoder weights
    sam_mask_decoder_state = {}
    prefix = "mask_decoder."
    for key, value in sam_state_dict.items():
        if key.startswith(prefix):
            new_key = "sam_mask_decoder." + key[len(prefix):]
            sam_mask_decoder_state[new_key] = value

    print(f"✓ Extracted {len(sam_mask_decoder_state)} parameters from SAM mask decoder")

    # Load SAM weights into MaskVQVAE
    print("\nLoading SAM weights into MaskVQVAE...")
    model_state = mask_vqvae.state_dict()

    loaded = 0
    mismatched = 0
    for key, value in sam_mask_decoder_state.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                model_state[key] = value
                loaded += 1
            else:
                print(f"  ⚠ Shape mismatch: {key} - model={model_state[key].shape}, sam={value.shape}")
                mismatched += 1

    mask_vqvae.load_state_dict(model_state, strict=False)
    print(f"✓ Loaded {loaded} parameters from SAM ({mismatched} mismatched)")

    # Freeze SAM-related parameters if requested
    if freeze_sam_parts:
        print("\nFreezing SAM-related parameters...")
        frozen = 0
        for name, param in mask_vqvae.named_parameters():
            if "sam_mask_decoder" in name:
                param.requires_grad = False
                frozen += param.numel()

        print(f"✓ Frozen {frozen:,} parameters (sam_mask_decoder)")

    # Count trainable parameters
    trainable = sum(p.numel() for p in mask_vqvae.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in mask_vqvae.parameters() if not p.requires_grad)

    print("\n" + "=" * 60)
    print("Parameter Summary:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen:    {frozen:,}")
    print("=" * 60)

    # Save checkpoint
    print(f"\nSaving checkpoint to: {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': mask_vqvae.state_dict(),
        'config': {
            'vocab_size': mask_vqvae.vocab_size,
            'z_channels': mask_vqvae.Cvae,
            'v_patch_nums': mask_vqvae.v_patch_nums,
            'img_feat_dim': mask_vqvae.img_feat_dim,
        },
        'initialized_from': {
            'sam': sam_checkpoint_path,
            'vqvae': vqvae_checkpoint_path,
        },
        'frozen_sam_parts': freeze_sam_parts,
    }

    torch.save(checkpoint, output_path)
    print("✓ Checkpoint saved successfully!")

    return mask_vqvae


def main():
    parser = argparse.ArgumentParser(
        description="Initialize MaskVQVAE checkpoint from SAM and VQVAE weights"
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        required=True,
        help="Path to SAM checkpoint (e.g., ckpt/sam_vit_b_01ec64.pth)"
    )
    parser.add_argument(
        "--vqvae_checkpoint",
        type=str,
        default=None,
        help="Path to VQVAE checkpoint for encoder/quantizer initialization"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ckpt/mask_vqvae_init.pth",
        help="Output path for initialized checkpoint (default: ckpt/mask_vqvae_init.pth)"
    )
    parser.add_argument(
        "--freeze_sam_parts",
        action="store_true",
        default=True,
        help="Freeze SAM-related parameters (default: True)"
    )
    parser.add_argument(
        "--no_freeze_sam_parts",
        action="store_true",
        help="Do not freeze SAM-related parameters"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (default: cpu)"
    )

    args = parser.parse_args()

    freeze_sam = args.freeze_sam_parts and not args.no_freeze_sam_parts

    init_mask_vqvae_from_sam(
        sam_checkpoint_path=args.sam_checkpoint,
        vqvae_checkpoint_path=args.vqvae_checkpoint,
        output_path=args.output,
        freeze_sam_parts=freeze_sam,
        device=args.device,
    )


if __name__ == "__main__":
    main()
