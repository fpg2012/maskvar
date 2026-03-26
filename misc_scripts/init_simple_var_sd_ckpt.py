#!/usr/bin/env python3
"""
Initialize simple_var_sd checkpoint from SAM pretrained weights.

This script loads SAM's mask decoder transformer weights and initializes
SimpleVARSamDecoder. The key is to load SAM's TwoWayTransformer weights
into AdaptedTwoWayTransformer.

Usage:
    python misc_scripts/init_simple_var_sd_ckpt.py \
        --sam_checkpoint ckpt/sam_vit_b_01ec64.pth \
        --output ckpt/simple_var_sd_init.pth
"""

import argparse
import torch
from pathlib import Path

from maskvar.maskseg_build_everything import (
    build_prompt_encoder,
    build_simple_var_sam_decoder,
)


def init_simple_var_sd_from_sam(
    sam_checkpoint_path: str,
    output_path: str,
    device: str = 'cpu',
):
    """
    Initialize SimpleVARSamDecoder from SAM pretrained weights.

    The key is to load SAM's mask decoder transformer weights into
    AdaptedTwoWayTransformer, which is compatible because AdaptedTwoWayTransformer
    is adapted from SAM's TwoWayTransformer.

    Args:
        sam_checkpoint_path: Path to SAM checkpoint (e.g., ckpt/sam_vit_b_01ec64.pth)
        output_path: Path to save the initialized checkpoint
        device: Device to load model on

    Returns:
        SimpleVARSamDecoder: Initialized model
    """
    print(f"Loading SAM checkpoint from: {sam_checkpoint_path}")

    # Load prompt encoder to get SAM PE
    print("Loading prompt encoder for SAM positional encoding...")
    prompt_encoder = build_prompt_encoder(sam_checkpoint_path)
    sam_pe = prompt_encoder.get_dense_pe()  # (1, 256, H, W)
    del prompt_encoder

    # Create SimpleVARSamDecoder using builder
    # Note: builder will call init_block_mask() automatically
    print("Creating SimpleVARSamDecoder using builder...")
    simple_var_sd = build_simple_var_sam_decoder(
        simple_var_checkpoint_path=None,  # No checkpoint, create new model
        sam_pe=sam_pe,
        device=device,
    )

    # Load SAM model to extract mask decoder transformer weights
    print("Loading SAM mask decoder transformer weights...")
    try:
        # Try loading with segment anything library
        from segment_anything import sam_model_registry
        sam = sam_model_registry["vit_b"](sam_checkpoint_path)
        sam_transformer_state = sam.mask_decoder.transformer.state_dict()
        del sam
    except ImportError:
        # Fallback: load directly from checkpoint
        print("segment_anything not found, loading from checkpoint directly...")
        sam_state_dict = torch.load(sam_checkpoint_path, map_location='cpu', weights_only=True)

        # Extract transformer weights
        sam_transformer_state = {}
        prefix = "mask_decoder.transformer."
        for key, value in sam_state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                sam_transformer_state[new_key] = value

    # Load SAM transformer weights into AdaptedTwoWayTransformer
    print("Loading SAM transformer weights into AdaptedTwoWayTransformer...")
    missing_keys, unexpected_keys = simple_var_sd.adapted_mask_decoder.transformer.load_state_dict(
        sam_transformer_state, strict=False
    )

    if missing_keys:
        print(f"Missing keys (will be randomly initialized): {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys (ignored): {unexpected_keys}")

    # Save checkpoint
    print(f"\nSaving initialized checkpoint to: {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(simple_var_sd.state_dict(), output_path)

    # Print model info
    total_params = sum(p.numel() for p in simple_var_sd.parameters())
    trainable_params = sum(p.numel() for p in simple_var_sd.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("Initialization complete!")
    print("=" * 60)
    print(f"Checkpoint saved to: {output_path}")
    print(f"\nModel info:")
    print(f"  - Device: {device}")
    print(f"  - Use SAM PE: True")
    print(f"  - Patch nums: {simple_var_sd.patch_num}")
    print(f"  - Vocab size: {simple_var_sd.vocab_size}")
    print(f"  - Dim: {simple_var_sd.dim}")
    print(f"\nParameter count:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    print("=" * 60)

    return simple_var_sd


def main():
    parser = argparse.ArgumentParser(
        description="Initialize simple_var_sd checkpoint from SAM pretrained weights"
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        required=True,
        help="Path to SAM checkpoint (e.g., ckpt/sam_vit_b_01ec64.pth)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ckpt/simple_var_sd_init.pth",
        help="Output path for initialized checkpoint (default: ckpt/simple_var_sd_init.pth)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (default: cpu)"
    )

    args = parser.parse_args()

    init_simple_var_sd_from_sam(
        sam_checkpoint_path=args.sam_checkpoint,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
