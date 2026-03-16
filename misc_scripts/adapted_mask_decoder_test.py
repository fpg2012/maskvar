import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import random
import numpy as np
import json
import argparse

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import adapted mask decoder components
from maskvar.models.simple_ar.adapted_mask_decoder import AdaptedMaskDecoder
from maskvar.models.simple_ar.adapted_twt import AdaptedTwoWayTransformer

def create_dummy_data(batch_size=2, embedding_dim=256, image_size=16,
                      num_sparse_prompts=3, num_ar_mask_tokens=4):
    """
    Create dummy data for AdaptedMaskDecoder testing.

    Returns:
        image_embeddings: (B, HW, C)
        image_pe: (B, HW, C)
        sparse_prompt_embeddings: (B, num_sparse_prompts, C)
        dense_prompt_embeddings: (B, C, H, W)
        mask_tokens: (B, num_ar_mask_tokens, C)
        mask_tokens_pe: (1, num_ar_mask_tokens, C)
        target_qs: (B, Lqs, C) - random target for qs
        target_qm: (B, num_ar_mask_tokens, C) - random target for qm
    """
    H = W = image_size
    HW = H * W

    # Image embeddings (from SAM image encoder) - already flattened to (B, HW, C)
    image_embeddings = torch.randn(batch_size, HW, embedding_dim, device=device)

    # Positional encoding for image - create as (B, C, H, W) to match transformer expectations
    # Note: AdaptedMaskDecoder expects (B, HW, C) but transformer inside expects (B, C, H, W)
    # We'll create (B, C, H, W) and see if it works
    image_pe = torch.randn(batch_size, embedding_dim, H, W, device=device)

    # Sparse prompt embeddings (points, boxes)
    sparse_prompt_embeddings = torch.randn(batch_size, num_sparse_prompts, embedding_dim, device=device)

    # Dense prompt embeddings (mask inputs) - (B, C, H, W)
    dense_prompt_embeddings = torch.randn(batch_size, embedding_dim, H, W, device=device)

    # Autoregressive mask tokens
    mask_tokens = torch.randn(batch_size, num_ar_mask_tokens, embedding_dim, device=device)

    # Mask tokens positional encoding (1, L, C) pattern (will be expanded in decoder)
    mask_tokens_pe = torch.randn(1, num_ar_mask_tokens, embedding_dim, device=device)

    # Create random targets for overfitting
    # qs includes: iou_token + mask_tokens (num_multimask_outputs+1) + sparse_prompt_embeddings
    num_mask_tokens_in_qs = 4  # iou_token + 3 mask_tokens (default num_multimask_outputs=3)
    Lqs = 1 + num_mask_tokens_in_qs + num_sparse_prompts
    target_qs = torch.randn(batch_size, Lqs, embedding_dim, device=device)
    target_qm = torch.randn(batch_size, num_ar_mask_tokens, embedding_dim, device=device)

    return (image_embeddings, image_pe, sparse_prompt_embeddings,
            dense_prompt_embeddings, mask_tokens, mask_tokens_pe,
            target_qs, target_qm)

def overfit(model, data, targets, n_epoch=500, lr=1e-3):
    """
    Overfit the model to random data.

    Args:
        model: AdaptedMaskDecoder instance
        data: tuple of input tensors
        targets: tuple of target tensors (target_qs, target_qm)
        n_epoch: number of training epochs
        lr: learning rate

    Returns:
        losses: list of loss values
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    (image_embeddings, image_pe, sparse_prompt_embeddings,
     dense_prompt_embeddings, mask_tokens, mask_tokens_pe) = data
    target_qs, target_qm = targets

    losses = []

    with tqdm(range(n_epoch), desc="Training AdaptedMaskDecoder") as pbar:
        for epoch in pbar:
            optimizer.zero_grad()

            # Forward pass
            qs, qm = model(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                mask_tokens=mask_tokens,
                mask_tokens_pe=mask_tokens_pe,
                self_attn_mask=None  # No mask for overfitting test
            )

            # Compute losses
            loss_qs = F.mse_loss(qs, target_qs)
            loss_qm = F.mse_loss(qm, target_qm)
            loss = loss_qs + loss_qm

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'qs_loss': f'{loss_qs.item():.6f}',
                'qm_loss': f'{loss_qm.item():.6f}'
            })

    return losses

def test_adapted_mask_decoder(args):
    """Test if AdaptedMaskDecoder can overfit to random data."""
    print("Testing AdaptedMaskDecoder overfitting...")
    print(f"Device: {args.device}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")

    # Create transformer for mask decoder
    depth = 2
    embedding_dim = args.embedding_dim
    num_heads = 4
    mlp_dim = 512

    transformer = AdaptedTwoWayTransformer(
        depth=depth,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        activation=nn.ReLU,
        attention_downsample_rate=2
    )

    # Create adapted mask decoder
    num_multimask_outputs = 3  # default
    mask_decoder = AdaptedMaskDecoder(
        transformer_dim=embedding_dim,
        transformer=transformer,
        num_multimask_outputs=num_multimask_outputs,
        activation=nn.GELU,
        iou_head_depth=2,
        iou_head_hidden_dim=128
    ).to(args.device)

    # mask_decoder = torch.compile(mask_decoder)  # Disabled for debugging

    # Create dummy data
    (image_embeddings, image_pe, sparse_prompt_embeddings,
     dense_prompt_embeddings, mask_tokens, mask_tokens_pe,
     target_qs, target_qm) = create_dummy_data(
        batch_size=args.batch_size,
        embedding_dim=embedding_dim,
        image_size=args.image_size,
        num_sparse_prompts=args.num_sparse_prompts,
        num_ar_mask_tokens=args.num_ar_mask_tokens
    )

    # Run overfitting test
    losses = overfit(
        model=mask_decoder,
        data=(image_embeddings, image_pe, sparse_prompt_embeddings,
              dense_prompt_embeddings, mask_tokens, mask_tokens_pe),
        targets=(target_qs, target_qm),
        n_epoch=args.n_epoch,
        lr=args.lr
    )

    # Check results
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss

    print(f"\nAdaptedMaskDecoder Results:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Loss reduction: {loss_reduction*100:.2f}%")
    print(f"  Number of parameters: {sum(p.numel() for p in mask_decoder.parameters()):,}")

    # Save loss curve if requested
    if args.save_loss:
        with open(f'loss_{args.exp}.json', 'w') as f:
            json.dump(losses, f)
        print(f"  Loss curve saved to loss_{args.exp}.json")

    # Determine if test passed
    if loss_reduction > 0.9:  # 90% reduction
        print("✅ AdaptedMaskDecoder can overfit random data!")
        return True, losses
    else:
        print("❌ AdaptedMaskDecoder failed to overfit random data")
        return False, losses

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Test AdaptedMaskDecoder overfitting")

    # Training parameters
    parser.add_argument('--n_epoch', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')

    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--image_size', type=int, default=16,
                       help='Image size (will be flattened to HW)')
    parser.add_argument('--num_sparse_prompts', type=int, default=3,
                       help='Number of sparse prompt embeddings')
    parser.add_argument('--num_ar_mask_tokens', type=int, default=4,
                       help='Number of autoregressive mask tokens')

    # Experiment parameters
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--exp', type=str, required=True,
                       help='Experiment name for saving results')
    parser.add_argument('--save_loss', action='store_true',
                       help='Save loss curve to JSON file')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("AdaptedMaskDecoder Overfitting Test")
    print("=" * 60)

    # Run test
    passed, losses = test_adapted_mask_decoder(args)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if passed:
        print("🎉 AdaptedMaskDecoder passed overfitting test!")
        return 0
    else:
        print("⚠️  AdaptedMaskDecoder failed overfitting test.")
        print("   Check implementation for potential issues.")
        return 1

if __name__ == "__main__":
    exit(main())