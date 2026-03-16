import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import random
import numpy as np

torch.set_float32_matmul_precision('high')

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Import SAM transformer components
from maskvar.models.sam.transformer import TwoWayTransformer, Attention
from maskvar.models.sam.mask_decoder import MaskDecoder
from maskvar.models.sam.common import MLPBlock

def create_dummy_data(batch_size=2, embedding_dim=256, image_size=16, num_points=5):
    """
    Create dummy data for TwoWayTransformer testing.

    Returns:
        image_embedding: (B, C, H, W)
        image_pe: (B, C, H, W)
        point_embedding: (B, N_points, C)
        target_queries: (B, N_points, C) - random target for queries
        target_keys: (B, H*W, C) - random target for keys
    """
    # Image embeddings (from SAM image encoder)
    H = W = image_size
    image_embedding = torch.randn(batch_size, embedding_dim, H, W, device=device)

    # Positional encoding for image (same shape as image_embedding)
    image_pe = torch.randn(batch_size, embedding_dim, H, W, device=device)

    # Point embeddings (prompt tokens: iou_token + mask_tokens + sparse prompts)
    point_embedding = torch.randn(batch_size, num_points, embedding_dim, device=device)

    # Create random targets for overfitting
    target_queries = torch.randn(batch_size, num_points, embedding_dim, device=device)
    target_keys = torch.randn(batch_size, H * W, embedding_dim, device=device)

    return image_embedding, image_pe, point_embedding, target_queries, target_keys

def test_two_way_transformer():
    """Test if TwoWayTransformer can overfit to random data."""
    print("Testing TwoWayTransformer overfitting...")

    # Create a small transformer
    depth = 2
    embedding_dim = 128
    num_heads = 4
    mlp_dim = 512

    transformer = TwoWayTransformer(
        depth=depth,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        activation=nn.ReLU,
        attention_downsample_rate=2
    ).to(device)

    transformer = torch.compile(transformer)

    # Create dummy data
    batch_size = 2
    image_size = 8  # Smaller for faster testing
    num_points = 3  # iou_token + 2 mask tokens

    image_embedding, image_pe, point_embedding, target_queries, target_keys = create_dummy_data(
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        image_size=image_size,
        num_points=num_points
    )

    # Create optimizer
    optimizer = AdamW(transformer.parameters(), lr=1e-3)

    # Training loop
    n_epochs = 1000
    losses = []

    transformer.train()

    with tqdm(range(n_epochs), desc="Training") as pbar:
        for epoch in pbar:
            optimizer.zero_grad()

            # Forward pass
            queries, keys = transformer(image_embedding, image_pe, point_embedding)

            # Compute loss - MSE between predictions and random targets
            loss_queries = F.mse_loss(queries, target_queries)
            loss_keys = F.mse_loss(keys, target_keys)
            loss = loss_queries + loss_keys

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'q_loss': f'{loss_queries.item():.6f}',
                'k_loss': f'{loss_keys.item():.6f}'
            })

    # Check if loss decreased significantly
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss

    print(f"\nResults:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Loss reduction: {loss_reduction*100:.2f}%")

    if loss_reduction > 0.9:  # 90% reduction
        print("✅ TwoWayTransformer can overfit random data!")
        return True
    else:
        print("❌ TwoWayTransformer failed to overfit random data")
        return False

def test_mask_decoder():
    """Test if MaskDecoder can overfit to random data."""
    print("\nTesting MaskDecoder overfitting...")

    # Create transformer for mask decoder
    depth = 2
    embedding_dim = 128
    num_heads = 4
    mlp_dim = 512

    transformer = TwoWayTransformer(
        depth=depth,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        activation=nn.ReLU,
        attention_downsample_rate=2
    )

    # Create mask decoder
    mask_decoder = MaskDecoder(
        transformer_dim=embedding_dim,
        transformer=transformer,
        num_multimask_outputs=2,
        activation=nn.GELU,
        iou_head_depth=2,
        iou_head_hidden_dim=128
    ).to(device)

    # Create dummy data for mask decoder
    batch_size = 2
    image_size = 8
    H = W = image_size

    # Image embeddings
    image_embeddings = torch.randn(batch_size, embedding_dim, H, W, device=device)
    image_pe = torch.randn(batch_size, embedding_dim, H, W, device=device)

    # Sparse prompt embeddings (points/boxes)
    sparse_prompt_embeddings = torch.randn(batch_size, 3, embedding_dim, device=device)

    # Dense prompt embeddings (mask inputs)
    dense_prompt_embeddings = torch.randn(batch_size, embedding_dim, H, W, device=device)

    # Create random targets for masks and IoU predictions
    # MaskDecoder outputs masks of shape (B, num_mask_tokens, H*4, W*4) after upscaling
    # and iou_pred of shape (B, num_mask_tokens)
    upscaled_H, upscaled_W = H * 4, W * 4  # Output upscaling factor is 4
    num_mask_tokens = 3  # 1 main mask + 2 multimask outputs

    target_masks = torch.randn(batch_size, num_mask_tokens, upscaled_H, upscaled_W, device=device)
    target_iou = torch.randn(batch_size, num_mask_tokens, device=device)

    # Training setup
    optimizer = AdamW(mask_decoder.parameters(), lr=1e-3)
    n_epochs = 1000
    losses = []

    mask_decoder.train()

    with tqdm(range(n_epochs), desc="Training MaskDecoder") as pbar:
        for epoch in pbar:
            optimizer.zero_grad()

            # Forward pass - get all masks (multimask_output=False to get all 3 masks)
            masks, iou_pred = mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=False  # Get all masks for training
            )

            # Compute losses
            mask_loss = F.mse_loss(masks, target_masks)
            iou_loss = F.mse_loss(iou_pred, target_iou)
            loss = mask_loss + iou_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'mask_loss': f'{mask_loss.item():.6f}',
                'iou_loss': f'{iou_loss.item():.6f}'
            })

    # Check results
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss

    print(f"\nMaskDecoder Results:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Loss reduction: {loss_reduction*100:.2f}%")

    if loss_reduction > 0.9:
        print("✅ MaskDecoder can overfit random data!")
        return True
    else:
        print("❌ MaskDecoder failed to overfit random data")
        return False

def test_adapted_two_way_transformer():
    """Test if AdaptedTwoWayTransformer can overfit to random data."""
    print("\nTesting AdaptedTwoWayTransformer overfitting...")

    try:
        from maskvar.models.simple_ar.adapted_twt import AdaptedTwoWayTransformer, AdaptedAttention
    except ImportError as e:
        print(f"❌ Cannot import AdaptedTwoWayTransformer: {e}")
        return False

    # Create a small adapted transformer
    depth = 2
    embedding_dim = 128
    num_heads = 4
    mlp_dim = 512

    transformer = AdaptedTwoWayTransformer(
        depth=depth,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        activation=nn.ReLU,
        attention_downsample_rate=2
    ).to(device)

    transformer = torch.compile(transformer)

    # Create dummy data for adapted transformer
    batch_size = 2
    image_size = 8
    H = W = image_size
    num_query_points = 3  # iou_token + mask tokens
    num_mask_tokens = 4   # autoregressive mask tokens

    # Image embeddings
    image_embedding = torch.randn(batch_size, embedding_dim, H, W, device=device)
    image_pe = torch.randn(batch_size, embedding_dim, H, W, device=device)

    # Query tokens (iou + mask tokens + prompts)
    point_embedding = torch.randn(batch_size, num_query_points, embedding_dim, device=device)

    # Autoregressive mask tokens
    mask_tokens = torch.randn(batch_size, num_mask_tokens, embedding_dim, device=device)
    mask_tokens_pe = torch.randn(1, num_mask_tokens, embedding_dim, device=device)  # (1, L, C) pattern
    # Expand to batch dimension as done in adapted_mask_decoder.py
    mask_tokens_pe = mask_tokens_pe.expand(batch_size, -1, -1)

    # Create random targets
    target_qs = torch.randn(batch_size, num_query_points, embedding_dim, device=device)
    target_qm = torch.randn(batch_size, num_mask_tokens, embedding_dim, device=device)
    target_keys = torch.randn(batch_size, H * W, embedding_dim, device=device)

    # Training setup
    optimizer = AdamW(transformer.parameters(), lr=1e-3)
    n_epochs = 1000
    losses = []

    transformer.train()

    with tqdm(range(n_epochs), desc="Training AdaptedTwoWayTransformer") as pbar:
        for epoch in pbar:
            optimizer.zero_grad()

            # Forward pass
            qs, keys, qm = transformer(
                image_embedding=image_embedding,
                image_pe=image_pe,
                point_embedding=point_embedding,
                mask_tokens=mask_tokens,
                mask_tokens_pe=mask_tokens_pe,
                self_attn_mask=None  # No mask for overfitting test
            )

            # Compute losses
            loss_qs = F.mse_loss(qs, target_qs)
            loss_qm = F.mse_loss(qm, target_qm)
            loss_keys = F.mse_loss(keys, target_keys)
            loss = loss_qs + loss_qm + loss_keys

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'qs_loss': f'{loss_qs.item():.6f}',
                'qm_loss': f'{loss_qm.item():.6f}',
                'keys_loss': f'{loss_keys.item():.6f}'
            })

    # Check results
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss

    print(f"\nAdaptedTwoWayTransformer Results:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Loss reduction: {loss_reduction*100:.2f}%")

    if loss_reduction > 0.9:
        print("✅ AdaptedTwoWayTransformer can overfit random data!")
        return True
    else:
        print("❌ AdaptedTwoWayTransformer failed to overfit random data")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SAM TwoWayTransformer Overfitting Test")
    print("=" * 60)

    # Run tests
    results = []

    # Test 1: Original TwoWayTransformer
    results.append(("TwoWayTransformer", test_two_way_transformer()))

    # Test 2: Original MaskDecoder (optional)
    # Uncomment if you want to test MaskDecoder as well
    # results.append(("MaskDecoder", test_mask_decoder()))

    # Test 3: AdaptedTwoWayTransformer
    results.append(("AdaptedTwoWayTransformer", test_adapted_two_way_transformer()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:30} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! The transformers can overfit random data.")
    else:
        print("\n⚠️  Some tests failed. Check the implementation.")

    return all_passed

if __name__ == "__main__":
    main()