"""
Simple test script for MaskVQVAE implementation.
"""

import torch

from maskvar.models.mask_vqvae import MaskVQVAE


def test_mask_vqvae():
    """Test MaskVQVAE basic functionality."""
    print("Testing MaskVQVAE...")

    # Create model
    model = MaskVQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=128,
        v_patch_nums=(1, 2, 4, 8, 16),
        img_feat_dim=256,
        transformer_dim=256,
        transformer_depth=2,
        transformer_num_heads=8,
        fusion_type="sum",
        use_sam_mask_decoder=True,
        test_mode=False,  # Enable gradients for testing
    )

    model.eval()

    # Test input dimensions
    batch_size = 2
    H, W = 256, 256  # Original image size
    H_feat, W_feat = H // 16, W // 16  # SAM image encoder output size (16x downsampled)

    # Create dummy inputs
    mask = torch.randn(batch_size, 1, H, W)  # Input mask
    image_features = torch.randn(batch_size, 256, H_feat, W_feat)  # SAM image features

    print(f"Input mask shape: {mask.shape}")
    print(f"Image features shape: {image_features.shape}")

    # Test forward pass with image features
    with torch.no_grad():
        try:
            rec_mask, vq_loss = model(mask, image_features, use_image_features=True)
            print(f"✓ Forward pass with image features successful")
            print(f"  Reconstructed mask shape: {rec_mask.shape}")
            print(f"  VQ loss: {vq_loss.item():.4f}")
        except Exception as e:
            print(f"✗ Forward pass with image features failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Test forward pass without image features (baseline)
    with torch.no_grad():
        try:
            rec_mask_baseline = model(mask, use_image_features=False)
            print(f"✓ Forward pass without image features successful")
            print(f"  Baseline reconstructed mask shape: {rec_mask_baseline.shape}")
        except Exception as e:
            print(f"✗ Forward pass without image features failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Test encoding
    with torch.no_grad():
        try:
            ms_idx_Bl = model.encode_to_indices(mask)
            print(f"✓ Encoding successful")
            print(f"  Number of scales: {len(ms_idx_Bl)}")
            for i, idx in enumerate(ms_idx_Bl):
                print(f"    Scale {i}: {idx.shape}")
        except Exception as e:
            print(f"✗ Encoding failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Test decoding from indices
    with torch.no_grad():
        try:
            rec_mask_from_idx = model.decode_from_indices(
                ms_idx_Bl, image_features, original_size=(H, W)
            )
            print(f"✓ Decoding from indices successful")
            print(f"  Reconstructed mask shape: {rec_mask_from_idx.shape}")
        except Exception as e:
            print(f"✗ Decoding from indices failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✓ All tests passed!")
    return True


def test_mask_decoder_module():
    """Test MaskDecoderModule in isolation."""
    print("\nTesting MaskDecoderModule...")

    from maskvar.models.mask_vqvae.mask_decoder import MaskDecoderModule

    decoder = MaskDecoderModule(
        cvae_dim=32,
        img_feat_dim=256,
        transformer_dim=256,
        v_patch_nums=(1, 2, 4, 8, 16),
        fusion_type="sum",
    )
    decoder.eval()

    batch_size = 2
    H, W = 256, 256
    H_feat, W_feat = H // 16, W // 16

    # Create dummy inputs
    image_features = torch.randn(batch_size, 256, H_feat, W_feat)
    ms_tokens = [
        torch.randn(batch_size, 32, pn, pn) for pn in (1, 2, 4, 8, 16)
    ]

    with torch.no_grad():
        try:
            fused_mask, all_masks = decoder(
                image_features, ms_tokens, original_size=(H, W)
            )
            print(f"✓ MaskDecoderModule forward successful")
            print(f"  Fused mask shape: {fused_mask.shape}")
            print(f"  Number of scales: {len(all_masks)}")
            for i, m in enumerate(all_masks):
                print(f"    Scale {i}: {m.shape}")
        except Exception as e:
            print(f"✗ MaskDecoderModule forward failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("✓ MaskDecoderModule tests passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("MaskVQVAE Implementation Tests")
    print("=" * 60)

    success = True
    success = test_mask_decoder_module() and success
    success = test_mask_vqvae() and success

    print("\n" + "=" * 60)
    if success:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)
