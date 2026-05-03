from functools import partial
from typing import Optional, Tuple

import torch
import timm
from safetensors.torch import load_file

from .models.vqvae_single import VQVAE_Single
from .models.mask_vqvae import MaskVQVAE
# from .models.maskgit import MaskGIT
# from .models.maskseg import MaskSeg
from .models.flex_maskvar import FlexMaskVAR
from .models.flex_maskvar_simple import FlexMaskVARSimple
from .models.sam import ImageEncoderViT as SamImageEncoder
from .models.sam import PromptEncoder
from .models.image_encoder import ImageEncoder, VarImageEncoder, NeckFPN
from .models.maskvar import MaskVAR
from .models.tinyvit import TinyViT
from .models.simple_ar import SimpleAR, SimpleVAR, SimpleVARSamDecoder
from .models.simple_ar.adapted_mask_decoder import AdaptedMaskDecoder
from .models.simple_ar.adapted_twt import AdaptedTwoWayTransformer
from .models.simple_mask_vqvae import (
    SimpleMaskVqvae, MaskEncoderLite, MaskEncoderLite16x16,
    SimpleMaskVqvaeV2, SimpleMaskVqvaeV3, SimpleMaskVqvaeV4,
    SimpleMaskVqvaeShapeOnly,
    SimpleMaskVqvaeMultiScale,
    SimpleMaskVqvaeMultiScaleResidual,
)
from .models.simple_mask_ar import SimpleMaskAR, SimpleMaskVAR
from .models.simple_mask_ar import SimpleQueryTokenAR
from .models.simple_mask_vae import SimpleMaskVAEV2
from .models.simple_mask_diffusion import SimpleMaskLatentDiT
from .models.dino_wrapper import DinoV3Wrapper

from .datasets.mask_level_dataset import MaskLevelDataset
from .datasets.coco_lvis import LvisDataset
from .datasets.hqseg44k import HQSeg44KTestDataset, HQSeg44KTrainDataset
from .datasets.coconut_hf import CoconutHFDataset


# def build_maskseg(vqvae_checkpoint_path: Optional[str] = None, maskgit_checkpoint_path: Optional[str] = None, sam_checkpoint_path: Optional[str] = None) -> MaskSeg:
#     vqvae = build_vqvae_single(vqvae_checkpoint_path)

#     prompt_encoder = build_prompt_encoder(sam_checkpoint_path)
#     image_encoder = build_image_encoder(sam_checkpoint_path)
#     maskgit = build_maskgit(vqvae, maskgit_checkpoint_path)

#     maskseg = MaskSeg(maskgit=maskgit, 
#                       prompt_encoder=prompt_encoder, 
#                       image_encoder=image_encoder, 
#                       freeze_prompt_encoder=True)

#     return maskseg

def build_maskvar(vqvae_checkpoint_path: Optional[str] = None, sam_checkpoint_path: Optional[str] = None, flash_if_available: bool = False, device='cpu'):
    vqvae = build_vqvae_single(vqvae_checkpoint_path).to(device)
    sam_image_encoder = build_sam_image_encoder(sam_checkpoint_path).to(device)
    var_image_encoder = build_var_image_encoder().to(device)
    prompt_encoder = build_prompt_encoder(sam_checkpoint_path).to(device)
    maskvar = MaskVAR(
        vae_local=vqvae, 
        image_encoder=var_image_encoder, 
        prompt_encoder=prompt_encoder,
        num_classes=1,
        depth=4,
        embed_dim=256,
        num_heads=16,
        mlp_ratio=4.,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32),
        flash_if_available=flash_if_available,
        fused_if_available=True,
        image_encoder_requires_grad=True,
        prompt_encoder_requires_grad=False,
    ).to(device)
    return vqvae, maskvar, sam_image_encoder

def build_maskvar_flex(vqvae_checkpoint_path: Optional[str] = None, sam_checkpoint_path: Optional[str] = None, device='cpu'):
    patch_num = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32)
    vqvae = build_vqvae_single(vqvae_checkpoint_path).to(device)
    sam_image_encoder = build_sam_image_encoder(sam_checkpoint_path).to(device)
    var_image_encoder = build_var_image_encoder().to(device)
    prompt_encoder = build_prompt_encoder(sam_checkpoint_path).to(device)
    maskvar = FlexMaskVAR(
        vae_local=vqvae, 
        image_encoder=var_image_encoder, 
        prompt_encoder=prompt_encoder,
        num_classes=1,
        depth=2,
        embed_dim=256,
        num_heads=4,
        mlp_ratio=2,
        drop_rate=0.1,
        drop_path_rate=0.1,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=patch_num,
        # fused_if_available=True,
        image_encoder_requires_grad=True,
        prompt_encoder_requires_grad=False,
    ).to(device)
    return vqvae, maskvar, sam_image_encoder

def build_maskvar_flex_5_stages(vqvae_checkpoint_path: Optional[str] = None, sam_checkpoint_path: Optional[str] = None, device='cpu'):
    patch_nums = (1, 8, 16, 24, 32)
    vqvae = build_vqvae_single_5_stages_v1(vqvae_checkpoint_path).to(device)
    sam_image_encoder = build_sam_image_encoder(sam_checkpoint_path).to(device)
    var_image_encoder = build_var_image_encoder(patch_nums=patch_nums).to(device)
    prompt_encoder = build_prompt_encoder(sam_checkpoint_path).to(device)
    maskvar = FlexMaskVAR(
        vae_local=vqvae, 
        image_encoder=var_image_encoder, 
        prompt_encoder=prompt_encoder,
        num_classes=1,
        depth=4,
        embed_dim=256,
        num_heads=4,
        mlp_ratio=2,
        drop_rate=0.1,
        drop_path_rate=0.1,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=patch_nums,
        # fused_if_available=True,
        image_encoder_requires_grad=True,
        prompt_encoder_requires_grad=False,
    ).to(device)
    return vqvae, maskvar, sam_image_encoder

def build_maskvar_flex_mobile_5_stages(vqvae_checkpoint_path: Optional[str] = None, sam_checkpoint_path: Optional[str] = None, device='cpu'):
    patch_nums = (1, 8, 16, 24, 32)
    vqvae = build_vqvae_single_5_stages_v1(vqvae_checkpoint_path).to(device)
    sam_image_encoder = build_mobile_sam_image_encoder(sam_checkpoint_path).to(device)
    var_image_encoder = build_var_image_encoder(patch_nums=patch_nums).to(device)
    prompt_encoder = build_prompt_encoder(sam_checkpoint_path).to(device)
    maskvar = FlexMaskVAR(
        vae_local=vqvae, 
        image_encoder=var_image_encoder, 
        prompt_encoder=prompt_encoder,
        num_classes=1,
        depth=4,
        embed_dim=256,
        num_heads=4,
        mlp_ratio=4,
        drop_rate=0.1,
        drop_path_rate=0.1,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=patch_nums,
        # fused_if_available=True,
        image_encoder_requires_grad=True,
        prompt_encoder_requires_grad=False,
    ).to(device)
    return vqvae, maskvar, sam_image_encoder

def build_maskvar_flex_simple_mobile_5_stages(vqvae_checkpoint_path: Optional[str] = None, sam_checkpoint_path: Optional[str] = None, device='cpu'):
    patch_nums = (1, 8, 16, 24, 32)
    vqvae = build_vqvae_single_5_stages_v1(vqvae_checkpoint_path).to(device)
    sam_image_encoder = build_mobile_sam_image_encoder(sam_checkpoint_path).to(device)
    prompt_encoder = build_prompt_encoder(sam_checkpoint_path).to(device)
    maskvar = FlexMaskVARSimple(
        vae_local=vqvae, 
        prompt_encoder=prompt_encoder,
        num_classes=1,
        depth=4,
        embed_dim=256,
        num_heads=4,
        mlp_ratio=4,
        drop_rate=0.1,
        drop_path_rate=0.1,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=patch_nums,
        prompt_encoder_requires_grad=False,
    ).to(device)
    return vqvae, maskvar, sam_image_encoder

def build_maskvar_flex_v2(vqvae_checkpoint_path: Optional[str] = None, sam_checkpoint_path: Optional[str] = None, device='cpu'):
    patch_nums = (8, 16, 24, 32)
    
    vqvae = build_vqvae_single_4_stages_v2(vqvae_checkpoint_path).to(device)
    sam_image_encoder = build_sam_image_encoder(sam_checkpoint_path).to(device)
    var_image_encoder = build_var_image_encoder(patch_nums=patch_nums).to(device)
    prompt_encoder = build_prompt_encoder(sam_checkpoint_path).to(device)
    maskvar = FlexMaskVAR(
        vae_local=vqvae, 
        image_encoder=var_image_encoder, 
        prompt_encoder=prompt_encoder,
        num_classes=1,
        depth=4,
        embed_dim=256,
        num_heads=16,
        mlp_ratio=4.,
        drop_rate=0.1,
        drop_path_rate=0.1,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=patch_nums,
        image_encoder_requires_grad=True,
        prompt_encoder_requires_grad=False,
    ).to(device)
    return vqvae, maskvar, sam_image_encoder

def build_maskvar_v2(vqvae_checkpoint_path: Optional[str] = None, sam_checkpoint_path: Optional[str] = None, flash_if_available: bool = False, device='cpu'):
    patch_nums = (8, 16, 24, 32)
    
    vqvae = build_vqvae_single_4_stages_v2(vqvae_checkpoint_path).to(device)
    sam_image_encoder = build_sam_image_encoder(sam_checkpoint_path).to(device)
    var_image_encoder = build_var_image_encoder(patch_nums=patch_nums).to(device)
    prompt_encoder = build_prompt_encoder(sam_checkpoint_path).to(device)
    maskvar = MaskVAR(
        vae_local=vqvae, 
        image_encoder=var_image_encoder, 
        prompt_encoder=prompt_encoder,
        num_classes=1,
        depth=4,
        embed_dim=256,
        num_heads=16,
        mlp_ratio=4.,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available,
        fused_if_available=True,
        image_encoder_requires_grad=True,
        prompt_encoder_requires_grad=False,
    ).to(device)
    return vqvae, maskvar, sam_image_encoder

def build_maskvar_v3(vqvae_checkpoint_path: Optional[str] = None, sam_checkpoint_path: Optional[str] = None, flash_if_available: bool = False, device='cpu'):
    vqvae = build_vqvae_single_4_stages_4_slices_v2(vqvae_checkpoint_path).to(device)
    sam_image_encoder = build_sam_image_encoder(sam_checkpoint_path).to(device)
    var_image_encoder = build_var_image_encoder().to(device)
    prompt_encoder = build_prompt_encoder(sam_checkpoint_path).to(device)
    maskvar = MaskVAR(
        vae_local=vqvae, 
        image_encoder=var_image_encoder, 
        prompt_encoder=prompt_encoder,
        num_classes=1,
        depth=4,
        embed_dim=256,
        num_heads=16,
        mlp_ratio=4.,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(8, 16, 24, 32),
        flash_if_available=flash_if_available,
        fused_if_available=True,
        image_encoder_requires_grad=True,
        prompt_encoder_requires_grad=False,
    ).to(device)
    return vqvae, maskvar, sam_image_encoder

def build_vqvae_single(vqvae_checkpoint_path: Optional[str] = None, require_grad = False) -> VQVAE_Single:
    vocab_size = 4096  # 码本大小
    z_channels = 32   # 潜在空间通道数
    base_channels = 128  # 基础通道数
    beta = 0.25  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        v_patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32),
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path)
        vqvae.load_state_dict(vqvae_state_dict)

    return vqvae

def build_vqvae_single_fewer_stages(vqvae_checkpoint_path: Optional[str] = None, require_grad = False) -> VQVAE_Single:
    vocab_size = 4096  # 码本大小
    z_channels = 32   # 潜在空间通道数
    base_channels = 128  # 基础通道数
    beta = 0.25  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        v_patch_nums=(1, 2, 4, 8, 16, 32),
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path)
        if 'model_state_dict' in vqvae_state_dict.keys():
            vqvae_state_dict = vqvae_state_dict['model_state_dict']
        vqvae.load_state_dict(vqvae_state_dict)
    
    return vqvae

def build_vqvae_single_4_stages(vqvae_checkpoint_path: Optional[str] = None, require_grad = False) -> VQVAE_Single:
    vocab_size = 4096  # 码本大小
    z_channels = 32   # 潜在空间通道数
    base_channels = 128  # 基础通道数
    beta = 0.25  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        v_patch_nums=(1, 8, 16, 32),
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path)
        if 'model_state_dict' in vqvae_state_dict.keys():
            vqvae_state_dict = vqvae_state_dict['model_state_dict']
        vqvae.load_state_dict(vqvae_state_dict)
    
    return vqvae

def build_vqvae_single_4_stages_v2(vqvae_checkpoint_path: Optional[str] = None, require_grad = False) -> VQVAE_Single:
    vocab_size = 4096  # 码本大小
    z_channels = 32   # 潜在空间通道数
    base_channels = 128  # 基础通道数
    beta = 0.25  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        v_patch_nums=(8, 16, 24, 32),
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path)
        if 'model_state_dict' in vqvae_state_dict.keys():
            vqvae_state_dict = vqvae_state_dict['model_state_dict']
        vqvae.load_state_dict(vqvae_state_dict)
    
    return vqvae

# NOTE: currently the best vqvae to use
def build_vqvae_single_5_stages_v1(vqvae_checkpoint_path: Optional[str] = None, require_grad = False) -> VQVAE_Single:
    vocab_size = 4096  # 码本大小
    z_channels = 32   # 潜在空间通道数
    base_channels = 128  # 基础通道数
    beta = 0.25  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        v_patch_nums=(1, 8, 16, 24, 32),
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path, weights_only=True)
        if 'model_state_dict' in vqvae_state_dict.keys():
            vqvae_state_dict = vqvae_state_dict['model_state_dict']
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in vqvae_state_dict.items()}
        vqvae.load_state_dict(new_state_dict)
    
    return vqvae

def build_mask_vqvae_v0(
    checkpoint_path: Optional[str] = None,
    vqvae_init_checkpoint: Optional[str] = None,
    require_grad: bool = True,
    device: str = 'cpu',
) -> MaskVQVAE:
    """
    Build MaskVQVAE v0 model.

    This is the initial version of MaskVQVAE with 5 scales using SAM-style mask decoder.

    Args:
        checkpoint_path: Path to full MaskVQVAE checkpoint (if resuming training)
        vqvae_init_checkpoint: Path to VQVAE checkpoint for initializing encoder/quantizer
        require_grad: Whether to require gradients for model parameters
        device: Device to load model on

    Returns:
        MaskVQVAE model
    """
    vocab_size = 4096
    z_channels = 32
    base_channels = 128
    beta = 0.25
    v_patch_nums = (1, 8, 16, 24, 32)

    model = MaskVQVAE(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        v_patch_nums=v_patch_nums,
        img_feat_dim=256,  # SAM ViT output dimension
        transformer_dim=256,
        transformer_depth=2,
        transformer_num_heads=8,
        transformer_mlp_dim=2048,
        fusion_type='sum',
        use_sam_mask_decoder=True,
        test_mode=False,  # Enable training mode
        ddconfig=dict(
            in_channels=1,
            ch_mult=(1, 1, 2, 4),
            num_res_blocks=2,
            using_sa=True,
            using_mid_sa=True,
        ),
    )

    # Initialize from pretrained VQVAE if specified
    if vqvae_init_checkpoint is not None:
        vqvae_state_dict = torch.load(vqvae_init_checkpoint, map_location='cpu', weights_only=True)
        if 'model_state_dict' in vqvae_state_dict:
            vqvae_state_dict = vqvae_state_dict['model_state_dict']

        # Remove _orig_mod prefix if present (from torch.compile)
        vqvae_state_dict = {k.replace('_orig_mod.', ''): v for k, v in vqvae_state_dict.items()}

        # Load only encoder/quantizer/decoder weights that exist in both models
        model_state = model.state_dict()
        pretrained_state = {}
        for k, v in vqvae_state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                pretrained_state[k] = v

        model_state.update(pretrained_state)
        model.load_state_dict(model_state, strict=False)

        print(f"Initialized MaskVQVAE encoder/quantizer from {vqvae_init_checkpoint}")
        print(f"Loaded {len(pretrained_state)}/{len(vqvae_state_dict)} parameters")

    # Load full checkpoint if specified
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded MaskVQVAE checkpoint from {checkpoint_path}")

    # Set requires_grad
    if not require_grad:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    else:
        model.train()

    return model.to(device)


def build_vqvae_single_4_stages_4_slices(vqvae_checkpoint_path: Optional[str] = None, require_grad=False) -> VQVAE_Single:
    vocab_size = 4096  # 码本大小
    z_channels = 32   # 潜在空间通道数
    base_channels = 128  # 基础通道数
    beta = 0.25  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        v_patch_nums=(4, 8, 12, 16),
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path)
        if 'model_state_dict' in vqvae_state_dict.keys():
            vqvae_state_dict = vqvae_state_dict['model_state_dict']
        vqvae.load_state_dict(vqvae_state_dict)
    
    return vqvae

def build_vqvae_single_4_stages_4_slices_v2(vqvae_checkpoint_path: Optional[str] = None, require_grad=False) -> VQVAE_Single:
    vocab_size = 4096  # 码本大小
    z_channels = 32   # 潜在空间通道数
    base_channels = 128  # 基础通道数
    beta = 0.25  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        v_patch_nums=(4, 8, 16),
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path)
        if 'model_state_dict' in vqvae_state_dict.keys():
            vqvae_state_dict = vqvae_state_dict['model_state_dict']
        vqvae.load_state_dict(vqvae_state_dict)
    
    return vqvae

def build_var_image_encoder(patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32)) -> VarImageEncoder:
    neck_fpn = NeckFPN(
        embed_dim=256, 
        in_dim=256, 
        in_size=(64, 64),
        real_size=(256, 256),
        patch_nums=patch_nums,
    )
    var_image_encoder = VarImageEncoder(neck_fpn)
    return var_image_encoder

# def build_maskvar(maskvar_checkpoint_path: Optional[str] = None, vqvae: VQVAE_Single = None, image_encoder: VarImageEncoder = None, prompt_encoder: PromptEncoder = None, flash_if_available: bool = False) -> MaskVAR:
#     maskvar = MaskVAR(
#         vae_local=vqvae,
#         image_encoder=image_encoder,
#         prompt_encoder=prompt_encoder,
#         num_classes=1,
#         depth=4,
#         embed_dim=256,
#         num_heads=16,
#         mlp_ratio=4.,
#         drop_rate=0.1,
#         attn_drop_rate=0.1,
#         drop_path_rate=0.1,
#         norm_eps=1e-6,
#         shared_aln=False,
#         cond_drop_rate=0.1,
#         attn_l2_norm=False,
#         patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32),
#         flash_if_available=flash_if_available,
#         fused_if_available=True,
#     )
#     if maskvar_checkpoint_path is not None:
#         maskvar_state_dict = torch.load(maskvar_checkpoint_path)
#         maskvar.load_state_dict(maskvar_state_dict)
#     return maskvar

def build_vqvae_single_v3(vqvae_checkpoint_path: Optional[str] = None, require_grad = False) -> VQVAE_Single:
    vocab_size = 4096  # 码本大小
    z_channels = 16   # 潜在空间通道数
    base_channels = 64  # 基础通道数
    beta = 0.25  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        v_patch_nums=(1, 2, 4, 8, 16, 24, 32),
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path)
        vqvae.load_state_dict(vqvae_state_dict)

    return vqvae

def build_vqvae_single_monoscale(vqvae_checkpoint_path: Optional[str] = None, require_grad = False) -> VQVAE_Single:
    # 模型参数
    vocab_size = 4096  # 码本大小
    z_channels = 32   # 潜在空间通道数
    base_channels = 128  # 基础通道数
    beta = 0.25  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        # v_patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32),
        v_patch_nums=[32],
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path)
        vqvae.load_state_dict(vqvae_state_dict)
    
    return vqvae

def build_vqvae_single_monoscale_fl(vqvae_checkpoint_path: Optional[str] = None, require_grad = False) -> VQVAE_Single:
    # 模型参数
    vocab_size = 4096  # 码本大小
    z_channels = 32   # 潜在空间通道数
    base_channels = 128  # 基础通道数
    beta = 0.05  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        # v_patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32),
        v_patch_nums=[32],
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path)
        vqvae.load_state_dict(vqvae_state_dict)
    
    return vqvae

def build_vqvae_single_monoscale_v2(vqvae_checkpoint_path: Optional[str] = None, require_grad = False) -> VQVAE_Single:
    # 模型参数
    vocab_size = 4096  # 码本大小
    z_channels = 32   # 潜在空间通道数
    base_channels = 64  # 基础通道数
    beta = 0.15  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        # v_patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32),
        v_patch_nums=[32],
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path)
        vqvae.load_state_dict(vqvae_state_dict)
    
    return vqvae

def build_vqvae_single_monoscale_v2_1(vqvae_checkpoint_path: Optional[str] = None, require_grad = False) -> VQVAE_Single:
    # 模型参数
    vocab_size = 5120  # 码本大小
    z_channels = 32   # 潜在空间通道数
    base_channels = 64  # 基础通道数
    beta = 0.15  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        # v_patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32),
        v_patch_nums=[32],
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path)
        vqvae.load_state_dict(vqvae_state_dict)
    
    return vqvae

def build_vqvae_single_monoscale_v2_2(vqvae_checkpoint_path: Optional[str] = None, require_grad = False) -> VQVAE_Single:
    # 模型参数
    vocab_size = 4096  # 码本大小
    z_channels = 64   # 潜在空间通道数
    base_channels = 64  # 基础通道数
    beta = 0.1  # commitment loss权重

    vqvae = VQVAE_Single(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=base_channels,
        beta=beta,
        # v_patch_nums=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32),
        v_patch_nums=[32],
        test_mode=False,
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 4), num_res_blocks=2,   # 通道数乘数，用于构建网络层
                    using_sa=True, using_mid_sa=True,)
    )

    if vqvae_checkpoint_path is not None:
        vqvae_state_dict = torch.load(vqvae_checkpoint_path)
        vqvae.load_state_dict(vqvae_state_dict)
    
    return vqvae

def build_sam_image_encoder(sam_checkpoint_path: Optional[str] = None) -> SamImageEncoder:
    dim = 256

    encoder_embed_dim=768
    encoder_depth=12
    encoder_num_heads=12
    encoder_global_attn_indexes=[2, 5, 8, 11]

    prompt_embed_dim = 256
    image_size = 1024 # keep the same as the original sam
    vit_patch_size = 16

    sam_image_encoder=SamImageEncoder(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    # load_state_dict
    if sam_checkpoint_path is not None:
        sam_state_dict = torch.load(sam_checkpoint_path)
        image_encoder_state_dict = {}
        for key, value in sam_state_dict.items():
            if "image_encoder" in key:
                image_encoder_state_dict[key.replace("image_encoder.", "")] = value
        sam_image_encoder.load_state_dict(image_encoder_state_dict)

    return sam_image_encoder

def build_mobile_sam_image_encoder(sam_checkpoint_path: Optional[str] = None) -> TinyViT:
    sam_image_encoder = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8
    )
     # load_state_dict
    if sam_checkpoint_path is not None:
        sam_state_dict = torch.load(sam_checkpoint_path)
        image_encoder_state_dict = {}
        for key, value in sam_state_dict.items():
            if "image_encoder" in key:
                image_encoder_state_dict[key.replace("image_encoder.", "")] = value
        sam_image_encoder.load_state_dict(image_encoder_state_dict)
    
    return sam_image_encoder

def build_image_encoder(sam_checkpoint_path: Optional[str] = None) -> ImageEncoder:
    dim = 256

    encoder_embed_dim=768
    encoder_depth=12
    encoder_num_heads=12
    encoder_global_attn_indexes=[2, 5, 8, 11]

    prompt_embed_dim = 256
    image_size = 1024 # keep the same as the original sam
    vit_patch_size = 16

    sam_image_encoder=SamImageEncoder(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    # load_state_dict
    if sam_checkpoint_path is not None:
        sam_state_dict = torch.load(sam_checkpoint_path)
        image_encoder_state_dict = {}
        for key, value in sam_state_dict.items():
            if "image_encoder" in key:
                image_encoder_state_dict[key.replace("image_encoder.", "")] = value
        sam_image_encoder.load_state_dict(image_encoder_state_dict)

    image_encoder = ImageEncoder(
        sam_embed_dim=prompt_embed_dim,
        embed_dim=dim,
        sam_encoder=sam_image_encoder,
        freeze_sam_encoder=True,
    )

    return image_encoder

def build_prompt_encoder(sam_checkpoint_path: Optional[str] = None) -> PromptEncoder:
    prompt_embed_dim = 256
    image_size = 256 # different from original sam, align with maskseg
    image_embedding_size = 256 // 8

    prompt_encoder=PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    if sam_checkpoint_path is not None:
        sam_state_dict = torch.load(sam_checkpoint_path)
        prompt_encoder_state_dict = {}
        for key, value in sam_state_dict.items():
            if "prompt_encoder" in key:
                prompt_encoder_state_dict[key.replace("prompt_encoder.", "")] = value
        prompt_encoder.load_state_dict(prompt_encoder_state_dict)

    return prompt_encoder

def build_simple_ar(simple_ar_checkpoint_path: Optional[str] = None, device: str = 'cpu') -> SimpleAR:
    simple_ar = SimpleAR(
        dim=256,
        depth=2,
        vocab_size=4096,
        device=device,
        patch_num=[1, 8, 16, 24, 32],
        num_heads=4,
    )

    if simple_ar_checkpoint_path is not None:
        simple_ar_state_dict = torch.load(simple_ar_checkpoint_path)
        simple_ar.load_state_dict(simple_ar_state_dict)

    return simple_ar.to(device)

def build_simple_var(simple_var_checkpoint_path: Optional[str] = None, sam_pe: Optional[torch.Tensor] = None, device: str = 'cpu', enable_prompt_tokens: bool = False) -> SimpleVAR:
    simple_var = SimpleVAR(
        dim=256,
        depth=2,
        vocab_size=4096,
        device=device,
        patch_num=[1, 8, 16, 24, 32],
        num_heads=4,
        vqvae_dim=32,
        use_sam_pe=(sam_pe is not None),
        sam_pe=sam_pe,
        enable_prompt_tokens=enable_prompt_tokens,
    )

    if simple_var_checkpoint_path is not None:
        simple_var_state_dict = torch.load(simple_var_checkpoint_path)
        if any(key.startswith('_orig_mod.') for key in simple_var_state_dict.keys()):
            # 创建一个新的字典，移除 '_orig_mod.' 前缀
            new_state_dict = {}
            for key, value in simple_var_state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            simple_var_state_dict = new_state_dict
        simple_var.load_state_dict(simple_var_state_dict)

    simple_var.init_block_mask()

    return simple_var.to(device)

def build_simple_var_sam_decoder(simple_var_checkpoint_path: Optional[str] = None, sam_pe: Optional[torch.Tensor] = None, device: str = 'cpu', enable_prompt_tokens: bool = False) -> SimpleVARSamDecoder:
    # NOTE: enable_prompt_tokens is kept for interface compatibility with build_simple_var.
    # SimpleVARSamDecoder inherently supports prompt tokens via AdaptedMaskDecoder,
    # so this parameter is not used in the function body.
    adapted_2way_transformer = AdaptedTwoWayTransformer(
        depth=2,
        embedding_dim=256,
        num_heads=4,
        mlp_dim=2048,
    )
    
    adapted_mask_decoder = AdaptedMaskDecoder(
        transformer_dim=256,
        transformer=adapted_2way_transformer,
    )
    
    simple_var = SimpleVARSamDecoder(
        adapted_mask_decoder=adapted_mask_decoder,
        dim=256,
        depth=2,
        vocab_size=4096,
        device=device,
        patch_num=[1, 8, 16, 24, 32],
        num_heads=4,
        vqvae_dim=32,
        use_sam_pe=(sam_pe is not None),
        sam_pe=sam_pe,
    )

    if simple_var_checkpoint_path is not None:
        simple_var_state_dict = torch.load(simple_var_checkpoint_path)
        if any(key.startswith('_orig_mod.') for key in simple_var_state_dict.keys()):
            # 创建一个新的字典，移除 '_orig_mod.' 前缀
            new_state_dict = {}
            for key, value in simple_var_state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            simple_var_state_dict = new_state_dict
        simple_var.load_state_dict(simple_var_state_dict)

    simple_var.init_block_mask()

    return simple_var.to(device)

def build_simple_var_sam_decoder_mlp_adapter(simple_var_checkpoint_path: Optional[str] = None, sam_pe: Optional[torch.Tensor] = None, device: str = 'cpu', enable_prompt_tokens: bool = False) -> SimpleVARSamDecoder:
    # NOTE: enable_prompt_tokens is kept for interface compatibility with build_simple_var.
    # SimpleVARSamDecoder inherently supports prompt tokens via AdaptedMaskDecoder,
    # so this parameter is not used in the function body.
    adapted_2way_transformer = AdaptedTwoWayTransformer(
        depth=2,
        embedding_dim=256,
        num_heads=4,
        mlp_dim=2048,
    )
    
    adapted_mask_decoder = AdaptedMaskDecoder(
        transformer_dim=256,
        transformer=adapted_2way_transformer,
    )
    
    simple_var = SimpleVARSamDecoder(
        adapted_mask_decoder=adapted_mask_decoder,
        dim=256,
        depth=2,
        vocab_size=4096,
        device=device,
        patch_num=[1, 8, 16, 24, 32],
        num_heads=4,
        vqvae_dim=32,
        use_sam_pe=(sam_pe is not None),
        sam_pe=sam_pe,
        linear_input_mapping=False,
        linear_output_mapping=False,
    )

    if simple_var_checkpoint_path is not None:
        simple_var_state_dict = torch.load(simple_var_checkpoint_path)
        if any(key.startswith('_orig_mod.') for key in simple_var_state_dict.keys()):
            # 创建一个新的字典，移除 '_orig_mod.' 前缀
            new_state_dict = {}
            for key, value in simple_var_state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            simple_var_state_dict = new_state_dict
        simple_var.load_state_dict(simple_var_state_dict)

    simple_var.init_block_mask()

    return simple_var.to(device)

def build_simple_var_6d(simple_var_checkpoint_path: Optional[str] = None, sam_pe: Optional[torch.Tensor] = None, device: str = 'cpu') -> SimpleVAR:
    simple_var = SimpleVAR(
        dim=256,
        depth=6,
        vocab_size=4096,
        device=device,
        patch_num=[1, 8, 16, 24, 32],
        num_heads=4,
        vqvae_dim=32,
        use_sam_pe=(sam_pe is not None),
        sam_pe=sam_pe,
    )

    if simple_var_checkpoint_path is not None:
        simple_var_state_dict = torch.load(simple_var_checkpoint_path)
        if any(key.startswith('_orig_mod.') for key in simple_var_state_dict.keys()):
            # 创建一个新的字典，移除 '_orig_mod.' 前缀
            new_state_dict = {}
            for key, value in simple_var_state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            simple_var_state_dict = new_state_dict
        simple_var.load_state_dict(simple_var_state_dict)

    simple_var.init_block_mask()

    return simple_var.to(device)

def build_simple_var_16d(simple_var_checkpoint_path: Optional[str] = None, sam_pe: Optional[torch.Tensor] = None, device: str = 'cpu') -> SimpleVAR:
    simple_var = SimpleVAR(
        dim=256,
        depth=16,
        vocab_size=4096,
        device=device,
        patch_num=[1, 8, 16, 24, 32],
        num_heads=4,
        vqvae_dim=32,
        use_sam_pe=(sam_pe is not None),
        sam_pe=sam_pe,
    )

    if simple_var_checkpoint_path is not None:
        simple_var_state_dict = torch.load(simple_var_checkpoint_path)
        if any(key.startswith('_orig_mod.') for key in simple_var_state_dict.keys()):
            # 创建一个新的字典，移除 '_orig_mod.' 前缀
            new_state_dict = {}
            for key, value in simple_var_state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            simple_var_state_dict = new_state_dict
        simple_var.load_state_dict(simple_var_state_dict)

    simple_var.init_block_mask()

    return simple_var.to(device)


def build_simple_mask_vqvae(
    simple_mask_vqvae_checkpoint_path: Optional[str] = None,
    image_encoder_checkpoint: Optional[str] = None,
    image_encoder_config_name: Optional[str] = 'dino_v3_vits',
    enable_vq=False,
    device: str = 'cpu',
    kmeans_centroids_path: Optional[str] = None,
) -> SimpleMaskVqvae:
    """
    Build SimpleMaskVqvae model.

    Both image and mask use the same encoder architecture (MobileSAM/TinyViT).

    Args:
        simple_mask_vqvae_checkpoint_path: Path to model checkpoint
        image_encoder_checkpoint: Path to image encoder checkpoint
        image_encoder_config_name: Name of image encoder config
        enable_vq: Whether to enable VQ
        device: Device to load model on
        kmeans_centroids_path: Path to KMeans centroids for initializing VQ codebook (.npy or .pt)
    """
    # Model config (hardcoded)
    vocab_size = 4096
    dim = 256
    beta = 0.25

    image_encoder = builder_map['image_encoder'][image_encoder_config_name](image_encoder_checkpoint)
    # mask_encoder = build_mobile_sam_image_encoder(sam_checkpoint_path)
    mask_encoder = MaskEncoderLite(dim=dim)

    simple_mask_vqvae = SimpleMaskVqvae(
        image_encoder=image_encoder,
        mask_encoder=mask_encoder,
        dim=dim,
        vocab_size=vocab_size,
        beta=beta,
        device=device,
        enable_vq=enable_vq
    )

    if simple_mask_vqvae_checkpoint_path is not None:
        checkpoint = torch.load(simple_mask_vqvae_checkpoint_path, map_location='cpu', weights_only=True)

        # Handle full checkpoint format from training (contains 'model_state_dict')
        if 'model_state_dict' in checkpoint:
            simple_mask_vqvae_state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            simple_mask_vqvae_state_dict = checkpoint

        # Remove '_orig_mod.' prefix from torch.compile
        if any(key.startswith('_orig_mod.') for key in simple_mask_vqvae_state_dict.keys()):
            new_state_dict = {}
            for key, value in simple_mask_vqvae_state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            simple_mask_vqvae_state_dict = new_state_dict

        simple_mask_vqvae.load_state_dict(simple_mask_vqvae_state_dict)
        print(f"Loaded SimpleMaskVqvae checkpoint from {simple_mask_vqvae_checkpoint_path}")

    # Initialize VQ codebook from KMeans centroids if provided
    if kmeans_centroids_path is not None:
        _initialize_codebook_from_kmeans(simple_mask_vqvae, kmeans_centroids_path)

    return simple_mask_vqvae.to(device)


def build_simple_mask_vqvae_dim384(
    simple_mask_vqvae_checkpoint_path: Optional[str] = None,
    image_encoder_checkpoint: Optional[str] = None,
    image_encoder_config_name: Optional[str] = 'dino_v3_vits',
    device: str = 'cpu',
    enable_vq=False,
    kmeans_centroids_path: Optional[str] = None,
) -> SimpleMaskVqvae:
    """
    Build SimpleMaskVqvae model with dim=384.

    Both image and mask use the same encoder architecture (MobileSAM/TinyViT).

    Args:
        simple_mask_vqvae_checkpoint_path: Path to model checkpoint
        image_encoder_checkpoint: Path to image encoder checkpoint
        image_encoder_config_name: Name of image encoder config
        device: Device to load model on
        enable_vq: Whether to enable VQ
        kmeans_centroids_path: Path to KMeans centroids for initializing VQ codebook (.npy or .pt)
    """
    # Model config (hardcoded)
    vocab_size = 4096
    dim = 384
    beta = 0.25

    image_encoder = builder_map['image_encoder'][image_encoder_config_name](image_encoder_checkpoint)
    # mask_encoder = build_mobile_sam_image_encoder(sam_checkpoint_path)
    mask_encoder = MaskEncoderLite(dim=dim)

    simple_mask_vqvae = SimpleMaskVqvae(
        image_encoder=image_encoder,
        mask_encoder=mask_encoder,
        dim=dim,
        vocab_size=vocab_size,
        beta=beta,
        device=device,
        enable_vq=enable_vq,
    )

    if simple_mask_vqvae_checkpoint_path is not None:
        checkpoint = torch.load(simple_mask_vqvae_checkpoint_path, map_location='cpu', weights_only=True)

        # Handle full checkpoint format from training (contains 'model_state_dict')
        if 'model_state_dict' in checkpoint:
            simple_mask_vqvae_state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            simple_mask_vqvae_state_dict = checkpoint

        # Remove '_orig_mod.' prefix from torch.compile
        if any(key.startswith('_orig_mod.') for key in simple_mask_vqvae_state_dict.keys()):
            new_state_dict = {}
            for key, value in simple_mask_vqvae_state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            simple_mask_vqvae_state_dict = new_state_dict

        simple_mask_vqvae.load_state_dict(simple_mask_vqvae_state_dict)
        print(f"Loaded SimpleMaskVqvae checkpoint from {simple_mask_vqvae_checkpoint_path}")

    # Initialize VQ codebook from KMeans centroids if provided
    if kmeans_centroids_path is not None:
        _initialize_codebook_from_kmeans(simple_mask_vqvae, kmeans_centroids_path)

    return simple_mask_vqvae.to(device)


def build_simple_mask_vqvae_v3_16x16_dim384(
    simple_mask_vqvae_checkpoint_path: Optional[str] = None,
    image_encoder_checkpoint: Optional[str] = None,
    image_encoder_config_name: Optional[str] = 'dino_v3_vits',
    device: str = 'cpu',
    enable_vq=False,
    kmeans_centroids_path: Optional[str] = None,
) -> SimpleMaskVqvae:
    """
    Build SimpleMaskVqvae with a 16x16 spatial mask-token grid.

    This keeps the V1-style spatial latent interface, but reduces the mask
    token sequence from 64x64=4096 to 16x16=256 for faster AR experiments.
    """
    vocab_size = 4096
    dim = 384
    beta = 0.25

    image_encoder = builder_map['image_encoder'][image_encoder_config_name](image_encoder_checkpoint)
    mask_encoder = MaskEncoderLite16x16(dim=dim)

    simple_mask_vqvae = SimpleMaskVqvae(
        image_encoder=image_encoder,
        mask_encoder=mask_encoder,
        dim=dim,
        vocab_size=vocab_size,
        beta=beta,
        device=device,
        enable_vq=enable_vq,
    )

    if simple_mask_vqvae_checkpoint_path is not None:
        checkpoint = torch.load(simple_mask_vqvae_checkpoint_path, map_location='cpu', weights_only=True)

        if 'model_state_dict' in checkpoint:
            simple_mask_vqvae_state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            simple_mask_vqvae_state_dict = checkpoint

        if any(key.startswith('_orig_mod.') for key in simple_mask_vqvae_state_dict.keys()):
            new_state_dict = {}
            for key, value in simple_mask_vqvae_state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            simple_mask_vqvae_state_dict = new_state_dict

        simple_mask_vqvae.load_state_dict(simple_mask_vqvae_state_dict)
        print(f"Loaded SimpleMaskVqvae checkpoint from {simple_mask_vqvae_checkpoint_path}")

    if kmeans_centroids_path is not None:
        _initialize_codebook_from_kmeans(simple_mask_vqvae, kmeans_centroids_path)

    return simple_mask_vqvae.to(device)


def build_simple_mask_vqvae_multiscale_dim384(
    simple_mask_vqvae_checkpoint_path: Optional[str] = None,
    image_encoder_checkpoint: Optional[str] = None,
    image_encoder_config_name: Optional[str] = 'dino_v3_vits',
    device: str = 'cpu',
    enable_vq=False,
    kmeans_centroids_path: Optional[str] = None,
) -> SimpleMaskVqvaeMultiScale:
    """
    Build multi-scale SimpleMaskVqvae with scales [1, 2, 4, 8, 16, 32, 64].

    The decoder is the existing SimpleMaskDecoder; only the tokenizer/fusion
    path is multi-scale.
    """
    vocab_size = 4096
    dim = 384
    beta = 0.25

    image_encoder = builder_map['image_encoder'][image_encoder_config_name](image_encoder_checkpoint)
    mask_encoder = MaskEncoderLite(dim=dim)

    simple_mask_vqvae = SimpleMaskVqvaeMultiScale(
        image_encoder=image_encoder,
        mask_encoder=mask_encoder,
        dim=dim,
        vocab_size=vocab_size,
        beta=beta,
        scales=(1, 2, 4, 8, 16, 32, 64),
        device=device,
        enable_vq=enable_vq,
    )

    if simple_mask_vqvae_checkpoint_path is not None:
        checkpoint = torch.load(simple_mask_vqvae_checkpoint_path, map_location='cpu', weights_only=True)

        if 'model_state_dict' in checkpoint:
            simple_mask_vqvae_state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            simple_mask_vqvae_state_dict = checkpoint

        if any(key.startswith('_orig_mod.') for key in simple_mask_vqvae_state_dict.keys()):
            simple_mask_vqvae_state_dict = {
                key.replace('_orig_mod.', ''): value
                for key, value in simple_mask_vqvae_state_dict.items()
            }

        simple_mask_vqvae.load_state_dict(simple_mask_vqvae_state_dict)
        print(f"Loaded SimpleMaskVqvaeMultiScale checkpoint from {simple_mask_vqvae_checkpoint_path}")

    if kmeans_centroids_path is not None:
        _initialize_codebook_from_kmeans(simple_mask_vqvae, kmeans_centroids_path)

    return simple_mask_vqvae.to(device)


def build_simple_mask_vqvae_multiscale_v2_dim384(
    simple_mask_vqvae_checkpoint_path: Optional[str] = None,
    image_encoder_checkpoint: Optional[str] = None,
    image_encoder_config_name: Optional[str] = 'dino_v3_vits',
    device: str = 'cpu',
    enable_vq=False,
    kmeans_centroids_path: Optional[str] = None,
) -> SimpleMaskVqvaeMultiScaleResidual:
    """
    Build SimpleMaskVqvae with VAR-style multi-scale residual quantization.

    The model intentionally keeps the SimpleMaskVqvae encoder/decoder layout
    and only swaps SimpleVectorQuantize for MultiscaleVectorQuantize, so a
    trained SimpleMaskVqvae checkpoint can initialize encoder, decoder, and the
    shared codebook.
    """
    vocab_size = 4096
    dim = 384
    beta = 0.25

    image_encoder = builder_map['image_encoder'][image_encoder_config_name](image_encoder_checkpoint)
    mask_encoder = MaskEncoderLite(dim=dim)

    simple_mask_vqvae = SimpleMaskVqvaeMultiScaleResidual(
        image_encoder=image_encoder,
        mask_encoder=mask_encoder,
        dim=dim,
        vocab_size=vocab_size,
        beta=beta,
        scales=(1, 2, 4, 8, 16, 32, 64),
        device=device,
        enable_vq=enable_vq,
    )

    if simple_mask_vqvae_checkpoint_path is not None:
        checkpoint = torch.load(simple_mask_vqvae_checkpoint_path, map_location='cpu', weights_only=True)

        if 'model_state_dict' in checkpoint:
            simple_mask_vqvae_state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            simple_mask_vqvae_state_dict = checkpoint

        if any(key.startswith('_orig_mod.') for key in simple_mask_vqvae_state_dict.keys()):
            simple_mask_vqvae_state_dict = {
                key.replace('_orig_mod.', ''): value
                for key, value in simple_mask_vqvae_state_dict.items()
            }

        missing_keys, unexpected_keys = simple_mask_vqvae.load_state_dict(simple_mask_vqvae_state_dict, strict=False)
        print(f"Loaded SimpleMaskVqvaeMultiScaleResidual checkpoint from {simple_mask_vqvae_checkpoint_path}")
        if missing_keys:
            print(f"  Missing keys initialized from scratch: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected checkpoint keys ignored: {unexpected_keys}")

    if kmeans_centroids_path is not None:
        _initialize_codebook_from_kmeans(simple_mask_vqvae, kmeans_centroids_path)

    return simple_mask_vqvae.to(device)


def build_simple_mask_vqvae_v2_dim384(
    simple_mask_vqvae_checkpoint_path: Optional[str] = None,
    image_encoder_checkpoint: Optional[str] = None,
    image_encoder_config_name: Optional[str] = 'dino_v3_vits',
    device: str = 'cpu',
    enable_vq=False,
    kmeans_centroids_path: Optional[str] = None,
) -> SimpleMaskVqvae:
    """
    Build SimpleMaskVqvae model with dim=384.

    Both image and mask use the same encoder architecture (MobileSAM/TinyViT).

    Args:
        simple_mask_vqvae_checkpoint_path: Path to model checkpoint
        image_encoder_checkpoint: Path to image encoder checkpoint
        image_encoder_config_name: Name of image encoder config
        device: Device to load model on
        enable_vq: Whether to enable VQ
        kmeans_centroids_path: Path to KMeans centroids for initializing VQ codebook (.npy or .pt)
    """
    # Model config (hardcoded)
    vocab_size = 4096
    dim = 384
    beta = 0.25

    image_encoder = builder_map['image_encoder'][image_encoder_config_name](image_encoder_checkpoint)
    # mask_encoder = build_mobile_sam_image_encoder(sam_checkpoint_path)
    mask_encoder = MaskEncoderLite(dim=dim)

    simple_mask_vqvae = SimpleMaskVqvaeV2(
        image_encoder=image_encoder,
        mask_encoder=mask_encoder,
        dim=dim,
        vocab_size=vocab_size,
        beta=beta,
        device=device,
        enable_vq=enable_vq,
    )

    if simple_mask_vqvae_checkpoint_path is not None:
        checkpoint = torch.load(simple_mask_vqvae_checkpoint_path, map_location='cpu', weights_only=True)

        # Handle full checkpoint format from training (contains 'model_state_dict')
        if 'model_state_dict' in checkpoint:
            simple_mask_vqvae_state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            simple_mask_vqvae_state_dict = checkpoint

        # Remove '_orig_mod.' prefix from torch.compile
        if any(key.startswith('_orig_mod.') for key in simple_mask_vqvae_state_dict.keys()):
            new_state_dict = {}
            for key, value in simple_mask_vqvae_state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            simple_mask_vqvae_state_dict = new_state_dict

        simple_mask_vqvae.load_state_dict(simple_mask_vqvae_state_dict)
        print(f"Loaded SimpleMaskVqvae checkpoint from {simple_mask_vqvae_checkpoint_path}")

    # Initialize VQ codebook from KMeans centroids if provided
    if kmeans_centroids_path is not None:
        _initialize_codebook_from_kmeans(simple_mask_vqvae, kmeans_centroids_path)

    return simple_mask_vqvae.to(device)


def _build_simple_mask_vqvae_query_variant_dim384(
    model_cls,
    simple_mask_vqvae_checkpoint_path: Optional[str] = None,
    image_encoder_checkpoint: Optional[str] = None,
    image_encoder_config_name: Optional[str] = 'dino_v3_vits',
    device: str = 'cpu',
    enable_vq=False,
    kmeans_centroids_path: Optional[str] = None,
):
    vocab_size = 4096
    dim = 384
    beta = 0.25

    image_encoder = builder_map['image_encoder'][image_encoder_config_name](image_encoder_checkpoint)
    mask_encoder = MaskEncoderLite(dim=dim)

    simple_mask_vqvae = model_cls(
        image_encoder=image_encoder,
        mask_encoder=mask_encoder,
        dim=dim,
        vocab_size=vocab_size,
        beta=beta,
        device=device,
        enable_vq=enable_vq,
    )

    if simple_mask_vqvae_checkpoint_path is not None:
        checkpoint = torch.load(simple_mask_vqvae_checkpoint_path, map_location='cpu', weights_only=True)

        if 'model_state_dict' in checkpoint:
            simple_mask_vqvae_state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            simple_mask_vqvae_state_dict = checkpoint

        if any(key.startswith('_orig_mod.') for key in simple_mask_vqvae_state_dict.keys()):
            simple_mask_vqvae_state_dict = {
                key.replace('_orig_mod.', ''): value
                for key, value in simple_mask_vqvae_state_dict.items()
            }

        simple_mask_vqvae.load_state_dict(simple_mask_vqvae_state_dict)
        print(f"Loaded {model_cls.__name__} checkpoint from {simple_mask_vqvae_checkpoint_path}")

    if kmeans_centroids_path is not None:
        _initialize_codebook_from_kmeans(simple_mask_vqvae, kmeans_centroids_path)

    return simple_mask_vqvae.to(device)


def build_simple_mask_vqvae_v3_dim384(
    simple_mask_vqvae_checkpoint_path: Optional[str] = None,
    image_encoder_checkpoint: Optional[str] = None,
    image_encoder_config_name: Optional[str] = 'dino_v3_vits',
    device: str = 'cpu',
    enable_vq=False,
    kmeans_centroids_path: Optional[str] = None,
) -> SimpleMaskVqvaeV3:
    return _build_simple_mask_vqvae_query_variant_dim384(
        SimpleMaskVqvaeV3,
        simple_mask_vqvae_checkpoint_path=simple_mask_vqvae_checkpoint_path,
        image_encoder_checkpoint=image_encoder_checkpoint,
        image_encoder_config_name=image_encoder_config_name,
        device=device,
        enable_vq=enable_vq,
        kmeans_centroids_path=kmeans_centroids_path,
    )


def build_simple_mask_vqvae_v4_dim384(
    simple_mask_vqvae_checkpoint_path: Optional[str] = None,
    image_encoder_checkpoint: Optional[str] = None,
    image_encoder_config_name: Optional[str] = 'dino_v3_vits',
    device: str = 'cpu',
    enable_vq=False,
    kmeans_centroids_path: Optional[str] = None,
) -> SimpleMaskVqvaeV4:
    return _build_simple_mask_vqvae_query_variant_dim384(
        SimpleMaskVqvaeV4,
        simple_mask_vqvae_checkpoint_path=simple_mask_vqvae_checkpoint_path,
        image_encoder_checkpoint=image_encoder_checkpoint,
        image_encoder_config_name=image_encoder_config_name,
        device=device,
        enable_vq=enable_vq,
        kmeans_centroids_path=kmeans_centroids_path,
    )


def build_simple_mask_vqvae_shape_only_dim384(
    simple_mask_vqvae_checkpoint_path: Optional[str] = None,
    image_encoder_checkpoint: Optional[str] = None,
    image_encoder_config_name: Optional[str] = 'dino_v3_vits',
    device: str = 'cpu',
    enable_vq=False,
    kmeans_centroids_path: Optional[str] = None,
) -> SimpleMaskVqvaeShapeOnly:
    """
    Build shape-only SimpleMaskVqvae model with dim=384.

    The signature intentionally matches other builders so the existing trainer
    can reuse the same argument plumbing. Image encoder arguments are ignored.

    Args:
        simple_mask_vqvae_checkpoint_path: Path to model checkpoint
        image_encoder_checkpoint: Unused, kept for trainer compatibility
        image_encoder_config_name: Unused, kept for trainer compatibility
        device: Device to load model on
        enable_vq: Whether to enable VQ
        kmeans_centroids_path: Path to KMeans centroids for initializing VQ codebook (.npy or .pt)
    """
    del image_encoder_checkpoint, image_encoder_config_name

    vocab_size = 4096
    dim = 384
    beta = 0.25
    num_queries = 8

    mask_encoder = MaskEncoderLite(dim=dim)

    simple_mask_vqvae = SimpleMaskVqvaeShapeOnly(
        mask_encoder=mask_encoder,
        dim=dim,
        num_queries=num_queries,
        vocab_size=vocab_size,
        beta=beta,
        device=device,
        enable_vq=enable_vq,
    )

    if simple_mask_vqvae_checkpoint_path is not None:
        checkpoint = torch.load(simple_mask_vqvae_checkpoint_path, map_location='cpu', weights_only=True)

        if 'model_state_dict' in checkpoint:
            simple_mask_vqvae_state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            simple_mask_vqvae_state_dict = checkpoint

        if any(key.startswith('_orig_mod.') for key in simple_mask_vqvae_state_dict.keys()):
            new_state_dict = {}
            for key, value in simple_mask_vqvae_state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            simple_mask_vqvae_state_dict = new_state_dict

        simple_mask_vqvae.load_state_dict(simple_mask_vqvae_state_dict)
        print(f"Loaded SimpleMaskVqvaeShapeOnly checkpoint from {simple_mask_vqvae_checkpoint_path}")

    if kmeans_centroids_path is not None:
        _initialize_codebook_from_kmeans(simple_mask_vqvae, kmeans_centroids_path)

    return simple_mask_vqvae.to(device)


def build_simple_mask_ar(
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
    enable_click: bool = False,
) -> SimpleMaskAR:
    """
    Build SimpleMaskAR model.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        SimpleMaskAR model
    """
    # Fixed hyperparameters
    dim = 384
    depth = 2
    vocab_size = 4096
    h = 64  # 1024 / 16
    w = 64  # 1024 / 16
    num_heads = 4

    model = SimpleMaskAR(
        dim=dim,
        depth=depth,
        vocab_size=vocab_size,
        h=h,
        w=w,
        num_heads=num_heads,
        enable_click=enable_click,
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Handle full checkpoint format from training
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            state_dict = checkpoint

        # Remove '_orig_mod.' prefix from torch.compile
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=not enable_click)
        print(f"Loaded SimpleMaskAR checkpoint from {checkpoint_path}")

    return model.to(device)


def build_simple_mask_var(
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
    enable_click: bool = False,
) -> SimpleMaskVAR:
    if enable_click:
        raise ValueError("SimpleMaskVAR does not implement click conditioning yet.")

    model = SimpleMaskVAR(
        dim=384,
        depth=2,
        vocab_size=4096,
        scales=(1, 2, 4, 8, 16, 32, 64),
        h=64,
        w=64,
        num_heads=4,
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            state_dict = checkpoint
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Loaded SimpleMaskVAR checkpoint from {checkpoint_path}")

    return model.to(device)


def build_simple_query_token_ar(
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
) -> SimpleQueryTokenAR:
    dim = 384
    depth = 2
    vocab_size = 4096
    num_queries = 8
    image_token_len = 64 * 64
    num_heads = 4

    model = SimpleQueryTokenAR(
        dim=dim,
        depth=depth,
        vocab_size=vocab_size,
        num_queries=num_queries,
        image_token_len=image_token_len,
        num_heads=num_heads,
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            state_dict = checkpoint

        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        print(f"Loaded SimpleQueryTokenAR checkpoint from {checkpoint_path}")

    return model.to(device)


def build_simple_mask_vae_v2_dim384(
    checkpoint_path: Optional[str] = None,
    image_encoder_checkpoint: Optional[str] = None,
    image_encoder_config_name: Optional[str] = 'dino_v3_vits',
    device: str = 'cpu',
) -> SimpleMaskVAEV2:
    dim = 384
    latent_dim = 128
    num_queries = 8
    beta_kl = 1e-4

    image_encoder = builder_map['image_encoder'][image_encoder_config_name](image_encoder_checkpoint)
    mask_encoder = MaskEncoderLite(dim=dim)
    model = SimpleMaskVAEV2(
        image_encoder=image_encoder,
        mask_encoder=mask_encoder,
        dim=dim,
        latent_dim=latent_dim,
        num_queries=num_queries,
        beta_kl=beta_kl,
        num_heads=4,
        device=device,
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            state_dict = checkpoint
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Loaded SimpleMaskVAEV2 checkpoint from {checkpoint_path}")

    return model.to(device)


def build_simple_mask_latent_dit(
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
) -> SimpleMaskLatentDiT:
    model = SimpleMaskLatentDiT(
        latent_dim=128,
        dim=384,
        depth=2,
        num_heads=4,
        num_queries=8,
        image_dim=384,
        cond_grid_size=8,
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            state_dict = checkpoint
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Loaded SimpleMaskLatentDiT checkpoint from {checkpoint_path}")

    return model.to(device)


def _initialize_codebook_from_kmeans(model: SimpleMaskVqvae, centroids_path: str):
    """
    Initialize VQ codebook using KMeans centroids.

    Args:
        model: SimpleMaskVqvae instance
        centroids_path: Path to KMeans centroids file (.npy or .pt)
    """
    import numpy as np

    # Load centroids based on file extension
    if centroids_path.endswith('.npy'):
        centroids = np.load(centroids_path)
        centroids = torch.from_numpy(centroids).float()
    elif centroids_path.endswith('.pt') or centroids_path.endswith('.pth'):
        centroids = torch.load(centroids_path, map_location='cpu')
        if isinstance(centroids, np.ndarray):
            centroids = torch.from_numpy(centroids).float()
    else:
        raise ValueError(f"Unsupported file format: {centroids_path}. Use .npy or .pt/.pth")

    # Validate dimensions
    if centroids.shape[0] != model.vocab_size:
        raise ValueError(
            f"Vocab size mismatch: centroids have {centroids.shape[0]} clusters, "
            f"but model expects vocab_size={model.vocab_size}"
        )
    if centroids.shape[1] != model.dim:
        raise ValueError(
            f"Dimension mismatch: centroids have dim={centroids.shape[1]}, "
            f"but model expects dim={model.dim}"
        )

    # Initialize the embedding weights
    with torch.no_grad():
        model.quant.embedding.weight.copy_(centroids)

    print(f"Initialized VQ codebook from KMeans centroids: {centroids_path}")
    print(f"  Centroids shape: {centroids.shape}")
    print(f"  Centroids mean: {centroids.mean().item():.4f}, std: {centroids.std().item():.4f}")


# def build_maskgit(vqvae: VQVAE_Single, maskgit_checkpoint_path: Optional[str] = None) -> MaskGIT:
#     maskgit = MaskGIT(
#         vqvae=vqvae,
#         image_size=256,
#         patch_size=8,
#         dim=256,
#         num_heads=8,
#         num_blocks=8,
#         vocab_size=4096,
#         image_cross_layers=[0, 4],
#         click_cross_layers=[2, 6],
#         freeze_vqvae=True,
#     )

#     if maskgit_checkpoint_path is not None:
#         maskgit_state_dict = torch.load(maskgit_checkpoint_path)
#         maskgit.load_state_dict(maskgit_state_dict)

#     return maskgit

# def build_maskgit_v0(vqvae: VQVAE_Single, maskgit_checkpoint_path: Optional[str] = None) -> MaskGIT:
#     maskgit = MaskGIT(
#         vqvae=vqvae,
#         image_size=256,
#         patch_size=8,
#         dim=256,
#         num_heads=8,
#         num_blocks=6,
#         vocab_size=4096,
#         image_cross_layers=[0, 3],
#         click_cross_layers=[1, 4],
#         freeze_vqvae=True,
#     )

#     if maskgit_checkpoint_path is not None:
#         maskgit_state_dict = torch.load(maskgit_checkpoint_path)
#         maskgit.load_state_dict(maskgit_state_dict)

#     return maskgit

def build_cocolvis_dataset(dataset_path='data/coco_lvis') -> Tuple[LvisDataset, LvisDataset]:
    trainset = LvisDataset(
        dataset_path=dataset_path,
        split='train',
        img_split='train',
        stuff_prob=0.0,
    )

    valset = LvisDataset(
        dataset_path=dataset_path,
        split='val',
        img_split='val',
        stuff_prob=0.0,
    )

    return trainset, valset

def build_hqseg44k_dataset(dataset_path='data/sam-hq') -> Tuple[HQSeg44KTrainDataset, HQSeg44KTestDataset]:
    trainset = HQSeg44KTrainDataset(
        data_root=dataset_path,
        img_size=(1024, 1024)
    )

    testset = HQSeg44KTestDataset(
        data_root=dataset_path,
        img_size=(1024, 1024)
    )

    return trainset, testset


def build_coconut_hf_dataset(
    dataset_path='data/coconut_hf',
    stuff_prob=1.0,
) -> Tuple[CoconutHFDataset, CoconutHFDataset]:
    """
    Build COCONut HF dataset for training and validation.

    Note: COCONut parquet files contain train split only.
    For validation, we typically use a subset or separate val parquet.
    """
    trainset = CoconutHFDataset(
        parquet_path=f"{dataset_path}/train/",
        image_root=f"{dataset_path}/train2017",
        stuff_prob=stuff_prob,
    )

    # For validation, use a subset of training data
    # In practice, you may want to use a separate val parquet file
    valset = CoconutHFDataset(
        parquet_path=f"{dataset_path}/val/",
        image_root=f"{dataset_path}/val2017",
        stuff_prob=stuff_prob,
    )

    return trainset, valset


def _build_dino_v3(model_name, checkpoint_path, device):
    model = timm.create_model(
        model_name,
        pretrained=False,  # 不从网络加载权重
        num_classes=0,     # 移除分类头
        img_size=1024,
        # features_only=True,
    )

    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    # model.set_input_size(1024)
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model

def build_dino_v3_vits(checkpoint_path='ckpt/dino_v3_vits.safetensors', device='cpu'):
    dinov3 = _build_dino_v3('vit_small_patch16_dinov3', checkpoint_path, device)
    return DinoV3Wrapper(dinov3)

def build_dino_v3_vitb(checkpoint_path='ckpt/dino_v3_vitb.safetensors', device='cpu'):
    dinov3 = _build_dino_v3('vit_base_patch16_dinov3', checkpoint_path, device)
    return DinoV3Wrapper(dinov3)

def build_dino_v3_transform(dino_v3_model):
    data_config = timm.data.resolve_model_data_config(dino_v3_model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return transforms


builder_map = {
    "maskvar": {
        "maskvar": build_maskvar,
        "maskvar_flex": build_maskvar_flex,
        "maskvar_flex_5_stages": build_maskvar_flex_5_stages,
        "maskvar_flex_mobile_5_stages": build_maskvar_flex_mobile_5_stages,
        "maskvar_flex_simple_mobile_5_stages": build_maskvar_flex_simple_mobile_5_stages,
        "maskvar_flex_v2": build_maskvar_flex_v2,
        "maskvar_v2": build_maskvar_v2,
        "maskvar_v3": build_maskvar_v3,
    },
    "vqvae": {
        "vqvae_single": build_vqvae_single,
        "vqvae_single_4_stages_v2": build_vqvae_single_4_stages_v2,
        "vqvae_single_5_stages_v1": build_vqvae_single_5_stages_v1,
        "vqvae_single_4_stages_4_slices": build_vqvae_single_4_stages_4_slices,
        "vqvae_single_4_stages_4_slices_v2": build_vqvae_single_4_stages_4_slices_v2,
    },
    "mask_vqvae": {
        "mask_vqvae_v0": build_mask_vqvae_v0,
    },
    "simple_mask_vqvae": {
        "simple_mask_vqvae": build_simple_mask_vqvae,
        "simple_mask_vqvae_dim384": build_simple_mask_vqvae_dim384,
        "simple_mask_vqvae_v3_16x16_dim384": build_simple_mask_vqvae_v3_16x16_dim384,
        "simple_mask_vqvae_multiscale_dim384": build_simple_mask_vqvae_multiscale_dim384,
        "simple_mask_vqvae_multiscale_v2_dim384": build_simple_mask_vqvae_multiscale_v2_dim384,
        "simple_mask_vqvae_v2_dim384": build_simple_mask_vqvae_v2_dim384,
        "simple_mask_vqvae_v3_dim384": build_simple_mask_vqvae_v3_dim384,
        "simple_mask_vqvae_v4_dim384": build_simple_mask_vqvae_v4_dim384,
        "simple_mask_vqvae_shape_only_dim384": build_simple_mask_vqvae_shape_only_dim384,
    },
    "image_encoder": {
        "sam_vitb": build_sam_image_encoder,
        "mobile_sam": build_mobile_sam_image_encoder,
        "dino_v3_vitb": build_dino_v3_vitb,
        "dino_v3_vits": build_dino_v3_vits,
    },
    "prompt_encoder": build_prompt_encoder,
    "simple_ar": build_simple_ar,
    "simple_var": {
        "simple_var": build_simple_var,
        "simple_var_16d": build_simple_var_16d,
        "simple_var_6d": build_simple_var_6d,
        "simple_var_sd": build_simple_var_sam_decoder,
        "simple_var_sd_mlp_adapter": build_simple_var_sam_decoder_mlp_adapter,
    },
    "simple_mask_ar": {
        "simple_mask_ar": build_simple_mask_ar,
        "simple_mask_var": build_simple_mask_var,
        "simple_query_token_ar": build_simple_query_token_ar,
    },
    "simple_mask_vae": {
        "simple_mask_vae_v2_dim384": build_simple_mask_vae_v2_dim384,
    },
    "simple_mask_diffusion": {
        "simple_mask_latent_dit": build_simple_mask_latent_dit,
    },
    "dataset": {
        "cocolvis": build_cocolvis_dataset,
        "hqseg44k": build_hqseg44k_dataset,
        "coconut_hf": build_coconut_hf_dataset,
    }
}
