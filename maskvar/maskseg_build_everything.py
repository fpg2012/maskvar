import torch
from functools import partial
from typing import Optional, Tuple

from .models.vqvae_single import VQVAE_Single
# from .models.maskgit import MaskGIT
# from .models.maskseg import MaskSeg
from .models.flex_maskvar import FlexMaskVAR
from .models.flex_maskvar_simple import FlexMaskVARSimple
from .models.sam import ImageEncoderViT as SamImageEncoder
from .models.sam import PromptEncoder
from .models.image_encoder import ImageEncoder, VarImageEncoder, NeckFPN
from .models.maskvar import MaskVAR
from .models.tinyvit import TinyViT
from .models.simple_ar import SimpleAR, SimpleVAR

from .datasets.mask_level_dataset import MaskLevelDataset
from .datasets.coco_lvis import LvisDataset
from .datasets.hqseg44k import HQSeg44KTestDataset, HQSeg44KTrainDataset


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

def build_simple_var(simple_var_checkpoint_path: Optional[str] = None, device: str = 'cpu') -> SimpleVAR:
    simple_var = SimpleVAR(
        dim=256,
        depth=2,
        vocab_size=4096,
        device=device,
        patch_num=[1, 8, 16, 24, 32],
        num_heads=4,
        vqvae_dim=32,
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

def build_simple_var_6d(simple_var_checkpoint_path: Optional[str] = None, device: str = 'cpu') -> SimpleVAR:
    simple_var = SimpleVAR(
        dim=256,
        depth=6,
        vocab_size=4096,
        device=device,
        patch_num=[1, 8, 16, 24, 32],
        num_heads=4,
        vqvae_dim=32,
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

def build_simple_var_16d(simple_var_checkpoint_path: Optional[str] = None, device: str = 'cpu') -> SimpleVAR:
    simple_var = SimpleVAR(
        dim=256,
        depth=16,
        vocab_size=4096,
        device=device,
        patch_num=[1, 8, 16, 24, 32],
        num_heads=4,
        vqvae_dim=32,
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
    "image_encoder": {
        "sam_vitb": build_sam_image_encoder,
        "mobile_sam": build_mobile_sam_image_encoder,
    },
    "prompt_encoder": build_prompt_encoder,
    "simple_ar": build_simple_ar,
    "simple_var": {
        "simple_var": build_simple_var,
        "simple_var_16d": build_simple_var_16d,
        "simple_var_6d": build_simple_var_6d,
    },
    "dataset": {
        "cocolvis": build_cocolvis_dataset,
        "hqseg44k": build_hqseg44k_dataset,
    }
}