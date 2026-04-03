# %%
import torch
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from maskvar.models.vqvae_single import VQVAE_Single
from maskvar.datasets import MaskLevelDatasetDummy
from maskvar.datasets.image_feature_cache import ImageFeatureCache
from maskvar.models.simple_ar import (
    SimpleVAR,
    simple_var_inference,
    simple_var_train_pass
)
from maskvar.maskseg_build_everything import (
    build_coconut_hf_dataset,
    build_simple_var,
    build_vqvae_single_5_stages_v1,
    build_prompt_encoder,
)
from maskvar.utils.clicker import init_clicks, to_sam_format

device = 'cuda:3'

# %%
# simple_var: SimpleVAR = build_simple_var(simple_var_checkpoint_path='../out/simple_var_1_debug/checkpoints/.simple_var.200.pt', device=device)

simple_var: SimpleVAR = build_simple_var(simple_var_checkpoint_path='../out/ddp_simple_var_coconut_v0_lr2e-3_bs32_sampe_clicks/checkpoints/.simple_var.286272.pt', device=device, enable_prompt_tokens=True)

vqvae: VQVAE_Single = build_vqvae_single_5_stages_v1('../out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth', require_grad=False)
vqvae = vqvae.to(device)

prompt_encoder = build_prompt_encoder('../ckpt/sam_vit_b_01ec64.pth').to(device)
prompt_encoder.eval()

# %%
def visualize(indices, ax, device='cpu', name='mask'):
    result = vqvae.idxBl_to_img(indices, same_shape=True)

    # for i in range(len(indices)):
    #     print(f'index {i}: {indices[i].shape}')
    # result_conv = [edge(item) for item in result]
    result = [mask for mask in result]
    chw = torchvision.utils.make_grid(torch.cat(result, dim=0), nrow=3, padding=1, pad_value=1.0)

    chw = chw.permute(1, 2, 0).cpu().numpy()
    ax.imshow(chw[:, :, 0] > 0)
    ax.axis('off')
    ax.set_title(name)

# %%
image_feature_cache_train = ImageFeatureCache(
    cache_dir=Path('../data/cache'),
    dataset='coconut_hf_train',
    model_name='sam_vitb',
)

image_feature_cache_val = ImageFeatureCache(
    cache_dir=Path('../data/cache'),
    dataset='coconut_hf_val',
    model_name='sam_vitb',
)

train_set, val_set = build_coconut_hf_dataset('../data/coconut_hf') # validate on train set
# train_set_masklevel = MaskLevelDatasetDummy(
#     dataset=train_set,
#     # sam_encoder=sam_image_encoder,
#     image_feature_cache=image_feature_cache_train,
#     with_image_embed=True,
#     device=device,
#     mask_filter_thresh=0.1,
#     seed=42,
#     count=5,
# )
val_set_masklevel = MaskLevelDatasetDummy(
    dataset=val_set,
    # sam_encoder=sam_image_encoder,
    image_feature_cache=image_feature_cache_val,
    with_image_embed=True,
    device=device,
    mask_filter_thresh=0.1,
    seed=42,
    count=5,
)

train_dataloader = DataLoader(val_set_masklevel, batch_size=1, shuffle=False, drop_last=True)

# %%
data_iter = iter(train_dataloader)
# _ = next(data_iter)

# %%
image, image_embed_sam, single_mask_normalized, single_mask = next(data_iter)
print("image.shape:", image.shape)
print("image_embed_sam.shape:", image_embed_sam.shape)
print("single_mask_normalized.shape:", single_mask_normalized.shape)
print("single_mask.shape:", single_mask.shape)

# %%
use_prompt_embedding = True
if use_prompt_embedding:
    # Generate 2 initial positive clicks from ground truth mask
    # single_mask is (B, 1, H, W), take first sample and squeeze
    mask_np = single_mask[0, 0].cpu().numpy()
    click_list, eroded_mask, dt = init_clicks(
        gt_mask=mask_np,
        num_random_clicks=2,
        random_sample=True
    )

    if len(click_list) > 0:
        # Convert clicks to SAM format
        coords, labels = to_sam_format(click_list, pad_size=4, device=device)
        # Add batch dimension: (N, 2) -> (1, N, 2), (N,) -> (1, N)
        coords = coords.unsqueeze(0)
        labels = labels.unsqueeze(0)

        # Get sparse embeddings from prompt encoder
        with torch.no_grad():
            sparse_embeddings, _ = prompt_encoder(
                points=(coords, labels),
                boxes=None,
                masks=None
            )
    else:
        print("Warning: No clicks generated (empty mask)")
        sparse_embeddings = None
else:
    sparse_embeddings = None

# %%
# inference
id_seq = simple_var_inference(
    image_feat=image_embed_sam.to(device), 
    simple_var=simple_var, 
    vqvae=vqvae,
    sparse_embeddings=sparse_embeddings
)
print("Inference with sparse embeddings")


# gt
idx = vqvae.img_to_idxBl(single_mask_normalized.to(device))

# inference with teacher forced input
logits = simple_var_train_pass(
    idx, 
    image_feat=image_embed_sam.to(device), 
    simple_var=simple_var, 
    vqvae=vqvae,
    sparse_embeddings=sparse_embeddings
)

# sample the max token
id_seq_teach = logits.argmax(dim=-1)
id_seq_teach_Bl = []
start_pos = 0
for pn in simple_var.patch_num:
    end_pos = start_pos + pn * pn
    id_seq_teach_Bl.append(id_seq_teach[:, start_pos:end_pos])
    start_pos = end_pos

# %%
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

visualize(idx, ax[0], name='gt')
visualize(id_seq, ax[1], name='pred')
visualize(id_seq_teach_Bl, ax[2], name='pred w/ teacher input')
plt.savefig('output_masks.png', dpi=150, bbox_inches='tight')
print("Saved: output_masks.png")

# %%
# Visualize clicks on the mask (only if use_prompt_embedding is True and clicks were generated)
if use_prompt_embedding and 'click_list' in dir() and len(click_list) > 0:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    # Original mask
    single_mask_np = single_mask[0, 0].cpu().numpy()
    ax[0].imshow(single_mask_np, cmap='gray')
    ax[0].set_title('Original Mask')
    ax[0].axis('off')

    # Distance transform (used for click sampling)
    im1 = ax[1].imshow(dt, cmap='hot')
    ax[1].set_title('Distance Transform')
    ax[1].axis('off')
    plt.colorbar(im1, ax=ax[1])

    # Mask with clicks overlay
    ax[2].imshow(single_mask_np, cmap='gray', alpha=0.7)
    for i, (y, x, label) in enumerate(click_list):
        color = 'green' if label == 1 else 'red'
        ax[2].scatter(x, y, c=color, s=100, marker='x', linewidths=2)
        ax[2].annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                       color=color, fontsize=12, fontweight='bold')
    ax[2].set_title(f'Mask with {len(click_list)} Clicks')
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig('output_clicks.png', dpi=150, bbox_inches='tight')
    print("Saved: output_clicks.png")
else:
    print("Skipping click visualization (no clicks generated or use_prompt_embedding=False)")


