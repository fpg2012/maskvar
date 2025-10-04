from maskseg_build_everything import build_maskvar, build_cocolvis_dataset, build_prompt_encoder, build_maskvar_v2
from utils.timer import profile_timer
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler, record_function
from torch.utils.data import DataLoader
from datasets.mask_level_dataset import MaskLevelDataset
import tensorboard
from utils.clicker import to_sam_format
import torch

import argparse

def profile_forward(vqvae, maskvar, prompt_encoder, args):
    train_set, val_set = build_cocolvis_dataset()
    train_set_masklevel = MaskLevelDataset(train_set, sam_image_encoder, device)

    train_dataloader = DataLoader(train_set_masklevel, batch_size=batch_size)

    label_B = torch.zeros(batch_size, dtype=torch.long, device=device)

    click_list = [(100, 100, 1), (200, 200, 1), (10, 330, -1), (23, 89, 1), (23, 300, -1)]
    coords, label = to_sam_format(click_list, pad_size=10)
    coords = coords.to(device)
    label = label.to(device)
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1)
    label = label.unsqueeze(0).repeat(batch_size, 1)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 profile_memory=True,
                 record_shapes=True,
                 schedule=schedule(wait=1, warmup=3, active=2, repeat=2), 
                 on_trace_ready=tensorboard_trace_handler(f"./logs/{args.exp_name}_forward")
    ) as prof:
        for steps, (image, image_embed, gt_mask_normalized, gt_mask) in enumerate(train_dataloader):
            with record_function("vqvae"):
                gt_idx_Bl = vqvae.img_to_idxBl(gt_mask_normalized)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            with record_function("vqvae_quantize"):
                x_BLCv_wo_first_l = vqvae.quantize.idxBl_to_var_input(gt_idx_Bl)
            with record_function("maskvar"):
                logits_BLV = maskvar(
                    label_B, 
                    x_BLCv_wo_first_l.detach(), 
                    image_embed.detach(), 
                    points_coords=coords, points_labels=label
                )
            prof.step()
            if steps >= 10:
                break
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

def profile_autogressive(vqvae, maskvar, prompt_encoder, args):
    train_set, val_set = build_cocolvis_dataset()
    train_set_masklevel = MaskLevelDataset(train_set, sam_image_encoder, device)

    train_dataloader = DataLoader(train_set_masklevel, batch_size=batch_size)

    label_B = torch.zeros(batch_size, dtype=torch.long, device=device)

    click_list = [(100, 100, 1), (200, 200, 1), (10, 330, -1), (23, 89, 1), (23, 300, -1)]
    coords, label = to_sam_format(click_list, pad_size=10)
    coords = coords.to(device)
    label = label.to(device)
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1)
    label = label.unsqueeze(0).repeat(batch_size, 1)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 profile_memory=True,
                 record_shapes=True,
                 schedule=schedule(wait=1, warmup=3, active=2, repeat=2), 
                 on_trace_ready=tensorboard_trace_handler(f"./logs/{args.exp_name}_autogressive")
    ) as prof:
        for steps, (image, image_embed, gt_mask_normalized, gt_mask) in enumerate(train_dataloader):
            with record_function("vqvae"):
                gt_idx_Bl = vqvae.img_to_idxBl(gt_mask_normalized)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            with record_function("vqvae_quantize"):
                x_BLCv_wo_first_l = vqvae.quantize.idxBl_to_var_input(gt_idx_Bl)
            with record_function("maskvar"):
                logits_BLV = maskvar.autoregressive_infer_cfg(
                    B=batch_size,
                    label_B=None,
                    sam_image_embedding=image_embed.detach(),
                    points_coords=coords, points_labels=label
                )
            prof.step()
            if steps >= 10:
                break
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--exp_name", type=str)
    args = parser.parse_args()

    device = args.device
    batch_size = args.batch_size

    # vqvae, maskvar, sam_image_encoder = build_maskvar(
    #     vqvae_checkpoint_path="ckpt/vqvae_single.pth",
    #     sam_checkpoint_path="ckpt/sam_vit_b_01ec64.pth",
    #     flash_if_available=True,
    #     device=device
    # )
    
    vqvae, maskvar, sam_image_encoder = build_maskvar_v2(
        vqvae_checkpoint_path="out_vqvae_4_stages_2/ckpt/vqvae_single_epoch_40.pth",
        sam_checkpoint_path="ckpt/sam_vit_b_01ec64.pth",
        flash_if_available=True,
        device=device
    )
    prompt_encoder = build_prompt_encoder("ckpt/sam_vit_b_01ec64.pth").to(device)

    profile_forward(vqvae, maskvar, prompt_encoder, args)
    profile_autogressive(vqvae, maskvar, prompt_encoder, args)
    

    

            