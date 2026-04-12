"""
DINOv3 模型测试脚本
使用随机生成的图像数据测试 DINOv3 模型
"""

import torch
import timm
from safetensors.torch import load_file


def load_dinov3_model(model_name: str, checkpoint_path: str, device: str = "cuda"):
    """
    使用 timm 创建 DINOv3 模型并加载本地 safetensors 权重

    Args:
        model_name: timm 模型名称，如 'vit_small_patch16_dinov3' 或 'vit_base_patch16_dinov3'
        checkpoint_path: 本地 safetensors 权重文件路径
        device: 运行设备

    Returns:
        model: 加载好权重的模型
        transforms: 模型对应的预处理 transforms
    """
    # 创建模型结构（不加载预训练权重）
    model = timm.create_model(
        model_name,
        pretrained=False,  # 不从网络加载权重
        num_classes=0,     # 移除分类头
    )

    # 加载本地 safetensors 权重
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)

    model = model.eval().to(device)

    # 获取模型特定的预处理配置
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    print(f"Loaded {model_name} from {checkpoint_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model, transforms


def test_dinov3_with_random_image(model_name: str, checkpoint_path: str, device: str = "cuda"):
    """
    使用随机生成的图像测试 DINOv3 模型

    Args:
        model_name: timm 模型名称
        checkpoint_path: 本地权重路径
        device: 运行设备
    """
    model, transforms = load_dinov3_model(model_name, checkpoint_path, device)

    print(f'transforms: {transforms}')

    # 生成随机图像数据 (H, W, C) - uint8 格式，范围 [0, 255]
    # DINOv3 使用 256x256 输入
    random_image = torch.randint(0, 256, (3, 256, 256), dtype=torch.float) / 255
    print(f"\nRandom image shape: {random_image.shape}, dtype: {random_image.dtype}")

    # 应用预处理（转换为 tensor 并归一化）
    # transforms 期望 PIL Image 或 tensor，这里我们直接构造 tensor
    input_tensor = transforms(random_image).unsqueeze(0).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")

    with torch.no_grad():
        # 方式1: forward_features - 获取未池化的特征
        features = model.forward_features(input_tensor)
        print(f"Forward features shape: {features.shape}")  # (1, num_patches+1, embed_dim)

        # 方式2: forward_head - 获取池化后的特征
        pooled_features = model.forward_head(features, pre_logits=True)
        print(f"Pooled features shape: {pooled_features.shape}")  # (1, embed_dim)

        # 方式3: 直接 forward - 等同于 forward_head(forward_features(x), pre_logits=True)
        output = model(input_tensor)
        print(f"Direct forward output shape: {output.shape}")

    return features, pooled_features


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # 配置：模型名称和对应的 checkpoint 路径
    models_config = {
        "vit_small": {
            "model_name": "vit_small_patch16_dinov3",
            "checkpoint": "/data/clc/maskseg/ckpt/dino_v3_vits.safetensors"
        },
        "vit_base": {
            "model_name": "vit_base_patch16_dinov3",
            "checkpoint": "/data/clc/maskseg/ckpt/dino_v3_vitb.safetensors"
        }
    }

    # 测试 vit_small
    print("=" * 50)
    print("Testing DINOv3 ViT-Small")
    print("=" * 50)
    test_dinov3_with_random_image(
        models_config["vit_small"]["model_name"],
        models_config["vit_small"]["checkpoint"],
        device
    )

    # 测试 vit_base
    print("\n" + "=" * 50)
    print("Testing DINOv3 ViT-Base")
    print("=" * 50)
    test_dinov3_with_random_image(
        models_config["vit_base"]["model_name"],
        models_config["vit_base"]["checkpoint"],
        device
    )

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
