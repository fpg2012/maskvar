import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import os
import argparse
import cv2
from datasets.instance_info import InstanceInfo

class HQSeg44KTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_root='datasets/sam-hq', transform=None, img_size=(256, 256)):
        self.data_root = Path(data_root)
        self.transform = transform
        self.data_list = []
        self.img_size = img_size
        
        # 定义子目录
        self.subdirs = [
            'DIS5K/DIS-TR',
            'thin_object_detection/ThinObject5K',
            'cascade_psp/fss_all',
            'cascade_psp/DUTS-TR',
            'cascade_psp/DUTS-TE',
            'cascade_psp/ecssd',
            'cascade_psp/MSRA_10K'
        ]
        
        # 加载数据列表
        self._load_data_list()
        
    def _load_data_list(self):
        for subdir in self.subdirs:
            img_files = {
                p.stem: p for p in
                (self.data_root / subdir).rglob('*.jpg')
                if 'images_test' not in str(p) and
                   'masks_test' not in str(p)}
            ann_files = {
                p.stem: p for p in
                (self.data_root / subdir).rglob('*.png')
                if 'images_test' not in str(p) and
                   'masks_test' not in str(p)}
            
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            for prefix in sorted(img_files.keys()):
                if prefix in prefixes:
                    self.data_list.append({
                        'img_path': str(img_files[prefix]),
                        'mask_path': str(ann_files[prefix])
                    })
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 加载图像
        with Image.open(item['img_path']) as img:
            img = img.convert('RGB')  # 确保图像为RGB格式
            img = np.array(img)  # 形状为 (H, W, 3)
        
        # 加载掩码
        with Image.open(item['mask_path']) as mask:
            mask = np.array(mask)
        
        # 确保掩码是二值的
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)  # 形状为 (H, W)
        
        # 裁剪最大正方形区域（图像和掩码同步裁剪）
        size = min(img.shape[0], img.shape[1])  # 获取最小边长
        img = img[:size, :size, :]  # 裁剪图像
        mask = mask[:size, :size]  # 裁剪掩码
        
        # 缩放到256x256（图像和掩码同步缩放）
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)  # 形状为 (256, 256, 3)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)  # 形状为 (256, 256)

        mask = np.expand_dims(mask, axis=-1)  # 形状为 (256, 256, 1)
        
        # 包装返回值以匹配 coco_lvis.py 的格式
        # 视为只有一个实例，实例ID为0
        instances_info = {
            0: InstanceInfo(
                mapping=(0, 1),  # 掩码在第0层，掩码ID为1
                parent=None,      # 无父实例
                children=[],      # 无子实例
                node_level=0      # 实例层级
            )
        }
        if self.transform:
            result = self.transform((img, mask, instances_info))
            return result
        return img, mask, instances_info  # 格式: (image, layers, instances_info)
    
class HQSeg44KTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root='datasets/sam-hq', transform=None, img_size=(256, 256)):
        """
        初始化测试数据集。

        Args:
            data_root (str): 数据根目录。
            transform (callable, optional): 数据预处理函数。
        """
        self.data_root = Path(data_root)
        self.data_list = []
        
        # 定义子目录
        self.subdirs = [
            'thin_object_detection/COIFT',
            'thin_object_detection/HRSOD',
            'thin_object_detection/ThinObject5K/',
            'DIS5K/DIS-VD'
        ]
        self.img_size = img_size
        
        # 加载数据列表
        self._load_data_list()
    
    def _load_data_list(self):
        """
        加载测试集数据列表，过滤掉训练集和验证集的数据。
        """
        for subdir in self.subdirs:
            img_files = {
                p.stem: p for p in
                (self.data_root / subdir).rglob('*.jpg')
                if 'train' not in str(p) and 'val' not in str(p)}
            ann_files = {
                p.stem: p for p in
                (self.data_root / subdir).rglob('*.png')
                if 'train' not in str(p) and 'val' not in str(p)}
            
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            for prefix in tqdm(sorted(img_files.keys()), desc=f"Loading {subdir}"):
                if prefix in prefixes:
                    self.data_list.append({
                        'img_path': str(img_files[prefix]),
                        'mask_path': str(ann_files[prefix])
                    })
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        获取单个样本。

        Args:
            idx (int): 样本索引。

        Returns:
            torch.Tensor: 单通道掩码图像，形状为 [1, H, W]。
        """
        item = self.data_list[idx]
        
        # 加载图像
        with Image.open(item['img_path']) as img:
            img = img.convert('RGB')  # 确保图像为RGB格式
            img = np.array(img)  # 形状为 (H, W, 3)
        
        # 加载掩码
        with Image.open(item['mask_path']) as mask:
            mask = np.array(mask)
        
        # 确保掩码是二值的
        # if len(mask.shape) == 3:
        #     mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)  # 形状为 (H, W)
        
        # 裁剪最大正方形区域（图像和掩码同步裁剪）
        size = min(img.shape[0], img.shape[1])  # 获取最小边长
        img = img[:size, :size, :]  # 裁剪图像
        mask = mask[:size, :size]  # 裁剪掩码
        
        # 缩放到256x256（图像和掩码同步缩放）
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)  # 形状为 (256, 256, 3)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)  # 形状为 (256, 256)

        mask = np.expand_dims(mask, axis=-1)  # 形状为 (256, 256, 1)
        
        # 包装返回值以匹配 coco_lvis.py 的格式
        # 视为只有一个实例，实例ID为0
        instances_info = {
            0: InstanceInfo(
                mapping=(0, 1),  # 掩码在第0层，掩码ID为1
                parent=None,      # 无父实例
                children=[],      # 无子实例
                node_level=0      # 实例层级
            )
        }
        return img, mask, instances_info  # 格式: (image, layers, instances_info)