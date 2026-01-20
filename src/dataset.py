"""
Brain MRI Segmentation Dataset
用于加载脑部MRI图像和分割掩码的数据集类

参考文档要求:
- 图像尺寸: 256x256
- 归一化: [0, 1]
- 数据增强: 随机旋转、水平翻转、对比度调整
- 数据划分: 7:3 (训练:验证)
"""

import os
import glob
import random
import numpy as np
from PIL import Image
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class BrainMRIDataset(Dataset):
    """
    脑部MRI分割数据集
    
    支持两种数据格式:
    1. Kaggle LGG-MRI-Segmentation: 图像和mask在同一目录，mask文件名包含'_mask'
    2. 通用格式: 图像和mask在不同目录
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
        augmentation: bool = True,
        mode: str = 'train'
    ):
        """
        Args:
            image_dir: 图像目录路径
            mask_dir: 掩码目录路径（如果为None，假设mask在同一目录）
            image_size: 输出图像尺寸 (H, W)
            transform: 自定义变换函数
            augmentation: 是否进行数据增强
            mode: 'train' 或 'val'
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform
        self.augmentation = augmentation and (mode == 'train')
        self.mode = mode
        
        # 收集图像路径
        self.image_paths = []
        self.mask_paths = []
        
        self._load_data_paths()
        
        print(f"[{mode.upper()}] 加载了 {len(self.image_paths)} 张图像")
    
    def _load_data_paths(self):
        """加载数据路径"""
        # 支持多种图像格式
        image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
        
        if self.mask_dir is None:
            # Kaggle LGG-MRI格式: 图像和mask在同一目录
            # 图像名: TCGA_CS_4941_19960909_1.tif
            # 掩码名: TCGA_CS_4941_19960909_1_mask.tif
            all_files = []
            for ext in image_extensions:
                all_files.extend(glob.glob(os.path.join(self.image_dir, '**', ext), recursive=True))
            
            # 分离图像和掩码
            mask_files = set([f for f in all_files if '_mask' in os.path.basename(f)])
            image_files = [f for f in all_files if f not in mask_files and '_mask' not in os.path.basename(f)]
            
            for img_path in image_files:
                # 构造对应的mask路径
                base, ext = os.path.splitext(img_path)
                mask_path = base + '_mask' + ext
                
                if mask_path in mask_files or os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
        else:
            # 通用格式: 图像和mask在不同目录
            for ext in image_extensions:
                image_files = glob.glob(os.path.join(self.image_dir, ext))
                for img_path in image_files:
                    basename = os.path.basename(img_path)
                    # 尝试找到对应的mask
                    for mask_ext in image_extensions:
                        mask_name = os.path.splitext(basename)[0] + mask_ext.replace('*', '')
                        mask_path = os.path.join(self.mask_dir, mask_name)
                        if os.path.exists(mask_path):
                            self.image_paths.append(img_path)
                            self.mask_paths.append(mask_path)
                            break
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        # Resize到指定尺寸
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)
        
        # 转换为numpy数组
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        
        # 数据增强
        if self.augmentation:
            image, mask = self._augment(image, mask)
        
        # 归一化图像到 [0, 1]
        image = image / 255.0
        
        # 二值化mask
        mask = (mask > 127).astype(np.float32)
        
        # 自定义变换
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # 转换为PyTorch张量
        # 图像: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        # 掩码: (H, W) -> (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask
    
    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据增强
        - 随机水平翻转
        - 随机垂直翻转
        - 随机旋转 (90度倍数)
        - 随机对比度调整
        """
        # 随机水平翻转
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        
        # 随机垂直翻转
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        
        # 随机90度旋转
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        
        # 随机对比度调整 (仅对图像)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = image.mean()
            image = (image - mean) * factor + mean
            image = np.clip(image, 0, 255)
        
        return image, mask


def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    image_size: Tuple[int, int] = (256, 256),
    train_ratio: float = 0.7,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    获取训练和验证数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批量大小
        image_size: 图像尺寸
        train_ratio: 训练集比例 (默认0.7，即7:3划分)
        num_workers: 数据加载线程数
        seed: 随机种子
    
    Returns:
        (train_loader, val_loader)
    """
    # 设置随机种子
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建完整数据集（不带增强，用于划分）
    full_dataset = BrainMRIDataset(
        image_dir=data_dir,
        image_size=image_size,
        augmentation=False,
        mode='train'
    )
    
    # 划分训练集和验证集
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    # 获取索引
    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 创建训练数据集（带增强）
    train_dataset = BrainMRIDataset(
        image_dir=data_dir,
        image_size=image_size,
        augmentation=True,
        mode='train'
    )
    
    # 创建验证数据集（不带增强）
    val_dataset = BrainMRIDataset(
        image_dir=data_dir,
        image_size=image_size,
        augmentation=False,
        mode='val'
    )
    
    # 使用Subset创建划分后的数据集
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"数据集划分: 训练集 {len(train_subset)} 张, 验证集 {len(val_subset)} 张")
    
    return train_loader, val_loader


# 测试代码
if __name__ == '__main__':
    # 测试数据集
    print("测试数据加载器...")
    
    # 这里需要替换为实际的数据路径
    data_dir = './data/lgg-mri-segmentation'
    
    if os.path.exists(data_dir):
        train_loader, val_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            image_size=(256, 256)
        )
        
        # 打印一个批次的信息
        for images, masks in train_loader:
            print(f"图像批次形状: {images.shape}")
            print(f"掩码批次形状: {masks.shape}")
            print(f"图像值范围: [{images.min():.3f}, {images.max():.3f}]")
            print(f"掩码唯一值: {torch.unique(masks)}")
            break
    else:
        print(f"数据目录不存在: {data_dir}")
        print("请先下载数据集到该目录")
