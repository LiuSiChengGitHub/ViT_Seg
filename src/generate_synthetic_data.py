"""
生成示例数据集
用于在没有真实数据时测试代码是否能正常运行

这个脚本会生成模拟的脑部MRI图像和对应的分割掩码
支持二分类和多分类（4类脑肿瘤分割）

多分类标签说明（参考BraTS数据集）：
- 0: 背景
- 1: 肿瘤核心 (Tumor Core)
- 2: 强化区域 (Enhancing Tumor)  
- 3: 水肿区域 (Edema)
"""

import os
import numpy as np
from PIL import Image
import random


def generate_synthetic_brain_mri(
    size: int = 256,
    add_tumor: bool = True
) -> tuple:
    """
    生成模拟的脑部MRI图像
    
    Args:
        size: 图像尺寸
        add_tumor: 是否添加肿瘤区域
    
    Returns:
        (image, mask) - 图像和掩码的numpy数组
    """
    # 创建背景
    image = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.float32)
    
    # 网格坐标
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    # 脑部轮廓 (椭圆)
    a = size // 2.5  # 水平半径
    b = size // 2.2  # 垂直半径
    brain_mask = ((x - center) / a) ** 2 + ((y - center) / b) ** 2 < 1
    
    # 添加脑部纹理
    noise = np.random.rand(size, size) * 0.15
    brain_intensity = 0.5 + noise
    image[brain_mask] = brain_intensity[brain_mask]
    
    # 添加脑室 (中间暗区)
    ventricle_a = size // 12
    ventricle_b = size // 8
    ventricle_mask = ((x - center) / ventricle_a) ** 2 + ((y - center + 10) / ventricle_b) ** 2 < 1
    image[ventricle_mask] = 0.2 + np.random.rand() * 0.1
    
    # 添加肿瘤区域
    if add_tumor and random.random() > 0.3:  # 70% 概率有肿瘤
        # 随机位置 (在脑部区域内)
        offset_x = random.randint(-size//6, size//6)
        offset_y = random.randint(-size//6, size//6)
        tumor_x = center + offset_x
        tumor_y = center + offset_y
        
        # 随机大小
        tumor_r = random.randint(size//20, size//8)
        
        # 不规则形状的肿瘤
        tumor_base = (x - tumor_x) ** 2 + (y - tumor_y) ** 2 < tumor_r ** 2
        
        # 添加一些不规则性
        noise_mask = np.random.rand(size, size) > 0.3
        tumor_mask = tumor_base & noise_mask & brain_mask
        
        # 肿瘤更亮
        image[tumor_mask] = 0.7 + np.random.rand() * 0.2
        mask[tumor_mask] = 1.0
    
    # 归一化到 0-255
    image = (image * 255).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    
    return image, mask


def generate_multiclass_brain_mri(
    size: int = 256,
    add_tumor: bool = True
) -> tuple:
    """
    生成多分类的模拟脑部MRI图像（4类标签）
    
    标签说明:
    - 0: 背景
    - 1: 肿瘤核心 (Tumor Core)
    - 2: 强化区域 (Enhancing Tumor)
    - 3: 水肿区域 (Edema)
    
    Args:
        size: 图像尺寸
        add_tumor: 是否添加肿瘤区域
    
    Returns:
        (image, mask) - 图像和多分类掩码的numpy数组
    """
    # 创建背景
    image = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)  # 多分类用整数
    
    # 网格坐标
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    # 脑部轮廓 (椭圆)
    a = size // 2.5  # 水平半径
    b = size // 2.2  # 垂直半径
    brain_mask = ((x - center) / a) ** 2 + ((y - center) / b) ** 2 < 1
    
    # 添加脑部纹理
    noise = np.random.rand(size, size) * 0.15
    brain_intensity = 0.5 + noise
    image[brain_mask] = brain_intensity[brain_mask]
    
    # 添加脑室 (中间暗区)
    ventricle_a = size // 12
    ventricle_b = size // 8
    ventricle_mask = ((x - center) / ventricle_a) ** 2 + ((y - center + 10) / ventricle_b) ** 2 < 1
    image[ventricle_mask] = 0.2 + np.random.rand() * 0.1
    
    # 添加多区域肿瘤
    if add_tumor and random.random() > 0.2:  # 80% 概率有肿瘤
        # 随机位置 (在脑部区域内)
        offset_x = random.randint(-size//6, size//6)
        offset_y = random.randint(-size//6, size//6)
        tumor_x = center + offset_x
        tumor_y = center + offset_y
        
        # 水肿区域 (最外层, 标签3) - 最大
        edema_r = random.randint(size//8, size//5)
        edema_base = (x - tumor_x) ** 2 + (y - tumor_y) ** 2 < edema_r ** 2
        edema_mask = edema_base & brain_mask
        image[edema_mask] = 0.55 + np.random.rand() * 0.1
        mask[edema_mask] = 3  # 水肿
        
        # 肿瘤核心 (中间层, 标签1)
        core_r = int(edema_r * 0.6)
        core_base = (x - tumor_x) ** 2 + (y - tumor_y) ** 2 < core_r ** 2
        core_mask = core_base & brain_mask
        image[core_mask] = 0.7 + np.random.rand() * 0.15
        mask[core_mask] = 1  # 肿瘤核心
        
        # 强化区域 (最内层, 标签2) - 不规则形状
        enhance_r = int(core_r * 0.5)
        enhance_base = (x - tumor_x) ** 2 + (y - tumor_y) ** 2 < enhance_r ** 2
        # 添加不规则性
        noise_mask = np.random.rand(size, size) > 0.35
        enhance_mask = enhance_base & noise_mask & brain_mask
        image[enhance_mask] = 0.85 + np.random.rand() * 0.1
        mask[enhance_mask] = 2  # 强化区域
    
    # 归一化图像到 0-255
    image = (image * 255).astype(np.uint8)
    # mask 保持为类别索引 (0, 1, 2, 3)
    
    return image, mask


def generate_dataset(
    output_dir: str,
    num_samples: int = 100,
    size: int = 256
):
    """
    生成完整的示例数据集（二分类）
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
        size: 图像尺寸
    """
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"生成示例数据集（二分类）...")
    print(f"  输出目录: {output_dir}")
    print(f"  样本数量: {num_samples}")
    print(f"  图像尺寸: {size}x{size}")
    
    for i in range(num_samples):
        # 生成图像和掩码
        image, mask = generate_synthetic_brain_mri(size, add_tumor=True)
        
        # 转换为RGB (某些预训练模型需要)
        image_rgb = np.stack([image, image, image], axis=-1)
        
        # 保存
        patient_name = f"Patient_{i+1:04d}"
        
        # 图像
        img_path = os.path.join(output_dir, f"{patient_name}.tif")
        Image.fromarray(image_rgb).save(img_path)
        
        # 掩码
        mask_path = os.path.join(output_dir, f"{patient_name}_mask.tif")
        Image.fromarray(mask).save(mask_path)
        
        if (i + 1) % 20 == 0:
            print(f"  生成进度: {i+1}/{num_samples}")
    
    print(f"\n数据集生成完成!")
    print(f"  图像文件: {num_samples} 个")
    print(f"  掩码文件: {num_samples} 个")


def generate_multiclass_dataset(
    output_dir: str,
    num_samples: int = 100,
    size: int = 256
):
    """
    生成多分类示例数据集（4类脑肿瘤分割）
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
        size: 图像尺寸
    """
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"生成多分类数据集（4类脑肿瘤）...")
    print(f"  输出目录: {output_dir}")
    print(f"  样本数量: {num_samples}")
    print(f"  图像尺寸: {size}x{size}")
    print(f"  标签类别: 0=背景, 1=肿瘤核心, 2=强化区域, 3=水肿")
    
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for i in range(num_samples):
        # 生成图像和多分类掩码
        image, mask = generate_multiclass_brain_mri(size, add_tumor=True)
        
        # 统计标签
        for label in range(4):
            if np.any(mask == label):
                label_counts[label] += 1
        
        # 转换为RGB (某些预训练模型需要)
        image_rgb = np.stack([image, image, image], axis=-1)
        
        # 保存
        patient_name = f"Patient_{i+1:04d}"
        
        # 图像
        img_path = os.path.join(output_dir, f"{patient_name}.tif")
        Image.fromarray(image_rgb).save(img_path)
        
        # 掩码 (保存为PNG以保持整数值)
        mask_path = os.path.join(output_dir, f"{patient_name}_mask.png")
        Image.fromarray(mask).save(mask_path)
        
        if (i + 1) % 20 == 0:
            print(f"  生成进度: {i+1}/{num_samples}")
    
    print(f"\n多分类数据集生成完成!")
    print(f"  图像文件: {num_samples} 个")
    print(f"  掩码文件: {num_samples} 个")
    print(f"  标签分布:")
    for label, count in label_counts.items():
        print(f"    类别 {label}: {count} 张图像包含该标签")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成示例MRI数据集')
    parser.add_argument('--output_dir', type=str, 
                        default='./data/synthetic_mri',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='样本数量')
    parser.add_argument('--size', type=int, default=256,
                        help='图像尺寸')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--multi_class', action='store_true',
                        help='生成多分类数据集（4类脑肿瘤）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.multi_class:
        generate_multiclass_dataset(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            size=args.size
        )
    else:
        generate_dataset(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            size=args.size
        )


if __name__ == '__main__':
    main()

