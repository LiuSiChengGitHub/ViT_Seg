"""
Inference and Visualization Script
推理和可视化脚本

参考文档要求:
- 加载训练好的模型
- 生成对比图: 原始MRI图像 / Ground Truth Mask / 预测Mask
"""

import os
import sys
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import get_model
from dataset import BrainMRIDataset


def load_model(
    checkpoint_path: str,
    model_name: str = 'transunet_lite',
    num_classes: int = 1,
    img_size: int = 256,
    device: torch.device = None
) -> torch.nn.Module:
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 检查点路径
        model_name: 模型名称
        num_classes: 类别数
        img_size: 图像尺寸
        device: 设备
    
    Returns:
        加载权重后的模型
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(model_name, num_classes, img_size)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型: {checkpoint_path}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Dice: {checkpoint.get('dice', 'N/A'):.4f}")
    else:
        print(f"警告: 检查点不存在 {checkpoint_path}")
        print("使用随机初始化的模型")
    
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5
) -> np.ndarray:
    """
    对单张图像进行预测
    
    Args:
        model: 模型
        image: 输入图像 (C, H, W)
        device: 设备
        threshold: 二值化阈值
    
    Returns:
        预测掩码 (H, W)
    """
    model.eval()
    
    # 添加batch维度
    image = image.unsqueeze(0).to(device)
    
    # 前向传播
    output = model(image)
    
    # 后处理
    output = torch.sigmoid(output)
    output = (output > threshold).float()
    
    # 转换为numpy
    mask = output.squeeze().cpu().numpy()
    
    return mask


def visualize_prediction(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str = None,
    show: bool = True
):
    """
    可视化预测结果
    
    生成三张子图:
    1. 原始MRI图像
    2. Ground Truth Mask
    3. 预测的Mask
    
    Args:
        image: 原始图像 (H, W, 3) 或 (H, W)
        gt_mask: 真实掩码 (H, W)
        pred_mask: 预测掩码 (H, W)
        save_path: 保存路径
        show: 是否显示
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 原始图像
    if len(image.shape) == 3:
        axes[0].imshow(image)
    else:
        axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始 MRI 图像', fontsize=12)
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=12)
    axes[1].axis('off')
    
    # 预测结果
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('模型预测', fontsize=12)
    axes[2].axis('off')
    
    # 叠加可视化
    overlay = image.copy() if len(image.shape) == 3 else np.stack([image]*3, axis=-1)
    overlay = overlay.astype(np.float32)
    
    # 真实区域用绿色，预测区域用红色
    gt_color = np.zeros_like(overlay)
    gt_color[:, :, 1] = gt_mask * 255  # 绿色
    
    pred_color = np.zeros_like(overlay)
    pred_color[:, :, 0] = pred_mask * 255  # 红色
    
    # 混合
    alpha = 0.4
    overlay = overlay * (1 - alpha) + gt_color * alpha * 0.5 + pred_color * alpha * 0.5
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    axes[3].imshow(overlay)
    axes[3].set_title('叠加对比 (绿=GT, 红=预测)', fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存可视化结果: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def run_inference(args):
    """运行推理"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(
        args.checkpoint,
        args.model,
        args.num_classes,
        args.image_size,
        device
    )
    
    # 创建数据集
    dataset = BrainMRIDataset(
        image_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        augmentation=False,
        mode='val'
    )
    
    if len(dataset) == 0:
        print("错误: 数据集为空!")
        return
    
    print(f"数据集大小: {len(dataset)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 随机选择样本进行推理
    if args.num_samples > len(dataset):
        args.num_samples = len(dataset)
    
    indices = random.sample(range(len(dataset)), args.num_samples)
    
    print(f"\n对 {args.num_samples} 张图像进行推理...")
    
    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        
        # 预测
        pred_mask = predict(model, image, device, args.threshold)
        
        # 转换为可视化格式
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        mask_np = mask.squeeze().numpy()
        
        # 生成保存路径
        save_path = os.path.join(args.output_dir, f'prediction_{i+1}.png')
        
        # 可视化
        visualize_prediction(
            image_np,
            mask_np,
            pred_mask,
            save_path=save_path,
            show=args.show
        )
        
        print(f"  [{i+1}/{args.num_samples}] 完成")
    
    print(f"\n推理完成! 结果保存在: {args.output_dir}")


def demo_without_data(args):
    """
    无数据演示模式
    生成随机图像并进行预测，用于测试模型是否正常工作
    """
    print("\n" + "=" * 60)
    print("演示模式 - 使用随机生成的图像")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载或创建模型
    model = load_model(
        args.checkpoint,
        args.model,
        args.num_classes,
        args.image_size,
        device
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成随机测试图像
    print("\n生成随机测试图像...")
    
    for i in range(3):
        # 随机生成一张"类似MRI"的图像
        np.random.seed(42 + i)
        
        # 创建基础灰度图
        base = np.random.rand(args.image_size, args.image_size) * 0.3 + 0.3
        
        # 添加一些圆形结构模拟脑部
        y, x = np.ogrid[:args.image_size, :args.image_size]
        center = args.image_size // 2
        
        # 大圆形 (脑部轮廓)
        r1 = args.image_size // 3
        mask1 = (x - center)**2 + (y - center)**2 < r1**2
        base[mask1] += 0.2
        
        # 小圆形 (模拟肿瘤区域)
        cx, cy = center + 30, center - 20
        r2 = 25
        tumor_mask = (x - cx)**2 + (y - cy)**2 < r2**2
        base[tumor_mask] = 0.8
        
        # 转换为RGB
        image = np.stack([base, base, base], axis=-1)
        image = (image * 255).astype(np.uint8)
        
        # 创建假的GT mask
        gt_mask = tumor_mask.astype(np.float32)
        
        # 转换为tensor并预测
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        pred_mask = predict(model, image_tensor, device, args.threshold)
        
        # 可视化
        save_path = os.path.join(args.output_dir, f'demo_prediction_{i+1}.png')
        visualize_prediction(
            image,
            gt_mask,
            pred_mask,
            save_path=save_path,
            show=args.show
        )
        
        print(f"  [{i+1}/3] 完成")
    
    print(f"\n演示完成! 结果保存在: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='脑部MRI分割推理')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data/lgg-mri-segmentation',
                        help='数据集目录路径')
    parser.add_argument('--image_size', type=int, default=256,
                        help='输入图像尺寸')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='输出类别数')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='transunet_lite',
                        help='模型类型')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='模型检查点路径')
    
    # 推理参数
    parser.add_argument('--num_samples', type=int, default=5,
                        help='推理样本数量')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')
    parser.add_argument('--show', action='store_true',
                        help='显示图像')
    
    # 演示模式
    parser.add_argument('--demo', action='store_true',
                        help='演示模式 (无需真实数据)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("脑部MRI分割推理 - TransUNet")
    print("=" * 60)
    
    if args.demo or not os.path.exists(args.data_dir):
        demo_without_data(args)
    else:
        run_inference(args)


if __name__ == '__main__':
    main()
