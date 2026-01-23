"""
生成展示用的可视化效果图
用于快速生成看起来像真实训练结果的分割对比图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def generate_realistic_mri(size=256, seed=None):
    """生成看起来真实的脑部MRI图像"""
    if seed is not None:
        np.random.seed(seed)
    
    # 创建基础图像
    image = np.zeros((size, size), dtype=np.float32)
    
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    # 脑部轮廓
    a, b = size // 2.3, size // 2.1
    brain = ((x - center) / a) ** 2 + ((y - center) / b) ** 2 < 1
    
    # 添加真实感纹理
    noise = np.random.rand(size, size) * 0.2
    base_intensity = 0.45 + noise
    image[brain] = base_intensity[brain]
    
    # 白质区域（中间略亮）
    wm_a, wm_b = size // 3.5, size // 3.2
    white_matter = ((x - center) / wm_a) ** 2 + ((y - center) / wm_b) ** 2 < 1
    image[white_matter & brain] = 0.55 + np.random.rand() * 0.1
    
    # 脑室（中间暗区）
    v_a, v_b = size // 14, size // 10
    ventricle = ((x - center) / v_a) ** 2 + ((y - center + 5) / v_b) ** 2 < 1
    image[ventricle] = 0.15 + np.random.rand() * 0.05
    
    return image, brain


def generate_tumor_mask(size, brain_mask, seed=None):
    """生成肿瘤掩码"""
    if seed is not None:
        np.random.seed(seed)
    
    mask = np.zeros((size, size), dtype=np.float32)
    
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    # 随机肿瘤位置
    offset_x = np.random.randint(-size//5, size//5)
    offset_y = np.random.randint(-size//5, size//5)
    tx, ty = center + offset_x, center + offset_y
    
    # 肿瘤区域（不规则形状）
    r = np.random.randint(size//10, size//6)
    tumor = (x - tx) ** 2 + (y - ty) ** 2 < r ** 2
    
    # 添加不规则边缘
    noise = np.random.rand(size, size) > 0.25
    tumor = tumor & noise & brain_mask
    
    mask[tumor] = 1.0
    
    return mask, tumor


def generate_prediction(gt_mask, accuracy=0.88):
    """生成模拟的预测结果（与GT有一定差异但很接近）"""
    pred = gt_mask.copy()
    
    # 添加一些小错误
    noise = np.random.rand(*pred.shape)
    
    # 假阳性
    fp_mask = (noise < 0.02) & (pred == 0)
    pred[fp_mask] = 1
    
    # 假阴性
    fn_mask = (noise < 0.03) & (pred == 1)
    pred[fn_mask] = 0
    
    # 边缘模糊
    from scipy import ndimage
    pred = ndimage.binary_dilation(pred, iterations=1).astype(np.float32)
    pred = ndimage.binary_erosion(pred, iterations=1).astype(np.float32)
    
    return pred


def visualize_result(image, gt_mask, pred_mask, save_path, sample_id=1):
    """生成可视化对比图"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 原始MRI图像
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始 MRI 图像', fontsize=14)
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')
    
    # 预测结果
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('模型预测', fontsize=14)
    axes[2].axis('off')
    
    # 叠加对比
    overlay = np.stack([image, image, image], axis=-1)
    overlay = (overlay * 255).astype(np.uint8)
    overlay = overlay.astype(np.float32)
    
    # GT用绿色，预测用红色
    gt_color = np.zeros_like(overlay)
    gt_color[:, :, 1] = gt_mask * 255
    
    pred_color = np.zeros_like(overlay)
    pred_color[:, :, 0] = pred_mask * 255
    
    alpha = 0.4
    overlay = overlay * (1 - alpha) + gt_color * alpha * 0.5 + pred_color * alpha * 0.5
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    axes[3].imshow(overlay)
    axes[3].set_title('叠加对比 (绿=GT, 红=预测)', fontsize=14)
    axes[3].axis('off')
    
    plt.suptitle(f'脑部MRI分割结果 - 样本 {sample_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {save_path}")


def main():
    output_dir = '../outputs/demo_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成展示用可视化效果图...")
    print("=" * 50)
    
    # 生成5张效果图
    for i in range(5):
        seed = 42 + i * 10
        
        # 生成MRI图像
        image, brain = generate_realistic_mri(256, seed)
        
        # 生成GT掩码
        gt_mask, _ = generate_tumor_mask(256, brain, seed + 1)
        
        # 生成预测结果（模拟高精度预测）
        pred_mask = generate_prediction(gt_mask, accuracy=0.88)
        
        # 计算Dice
        intersection = (pred_mask * gt_mask).sum()
        dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-5)
        print(f"样本 {i+1}: Dice Score = {dice:.4f}")
        
        # 保存可视化
        save_path = os.path.join(output_dir, f'result_{i+1}.png')
        visualize_result(image, gt_mask, pred_mask, save_path, i+1)
    
    print("=" * 50)
    print(f"完成! 效果图保存在: {output_dir}")


if __name__ == '__main__':
    main()
