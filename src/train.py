"""
Training Script for Brain MRI Segmentation
训练脚本

参考文档要求:
- 损失函数: DiceLoss + CrossEntropyLoss 组合
- 优化器: AdamW
- 评价指标: Dice Score 和 IoU
- 数据划分: 7:3 (训练:验证)
- 保存验证集分数最高的模型权重
"""

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import get_model, TransUNetLite
from dataset import get_dataloaders, BrainMRIDataset
from losses import get_loss_function
from metrics import MetricTracker, dice_score, iou_score


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    use_amp: bool = False
) -> dict:
    """
    训练一个epoch
    
    Returns:
        包含训练损失和指标的字典
    """
    model.train()
    
    total_loss = 0.0
    metric_tracker = MetricTracker()
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # 更新指标（不使用梯度）
        with torch.no_grad():
            metric_tracker.update(outputs, masks)
    
    avg_loss = total_loss / len(train_loader)
    metrics = metric_tracker.get_metrics()
    
    return {
        'loss': avg_loss,
        'dice': metrics['dice'],
        'iou': metrics['iou']
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """
    验证模型
    
    Returns:
        包含验证损失和指标的字典
    """
    model.eval()
    
    total_loss = 0.0
    metric_tracker = MetricTracker()
    
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        total_loss += loss.item()
        metric_tracker.update(outputs, masks)
    
    avg_loss = total_loss / len(val_loader)
    metrics = metric_tracker.get_metrics()
    
    return {
        'loss': avg_loss,
        'dice': metrics['dice'],
        'iou': metrics['iou']
    }


def train(args):
    """主训练函数"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据
    print(f"\n加载数据: {args.data_dir}")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        train_ratio=0.7,  # 7:3 划分
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # 创建模型
    print(f"\n创建模型: {args.model}")
    model = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        img_size=args.image_size
    )
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 定义损失函数和优化器
    criterion = get_loss_function(num_classes=args.num_classes)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # 混合精度训练
    use_amp = args.amp and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("启用混合精度训练 (AMP)")
    
    # 训练循环
    best_dice = 0.0
    best_epoch = 0
    
    print(f"\n开始训练 (共 {args.epochs} 轮)...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, use_amp
        )
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 计算耗时
        epoch_time = time.time() - epoch_start
        
        # 打印日志
        print(f"Epoch [{epoch:3d}/{args.epochs}] | "
              f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
                'iou': val_metrics['iou']
            }, save_path)
            print(f"  >>> 保存最佳模型 (Dice: {best_dice:.4f})")
        
        # 定期保存检查点
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'dice': val_metrics['dice']
            }, checkpoint_path)
    
    print("=" * 60)
    print(f"训练完成!")
    print(f"最佳验证 Dice Score: {best_dice:.4f} (Epoch {best_epoch})")
    print(f"模型保存于: {os.path.join(args.save_dir, 'best_model.pth')}")


def main():
    parser = argparse.ArgumentParser(description='脑部MRI分割训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data/lgg-mri-segmentation',
                        help='数据集目录路径')
    parser.add_argument('--image_size', type=int, default=256,
                        help='输入图像尺寸')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='输出类别数 (1=二分类)')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='transunet_lite',
                        choices=['transunet', 'transunet_lite'],
                        help='模型类型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='保存检查点间隔')
    parser.add_argument('--amp', action='store_true',
                        help='启用混合精度训练')
    
    args = parser.parse_args()
    
    # 打印配置
    print("\n" + "=" * 60)
    print("脑部MRI分割训练 - TransUNet")
    print("=" * 60)
    print(f"配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    train(args)


if __name__ == '__main__':
    main()
