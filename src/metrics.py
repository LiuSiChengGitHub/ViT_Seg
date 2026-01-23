"""
Evaluation Metrics for Medical Image Segmentation
评价指标: Dice Score 和 IoU

参考文档要求:
- 在验证集上计算 Dice Score 和 IoU
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5
) -> torch.Tensor:
    """
    计算Dice Score (Dice Coefficient)
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        pred: 预测值 (B, 1, H, W) - logits或概率
        target: 目标值 (B, 1, H, W) - 0/1二值
        threshold: 二值化阈值
        smooth: 平滑因子
    
    Returns:
        Dice分数 (标量)
    """
    # 应用sigmoid并二值化
    if pred.shape[1] == 1:
        pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    # 展平
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # 计算Dice
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )
    
    return dice


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5
) -> torch.Tensor:
    """
    计算IoU (Intersection over Union) / Jaccard Index
    
    IoU = |X ∩ Y| / |X ∪ Y|
    
    Args:
        pred: 预测值 (B, 1, H, W)
        target: 目标值 (B, 1, H, W)
        threshold: 二值化阈值
        smooth: 平滑因子
    
    Returns:
        IoU分数 (标量)
    """
    # 应用sigmoid并二值化
    if pred.shape[1] == 1:
        pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    # 展平
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # 计算IoU
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    计算像素准确率
    
    Args:
        pred: 预测值 (B, 1, H, W)
        target: 目标值 (B, 1, H, W)
        threshold: 二值化阈值
    
    Returns:
        像素准确率 (标量)
    """
    if pred.shape[1] == 1:
        pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    correct = (pred == target).sum()
    total = target.numel()
    
    return correct / total


def precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算精确率和召回率
    
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    
    Args:
        pred: 预测值 (B, 1, H, W)
        target: 目标值 (B, 1, H, W)
        threshold: 二值化阈值
        smooth: 平滑因子
    
    Returns:
        (precision, recall)
    """
    if pred.shape[1] == 1:
        pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    tp = (pred_flat * target_flat).sum()  # True Positives
    fp = (pred_flat * (1 - target_flat)).sum()  # False Positives
    fn = ((1 - pred_flat) * target_flat).sum()  # False Negatives
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return precision, recall


class MetricTracker:
    """
    指标追踪器
    用于在训练/验证过程中累积和计算平均指标
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.dice_sum = 0.0
        self.iou_sum = 0.0
        self.acc_sum = 0.0
        self.count = 0
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5
    ):
        """
        更新指标
        
        Args:
            pred: 预测值 (B, C, H, W) - C=1为二分类，C>1为多分类
            target: 目标值 (B, 1, H, W)
            threshold: 二值化阈值（仅用于二分类）
        """
        batch_size = pred.shape[0]
        num_classes = pred.shape[1]
        
        if num_classes > 1:
            # 多分类情况：将pred转换为类别索引，计算整体准确率
            pred_classes = torch.argmax(pred, dim=1, keepdim=True)  # (B, 1, H, W)
            target_classes = target.long()  # (B, 1, H, W)
            
            # 计算像素准确率
            correct = (pred_classes == target_classes).sum().float()
            total = target_classes.numel()
            acc = correct / total
            
            # 计算多分类Dice和IoU（对每个类别取平均）
            dice_sum = 0.0
            iou_sum = 0.0
            valid_classes = 0
            
            for c in range(num_classes):
                pred_c = (pred_classes == c).float()
                target_c = (target_classes == c).float()
                
                # 只计算存在该类别的情况
                if target_c.sum() > 0 or pred_c.sum() > 0:
                    intersection = (pred_c * target_c).sum()
                    union = pred_c.sum() + target_c.sum()
                    
                    dice_c = (2.0 * intersection + 1e-5) / (union + 1e-5)
                    iou_c = (intersection + 1e-5) / (union - intersection + 1e-5)
                    
                    dice_sum += dice_c.item()
                    iou_sum += iou_c.item()
                    valid_classes += 1
            
            if valid_classes > 0:
                dice = dice_sum / valid_classes
                iou = iou_sum / valid_classes
            else:
                dice = 0.0
                iou = 0.0
            
            self.dice_sum += dice * batch_size
            self.iou_sum += iou * batch_size
            self.acc_sum += acc.item() * batch_size
        else:
            # 二分类情况：使用原有逻辑
            self.dice_sum += dice_score(pred, target, threshold).item() * batch_size
            self.iou_sum += iou_score(pred, target, threshold).item() * batch_size
            self.acc_sum += pixel_accuracy(pred, target, threshold).item() * batch_size
        
        self.count += batch_size
    
    def get_metrics(self) -> dict:
        """
        获取平均指标
        
        Returns:
            包含各指标的字典
        """
        if self.count == 0:
            return {'dice': 0.0, 'iou': 0.0, 'accuracy': 0.0}
        
        return {
            'dice': self.dice_sum / self.count,
            'iou': self.iou_sum / self.count,
            'accuracy': self.acc_sum / self.count
        }


# 测试代码
if __name__ == '__main__':
    print("测试评价指标...")
    
    # 创建测试数据
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    # 计算指标
    dice = dice_score(pred, target)
    iou = iou_score(pred, target)
    acc = pixel_accuracy(pred, target)
    prec, rec = precision_recall(pred, target)
    
    print(f"Dice Score: {dice.item():.4f}")
    print(f"IoU Score: {iou.item():.4f}")
    print(f"Pixel Accuracy: {acc.item():.4f}")
    print(f"Precision: {prec.item():.4f}")
    print(f"Recall: {rec.item():.4f}")
    
    # 测试MetricTracker
    print("\n测试MetricTracker...")
    tracker = MetricTracker()
    tracker.update(pred, target)
    metrics = tracker.get_metrics()
    print(f"平均指标: {metrics}")
