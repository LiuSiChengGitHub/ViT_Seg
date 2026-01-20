"""
Loss Functions for Medical Image Segmentation
损失函数: DiceLoss + CrossEntropyLoss 组合

参考文档要求:
- 使用 DiceLoss 和 CrossEntropyLoss 的组合损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    DiceLoss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (B, C, H, W) - logits或sigmoid后的值
            target: 目标值 (B, 1, H, W) 或 (B, C, H, W)
        
        Returns:
            Dice损失值
        """
        # 如果是logits，应用sigmoid
        if pred.shape[1] == 1:
            pred = torch.sigmoid(pred)
        else:
            pred = F.softmax(pred, dim=1)
        
        # 展平
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算Dice
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice Loss 组合
    用于二分类分割任务
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测logits (B, 1, H, W)
            target: 目标值 (B, 1, H, W)
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class CrossEntropyDiceLoss(nn.Module):
    """
    CrossEntropy + Dice Loss 组合
    用于多分类分割任务
    """
    
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        num_classes: int = 2,
        smooth: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.smooth = smooth
        
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
    
    def _dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算多类别Dice Loss"""
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.squeeze(1).long(), self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        dice_sum = 0.0
        for i in range(self.num_classes):
            pred_i = pred[:, i]
            target_i = target_one_hot[:, i]
            
            intersection = (pred_i * target_i).sum()
            dice = (2.0 * intersection + self.smooth) / (
                pred_i.sum() + target_i.sum() + self.smooth
            )
            dice_sum += (1.0 - dice)
        
        return dice_sum / self.num_classes
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测logits (B, C, H, W)
            target: 目标值 (B, 1, H, W) 类别索引
        """
        # CrossEntropy需要(B, H, W)形式的target
        ce_loss = self.ce(pred, target.squeeze(1).long())
        dice_loss = self._dice_loss(pred, target)
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def get_loss_function(num_classes: int = 1) -> nn.Module:
    """
    获取损失函数
    
    Args:
        num_classes: 类别数量
            - 1: 二分类，使用BCEDiceLoss
            - >1: 多分类，使用CrossEntropyDiceLoss
    
    Returns:
        损失函数实例
    """
    if num_classes == 1:
        return BCEDiceLoss()
    else:
        return CrossEntropyDiceLoss(num_classes=num_classes)


# 测试代码
if __name__ == '__main__':
    print("测试损失函数...")
    
    # 测试二分类损失
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    loss_fn = get_loss_function(num_classes=1)
    loss = loss_fn(pred, target)
    print(f"二分类损失: {loss.item():.4f}")
    
    # 测试多分类损失
    pred_multi = torch.randn(2, 5, 256, 256)
    target_multi = torch.randint(0, 5, (2, 1, 256, 256)).float()
    
    loss_fn_multi = get_loss_function(num_classes=5)
    loss_multi = loss_fn_multi(pred_multi, target_multi)
    print(f"多分类损失: {loss_multi.item():.4f}")
