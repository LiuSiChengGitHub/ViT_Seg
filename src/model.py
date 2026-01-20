"""
TransUNet Model for Medical Image Segmentation
混合架构: Transformer (ViT) 提取全局特征 + CNN 提取局部细节

参考文档要求:
- 结合 Transformer 和 CNN 的混合架构
- 输出层通道数可配置 (支持二分类和多分类)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# Vision Transformer (ViT) Encoder - 用于提取全局特征 (Global Context)
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    将图像分割成patches并进行嵌入
    输入: (B, C, H, W)
    输出: (B, num_patches, embed_dim)
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 使用卷积进行patch嵌入
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)   # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # 计算Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """前馈神经网络"""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm架构
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer 编码器
    用于提取全局特征 (Global Context)
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        
        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 特征图尺寸
        self.patch_size = patch_size
        self.num_patches_side = img_size // patch_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (B, C, H, W)
        输出: (B, embed_dim, H/P, W/P)
        """
        B = x.shape[0]
        
        # Patch嵌入
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Transformer块
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 重塑为特征图形式
        x = x.transpose(1, 2)  # (B, embed_dim, num_patches)
        x = x.reshape(B, -1, self.num_patches_side, self.num_patches_side)
        
        return x


# ============================================================================
# CNN Decoder - 用于提取局部细节 (Local Details) 和上采样
# ============================================================================

class ConvBlock(nn.Module):
    """卷积块: Conv -> BN -> ReLU"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    """上采样块: 转置卷积 + 跳跃连接 + 卷积"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        
        if skip is not None:
            # 处理尺寸不匹配
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class CNNEncoder(nn.Module):
    """
    CNN编码器 - 用于提取多尺度局部特征
    提供跳跃连接给解码器
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        
        self.enc1 = ConvBlock(in_channels, base_channels)      # 256 -> 256
        self.pool1 = nn.MaxPool2d(2)                            # 256 -> 128
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2)  # 128 -> 128
        self.pool2 = nn.MaxPool2d(2)                              # 128 -> 64
        
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)  # 64 -> 64
        self.pool3 = nn.MaxPool2d(2)                                  # 64 -> 32
        
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)  # 32 -> 32
        self.pool4 = nn.MaxPool2d(2)                                  # 32 -> 16
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """返回多尺度特征用于跳跃连接"""
        e1 = self.enc1(x)    # (B, 64, 256, 256)
        e2 = self.enc2(self.pool1(e1))  # (B, 128, 128, 128)
        e3 = self.enc3(self.pool2(e2))  # (B, 256, 64, 64)
        e4 = self.enc4(self.pool3(e3))  # (B, 512, 32, 32)
        out = self.pool4(e4)  # (B, 512, 16, 16)
        
        return out, (e1, e2, e3, e4)


class Decoder(nn.Module):
    """
    解码器 - 融合ViT全局特征和CNN局部特征
    """
    def __init__(
        self,
        vit_channels: int = 768,
        cnn_channels: Tuple[int, ...] = (64, 128, 256, 512),
        out_channels: int = 1
    ):
        super().__init__()
        
        # 融合ViT和CNN最深层特征
        self.fusion = nn.Conv2d(vit_channels + cnn_channels[-1], 512, 1)
        
        # 上采样路径
        self.up1 = UpBlock(512, cnn_channels[3], 256)   # 16 -> 32
        self.up2 = UpBlock(256, cnn_channels[2], 128)   # 32 -> 64
        self.up3 = UpBlock(128, cnn_channels[1], 64)    # 64 -> 128
        self.up4 = UpBlock(64, cnn_channels[0], 32)     # 128 -> 256
        
        # 输出层
        self.out_conv = nn.Conv2d(32, out_channels, 1)
    
    def forward(
        self,
        vit_features: torch.Tensor,
        cnn_features: torch.Tensor,
        skip_connections: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Args:
            vit_features: (B, vit_channels, H/P, W/P) - ViT全局特征
            cnn_features: (B, 512, H/16, W/16) - CNN最深层特征
            skip_connections: CNN各层跳跃连接 (e1, e2, e3, e4)
        """
        e1, e2, e3, e4 = skip_connections
        
        # 确保ViT特征和CNN特征尺寸匹配
        if vit_features.shape[2:] != cnn_features.shape[2:]:
            vit_features = F.interpolate(
                vit_features, size=cnn_features.shape[2:],
                mode='bilinear', align_corners=False
            )
        
        # 融合全局和局部特征
        x = torch.cat([vit_features, cnn_features], dim=1)
        x = self.fusion(x)
        
        # 上采样路径（结合跳跃连接）
        x = self.up1(x, e4)
        x = self.up2(x, e3)
        x = self.up3(x, e2)
        x = self.up4(x, e1)
        
        # 输出
        x = self.out_conv(x)
        
        return x


# ============================================================================
# TransUNet - 完整模型
# ============================================================================

class TransUNet(nn.Module):
    """
    TransUNet: 结合Transformer和CNN的混合架构
    
    - Transformer (ViT): 提取全局特征 (Global Context)
    - CNN: 提取局部细节 (Local Details)
    - Skip Connections: 保留细节信息
    
    参考文档: 结合 Transformer 提取全局特征和 CNN 提取局部细节的混合架构
    """
    
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 3,
        num_classes: int = 1,
        vit_patch_size: int = 16,
        vit_embed_dim: int = 768,
        vit_depth: int = 12,
        vit_num_heads: int = 12,
        cnn_base_channels: int = 64,
        dropout: float = 0.0
    ):
        """
        Args:
            img_size: 输入图像尺寸
            in_channels: 输入通道数
            num_classes: 输出类别数 (1=二分类, >1=多分类)
            vit_patch_size: ViT的patch大小
            vit_embed_dim: ViT嵌入维度
            vit_depth: Transformer块数量
            vit_num_heads: 注意力头数量
            cnn_base_channels: CNN基础通道数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # ViT编码器 - 提取全局特征
        self.vit_encoder = ViTEncoder(
            img_size=img_size,
            patch_size=vit_patch_size,
            in_channels=in_channels,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            dropout=dropout
        )
        
        # CNN编码器 - 提取局部特征
        self.cnn_encoder = CNNEncoder(
            in_channels=in_channels,
            base_channels=cnn_base_channels
        )
        
        # 解码器 - 融合特征并上采样
        cnn_channels = (
            cnn_base_channels,
            cnn_base_channels * 2,
            cnn_base_channels * 4,
            cnn_base_channels * 8
        )
        self.decoder = Decoder(
            vit_channels=vit_embed_dim,
            cnn_channels=cnn_channels,
            out_channels=num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, C, H, W)
        
        Returns:
            分割预测 (B, num_classes, H, W)
        """
        # ViT编码器 - 全局特征
        vit_features = self.vit_encoder(x)
        
        # CNN编码器 - 局部特征 + 跳跃连接
        cnn_features, skip_connections = self.cnn_encoder(x)
        
        # 解码器 - 融合并输出
        output = self.decoder(vit_features, cnn_features, skip_connections)
        
        return output


# ============================================================================
# 轻量级版本 (适合显存有限的情况)
# ============================================================================

class TransUNetLite(TransUNet):
    """
    轻量级TransUNet
    减少ViT深度和嵌入维度，适合显存有限的情况
    """
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 3,
        num_classes: int = 1,
        **kwargs
    ):
        super().__init__(
            img_size=img_size,
            in_channels=in_channels,
            num_classes=num_classes,
            vit_patch_size=16,
            vit_embed_dim=256,  # 减小嵌入维度
            vit_depth=6,         # 减少深度
            vit_num_heads=8,
            cnn_base_channels=32,  # 减少CNN通道
            dropout=0.1,
            **kwargs
        )


def get_model(
    model_name: str = 'transunet',
    num_classes: int = 1,
    img_size: int = 256,
    pretrained: bool = False
) -> nn.Module:
    """
    获取模型
    
    Args:
        model_name: 模型名称 ('transunet' 或 'transunet_lite')
        num_classes: 输出类别数
        img_size: 输入图像尺寸
        pretrained: 是否加载预训练权重 (暂不支持)
    
    Returns:
        模型实例
    """
    if model_name == 'transunet':
        model = TransUNet(
            img_size=img_size,
            num_classes=num_classes,
            vit_depth=12,
            vit_embed_dim=768
        )
    elif model_name == 'transunet_lite':
        model = TransUNetLite(
            img_size=img_size,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"未知模型: {model_name}")
    
    return model


# 测试代码
if __name__ == '__main__':
    print("测试TransUNet模型...")
    
    # 创建模型
    model = get_model('transunet_lite', num_classes=1, img_size=256)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试标准版
    print("\n测试标准TransUNet...")
    model_full = get_model('transunet', num_classes=1, img_size=256)
    total_params_full = sum(p.numel() for p in model_full.parameters())
    print(f"标准版参数量: {total_params_full:,}")
