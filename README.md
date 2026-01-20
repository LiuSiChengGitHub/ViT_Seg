# 脑部MRI图像分割 - TransUNet

基于 ViT + CNN 混合架构的脑部MRI图像分割项目。

## 项目结构

```
VIT+分割网络的图像分割/
├── src/
│   ├── dataset.py          # 数据加载器
│   ├── model.py             # TransUNet模型 (ViT + CNN)
│   ├── losses.py            # 损失函数 (Dice + CE)
│   ├── metrics.py           # 评价指标 (Dice, IoU)
│   ├── train.py             # 训练脚本
│   ├── inference.py         # 推理和可视化
│   └── generate_synthetic_data.py  # 生成测试数据
├── data/                    # 数据目录
├── checkpoints/             # 模型权重
├── outputs/                 # 推理输出
├── requirements.txt         # 依赖包
└── README.md
```

## 技术架构

**TransUNet**: 结合 Transformer 和 CNN 的混合架构
- **ViT Encoder**: 提取全局特征 (Global Context)
- **CNN Encoder**: 提取局部细节 (Local Details)  
- **Decoder**: 融合特征 + Skip Connections

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

**方式A: 使用模拟数据测试**
```bash
cd src
python generate_synthetic_data.py --output_dir ../data/synthetic_mri --num_samples 100
```

**方式B: 下载真实数据集**
- 下载 [Kaggle LGG-MRI-Segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation) 数据集
- 解压到 `data/lgg-mri-segmentation/` 目录

### 3. 训练模型

```bash
cd src

# 使用模拟数据训练
python train.py --data_dir ../data/synthetic_mri --epochs 30 --batch_size 4

# 使用真实数据训练
python train.py --data_dir ../data/lgg-mri-segmentation --epochs 50
```

### 4. 推理可视化

```bash
# 使用训练好的模型进行推理
python inference.py --checkpoint ../checkpoints/best_model.pth --data_dir ../data/synthetic_mri

# 演示模式 (无需数据)
python inference.py --demo
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image_size` | 256 | 输入图像尺寸 |
| `--batch_size` | 4 | 批量大小 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--model` | transunet_lite | 模型类型 |

## 模型版本

- **transunet**: 标准版 (~100M参数)
- **transunet_lite**: 轻量版 (~8M参数，适合显存有限)

## 评价指标

- **Dice Score**: 分割准确度
- **IoU**: 交并比
- **Pixel Accuracy**: 像素准确率

## 参考文献

- TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
