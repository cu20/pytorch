# PyTorch 学习项目

这是一个 PyTorch 深度学习学习项目，包含了从基础到进阶的多个示例代码和完整的 CIFAR-10 图像分类项目。

## 项目结构

```
.
├── 1_dataset_test.py              # 数据集基础测试
├── 2_tensorboard_test.py          # TensorBoard 使用示例
├── 3_tensorboard_img_test.py      # TensorBoard 图像可视化
├── 4_transforms_test.py           # 数据变换测试
├── 5_dataset_transforms.py        # 数据集与变换结合
├── 6_dataloader_test.py           # 数据加载器测试
├── 7_nn_moudule_test.py           # 神经网络模块测试
├── 8_nn._conv.py                  # 卷积层示例
├── 9_nn_conv2d.py                 # 2D 卷积示例
├── 10_nn_maxpool.py               # 最大池化示例
├── 11_relu_sigmoid.py             # 激活函数示例
├── 12_linear.py                   # 线性层示例
├── 13_sequential.py               # 序列模型示例
├── 14_loss.py                     # 损失函数示例
├── 15_optim.py                    # 优化器示例
├── 16_pretrained.py               # 预训练模型示例
├── 17_model_save.py               # 模型保存示例
├── 18_model_load.py               # 模型加载示例
└── project_CIFA10/                # CIFAR-10 分类项目
    ├── model.py                   # 网络模型定义
    ├── train.py                   # 训练脚本
    ├── train_gpu_1.py             # GPU 训练脚本
    └── test.py                    # 测试脚本
```

## 环境要求

- Python 3.7+
- PyTorch 1.0+
- torchvision
- PIL (Pillow)
- tensorboard

## 安装依赖

```bash
pip install torch torchvision pillow tensorboard
```

## 使用说明

### 基础示例

项目包含了 18 个基础示例文件，涵盖了 PyTorch 的核心功能：

1. **数据集操作** (1_dataset_test.py, 5_dataset_transforms.py)
   - 自定义数据集类
   - 数据集变换

2. **可视化** (2_tensorboard_test.py, 3_tensorboard_img_test.py)
   - TensorBoard 使用
   - 图像可视化

3. **数据加载** (6_dataloader_test.py)
   - DataLoader 使用

4. **神经网络模块** (7-13)
   - 卷积层、池化层
   - 激活函数
   - 线性层
   - 序列模型

5. **训练相关** (14-18)
   - 损失函数
   - 优化器
   - 预训练模型
   - 模型保存与加载

### CIFAR-10 分类项目

进入 `project_CIFA10` 目录：

```bash
cd project_CIFA10
```

#### 训练模型

使用 CPU 训练：
```bash
python train.py
```

使用 GPU 训练：
```bash
python train_gpu_1.py
```

#### 测试模型

```bash
python test.py
```

#### 模型结构

项目使用了一个简单的 CNN 模型（Tudui），包含：
- 3 个卷积层 + 最大池化层
- 1 个全连接层
- 用于 CIFAR-10 的 10 分类任务

## 注意事项

1. 数据文件（`data/` 目录）需要单独下载，CIFAR-10 数据集可以通过 torchvision 自动下载
2. 训练日志保存在 `logs_train/` 目录，可通过 TensorBoard 查看：
   ```bash
   tensorboard --logdir=logs_train
   ```
3. 模型文件（`.pth`）会在训练过程中自动保存

## 学习路径建议

1. 从 `1_dataset_test.py` 开始，了解 PyTorch 数据集基础
2. 学习数据加载和变换（4-6）
3. 掌握神经网络模块（7-13）
4. 学习训练流程（14-18）
5. 最后完成完整的 CIFAR-10 项目

## 许可证

本项目仅用于学习目的。
