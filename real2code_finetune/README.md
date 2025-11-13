# Real2Code Fine-tuning 项目说明

## 项目简介

这个项目用于对 CodeLlama-7B 模型进行微调，使其能够理解3D场景中的物体关系，并根据定向包围盒（Oriented Bounding Box, OBB）信息生成物体部件之间的关节（articulation）描述。

## 主要功能

1. **数据准备** (`prepare_shape_data.py`): 将shape输出数据转换为训练格式
2. **模型训练** (`train_shape.py`): 使用LoRA方法微调CodeLlama模型

## 训练模式

支持三种训练模式：
- `absolute`: 绝对坐标模式
- `obb_rel`: OBB相对坐标模式（默认）
- `obb_rot`: OBB旋转模式

## 使用方法

### 1. 准备训练数据

首先需要运行数据准备脚本生成 `train.json` 和 `test.json`：

```bash
python prepare_shape_data.py --shape_dir <path_to_shape_data> --out_dir .
```

### 2. 运行训练

确保配置正确后，运行训练脚本：

```bash
python train_shape.py
```

### 3. 训练配置参数

在 `train_shape.py` 中可以修改以下关键参数：

```python
# 数据路径
TRAIN_JSON_PATH = "train.json"
TEST_JSON_PATH = "test.json"

# 训练模式
MODE = "obb_rel"

# 模型路径
BASE_MODEL = "/mnt/data/zhangzhaodong/real2code/models/codellama-7b"

# 训练参数
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16  # 有效batch = 1*16*8GPUs = 128
MAX_STEPS = 400
LEARNING_RATE = 3e-4
```

## 技术细节

### LoRA配置
- Rank (r): 16
- Alpha: 16
- Target modules: q_proj, k_proj, v_proj, o_proj
- Dropout: 0.05

### 训练优化
- 使用梯度检查点（gradient checkpointing）节省显存
- 支持多GPU训练（自动检测可用GPU数量）
- 自动选择最佳精度类型：
  - **BF16**: 如果GPU支持（安培架构及以上：A100、RTX 3090/4090等）
  - **FP32**: 如果GPU不支持BF16（更稳定但速度较慢）

## 常见问题及解决方案

### 问题1: ValueError: Attempting to unscale FP16 gradients

**原因**: 使用FP16混合精度训练时，PyTorch的梯度缩放器（GradScaler）在某些配置下无法正确处理梯度的unscale操作，导致训练崩溃。这个问题在使用梯度检查点和多GPU训练时尤其常见。

**已解决**: 脚本已自动切换到更稳定的精度方案：
- **BF16 (推荐)**: 如果GPU支持（计算能力≥8.0，如A100、RTX 3090/4090），自动使用BF16
  - BF16不需要梯度缩放器，避免了FP16的问题
  - 数值稳定性更好，训练更可靠
  
- **FP32 (备选)**: 如果GPU不支持BF16，自动降级到FP32
  - 完全稳定，但速度较慢
  - 显存占用略高

脚本会自动检测并选择最佳方案，无需手动配置。

**技术细节**:
```python
# 自动检测GPU能力
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

training_args = TrainingArguments(
    ...
    bf16=use_bf16,  # 使用BF16（如果支持）
    fp16=False,     # 禁用FP16
    ...
)
```

### 问题2: CUDA Out of Memory (显存不足)

**常见场景**:
1. **训练时OOM**: 训练过程中显存不足
2. **评估时OOM**: 训练正常,但在评估(evaluation)阶段显存不足

**已优化**: 脚本已针对显存使用进行优化：
- 评估batch size设置为1 (`PER_DEVICE_EVAL_BATCH_SIZE = 1`)
- 评估累积步数设置为4 (`eval_accumulation_steps = 4`)
- 评估频率降低到每100步一次 (从20步改为100步)
- 启用梯度检查点 (`gradient_checkpointing = True`)

**如果仍然遇到OOM**:

1. **进一步减少评估频率**:
   ```python
   EVAL_STEPS = 200  # 或更大的值
   SAVE_STEPS = 200
   ```

2. **完全禁用评估**（如果只需要训练）:
   ```python
   # 在TrainingArguments中修改
   evaluation_strategy="no",  # 禁用评估
   # 删除或注释掉eval_dataset参数
   ```

3. **减小训练batch size**:
   ```python
   PER_DEVICE_TRAIN_BATCH_SIZE = 1
   GRADIENT_ACCUMULATION_STEPS = 32  # 增加以保持有效batch size
   ```

4. **限制序列长度**:
   ```python
   # 在tokenize函数中
   max_length=600  # 从800减小到600
   ```

5. **使用环境变量优化显存分配**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   python train_shape.py
   ```

6. **清理之前的进程**:
   ```bash
   # 检查是否有其他进程占用GPU
   nvidia-smi
   # 如果有,杀死进程
   kill -9 <PID>
   ```

### 问题3: 模型路径不存在

**解决方案**: 修改 `BASE_MODEL` 变量为正确的 CodeLlama-7B 模型路径。

## 输出说明

训练完成后，模型将保存在 `code-llama-shape-ft` 目录下，包含：
- LoRA权重
- 训练日志
- 检查点文件（checkpoint-100, checkpoint-200, checkpoint-300, checkpoint-400等）

## 模型推理

训练完成后，可以使用微调后的模型进行推理。

### 快速推理

```bash
# 使用最终checkpoint进行推理
python run_inference.py --lora_model code-llama-shape-ft/checkpoint-400
```

### 推理文件说明

- **sample_bbox.json**: 包含3个测试样本（简单、中等、复杂）
- **run_inference.py**: 推理脚本，支持批量测试和结果对比
- **INFERENCE_GUIDE.md**: 详细的推理使用指南

### 更多推理选项

查看详细的推理指南：
```bash
cat INFERENCE_GUIDE.md
```

或查看脚本帮助：
```bash
python run_inference.py --help
```

## 系统要求

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (Parameter-Efficient Fine-Tuning)
- CUDA 11.8+
- 建议：8 x GPU (每个GPU至少24GB显存)

## 环境配置

使用conda环境：
```bash
conda activate real2code
```

## 模型输入输出格式

### 输入格式
OBB信息，描述物体各部件的空间位置和姿态。

### 输出格式
```python
root_geom=[root_geom_number]
child_joints = [
    dict(box=[child_OBB_ID], type=[joint_type], idx=[axis_index], edge=[edge_coords], sign=[direction]),
    # 更多关节...
]
```

其中：
- `root_geom`: 基础物体的OBB ID
- `box`: 子部件的OBB ID
- `type`: 关节类型（'hinge'表示旋转关节，'slide'表示平移关节）
- `idx`: 旋转轴索引（0=x轴, 1=y轴, 2=z轴）
- `edge`: OBB边缘坐标
- `sign`: 关节方向（+1或-1）

## 项目改进建议

1. **数据增强**: 可以考虑添加数据增强策略，如随机旋转、缩放等
2. **超参数调优**: 使用更系统的方法（如网格搜索）优化学习率、LoRA rank等参数
3. **评估指标**: 添加更详细的评估指标，如精确率、召回率等
4. **早停机制**: 添加早停（early stopping）避免过拟合
5. **模型推理脚本**: 添加独立的推理脚本方便测试训练好的模型

## 更新日志

### 2025-11-12 (第三次更新 - 显存优化)
- **解决评估时OOM问题**：优化显存使用策略
- 增加评估频率至每100步（原20步）
- 添加 `per_device_eval_batch_size` 和 `eval_accumulation_steps` 参数
- 更新文档，详细说明OOM问题的多种解决方案
- 改进输出信息，显示评估频率

### 2025-11-12 (第二次更新 - 精度优化)
- **彻底解决FP16梯度unscale错误**：改用BF16/FP32代替FP16
- 添加GPU能力自动检测，智能选择最佳精度类型
- BF16不需要梯度缩放器，彻底避免unscale问题
- 更新文档，详细说明精度选择逻辑

### 2025-11-12 (第一次更新)
- 尝试通过禁用梯度裁剪解决FP16问题（未完全解决）
- 创建项目README文档

## 联系信息

如有问题，请查看项目主目录下的文档或联系项目维护者。
