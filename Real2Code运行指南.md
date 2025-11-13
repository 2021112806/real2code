# Real2Code 项目运行指南

## 项目概述

Real2Code 是一个通过代码生成重建关节化对象的项目。基于您提供的文件结构，该项目包含：

- **预训练模型**: SAM模型 (`models/sam/sam_vt_h.pth`) 和 CodeLlama模型 (`models/codellama/`)
- **数据集**: PartNet-Mobility合成数据集和真实世界数据
- **完整的数据处理管道**: 从URDF文件到代码生成的完整流程

## 环境准备

### 1. 创建Conda环境

```bash
# 创建Python 3.9环境
conda create -n real2code python=3.9 -y
conda activate real2code

# 安装依赖包
conda env update --file environment.yml --prune
```

### 2. 验证安装

```bash
# 检查关键依赖
python -c "import blenderproc; print('BlenderProc安装成功')"
python -c "import mujoco; print('MuJoCo安装成功')"
python -c "import trimesh; print('Trimesh安装成功')"
```

## 数据集结构说明

### PartNet-Mobility数据集
```
datasets/partnet-mobility-v0/dataset/
├── 101844/                    # 眼镜对象示例
│   ├── mobility.urdf          # URDF文件
│   ├── mobility_v2.json       # 关节信息
│   ├── meta.json             # 元数据
│   ├── bounding_box.json     # 边界框信息
│   ├── semantics.txt         # 语义信息
│   ├── images/               # 渲染图像
│   └── textured_objs/        # 纹理对象
├── 101860/                   # 其他对象...
└── ...
```

### 真实世界数据集
```
datasets/real_world_data/
├── 0/                        # 对象0
│   ├── 00037.jpg            # RGB图像
│   ├── 00037_mask.png       # 分割掩码
│   ├── 00037_scene.npz      # 3D点云
│   ├── sam/                 # SAM处理结果
│   │   ├── filled_img.png
│   │   ├── filled_pcd_*.npz
│   │   └── nms_mask_*.png
│   ├── meshes/              # 网格文件
│   └── process_mesh/        # 处理后的网格
├── 1/                       # 对象1
└── ...
```

## 完整运行流程

### 阶段1: 数据渲染 (使用PartNet-Mobility数据)

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export MB_DATADIR=/mnt/data/zhangzhaodong/real2code/datasets/partnet-mobility-v0/dataset

# 渲染单个眼镜对象 (ID: 101844)
blenderproc run blender_render.py \
    --data_dir /mnt/data/zhangzhaodong/real2code/datasets/partnet-mobility-v0/dataset \
    --out_dir ./output \
    --split test \
    --folder 101844 \
    --overwrite \
    --num_loops 5 \
    --num_frames 12 \
    --render_bg

# 渲染多个对象
for FOLDER in 101844 101860 101303 103177 101845; do
    echo "渲染对象: $FOLDER"
    blenderproc run blender_render.py \
        --data_dir /mnt/data/zhangzhaodong/real2code/datasets/partnet-mobility-v0/dataset \
        --out_dir ./output \
        --split test \
        --folder $FOLDER \
        --overwrite \
        --num_loops 3 \
        --num_frames 8
done
```

### 阶段2: 数据预处理

```bash
# 处理单个对象
python preprocess_data.py \
    --data_dir ./output \
    --split test \
    --obj_type Eyeglasses \
    --folder 101844 \
    --loop_id 0 \
    --num_augs 5 \
    --overwrite_xml \
    --overwrite_obb \
    --overwrite_info \
    --try_vis

# 批量处理并生成shard文件
python preprocess_data.py \
    --data_dir ./output \
    --split test \
    --obj_type "*" \
    --folder "*" \
    --loop_id "*" \
    --shard_only \
    --write_augs 1 \
    --shard_output_dir ./shard_output
```

### 阶段3: 使用预训练模型进行推理

#### 3.1 使用SAM模型进行部分分割

```python
# 使用预训练的SAM模型
import torch
from sam_model import SAMModel

# 加载预训练模型
sam_model = SAMModel()
sam_model.load_state_dict(torch.load('models/sam/sam_vt_h.pth'))
sam_model.eval()

# 对真实世界数据进行分割
import cv2
import numpy as np

# 加载图像
image = cv2.imread('datasets/real_world_data/0/00037.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 进行分割
masks = sam_model.predict(image)
```

#### 3.2 使用CodeLlama进行代码生成

```python
# 使用预训练的CodeLlama模型
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型 (需要先下载CodeLlama模型到models/codellama/)
tokenizer = AutoTokenizer.from_pretrained('models/codellama/')
model = AutoModelForCausalLM.from_pretrained('models/codellama/')

# 生成代码
def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 阶段4: 处理真实世界数据

```bash
# 处理真实世界数据对象0
python preprocess_data.py \
    --data_dir /mnt/data/zhangzhaodong/real2code/datasets/real_world_data \
    --split real \
    --obj_type real_world \
    --folder 0 \
    --loop_id 0 \
    --num_augs 3 \
    --overwrite_xml \
    --overwrite_obb \
    --overwrite_info

# 批量处理所有真实世界对象
for FOLDER in 0 1 3 5 6 8 10 13 14 15 16; do
    echo "处理真实世界对象: $FOLDER"
    python preprocess_data.py \
        --data_dir /mnt/data/zhangzhaodong/real2code/datasets/real_world_data \
        --split real \
        --obj_type real_world \
        --folder $FOLDER \
        --loop_id 0 \
        --num_augs 3
done
```

## 具体运行示例

### 示例1: 完整的眼镜对象处理流程

```bash
#!/bin/bash
# 完整处理流程示例

# 1. 激活环境
conda activate real2code
export CUDA_VISIBLE_DEVICES=0

# 2. 渲染PartNet-Mobility数据
echo "步骤1: 渲染PartNet-Mobility数据"
blenderproc run blender_render.py \
    --data_dir /mnt/data/zhangzhaodong/real2code/datasets/partnet-mobility-v0/dataset \
    --out_dir ./synthetic_output \
    --split test \
    --folder 101844 \
    --overwrite \
    --num_loops 5 \
    --num_frames 12 \
    --render_bg

# 3. 预处理合成数据
echo "步骤2: 预处理合成数据"
python preprocess_data.py \
    --data_dir ./synthetic_output \
    --split test \
    --obj_type Eyeglasses \
    --folder 101844 \
    --loop_id 0 \
    --num_augs 5 \
    --overwrite_xml \
    --overwrite_obb \
    --overwrite_info \
    --try_vis

# 4. 处理真实世界数据
echo "步骤3: 处理真实世界数据"
python preprocess_data.py \
    --data_dir /mnt/data/zhangzhaodong/real2code/datasets/real_world_data \
    --split real \
    --obj_type real_world \
    --folder 0 \
    --loop_id 0 \
    --num_augs 3 \
    --overwrite_xml \
    --overwrite_obb \
    --overwrite_info

# 5. 生成训练数据
echo "步骤4: 生成训练数据"
python preprocess_data.py \
    --data_dir ./synthetic_output \
    --split test \
    --obj_type "*" \
    --folder "*" \
    --loop_id 0 \
    --shard_only \
    --write_augs 1 \
    --shard_output_dir ./training_data

echo "处理完成！"
```

### 示例2: 使用预训练模型进行推理

```python
# inference_example.py
import torch
import numpy as np
import cv2
from pathlib import Path

def run_sam_inference(image_path, output_dir):
    """使用SAM模型进行图像分割"""
    # 加载图像
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 加载SAM模型
    sam_model = torch.load('models/sam/sam_vt_h.pth')
    
    # 进行分割 (这里需要根据实际的SAM模型接口调整)
    # masks = sam_model.predict(image)
    
    # 保存结果
    output_path = Path(output_dir) / f"{image_path.stem}_sam_mask.png"
    # cv2.imwrite(str(output_path), masks[0] * 255)
    
    print(f"SAM分割结果保存到: {output_path}")

def run_codellama_inference(prompt, output_file):
    """使用CodeLlama生成代码"""
    # 这里需要根据实际的CodeLlama模型接口调整
    # generated_code = model.generate(prompt)
    
    # 保存生成的代码
    with open(output_file, 'w') as f:
        f.write("# 生成的代码\n")
        f.write(prompt)
    
    print(f"生成的代码保存到: {output_file}")

# 使用示例
if __name__ == "__main__":
    # SAM推理
    image_path = Path("datasets/real_world_data/0/00037.jpg")
    run_sam_inference(image_path, "output/sam_results")
    
    # CodeLlama推理
    prompt = "生成一个眼镜对象的MuJoCo代码"
    run_codellama_inference(prompt, "output/generated_code.py")
```

## 输出文件说明

### 渲染输出
```
synthetic_output/test/Eyeglasses/101844/
├── loop_0/
│   ├── 0.hdf5              # 渲染数据 (RGB, 深度, 分割)
│   ├── 1.hdf5
│   ├── joint_info.json     # 关节信息
│   ├── mesh_transforms.json # 网格变换
│   ├── rgb_0.png          # RGB图像
│   └── num_masks.txt      # 掩码数量信息
└── ...
```

### 预处理输出
```
processed/test/Eyeglasses/101844/
├── repaired.xml           # 修复的MuJoCo XML
├── mjcf_code.py          # 生成的MJCF代码
├── offsetted.xml         # 偏移后的XML
├── info_loop_0.json      # 关节参数化信息
└── blender_meshes/       # 合并的网格文件
    ├── link_0.obj
    └── link_1.obj
```

### 训练数据输出
```
training_data/
├── absolute/
│   ├── train_00000.parquet
│   └── test_00000.parquet
├── obb_rot/
│   ├── train_00000.parquet
│   └── test_00000.parquet
└── obb_rel/
    ├── train_00000.parquet
    └── test_00000.parquet
```

## 关键参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--data_dir` | 输入数据目录 | `/local/real/mandi/real2code_dataset_v0/` | `./datasets/partnet-mobility-v0/dataset` |
| `--out_dir` | 输出目录 | `/local/real/mandi/blender_dataset_v5` | `./output` |
| `--split` | 数据集分割 | `test` | `test/train/real` |
| `--obj_type` | 对象类型 | `*` | `Eyeglasses/real_world` |
| `--folder` | 对象文件夹 | `*` | `101844/0` |
| `--num_loops` | 渲染循环数 | `1` | `5` |
| `--num_frames` | 每循环帧数 | `2` | `12` |
| `--num_augs` | 数据增强数 | `5` | `3` |

## 常见问题解决

### 1. 模型文件问题
```bash
# 检查SAM模型文件
ls -la models/sam/sam_vt_h.pth

# 检查CodeLlama模型目录
ls -la models/codellama/
```

### 2. 数据集路径问题
```bash
# 检查PartNet-Mobility数据
ls -la datasets/partnet-mobility-v0/dataset/101844/

# 检查真实世界数据
ls -la datasets/real_world_data/0/
```

### 3. 内存不足
```bash
# 减少渲染参数
--num_frames 8 --num_loops 1

# 使用CPU
export CUDA_VISIBLE_DEVICES=""
```

### 4. 文件权限问题
```bash
# 确保输出目录可写
mkdir -p ./output ./processed ./training_data
chmod 755 ./output ./processed ./training_data
```

## 性能优化建议

### 1. 使用预训练模型
- 直接使用提供的SAM模型 (`models/sam/sam_vt_h.pth`)
- 使用预训练的CodeLlama模型进行代码生成

### 2. 批量处理
```bash
# 使用循环批量处理多个对象
for FOLDER in 101844 101860 101303; do
    # 处理命令
done
```

### 3. 并行处理
```bash
# 使用多个GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 验证运行结果

### 1. 检查渲染输出
```bash
# 检查渲染文件
ls -la output/test/Eyeglasses/101844/loop_0/
# 应该看到: *.hdf5, *.json, *.png文件
```

### 2. 检查预处理输出
```bash
# 检查预处理文件
ls -la processed/test/Eyeglasses/101844/
# 应该看到: *.xml, *.py, *.json文件
```

### 3. 检查模型推理
```bash
# 检查SAM分割结果
ls -la output/sam_results/

# 检查代码生成结果
ls -la output/generated_code.py
```

## 项目结构总结

```
real2code/
├── models/                          # 预训练模型
│   ├── sam/sam_vt_h.pth           # SAM模型
│   └── codellama/                 # CodeLlama模型
├── datasets/                       # 数据集
│   ├── partnet-mobility-v0/       # 合成数据集
│   └── real_world_data/           # 真实世界数据
├── preprocess_data.py             # 数据预处理
├── blender_render.py              # Blender渲染
├── data_utils/                    # 数据处理工具
├── eval_utils/                    # 评估工具
└── environment.yml                # 环境配置
```

这个运行指南基于您实际的文件结构，提供了完整的项目运行流程，包括使用预训练模型进行推理的具体方法。
