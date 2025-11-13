"""
基于shape_output数据训练CodeLlama的脚本

使用方法:
1. 首先运行prepare_shape_data.py生成train.json和test.json
2. 然后运行此脚本进行fine-tuning
"""

from datasets import load_dataset
import json
import random
import os
import sys
import torch
from datetime import datetime

# 环境配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_kbit_training,  # 不使用8-bit量化时不需要
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,  # 使用LlamaTokenizer代替CodeLlamaTokenizer
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


def read_json(file_path):
    """读取JSON文件"""
    with open(file_path, "r") as f:
        return json.load(f)


def prepare_jsonl_data(train_json_path, test_json_path, mode="obb_rel"):
    """
    从train.json和test.json准备JSONL格式的训练数据
    
    参数:
        train_json_path: 训练数据JSON文件路径
        test_json_path: 测试数据JSON文件路径
        mode: 训练模式，可选 "absolute", "obb_rel", "obb_rot"
    
    返回:
        train_entries: 训练数据列表
        test_entries: 测试数据列表
    """
    print(f"\n正在准备{mode}模式的训练数据...")
    
    # 读取数据
    train_data_json = read_json(train_json_path)
    test_data_json = read_json(test_json_path)
    
    all_train_entries = []
    all_test_entries = []
    
    # 处理训练数据
    for obj_id, obj_info in train_data_json.items():
        mode_info = obj_info.get(mode, {})
        bbox_code = mode_info.get("bbox_code", "")
        label_code = mode_info.get("label_code", "")
        
        # 确保bbox_code和label_code存在
        if bbox_code and label_code:
            all_train_entries.append((bbox_code, label_code))
    
    # 处理测试数据
    for obj_id, obj_info in test_data_json.items():
        mode_info = obj_info.get(mode, {})
        bbox_code = mode_info.get("bbox_code", "")
        label_code = mode_info.get("label_code", "")
        
        if bbox_code and label_code:
            all_test_entries.append((bbox_code, label_code))
    
    # 随机打乱
    random.shuffle(all_train_entries)
    random.shuffle(all_test_entries)
    
    print(f"训练样本数: {len(all_train_entries)}")
    print(f"测试样本数: {len(all_test_entries)}")
    
    return all_train_entries, all_test_entries


def write_jsonl_files(train_entries, test_entries, mode="obb_rel", output_dir="."):
    """
    将训练和测试数据写入JSONL文件
    
    参数:
        train_entries: 训练数据列表 [(bbox_code, label_code), ...]
        test_entries: 测试数据列表
        mode: 模式名称
        output_dir: 输出目录
    
    返回:
        train_file: 训练文件路径
        test_file: 测试文件路径
    """
    train_file = os.path.join(output_dir, f"{mode}_train_data.jsonl")
    test_file = os.path.join(output_dir, f"{mode}_test_data.jsonl")
    
    # 写入训练集
    with open(train_file, "w") as f:
        for bbox, label in train_entries:
            entry = {"context": "", "question": bbox, "answer": label}
            f.write(json.dumps(entry) + "\n")
    
    # 写入测试集
    with open(test_file, "w") as f:
        for bbox, label in test_entries:
            entry = {"context": "", "question": bbox, "answer": label}
            f.write(json.dumps(entry) + "\n")
    
    print(f"已写入: {train_file}")
    print(f"已写入: {test_file}")
    
    return train_file, test_file


def tokenize(tokenizer, prompt, max_length=800):
    """分词函数"""
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()
    
    return result


def generate_and_tokenize_prompt(data_point, tokenizer):
    """生成并分词prompt"""
    full_prompt = f"""You are an AI assistant trained to understand 3D scenes and object relationships. Given the following Oriented Bounding Box (OBB) information, your task is to generate a list of child joints that describes the articulations between object parts.

OBB Information:
### Input:
{data_point["question"]}

Generate a number of root_geom,which means the base object,relative to OBB ID
- root_geom: Integer relative to/ selected from  input OBB ID
Generate a list of child joints. Each joint should be described by a dictionary with the following keys:
- box: The ID of the child bounding box
- type: The joint type ('hinge' for revolute joints, 'slide' for prismatic joints)
- idx: The rotation axis index (0 for x-axis, 1 for y-axis, 2 for z-axis)
- edge: Edge coordinates on the OBB, for example [1, -1]
- sign: Direction of the joint (+1 or -1)

IMPORTANT: Your response must contain ONLY the root_geom number and child_joints list, exactly as shown below, with no additional text before or after:

root_geom=[root_geom_number] 
child_joints = [
    dict(box=[child OBB ID], type=[joint type], idx=[rotation axis index], edge=[edge coordinates], sign=[direction]),
    # Additional joints as needed
]


Generate the geom_number and child_joints list:

### Response:
{data_point["answer"]}
"""
    return tokenize(tokenizer, full_prompt)


def main():
    """主函数"""
    print("="*80)
    print("CodeLlama Shape Output Fine-tuning")
    print("="*80)
    
    # ========== 配置参数 ==========
    # 数据路径
    TRAIN_JSON_PATH = "train.json"
    TEST_JSON_PATH = "test.json"
    
    # 训练模式
    MODE = "obb_rel"  # 可选: "absolute", "obb_rel", "obb_rot"
    
    # 模型路径 - 请根据实际情况修改
    BASE_MODEL = "/mnt/data/zhangzhaodong/real2code/models/codellama-7b"
    
    # 输出目录
    OUTPUT_DIR = "code-llama-shape-ft"

    # 训练参数（显存允许下进一步增大批次大小以提升效率）
    BATCH_SIZE = 16  # 每个GPU的批次大小，进一步加大至16
    PER_DEVICE_TRAIN_BATCH_SIZE = 16
    PER_DEVICE_EVAL_BATCH_SIZE = 16  # 评估时的batch size
    GRADIENT_ACCUMULATION_STEPS = 2  # 保证有效batch=16*2*8=256
    MAX_STEPS = 400
    LEARNING_RATE = 1e-4  # 降低学习率，从3e-4改为1e-4，提高稳定性
    WARMUP_STEPS = 100
    EVAL_STEPS = 100  # 增加评估频率
    SAVE_STEPS = 100  # 同步保存频率

    
    # ========== 步骤1: 准备JSONL数据 ==========
    print("\n步骤1: 准备训练数据...")
    
    if not os.path.exists(TRAIN_JSON_PATH):
        print(f"错误: 找不到 {TRAIN_JSON_PATH}")
        print("请先运行: python prepare_shape_data.py --shape_dir <path> --out_dir .")
        sys.exit(1)
    
    if not os.path.exists(TEST_JSON_PATH):
        print(f"错误: 找不到 {TEST_JSON_PATH}")
        sys.exit(1)
    
    # 准备数据
    train_entries, test_entries = prepare_jsonl_data(
        TRAIN_JSON_PATH, TEST_JSON_PATH, mode=MODE
    )
    
    if len(train_entries) == 0:
        print("错误: 没有找到训练数据!")
        sys.exit(1)
    
    # 写入JSONL文件
    train_file, test_file = write_jsonl_files(
        train_entries, test_entries, mode=MODE
    )
    
    # ========== 步骤2: 加载数据集 ==========
    print("\n步骤2: 加载数据集...")
    
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    test_dataset = load_dataset("json", data_files=test_file, split="train")
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # ========== 步骤3: 加载模型和分词器 ==========
    print("\n步骤3: 加载模型和分词器...")
    print(f"模型路径: {BASE_MODEL}")
    
    if not os.path.exists(BASE_MODEL):
        print(f"错误: 模型路径不存在: {BASE_MODEL}")
        print("请修改 BASE_MODEL 变量为正确的模型路径")
        sys.exit(1)
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,  # 使用FP32确保数值稳定
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # 使用LlamaTokenizer，因为旧版transformers不支持CodeLlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    print("模型加载完成")
    
    # ========== 步骤4: 数据预处理 ==========
    print("\n步骤4: 数据预处理和分词...")
    
    tokenized_train_dataset = train_dataset.map(
        lambda x: generate_and_tokenize_prompt(x, tokenizer)
    )
    tokenized_test_dataset = test_dataset.map(
        lambda x: generate_and_tokenize_prompt(x, tokenizer)
    )
    
    print("数据预处理完成")
    
    # ========== 步骤5: 配置LoRA ==========
    print("\n步骤5: 配置LoRA...")
    
    model.train()
    # model = prepare_model_for_kbit_training(model)  # 不使用8-bit量化时不需要此步骤
    
    # 手动启用梯度检查点（如果需要节省显存）
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, config)
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
        print(f"使用 {torch.cuda.device_count()} 个GPU")
    
    print("LoRA配置完成")
    
    # ========== 步骤6: 配置训练参数 ==========
    print("\n步骤6: 配置训练参数...")
    
    # 使用定义的梯度累积步数
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    
    # 使用FP32以确保数值稳定性
    print("使用FP32训练（数值稳定）")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        max_grad_norm=1.0,  # 启用梯度裁剪，防止梯度爆炸
        fp16=False,
        bf16=False,  # 禁用混合精度，使用FP32
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        output_dir=OUTPUT_DIR,
        load_best_model_at_end=False,
        group_by_length=True,
        report_to="none",
        run_name=f"codellama-shape-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        eval_accumulation_steps=4,
    )
    
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"训练批次大小: {BATCH_SIZE}")
    print(f"评估批次大小: {PER_DEVICE_EVAL_BATCH_SIZE}")
    print(f"最大步数: {MAX_STEPS}")
    print(f"评估频率: 每 {EVAL_STEPS} 步")
    print(f"学习率: {LEARNING_RATE}")
    
    # ========== 步骤7: 创建Trainer并开始训练 ==========
    print("\n步骤7: 开始训练...")
    print("="*80)
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    
    trainer.train()
    
    print("\n"+"="*80)
    print("训练完成!")
    print(f"模型保存在: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()

