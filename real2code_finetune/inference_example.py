"""
使用微调后的CodeLlama模型对sample_bbox.json进行推理

用法:
# 测试所有样本
python inference_example.py --lora_model code-llama-shape-ft/checkpoint-400

# 测试单个样本
python inference_example.py --lora_model code-llama-shape-ft/checkpoint-400 --sample_name StorageFurniture_simple

# 使用基础模型（未微调）
python inference_example.py
"""

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from peft import PeftModel
import json
import argparse
import os


def load_model(base_model_path, lora_model_path=None):
    """
    加载模型和分词器
    
    参数:
        base_model_path: 基础模型路径
        lora_model_path: LoRA权重路径（可选）
    
    返回:
        model: 加载的模型
        tokenizer: 分词器
    """
    print(f"加载基础模型: {base_model_path}")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # 如果提供了LoRA权重，加载它
    if lora_model_path and os.path.exists(lora_model_path):
        print(f"加载LoRA权重: {lora_model_path}")
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        model = model.merge_and_unload()  # 合并LoRA权重到基础模型
        print("✓ 使用微调后的模型")
    else:
        model = base_model
        if lora_model_path:
            print(f"⚠️  LoRA权重不存在: {lora_model_path}")
        print("✓ 使用基础模型（未微调）")
    
    # 加载分词器 (使用LlamaTokenizer代替AutoTokenizer)
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    print(f"模型加载完成")
    print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")
    print(f"Model eos_token_id: {model.config.eos_token_id}")
    print()
    return model, tokenizer


def compare_outputs(generated, expected):
    """比较生成的输出和期望的输出"""
    print("\n" + "="*80)
    print("📊 结果比较")
    print("="*80)
    
    print("\n期望输出:")
    print("-"*80)
    print(expected)
    
    print("\n实际生成:")
    print("-"*80)
    print(generated)
    
    print("\n" + "="*80)
    if generated.strip() == expected.strip():
        print("✅ 完全匹配！")
    else:
        match_score = 0
        total_checks = 2
        
        # 检查root_geom
        if 'root_geom' in generated:
            gen_root = generated.split('root_geom')[1].split('\n')[0].strip()
            exp_root = expected.split('root_geom')[1].split('\n')[0].strip()
            if gen_root == exp_root:
                print("✅ root_geom 匹配")
                match_score += 1
            else:
                print(f"❌ root_geom 不匹配: 期望 '{exp_root}', 实际 '{gen_root}'")
        
        # 检查child_joints数量
        gen_dict_count = generated.count('dict(')
        exp_dict_count = expected.count('dict(')
        if gen_dict_count == exp_dict_count:
            print(f"✅ 关节数量匹配: {gen_dict_count}")
            match_score += 1
        else:
            print(f"❌ 关节数量不匹配: 期望 {exp_dict_count}, 实际 {gen_dict_count}")
        
        accuracy = (match_score / total_checks) * 100
        print(f"\n匹配度: {accuracy:.1f}% ({match_score}/{total_checks})")
    print("="*80)


def create_prompt(bbox_code):
    """
    创建推理prompt
    
    参数:
        bbox_code: OBB代码字符串
    
    返回:
        prompt: 完整的prompt
    """
    prompt = f"""You are an AI assistant trained to understand 3D scenes and object relationships. Given the following Oriented Bounding Box (OBB) information, your task is to generate a list of child joints that describes the articulations between object parts.

OBB Information:
### Input:
{bbox_code}

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
"""
    return prompt


def generate_joint_code(model, tokenizer, bbox_code, max_new_tokens=256):
    """
    生成关节代码
    
    参数:
        model: 模型
        tokenizer: 分词器
        bbox_code: OBB代码
        max_new_tokens: 最大生成token数
    
    返回:
        generated_code: 生成的关节代码
    """
    # 创建prompt
    prompt = create_prompt(bbox_code)
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成（使用贪婪解码避免采样问题）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 贪婪解码，更稳定
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 调试：打印生成的token数量和部分内容
    input_length = inputs['input_ids'].shape[1]
    output_length = outputs.shape[1]
    generated_length = output_length - input_length
    print(f"生成了 {generated_length} 个新token")
    
    # 调试：打印完整输出的最后200个字符
    print(f"完整输出末尾: ...{full_output[-200:]}")
    
    # 提取生成的部分（在"### Response:"之后）
    if "### Response:" in full_output:
        generated_code = full_output.split("### Response:")[-1].strip()
        print(f"提取的生成内容长度: {len(generated_code)}")
    else:
        print("⚠️  警告: 输出中没有找到'### Response:'标记")
        generated_code = full_output
    
    # 如果生成为空，返回完整输出用于调试
    if not generated_code:
        print("⚠️  警告: 生成内容为空")
    
    return generated_code


def main():
    parser = argparse.ArgumentParser(description="使用微调后的CodeLlama进行推理")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/mnt/data/zhangzhaodong/real2code/models/codellama-7b",
        help="基础CodeLlama模型路径",
    )
    parser.add_argument(
        "--lora_model",
        type=str,
        default=None,
        help="LoRA权重路径（可选）",
    )
    parser.add_argument(
        "--sample_file",
        type=str,
        default="sample_bbox.json",
        help="样本文件路径（默认: sample_bbox.json）",
    )
    parser.add_argument(
        "--sample_name",
        type=str,
        default=None,
        help="要测试的样本名称（如果不指定，将测试所有样本）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="最大生成token数",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_results.txt",
        help="结果保存文件",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🤖 CodeLlama 关节代码生成推理")
    print("=" * 80)
    print()
    
    # 加载模型
    model, tokenizer = load_model(args.base_model, args.lora_model)
    model.eval()  # 确保模型处于评估模式
    
    # 读取样本数据
    if not os.path.exists(args.sample_file):
        print(f"❌ 错误: 找不到样本文件 {args.sample_file}")
        return
    
    with open(args.sample_file, "r") as f:
        samples = json.load(f)
    
    # 确定要测试的样本
    if args.sample_name:
        if args.sample_name not in samples:
            print(f"❌ 错误: 样本 '{args.sample_name}' 不存在")
            print(f"可用样本: {list(samples.keys())}")
            return
        test_samples = {args.sample_name: samples[args.sample_name]}
    else:
        test_samples = samples
    
    print(f"📋 将测试 {len(test_samples)} 个样本\n")
    
    # 保存所有结果
    all_results = []
    
    # 对每个样本进行推理
    for sample_name, sample_data in test_samples.items():
        print("=" * 80)
        print(f"🔍 测试样本: {sample_name}")
        print("=" * 80)
        
        if "description" in sample_data:
            print(f"说明: {sample_data['description']}")
        
        bbox_code = sample_data["obb_rel"]["bbox_code"]
        expected_output = sample_data["obb_rel"].get("expected_output", "")
        
        print("\n📥 输入OBB代码:")
        print("-" * 80)
        print(bbox_code)
        print("-" * 80)
        
        # 生成关节代码
        print("\n⚙️  正在生成...")
        generated_code = generate_joint_code(
            model, tokenizer, bbox_code,
            max_new_tokens=args.max_new_tokens
        )
        
        # 如果有期望输出，进行比较
        if expected_output:
            compare_outputs(generated_code, expected_output)
        else:
            print("\n📤 生成的关节代码:")
            print("=" * 80)
            print(generated_code)
            print("=" * 80)
        
        # 保存结果
        all_results.append({
            "sample_name": sample_name,
            "bbox_code": bbox_code,
            "expected_output": expected_output,
            "generated_output": generated_code
        })
        
        print()
    
    # 保存所有结果到文件
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("CodeLlama 关节代码生成推理结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"模型: {args.base_model}\n")
        if args.lora_model:
            f.write(f"LoRA: {args.lora_model}\n")
        f.write(f"测试样本数: {len(test_samples)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        for result in all_results:
            f.write(f"样本: {result['sample_name']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"\n输入OBB代码:\n{result['bbox_code']}\n")
            f.write(f"\n期望输出:\n{result['expected_output']}\n")
            f.write(f"\n生成输出:\n{result['generated_output']}\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    print("=" * 80)
    print(f"✅ 所有结果已保存到: {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

