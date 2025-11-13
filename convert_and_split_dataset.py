#!/usr/bin/env python3
"""
转换和划分mobility_dataset为mobility_dataset_v2
1. 检查每个对象的格式
2. 转换为标准格式（只保留必要文件）
3. 每个类别按8:2划分训练集和测试集
4. 训练集和测试集都包含所有类别
"""
import os
import json
import shutil
from pathlib import Path

# 需要保留的文件和目录
REQUIRED_FILES = [
    'mobility.urdf',
    'object_meta_info.json',
    'bounding_box.json'
]

REQUIRED_DIRS = [
    'images',
    'textured_objs'
]

def convert_to_standard_format(source_dir, target_dir):
    """
    转换为标准格式，只保留必要的文件
    """
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. 复制mobility.urdf
    urdf_source = os.path.join(source_dir, 'mobility.urdf')
    urdf_target = os.path.join(target_dir, 'mobility.urdf')
    if os.path.exists(urdf_source):
        shutil.copy2(urdf_source, urdf_target)
    else:
        raise FileNotFoundError(f"mobility.urdf not found in {source_dir}")
    
    # 2. 复制或创建object_meta_info.json
    object_meta_info_source = os.path.join(source_dir, 'object_meta_info.json')
    object_meta_info_target = os.path.join(target_dir, 'object_meta_info.json')
    
    if os.path.exists(object_meta_info_source):
        # 已经有object_meta_info.json，直接复制
        shutil.copy2(object_meta_info_source, object_meta_info_target)
    else:
        # 需要从其他文件生成
        object_meta_info = create_object_meta_info(source_dir)
        with open(object_meta_info_target, 'w') as f:
            json.dump(object_meta_info, f)
    
    # 3. 复制bounding_box.json
    bbox_source = os.path.join(source_dir, 'bounding_box.json')
    bbox_target = os.path.join(target_dir, 'bounding_box.json')
    if os.path.exists(bbox_source):
        shutil.copy2(bbox_source, bbox_target)
    else:
        # 创建默认的bounding_box.json
        default_bbox = {"min": [-1, -1, -1], "max": [1, 1, 1]}
        with open(bbox_target, 'w') as f:
            json.dump(default_bbox, f)
    
    # 4. 复制images目录（如果存在）
    images_source = os.path.join(source_dir, 'images')
    images_target = os.path.join(target_dir, 'images')
    if os.path.exists(images_source):
        if os.path.exists(images_target):
            shutil.rmtree(images_target)
        shutil.copytree(images_source, images_target)
    else:
        # 创建空的images目录
        os.makedirs(images_target, exist_ok=True)
    
    # 5. 复制textured_objs目录
    textured_source = os.path.join(source_dir, 'textured_objs')
    textured_target = os.path.join(target_dir, 'textured_objs')
    if os.path.exists(textured_source):
        if os.path.exists(textured_target):
            shutil.rmtree(textured_target)
        shutil.copytree(textured_source, textured_target)
    else:
        raise FileNotFoundError(f"textured_objs not found in {source_dir}")
    
    return True

def create_object_meta_info(source_dir):
    """
    从其他元数据文件创建object_meta_info.json
    """
    object_meta_info = {
        "moveable_link": {},
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "offset_z": 0.0,
        "scale": 1.0
    }
    
    # 尝试从mobility_v2.json提取
    mobility_v2_file = os.path.join(source_dir, 'mobility_v2.json')
    if os.path.exists(mobility_v2_file):
        try:
            with open(mobility_v2_file, 'r') as f:
                mobility_data = json.load(f)
            
            for joint_info in mobility_data:
                if joint_info.get("joint") != "free":
                    link_name = joint_info["name"]
                    joint_type = joint_info["joint"]
                    
                    if joint_type == "hinge":
                        joint_type = "hinge"
                    elif joint_type == "prismatic":
                        joint_type = "slider"
                    else:
                        joint_type = "fixed"
                    
                    link_id = f"link_{joint_info.get('id', 0)}"
                    object_meta_info["moveable_link"][link_id] = {
                        "link_id": link_id,
                        "link_name": link_name,
                        "joint_type": joint_type
                    }
        except Exception as e:
            pass
    
    # 尝试从semantics.txt提取
    semantics_file = os.path.join(source_dir, 'semantics.txt')
    if os.path.exists(semantics_file):
        try:
            with open(semantics_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    link_id, joint_type, link_name = parts[0], parts[1], parts[2]
                    if joint_type != "free":
                        object_meta_info["moveable_link"][link_id] = {
                            "link_id": link_id,
                            "link_name": link_name,
                            "joint_type": joint_type
                        }
        except Exception as e:
            pass
    
    return object_meta_info

def process_category(category, source_base_dir, target_base_dir):
    """
    处理单个类别，划分训练集和测试集
    """
    category_dir = os.path.join(source_base_dir, category)
    
    if not os.path.isdir(category_dir):
        return 0, 0, 0
    
    # 获取所有对象ID
    all_obj_ids = sorted([d for d in os.listdir(category_dir) 
                          if os.path.isdir(os.path.join(category_dir, d))])
    
    if len(all_obj_ids) == 0:
        return 0, 0, 0
    
    # 8:2划分
    split_point = max(1, int(len(all_obj_ids) * 0.8))
    train_ids = all_obj_ids[:split_point]
    test_ids = all_obj_ids[split_point:]
    
    print(f"\n处理类别: {category}")
    print(f"  总对象数: {len(all_obj_ids)}")
    print(f"  训练集: {len(train_ids)} 个")
    print(f"  测试集: {len(test_ids)} 个")
    
    success_count = 0
    failed_count = 0
    
    # 处理训练集
    for obj_id in train_ids:
        source_obj_dir = os.path.join(category_dir, obj_id)
        target_obj_dir = os.path.join(target_base_dir, 'train', category, obj_id)
        
        try:
            convert_to_standard_format(source_obj_dir, target_obj_dir)
            success_count += 1
        except Exception as e:
            print(f"  错误: 处理训练集 {category}/{obj_id} 失败: {e}")
            failed_count += 1
    
    # 处理测试集
    for obj_id in test_ids:
        source_obj_dir = os.path.join(category_dir, obj_id)
        target_obj_dir = os.path.join(target_base_dir, 'test', category, obj_id)
        
        try:
            convert_to_standard_format(source_obj_dir, target_obj_dir)
            success_count += 1
        except Exception as e:
            print(f"  错误: 处理测试集 {category}/{obj_id} 失败: {e}")
            failed_count += 1
    
    return len(train_ids), len(test_ids), failed_count

def main():
    """
    主函数
    """
    source_base_dir = "/mnt/data/zhangzhaodong/real2code/datasets/mobility_dataset"
    target_base_dir = "/mnt/data/zhangzhaodong/real2code/datasets/mobility_dataset_v2"
    
    print("="*60)
    print("开始转换和划分mobility_dataset...")
    print(f"源目录: {source_base_dir}")
    print(f"目标目录: {target_base_dir}")
    print("="*60)
    
    # 删除旧的目标目录
    if os.path.exists(target_base_dir):
        print(f"\n删除旧的目标目录...")
        shutil.rmtree(target_base_dir)
    
    # 创建新的目录结构
    os.makedirs(os.path.join(target_base_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_base_dir, 'test'), exist_ok=True)
    
    # 获取所有类别（排除split-full.json文件）
    categories = sorted([d for d in os.listdir(source_base_dir) 
                        if os.path.isdir(os.path.join(source_base_dir, d))])
    
    print(f"\n找到 {len(categories)} 个类别")
    
    # 统计信息
    stats = {
        'categories': {},
        'total_train': 0,
        'total_test': 0,
        'total_failed': 0
    }
    
    # 处理每个类别
    for category in categories:
        train_count, test_count, failed_count = process_category(
            category, source_base_dir, target_base_dir
        )
        
        if train_count > 0 or test_count > 0:
            stats['categories'][category] = {
                'train': train_count,
                'test': test_count,
                'failed': failed_count
            }
            stats['total_train'] += train_count
            stats['total_test'] += test_count
            stats['total_failed'] += failed_count
    
    # 输出统计信息
    print("\n" + "="*60)
    print("处理完成！统计信息:")
    print(f"  处理的类别数: {len(stats['categories'])}")
    print(f"  训练集总数: {stats['total_train']}")
    print(f"  测试集总数: {stats['total_test']}")
    print(f"  失败总数: {stats['total_failed']}")
    print("="*60)
    
    # 显示每个类别的统计
    print("\n各类别统计:")
    for category in sorted(stats['categories'].keys()):
        cat_stats = stats['categories'][category]
        print(f"  {category:20s}: 训练={cat_stats['train']:3d}, 测试={cat_stats['test']:3d}, 失败={cat_stats['failed']:3d}")
    
    # 保存统计信息
    stats_file = os.path.join(target_base_dir, 'split_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\n统计信息已保存到: {stats_file}")
    print(f"\n所有数据已处理完成！")
    print(f"  - 训练集: {target_base_dir}/train/")
    print(f"  - 测试集: {target_base_dir}/test/")

if __name__ == "__main__":
    main()
