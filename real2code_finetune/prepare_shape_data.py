import json
import os
from glob import glob
from natsort import natsorted
import argparse
import sys
import numpy as np
import xml.etree.ElementTree as ET

# 添加项目根目录到路径
sys.path.insert(0, '/mnt/data/zhangzhaodong/real2code')

"""
从shape_output和mobility_dataset_v2文件夹准备训练数据

用法:
python prepare_shape_data.py \
  --shape_dir /mnt/data/zhangzhaodong/real2code/datasets/shape_output \
  --mobility_dir /mnt/data/zhangzhaodong/real2code/datasets/mobility_dataset_v2 \
  --out_dir /mnt/data/zhangzhaodong/real2code/real2code_finetune

这个脚本会：
1. 从shape_output读取OBB数据
2. 从mobility_dataset_v2读取URDF文件并生成label_code  
3. 生成train.json和test.json供fine-tuning使用
"""


def translate_dict_to_code(bbox_dict):
    """将OBB字典转换为代码字符串"""
    bbox_lines = ["bboxes={"]
    for bbox_id, info_listed in bbox_dict.items():
        line = str(info_listed)
        line = line.replace(" ", "")
        new_line = f"{bbox_id}:" + line + ","
        bbox_lines.append(new_line)
    bbox_lines.append("}")
    return "\n".join(bbox_lines)


def load_obb_from_loop(loop_folder):
    """
    从loop文件夹读取所有OBB文件
    返回: {0: {center, R, extent}, 1: {...}, ...}
    """
    obb_files = natsorted(glob(os.path.join(loop_folder, "obb_link_*.json")))
    bboxes = {}
    for obb_file in obb_files:
        # 从文件名提取link_id
        basename = os.path.basename(obb_file)  # obb_link_0.json
        link_id = int(basename.replace("obb_link_", "").replace(".json", ""))
        
        with open(obb_file, "r") as f:
            obb_data = json.load(f)
        
        # 四舍五入到小数点后2位
        bboxes[str(link_id)] = {
            "center": [round(x, 2) for x in obb_data["center"]],
            "R": [[round(x, 2) for x in row] for row in obb_data["R"]],
            "extent": [round(x, 2) for x in obb_data["extent"]],
        }
    
    return bboxes


def parse_urdf_for_joints(urdf_file):
    """
    从URDF文件解析关节信息
    
    返回: {
        'root_link': str,  # 基础link名称
        'joints': [
            {
                'name': str,
                'type': str,  # 'revolute', 'prismatic', 'fixed'
                'child': str,
                'parent': str,
                'axis': [x, y, z],
                'origin_xyz': [x, y, z]
            },
            ...
        ]
    }
    """
    if not os.path.exists(urdf_file):
        return None
    
    try:
        tree = ET.parse(urdf_file)
        root = tree.getroot()
        
        joints_data = []
        all_links = set()
        child_links = set()
        
        # 收集所有link
        for link in root.findall('link'):
            link_name = link.get('name')
            all_links.add(link_name)
        
        # 解析关节
        for joint in root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')
            
            child_elem = joint.find('child')
            parent_elem = joint.find('parent')
            axis_elem = joint.find('axis')
            origin_elem = joint.find('origin')
            
            if child_elem is not None and parent_elem is not None:
                child = child_elem.get('link')
                parent = parent_elem.get('link')
                child_links.add(child)
                
                # 解析轴向
                axis = [0, 0, 1]  # 默认z轴
                if axis_elem is not None:
                    axis_str = axis_elem.get('xyz', '0 0 1')
                    axis = [float(x) for x in axis_str.split()]
                
                # 解析位置
                origin_xyz = [0, 0, 0]
                if origin_elem is not None:
                    xyz_str = origin_elem.get('xyz', '0 0 0')
                    origin_xyz = [float(x) for x in xyz_str.split()]
                
                joints_data.append({
                    'name': joint_name,
                    'type': joint_type,
                    'child': child,
                    'parent': parent,
                    'axis': axis,
                    'origin_xyz': origin_xyz
                })
        
        # 找出root link (不是任何关节的子节点)
        root_links = all_links - child_links
        root_link = list(root_links)[0] if root_links else 'base'
        
        return {
            'root_link': root_link,
            'joints': joints_data
        }
        
    except Exception as e:
        print(f"警告: 解析URDF失败 {urdf_file}: {e}")
        return None


def generate_label_code_from_urdf(urdf_data, num_obbs):
    """
    根据URDF数据和OBB数量生成label_code
    
    参数:
        urdf_data: parse_urdf_for_joints的返回值
        num_obbs: OBB数量
    
    返回:
        label_code: 字符串格式的关节代码
    """
    if not urdf_data or num_obbs == 0:
        return ""
    
    # 假设最后一个OBB是root (通常是固定的基座)
    root_geom = num_obbs - 1
    
    child_joints = []
    
    # 处理每个关节
    for joint in urdf_data['joints']:
        if joint['type'] == 'fixed':
            continue  # 跳过固定关节
        
        # 从link名称提取ID (link_0 -> 0)
        try:
            child_link = joint['child']
            if 'link_' in child_link:
                box_id = int(child_link.split('_')[-1])
            else:
                continue
                
            # 确定关节类型
            if joint['type'] == 'revolute' or joint['type'] == 'continuous':
                joint_type = 'hinge'
            elif joint['type'] == 'prismatic':
                joint_type = 'slide'
            else:
                continue
            
            # 确定旋转轴索引
            axis = joint['axis']
            # 找到最大分量的索引
            abs_axis = [abs(a) for a in axis]
            idx = abs_axis.index(max(abs_axis))
            
            # 确定符号
            sign = 1 if axis[idx] > 0 else -1
            
            # 边缘坐标（简化处理，使用原点位置）
            origin = joint['origin_xyz']
            # 使用非主轴的坐标作为edge
            edge = [round(origin[i], 2) for i in range(3) if i != idx][:2]
            if len(edge) < 2:
                edge = [0.0, 0.0]
            
            child_joints.append(
                f"dict(box={box_id},type='{joint_type}',idx={idx},edge={edge},sign={sign})"
            )
            
        except Exception as e:
            print(f"警告: 处理关节 {joint['name']} 时出错: {e}")
            continue
    
    # 生成label_code
    label_code = f"root_geom = {root_geom}\n"
    label_code += "child_joints = [\n"
    for joint_str in child_joints:
        label_code += joint_str + ",\n"
    label_code += "]"
    
    return label_code




def process_object(shape_obj_folder, mobility_obj_folder, mode="obb_rel"):
    """
    处理单个对象的所有loop
    
    参数:
        shape_obj_folder: shape_output中的对象文件夹
        mobility_obj_folder: mobility_dataset_v2中的对象文件夹
        mode: 训练模式
    
    返回: {loop_id: {bbox_code, label_code}, ...}
    """
    obj_data = {}
    
    # 从mobility_dataset_v2读取URDF文件
    urdf_data = None
    
    if mobility_obj_folder and os.path.exists(mobility_obj_folder):
        urdf_file = os.path.join(mobility_obj_folder, "mobility.urdf")
        if os.path.exists(urdf_file):
            urdf_data = parse_urdf_for_joints(urdf_file)
    
    if not urdf_data:
        print(f"  ⚠ 未找到URDF文件，跳过此对象")
        return obj_data
    
    print(f"  ✓ 解析到 {len(urdf_data['joints'])} 个关节")
    
    # 遍历所有loop文件夹
    loop_folders = natsorted(glob(os.path.join(shape_obj_folder, "loop_*")))
    
    for loop_folder in loop_folders:
        loop_id = os.path.basename(loop_folder).replace("loop_", "")
        
        # 读取OBB数据
        try:
            bboxes = load_obb_from_loop(loop_folder)
            if len(bboxes) == 0:
                continue
            
            # 生成bbox_code
            bbox_code = translate_dict_to_code(bboxes)
            
            # 生成label_code
            label_code = generate_label_code_from_urdf(urdf_data, len(bboxes))
            
            obj_data[loop_id] = {
                "bbox_code": bbox_code,
                "label_code": label_code,
            }
        except Exception as e:
            print(f"  ✗ Error processing {loop_folder}: {e}")
            continue
    
    return obj_data


def prepare_split(shape_root, mobility_root, split="train"):
    """
    准备train或test数据
    
    参数:
        shape_root: shape_output根目录
        mobility_root: mobility_dataset_v2根目录
        split: "train" 或 "test"
    
    返回: {obj_id_loop_id: {mode: {bbox_code, label_code}}, ...}
    """
    merged_data = {}
    
    # 查找所有对象类型
    shape_split_dir = os.path.join(shape_root, split)
    mobility_split_dir = os.path.join(mobility_root, split)
    
    if not os.path.exists(shape_split_dir):
        print(f"❌ Shape directory not found: {shape_split_dir}")
        return merged_data
    
    if not os.path.exists(mobility_split_dir):
        print(f"❌ Mobility directory not found: {mobility_split_dir}")
        return merged_data
    
    obj_types = [d for d in os.listdir(shape_split_dir) 
                 if os.path.isdir(os.path.join(shape_split_dir, d))]
    
    print(f"\n处理 {split} 集，发现 {len(obj_types)} 个对象类型")
    
    total_objects = 0
    total_loops = 0
    
    for obj_type in obj_types:
        shape_type_dir = os.path.join(shape_split_dir, obj_type)
        mobility_type_dir = os.path.join(mobility_split_dir, obj_type)
        
        if not os.path.exists(mobility_type_dir):
            print(f"⚠ {obj_type}: mobility目录不存在，跳过")
            continue
        
        # 查找所有对象ID
        obj_ids = [d for d in os.listdir(shape_type_dir) 
                   if os.path.isdir(os.path.join(shape_type_dir, d))]
        
        print(f"\n类型 {obj_type}: {len(obj_ids)} 个对象")
        
        for obj_id in obj_ids:
            shape_obj_folder = os.path.join(shape_type_dir, obj_id)
            mobility_obj_folder = os.path.join(mobility_type_dir, obj_id)
            
            print(f"  {obj_id}:", end=" ")
            
            # 处理这个对象的所有loop
            obj_data = process_object(shape_obj_folder, mobility_obj_folder)
            
            # 将每个loop作为独立样本
            for loop_id, loop_data in obj_data.items():
                sample_id = f"{obj_type}_{obj_id}_loop_{loop_id}"
                merged_data[sample_id] = {
                    "obb_rel": loop_data
                }
                total_loops += 1
            
            if obj_data:
                print(f"✓ {len(obj_data)} 个loop")
                total_objects += 1
            else:
                print("⚠ 无有效数据")
    
    print(f"\n总计: {total_objects} 个对象, {total_loops} 个训练样本")
    
    return merged_data


def main():
    parser = argparse.ArgumentParser(
        description="从shape_output和mobility_dataset_v2准备训练数据"
    )
    parser.add_argument(
        "--shape_dir",
        type=str,
        required=True,
        help="shape_output目录路径",
    )
    parser.add_argument(
        "--mobility_dir",
        type=str,
        default="/mnt/data/zhangzhaodong/real2code/datasets/mobility_dataset_v2",
        help="mobility_dataset_v2目录路径（包含URDF文件）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="输出目录，将保存train.json和test.json",
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 80)
    print("Real2Code 数据准备脚本")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  Shape目录: {args.shape_dir}")
    print(f"  Mobility目录: {args.mobility_dir}")
    print(f"  输出目录: {args.out_dir}")
    print()
    
    # 处理train和test数据
    print("=" * 80)
    print("处理训练数据...")
    print("=" * 80)
    train_data = prepare_split(args.shape_dir, args.mobility_dir, split="train")
    
    print("\n" + "=" * 80)
    print("处理测试数据...")
    print("=" * 80)
    test_data = prepare_split(args.shape_dir, args.mobility_dir, split="test")
    
    # 保存结果
    train_file = os.path.join(args.out_dir, "train.json")
    test_file = os.path.join(args.out_dir, "test.json")
    
    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_file, "w") as f:
        json.dump(test_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ 数据准备完成!")
    print("=" * 80)
    print(f"\n统计信息:")
    print(f"  训练样本数: {len(train_data)}")
    print(f"  测试样本数: {len(test_data)}")
    print(f"  总样本数: {len(train_data) + len(test_data)}")
    print(f"\n输出文件:")
    print(f"  训练数据: {train_file}")
    print(f"  测试数据: {test_file}")
    print(f"\n下一步:")
    print(f"  cd {args.out_dir}")
    print(f"  python train_shape.py")
    print()


if __name__ == "__main__":
    main()

