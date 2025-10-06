"""
修复合成数据中的标签类别ID
确保所有标签文件的类别ID都是0
"""

import os
import sys
from typing import List


def fix_label_class_id(label_path: str) -> bool:
    """
    修复单个标签文件的类别ID
    
    Args:
        label_path: 标签文件路径
        
    Returns:
        bool: 修复是否成功
    """
    try:
        # 读取标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 修复类别ID
        fixed_lines = []
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 5:
                    # 将类别ID（第一个元素）设置为0
                    parts[0] = '0'
                    fixed_lines.append(' '.join(parts) + '\n')
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # 写回文件
        with open(label_path, 'w') as f:
            f.writelines(fixed_lines)
        
        return True
        
    except Exception as e:
        print(f"修复标签文件失败 {label_path}: {e}")
        return False


def batch_fix_label_class_ids(labels_dir: str) -> int:
    """
    批量修复标签文件的类别ID
    
    Args:
        labels_dir: 标签文件目录
        
    Returns:
        int: 成功修复的文件数量
    """
    if not os.path.exists(labels_dir):
        print(f"目录不存在: {labels_dir}")
        return 0
    
    # 获取所有标签文件
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(f"找到 {len(label_files)} 个标签文件")
    
    success_count = 0
    
    for i, filename in enumerate(label_files):
        label_path = os.path.join(labels_dir, filename)
        
        if fix_label_class_id(label_path):
            success_count += 1
        
        # 显示进度
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1}/{len(label_files)} 个标签文件")
    
    print(f"批量处理完成，成功修复 {success_count}/{len(label_files)} 个标签文件")
    return success_count


def main():
    """主函数"""
    print("=== 修复合成数据标签类别ID ===")
    
    # 修复合成数据标签
    labels_dir = "data/synthetic/labels"
    if os.path.exists(labels_dir):
        batch_fix_label_class_ids(labels_dir)
    else:
        print(f"标签目录不存在: {labels_dir}")


if __name__ == "__main__":
    main()