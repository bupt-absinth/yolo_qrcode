"""
生成测试样本以验证修复后的类别ID
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_generation.synthetic_data_generator import SyntheticDataGenerator


def main():
    """主函数"""
    print("=== 测试样本生成器 ===")
    
    # 创建生成器
    generator = SyntheticDataGenerator()
    
    # 生成少量测试样本以验证修复
    print("\n开始生成测试样本...")
    success_count = generator.generate_batch(10)  # 生成10个样本进行测试
    
    print(f"\n=== 生成完成 ===")
    print(f"成功生成 {success_count} 个测试样本")
    
    # 检查生成的标签文件
    print("\n检查生成的标签文件:")
    labels_dir = os.path.join(generator.output_dir, "labels")
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    for label_file in label_files[:5]:  # 检查前5个标签文件
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            content = f.read().strip()
            print(f"{label_file}: {content}")
    
    print(f"\n标签文件保存在: {labels_dir}")


if __name__ == "__main__":
    main()