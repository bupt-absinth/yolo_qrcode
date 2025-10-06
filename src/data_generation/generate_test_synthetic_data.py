"""
测试版合成数据生成器
快速生成小规模合成数据集用于测试
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_generation.synthetic_data_generator import SyntheticDataGenerator


def main():
    """主函数"""
    print("=== 测试版合成数据生成器 ===")
    
    # 创建生成器
    generator = SyntheticDataGenerator()
    
    # 生成测试规模的合成样本 (1000个)
    print("\n开始生成测试规模合成数据...")
    print("预计生成1,000个合成样本...")
    
    success_count = generator.generate_batch(1000)
    
    print(f"\n=== 生成完成 ===")
    print(f"成功生成 {success_count} 个测试合成样本")
    print(f"合成数据保存在: {generator.output_dir}")


if __name__ == "__main__":
    main()