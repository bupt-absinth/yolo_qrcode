"""
测试小程序码增强功能
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_generation.enhance_miniprogram_codes import MiniProgramCodeEnhancer


def main():
    """测试主函数"""
    print("测试小程序码增强功能...")
    
    # 创建增强器
    enhancer = MiniProgramCodeEnhancer(
        input_dir="data/mini_program_codes",
        output_dir="data/test_enhanced"
    )
    
    # 测试增强少量图像
    print("开始测试增强功能...")
    success_count = enhancer.enhance_batch(5)  # 只测试5张图像
    
    print(f"\n测试完成!")
    print(f"成功增强图像数量: {success_count}")
    print(f"测试结果保存在: data/test_enhanced/")


if __name__ == "__main__":
    main()