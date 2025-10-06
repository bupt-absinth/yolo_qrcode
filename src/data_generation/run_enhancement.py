"""
运行小程序码增强的脚本
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_generation.enhance_miniprogram_codes import MiniProgramCodeEnhancer


def main():
    """主函数"""
    print("开始增强小程序码图像...")
    
    # 创建增强器
    enhancer = MiniProgramCodeEnhancer(
        input_dir="data/mini_program_codes",
        output_dir="data/enhanced_miniprogram_codes"
    )
    
    # 增强所有图像
    success_count = enhancer.enhance_batch()
    
    print(f"\n增强完成!")
    print(f"成功增强图像数量: {success_count}")
    print(f"增强后的图像保存在: data/enhanced_miniprogram_codes/")


if __name__ == "__main__":
    main()