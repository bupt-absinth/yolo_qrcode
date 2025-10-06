#!/usr/bin/env python3
"""
测试小程序码增强功能
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
data_generation_path = os.path.join(src_path, 'data_generation')

sys.path.insert(0, src_path)
sys.path.insert(0, data_generation_path)

from enhance_miniprogram_codes import MiniProgramCodeEnhancer

def main():
    # 创建增强器实例
    enhancer = MiniProgramCodeEnhancer()
    
    # 测试增强前几张图片
    print("开始测试小程序码增强功能...")
    success_count = enhancer.enhance_batch(5)  # 只测试前5张图片
    print(f"测试完成，成功增强 {success_count}/5 张图片")

if __name__ == "__main__":
    main()