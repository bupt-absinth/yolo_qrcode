#!/usr/bin/env python3
"""
对比原图和增强后的图片
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

def compare_images(original_path, enhanced_path):
    """对比原图和增强后的图片"""
    # 打开图片
    original = Image.open(original_path)
    enhanced = Image.open(enhanced_path)
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示原图
    ax1.imshow(original)
    ax1.set_title("Original Mini Program Code")
    ax1.axis('off')
    
    # 显示增强后的图片
    ax2.imshow(enhanced)
    ax2.set_title("Enhanced Mini Program Code")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 图片路径
    original_path = "data/mini_program_codes/wechat_mp_000000.png"
    enhanced_path = "data/enhanced_miniprogram_codes/enhanced_wechat_mp_000000.png"
    
    if os.path.exists(original_path) and os.path.exists(enhanced_path):
        compare_images(original_path, enhanced_path)
    else:
        print("图片文件不存在，请确保已运行增强脚本")