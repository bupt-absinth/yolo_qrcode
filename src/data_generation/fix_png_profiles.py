"""
修复PNG图像中的ICC配置文件问题
消除libpng警告信息
"""

import os
import sys
from PIL import Image
import numpy as np
from typing import List, Optional
import warnings


def fix_png_profile(image_path: str, output_path: Optional[str] = None) -> bool:
    """
    修复PNG图像中的ICC配置文件问题
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径（如果为None，则覆盖原文件）
        
    Returns:
        bool: 修复是否成功
    """
    try:
        # 打开图像
        with Image.open(image_path) as img:
            # 转换为RGB模式（移除ICC配置文件）
            if img.mode in ('RGBA', 'LA', 'P'):
                # 保留透明通道
                if img.mode == 'P':
                    img = img.convert('RGBA')
                
                # 创建新的RGB图像
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))  # type: ignore
                if img.mode == 'RGBA':
                    rgb_img.paste(img, mask=img.split()[-1])  # 使用alpha通道作为掩码
                else:
                    rgb_img.paste(img)
            else:
                # 直接转换为RGB
                rgb_img = img.convert('RGB')
            
            # 保存图像（不包含ICC配置文件）
            save_path = output_path if output_path else image_path
            rgb_img.save(save_path, 'PNG', icc_profile=None)
            
        return True
        
    except Exception as e:
        print(f"修复图像失败 {image_path}: {e}")
        return False


def batch_fix_png_profiles(input_dir: str, output_dir: Optional[str] = None) -> int:
    """
    批量修复PNG图像中的ICC配置文件问题
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（如果为None，则覆盖原文件）
        
    Returns:
        int: 成功修复的图像数量
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有PNG文件
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    print(f"找到 {len(png_files)} 个PNG文件")
    
    success_count = 0
    
    for i, filename in enumerate(png_files):
        input_path = os.path.join(input_dir, filename)
        
        if output_dir:
            output_path = os.path.join(output_dir, filename)
        else:
            output_path = None  # 覆盖原文件
        
        if fix_png_profile(input_path, output_path):
            success_count += 1
        
        # 显示进度
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{len(png_files)} 个文件")
    
    print(f"批量处理完成，成功修复 {success_count}/{len(png_files)} 个文件")
    return success_count


def fix_synthetic_data_profiles() -> None:
    """修复合成数据中的PNG图像配置文件"""
    print("=== 修复合成数据中的PNG图像配置文件 ===")
    
    # 修复合成数据中的图像
    synthetic_images_dir = "data/synthetic/images"
    if os.path.exists(synthetic_images_dir):
        print(f"修复合成数据图像: {synthetic_images_dir}")
        batch_fix_png_profiles(synthetic_images_dir)
    
    # 修复其他可能的PNG图像目录
    png_dirs = [
        "data/mini_program_codes",
        "data/enhanced_miniprogram_codes",
        "data/square_qr_codes",
        "data/enhanced_square_qr_codes",
        "data/croped_square_qr_codes",
        "data/qr_codes",
        "data/swift_trail_formatted/images"
    ]
    
    for png_dir in png_dirs:
        if os.path.exists(png_dir):
            print(f"修复图像目录: {png_dir}")
            batch_fix_png_profiles(png_dir)


def main():
    """主函数"""
    print("=== PNG图像ICC配置文件修复工具 ===")
    
    # 询问用户要修复哪个目录
    print("\n选择要修复的目录:")
    print("1. 合成数据 (data/synthetic/images)")
    print("2. 所有PNG图像目录")
    print("3. 自定义目录")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        # 仅修复合成数据
        synthetic_images_dir = "data/synthetic/images"
        if os.path.exists(synthetic_images_dir):
            batch_fix_png_profiles(synthetic_images_dir)
        else:
            print(f"目录不存在: {synthetic_images_dir}")
            
    elif choice == "2":
        # 修复所有PNG图像目录
        fix_synthetic_data_profiles()
        
    elif choice == "3":
        # 自定义目录
        custom_dir = input("请输入要修复的目录路径: ").strip()
        if os.path.exists(custom_dir):
            backup = input("是否备份原文件? (y/n): ").strip().lower() == 'y'
            if backup:
                output_dir = custom_dir + "_fixed"
                batch_fix_png_profiles(custom_dir, output_dir)
            else:
                batch_fix_png_profiles(custom_dir)
        else:
            print(f"目录不存在: {custom_dir}")
    else:
        print("无效选择")


if __name__ == "__main__":
    main()