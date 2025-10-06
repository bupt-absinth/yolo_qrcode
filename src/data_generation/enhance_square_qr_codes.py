"""
增强方形二维码图像，使用背景图片替换中心圆形图片
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_generation.generate_and_enhance_miniprogram_codes import SquareQRCodeEnhancer


def main():
    """主函数"""
    print("=== 方形二维码增强流程 ===")
    
    # 增强方形二维码
    print("\n开始增强方形二维码（替换中心圆形图片）...")
    square_enhancer = SquareQRCodeEnhancer(
        input_dir="data/square_qr_codes",
        output_dir="data/enhanced_square_qr_codes"
    )
    enhanced_square_count = square_enhancer.enhance_batch()
    print(f"增强完成，共增强 {enhanced_square_count} 个方形二维码")
    
    print("\n=== 流程完成 ===")
    print("增强后的方形二维码保存在: data/enhanced_square_qr_codes/")


if __name__ == "__main__":
    main()