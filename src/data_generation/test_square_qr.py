"""
测试方形二维码生成和增强功能
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_generation.wechat_miniprogram_generator import WeChatMiniProgramCodeGenerator
from src.data_generation.generate_and_enhance_miniprogram_codes import SquareQRCodeEnhancer
from PIL import Image
from io import BytesIO


def test_square_qr_generation():
    """测试方形二维码生成功能"""
    print("测试方形二维码生成功能...")
    
    # 使用您提供的微信小程序凭证
    APPID = "wx2feb1f6c99f73709"
    APPSECRET = "d5c822694247fcf18f72686812b07872"
    
    # 创建生成器
    generator = WeChatMiniProgramCodeGenerator(APPID, APPSECRET)
    
    # 创建保存方形二维码的目录
    square_qr_dir = "data/test_square_qr_codes"
    os.makedirs(square_qr_dir, exist_ok=True)
    
    # 生成一个方形二维码作为测试
    path = "pages/index/index?param=test_000001"
    image_data = generator.generate_qr_code(path=path, width=430)
    
    if image_data:
        try:
            # 保存图像，文件名为 wechat_sp_000001.png
            image = Image.open(BytesIO(image_data))
            filename = "wechat_sp_000001.png"
            filepath = os.path.join(square_qr_dir, filename)
            image.save(filepath, "PNG")
            print(f"成功生成方形二维码: {filename}")
            return True
        except Exception as e:
            print(f"保存方形二维码失败: {e}")
            return False
    else:
        print("生成方形二维码失败")
        return False


def test_square_qr_enhancement():
    """测试方形二维码增强功能"""
    print("\n测试方形二维码增强功能...")
    
    # 创建增强器
    enhancer = SquareQRCodeEnhancer(
        input_dir="data/test_square_qr_codes",
        output_dir="data/test_enhanced_square_qr_codes"
    )
    
    # 增强一张图像作为测试
    success_count = enhancer.enhance_batch(1)
    print(f"成功增强图像数量: {success_count}")
    return success_count > 0


def main():
    """主函数"""
    print("=== 测试方形二维码生成与增强功能 ===")
    
    # 测试方形二维码生成
    if test_square_qr_generation():
        # 测试方形二维码增强
        if test_square_qr_enhancement():
            print("\n所有测试通过!")
        else:
            print("\n增强功能测试失败!")
    else:
        print("\n生成功能测试失败!")


if __name__ == "__main__":
    main()