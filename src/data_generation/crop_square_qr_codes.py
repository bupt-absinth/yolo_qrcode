"""
裁剪增强后的方形二维码图像
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from PIL import Image


def crop_image(image: Image.Image) -> Image.Image:
    """裁剪图片，去除上部、左侧、右侧各约1/10，底部约1/5"""
    width, height = image.size
    
    # 计算裁剪边界
    # 上部裁剪1/10
    m = 20
    top_crop = height // m
    # 左侧裁剪1/10
    left_crop = width // m
    # 右侧裁剪1/10
    right_crop = width - (width // m)
    # 底部裁剪1/5
    bottom_crop = height - (height // 6.2)
    
    # 执行裁剪
    cropped_image = image.crop((left_crop, top_crop, right_crop, bottom_crop))
    return cropped_image


def crop_batch_images(input_dir: str = "data/enhanced_square_qr_codes", 
                     output_dir: str = "data/croped_square_qr_codes") -> int:
    """批量裁剪图片"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入目录中的所有图片文件
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"找到 {len(image_files)} 个待裁剪图像")
    
    success_count = 0
    
    for i, image_file in enumerate(image_files):
        try:
            # 加载图像
            image_path = os.path.join(input_dir, image_file)
            image = Image.open(image_path).convert('RGBA')
            
            # 裁剪图像
            cropped_image = crop_image(image)
            
            # 保存裁剪后的图像
            output_path = os.path.join(output_dir, image_file)
            cropped_image.save(output_path, "PNG")
            
            success_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"已裁剪 {i + 1}/{len(image_files)} 张图像")
                
        except Exception as e:
            print(f"裁剪图像失败 {image_file}: {e}")
    
    print(f"批量裁剪完成，成功裁剪 {success_count}/{len(image_files)} 张图像")
    return success_count


def main():
    """主函数"""
    print("=== 方形二维码裁剪流程 ===")
    
    # 裁剪增强后的方形二维码
    print("\n开始裁剪增强后的方形二维码...")
    cropped_count = crop_batch_images()
    print(f"裁剪完成，共裁剪 {cropped_count} 个方形二维码")
    
    print("\n=== 流程完成 ===")
    print("裁剪后的方形二维码保存在: data/croped_square_qr_codes/")


if __name__ == "__main__":
    main()