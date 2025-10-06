"""
增强小程序码图像，使用背景图片替换中心圆形图片
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from typing import List, Tuple, Optional


class MiniProgramCodeEnhancer:
    """小程序码增强器"""
    
    def __init__(self, input_dir: str = "data/mini_program_codes", 
                 background_dir: str = "data/backgrounds",
                 output_dir: str = "data/enhanced_miniprogram_codes"):
        self.input_dir = input_dir
        self.background_dir = background_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取输入目录中的所有图片文件
        self.image_files = [f for f in os.listdir(input_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 获取背景图片文件
        self.background_files = [f for f in os.listdir(background_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"找到 {len(self.image_files)} 个小程序码图像")
        print(f"找到 {len(self.background_files)} 张背景图片")
    
    def extract_random_circle_from_background(self, size: Tuple[int, int]) -> Image.Image:
        """从随机背景图片中提取圆形区域"""
        width, height = size
        
        # 随机选择一张背景图片
        if not self.background_files:
            # 如果没有背景图片，生成纯色圆形作为备选
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            circle_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))  # type: ignore
            draw = ImageDraw.Draw(circle_img)
            draw.ellipse([0, 0, width, height], fill=color)
            return circle_img
        
        background_file = random.choice(self.background_files)
        background_path = os.path.join(self.background_dir, background_file)
        
        try:
            # 打开背景图片
            background = Image.open(background_path).convert('RGBA')
            bg_width, bg_height = background.size
            
            # 确保背景图片足够大
            if bg_width < width or bg_height < height:
                # 如果背景图片太小，调整大小
                scale = max(width / bg_width, height / bg_height)
                new_width = int(bg_width * scale * 1.5)  # 稍微大一些
                new_height = int(bg_height * scale * 1.5)
                background = background.resize((new_width, new_height), Image.Resampling.LANCZOS)
                bg_width, bg_height = background.size
            
            # 随机选择截取位置
            max_x = bg_width - width
            max_y = bg_height - height
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # 截取方形区域
            square_region = background.crop((x, y, x + width, y + height))
            
            # 创建圆形遮罩
            mask = Image.new('L', (width, height), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.ellipse([0, 0, width, height], fill=255)
            
            # 应用圆形遮罩，只保留圆形区域
            circular_region = Image.new('RGBA', (width, height), (0, 0, 0, 0))  # type: ignore
            circular_region.paste(square_region, (0, 0), mask)
            
            return circular_region
            
        except Exception as e:
            print(f"从背景图片提取圆形区域失败: {e}")
            # 出错时返回纯色圆形
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            circle_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))  # type: ignore
            draw = ImageDraw.Draw(circle_img)
            draw.ellipse([0, 0, width, height], fill=color)
            return circle_img
    
    def replace_center_circle(self, image: Image.Image) -> Image.Image:
        """使用背景图片中的圆形区域替换小程序码中心的圆形部分"""
        # 复制原图像
        enhanced_image = image.copy()
        width, height = enhanced_image.size
        
        # 计算中心圆形区域（小程序码中心的同心圆结构）
        # 中心圆形区域的直径是整张图片大小的一半
        # center_size = min(width, height) // 2
        center_size = int(min(width, height) * 0.449)
        center_x = width // 2
        center_y = height // 2
        
        # 从背景图片中提取圆形区域
        background_circle = self.extract_random_circle_from_background((center_size, center_size))
        
        # 创建用于粘贴的圆形图像（与原图同样大小）
        circle_on_canvas = Image.new('RGBA', (width, height), (0, 0, 0, 0))  # type: ignore
        circle_x = center_x - center_size // 2
        circle_y = center_y - center_size // 2
        circle_on_canvas.paste(background_circle, (circle_x, circle_y))
        
        # 创建遮罩，标识需要替换的中心圆形区域
        mask = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([
            center_x - center_size // 2,
            center_y - center_size // 2,
            center_x + center_size // 2,
            center_y + center_size // 2
        ], fill=255)
        
        # 使用遮罩将原图的中心圆形区域替换为背景圆形区域
        enhanced_image.paste(circle_on_canvas, (0, 0), mask)
        
        return enhanced_image
    
    def enhance_single_image(self, image_filename: str) -> bool:
        """增强单张图像"""
        try:
            # 加载图像
            image_path = os.path.join(self.input_dir, image_filename)
            image = Image.open(image_path).convert('RGBA')
            
            # 使用背景图片替换中心圆形部分
            enhanced_image = self.replace_center_circle(image)
            
            # 保存增强后的图像
            output_path = os.path.join(self.output_dir, f"enhanced_{image_filename}")
            enhanced_image.save(output_path, "PNG")
            
            return True
        except Exception as e:
            print(f"增强图像失败 {image_filename}: {e}")
            return False
    
    def enhance_batch(self, count: Optional[int] = None) -> int:
        """批量增强图像"""
        if count is None:
            count = len(self.image_files)
        
        success_count = 0
        
        for i, image_file in enumerate(self.image_files[:count]):
            if self.enhance_single_image(image_file):
                success_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"已增强 {i + 1}/{min(count, len(self.image_files))} 张图像")
        
        print(f"批量增强完成，成功增强 {success_count}/{min(count, len(self.image_files))} 张图像")
        return success_count


if __name__ == "__main__":
    # 示例使用
    enhancer = MiniProgramCodeEnhancer()
    
    # 增强前10张图像作为测试
    success_count = enhancer.enhance_batch(10)