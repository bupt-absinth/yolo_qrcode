"""
小程序码自动生成模块
"""

import os
import random
import string
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFilter
import numpy as np


class MiniProgramCodeGenerator:
    """小程序码生成器"""
    
    def __init__(self, save_dir: str = "data/mini_program_codes"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 颜色选项 (RGB格式)
        self.colors = [
            ((0, 0, 0), (255, 255, 255)),      # black, white
            ((0, 0, 139), (211, 211, 211)),    # darkblue, lightgray
            ((0, 100, 0), (255, 255, 224)),    # darkgreen, lightyellow
            ((139, 0, 0), (224, 255, 255)),    # darkred, lightcyan
            ((128, 0, 128), (255, 182, 193)),  # purple, lightpink
            ((165, 42, 42), (173, 216, 230))   # brown, lightblue
        ]
        
        # Logo形状选项
        self.logo_shapes = ['circle', 'square', 'triangle']
    
    def generate_random_string(self, length: int = 50) -> str:
        """生成随机字符串作为小程序码内容"""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))
    
    def generate_logo(self, size: int = 60, shape: Optional[str] = None) -> Image.Image:
        """生成随机Logo"""
        if shape is None:
            shape = random.choice(self.logo_shapes)
            
        logo = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(logo)
        
        # 随机颜色
        color = tuple(random.choices(range(50, 200), k=3))
        
        if shape == 'circle':
            draw.ellipse([0, 0, size, size], fill=color)
        elif shape == 'square':
            draw.rectangle([0, 0, size, size], fill=color)
        elif shape == 'triangle':
            points = [(size//2, 0), (0, size), (size, size)]
            draw.polygon(points, fill=color)
            
        return logo
    
    def generate_mini_program_code(self,
                                 content: Optional[str] = None,
                                 scale: int = 10,
                                 dark: Optional[Tuple[int, int, int]] = None,
                                 light: Optional[Tuple[int, int, int]] = None,
                                 add_logo: bool = True) -> Image.Image:
        """生成单个小程序码图像（简化版本，使用二维码替代）"""
        if content is None:
            content = f"wxp://mini-program/{self.generate_random_string(20)}"
            
        if dark is None or light is None:
            dark, light = random.choice(self.colors)
        
        # 创建一个圆形的小程序码替代图像
        size = scale * 25  # 大约25x25模块
        img = Image.new('RGB', (size, size), light)
        draw = ImageDraw.Draw(img)
        
        # 绘制外圆
        draw.ellipse([0, 0, size, size], outline=dark, width=2)
        
        # 绘制内部图案
        inner_size = size // 2
        inner_pos = (size - inner_size) // 2
        draw.ellipse([inner_pos, inner_pos, inner_pos + inner_size, inner_pos + inner_size], fill=dark)
        
        # 添加Logo
        if add_logo:
            logo = self.generate_logo(size=min(img.size)//5)
            logo_pos = (img.size[0]//2 - logo.size[0]//2, 
                       img.size[1]//2 - logo.size[1]//2)
            img.paste(logo, logo_pos, logo)
        
        return img
    
    def generate_batch(self, count: int = 1000, prefix: str = "mp") -> List[str]:
        """批量生成小程序码并保存"""
        saved_files = []
        
        for i in range(count):
            # 生成随机内容和参数
            content = f"wxp://mini-program/{self.generate_random_string(20)}"
            dark, light = random.choice(self.colors)
            add_logo = random.choice([True, False])
            
            # 生成小程序码
            img = self.generate_mini_program_code(
                content=content,
                dark=dark,
                light=light,
                add_logo=add_logo
            )
            
            # 保存图像
            filename = f"{prefix}_{i:04d}.png"
            filepath = os.path.join(self.save_dir, filename)
            img.save(filepath)
            saved_files.append(filename)
            
            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1}/{count} 个小程序码")
        
        print(f"成功生成 {count} 个小程序码，保存在 {self.save_dir}")
        return saved_files


if __name__ == "__main__":
    # 示例使用
    generator = MiniProgramCodeGenerator("data/mini_program_codes")
    files = generator.generate_batch(100)  # 生成100个小程序码作为示例