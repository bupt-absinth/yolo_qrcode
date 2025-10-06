"""
二维码自动生成模块
"""

import os
import qrcode
import random
import string
from typing import List, Optional
from PIL import Image
import numpy as np


class QRCodeGenerator:
    """二维码生成器"""
    
    def __init__(self, save_dir: str = "data/qr_codes"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 容错级别选项
        self.error_correction_levels = [
            qrcode.ERROR_CORRECT_L,  # 7%
            qrcode.ERROR_CORRECT_M,  # 15%
            qrcode.ERROR_CORRECT_Q,  # 25%
            qrcode.ERROR_CORRECT_H   # 30%
        ]
        
        # 颜色选项
        self.colors = [
            ("black", "white"),
            ("darkblue", "lightgray"),
            ("darkgreen", "lightyellow"),
            ("darkred", "lightcyan"),
            ("purple", "lightpink"),
            ("brown", "lightblue")
        ]
    
    def generate_random_string(self, length: int = 50) -> str:
        """生成随机字符串作为二维码内容"""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))
    
    def generate_qr_code(self, 
                        content: Optional[str] = None, 
                        error_correction: Optional[int] = None, 
                        fill_color: Optional[str] = None, 
                        back_color: Optional[str] = None,
                        box_size: int = 10,
                        border: int = 4) -> Image.Image:
        """生成单个二维码图像"""
        if content is None:
            content = self.generate_random_string()
            
        if error_correction is None:
            error_correction = random.choice(self.error_correction_levels)
            
        if fill_color is None or back_color is None:
            fill_color, back_color = random.choice(self.colors)
        
        # 创建QR码实例
        qr = qrcode.QRCode(
            version=1,
            error_correction=error_correction,
            box_size=box_size,
            border=border,
        )
        
        # 添加数据
        qr.add_data(content)
        qr.make(fit=True)
        
        # 创建图像
        img = qr.make_image(fill_color=fill_color, back_color=back_color)
        # 转换为PIL Image对象以确保类型正确
        return img.get_image()
    
    def generate_batch(self, count: int = 1000, prefix: str = "qr") -> List[str]:
        """批量生成二维码并保存"""
        saved_files = []
        
        for i in range(count):
            # 生成随机内容和参数
            content = f"https://example.com/{self.generate_random_string(20)}"
            error_correction = random.choice(self.error_correction_levels)
            fill_color, back_color = random.choice(self.colors)
            
            # 生成二维码
            img = self.generate_qr_code(
                content=content,
                error_correction=error_correction,
                fill_color=fill_color,
                back_color=back_color
            )
            
            # 保存图像
            filename = f"{prefix}_{i:04d}.png"
            filepath = os.path.join(self.save_dir, filename)
            img.save(filepath)
            saved_files.append(filename)
            
            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1}/{count} 个二维码")
        
        print(f"成功生成 {count} 个二维码，保存在 {self.save_dir}")
        return saved_files


if __name__ == "__main__":
    # 示例使用
    generator = QRCodeGenerator("data/qr_codes")
    files = generator.generate_batch(100)  # 生成100个二维码作为示例