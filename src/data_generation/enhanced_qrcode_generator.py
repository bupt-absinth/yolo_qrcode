"""
增强版二维码生成器
支持Logo嵌入、颜色定制、样式调整等功能
"""

import os
import qrcode
import random
import string
import colorsys
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw


class EnhancedQRCodeGenerator:
    """增强版二维码生成器"""
    
    def __init__(self, 
                 save_dir: str = "data/qr_codes",
                 logo_dir: str = "data/logos",
                 background_dir: str = "data/backgrounds"):
        self.save_dir = save_dir
        self.logo_dir = logo_dir
        self.background_dir = background_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 容错级别选项
        self.error_correction_levels = [
            qrcode.ERROR_CORRECT_L,  # 7%
            qrcode.ERROR_CORRECT_M,  # 15%
            qrcode.ERROR_CORRECT_Q,  # 25%
            qrcode.ERROR_CORRECT_H   # 30%
        ]
        
        # 获取Logo文件
        self.logo_files = self._get_logo_files()
        print(f"找到 {len(self.logo_files)} 个Logo文件")
        
        # 获取背景文件
        self.background_files = self._get_background_files()
        print(f"找到 {len(self.background_files)} 张背景图片")
    
    def _get_logo_files(self) -> List[str]:
        """获取Logo文件列表"""
        if not os.path.exists(self.logo_dir):
            return []
        return [f for f in os.listdir(self.logo_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def _get_background_files(self) -> List[str]:
        """获取背景文件列表"""
        if not os.path.exists(self.background_dir):
            return []
        return [f for f in os.listdir(self.background_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def generate_random_string(self, length: int = 50) -> str:
        """生成随机字符串作为二维码内容"""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))
    
    def generate_random_color(self) -> Tuple[str, str]:
        """生成随机颜色对（前景色，背景色）"""
        # 70%概率使用HSL随机色，30%概率使用预定义颜色
        if random.random() < 0.7:
            # 生成HSL随机色
            hue = random.random()  # 色相 (0-1)
            saturation = random.uniform(0.5, 1.0)  # 饱和度 (0.5-1.0)
            lightness = random.uniform(0.2, 0.8)   # 亮度 (0.2-0.8)
            
            # 转换为RGB
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            fill_color = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
            
            # 背景色使用对比色或白色
            if lightness > 0.5:
                back_color = "white"
            else:
                back_color = "rgb(240, 240, 240)"  # 浅灰色
        else:
            # 使用预定义颜色
            colors = [
                ("black", "white"),
                ("darkblue", "lightgray"),
                ("darkgreen", "lightyellow"),
                ("darkred", "lightcyan"),
                ("purple", "lightpink"),
                ("brown", "lightblue"),
                ("#FFD700", "white"),  # 金色
                ("#FFA500", "white"),  # 橙色
                ("#800080", "lightyellow")  # 紫色
            ]
            fill_color, back_color = random.choice(colors)
        
        return fill_color, back_color
    
    def embed_logo(self, qr_img: Image.Image, logo_path: str, error_correction: int) -> Image.Image:
        """在二维码中心嵌入Logo"""
        try:
            # 打开Logo
            logo = Image.open(logo_path)
            
            # 根据容错级别确定Logo尺寸
            qr_width, qr_height = qr_img.size
            if error_correction == qrcode.ERROR_CORRECT_L:
                logo_size = min(qr_width, qr_height) // 6  # 最大Logo尺寸
            elif error_correction == qrcode.ERROR_CORRECT_M:
                logo_size = min(qr_width, qr_height) // 5
            elif error_correction == qrcode.ERROR_CORRECT_Q:
                logo_size = min(qr_width, qr_height) // 4
            else:  # ERROR_CORRECT_H
                logo_size = min(qr_width, qr_height) // 3  # 最小Logo尺寸
            
            # 调整Logo尺寸
            logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
            
            # 计算Logo位置（中心）
            logo_x = (qr_width - logo_size) // 2
            logo_y = (qr_height - logo_size) // 2
            
            # 创建透明背景的Logo图像
            logo_rgba = Image.new("RGBA", qr_img.size, (0, 0, 0, 0))  # type: ignore
            
            logo_rgba.paste(logo, (logo_x, logo_y))
            
            # 合并二维码和Logo
            combined = Image.alpha_composite(qr_img.convert("RGBA"), logo_rgba)
            return combined
        except Exception as e:
            print(f"嵌入Logo失败: {e}")
            return qr_img
    
    def apply_style_effects(self, img: Image.Image) -> Image.Image:
        """应用样式效果（如圆角点阵等）"""
        # 这里可以添加更多样式效果
        # 目前我们只返回原图，后续可以扩展
        return img
    
    def generate_enhanced_qr_code(self,
                                 content: Optional[str] = None,
                                 error_correction: Optional[int] = None,
                                 fill_color: Optional[str] = None,
                                 back_color: Optional[str] = None,
                                 with_logo: bool = True,
                                 with_style: bool = True) -> Image.Image:
        """生成增强版二维码"""
        if content is None:
            content = self.generate_random_string()
            
        if error_correction is None:
            error_correction = random.choice(self.error_correction_levels)
            
        if fill_color is None or back_color is None:
            fill_color, back_color = self.generate_random_color()
        
        # 创建QR码实例
        qr = qrcode.QRCode(
            version=1,
            error_correction=error_correction,
            box_size=10,
            border=4,
        )
        
        # 添加数据
        qr.add_data(content)
        qr.make(fit=True)
        
        # 创建图像
        img = qr.make_image(fill_color=fill_color, back_color=back_color)
        qr_img = img.get_image()
        
        # 应用样式效果
        if with_style:
            qr_img = self.apply_style_effects(qr_img)
        
        # 嵌入Logo
        if with_logo and self.logo_files:
            logo_file = random.choice(self.logo_files)
            logo_path = os.path.join(self.logo_dir, logo_file)
            qr_img = self.embed_logo(qr_img, logo_path, error_correction)
        
        return qr_img
    
    def generate_batch(self, count: int = 1000, prefix: str = "enhanced_qr") -> List[str]:
        """批量生成增强版二维码并保存"""
        saved_files = []
        
        for i in range(count):
            # 生成随机内容和参数
            content = f"https://example.com/{self.generate_random_string(20)}"
            error_correction = random.choice(self.error_correction_levels)
            
            # 随机决定是否添加Logo（70%概率）
            with_logo = random.random() < 0.7
            
            # 随机决定是否应用样式（80%概率）
            with_style = random.random() < 0.8
            
            # 生成增强版二维码
            try:
                img = self.generate_enhanced_qr_code(
                    content=content,
                    error_correction=error_correction,
                    with_logo=with_logo,
                    with_style=with_style
                )
                
                # 保存图像
                filename = f"{prefix}_{i:06d}.png"
                filepath = os.path.join(self.save_dir, filename)
                img.save(filepath, "PNG")
                saved_files.append(filename)
                
                if (i + 1) % 100 == 0:
                    print(f"已生成 {i + 1}/{count} 个增强版二维码")
            except Exception as e:
                print(f"生成二维码失败 {i}: {e}")
        
        print(f"成功生成 {len(saved_files)} 个增强版二维码，保存在 {self.save_dir}")
        return saved_files


def main():
    """主函数"""
    print("=== 增强版二维码生成器 ===")
    
    # 创建生成器
    generator = EnhancedQRCodeGenerator()
    
    # 生成示例二维码
    print("\n开始生成增强版二维码...")
    files = generator.generate_batch(100)  # 生成100个作为示例
    
    print(f"\n生成完成，共生成 {len(files)} 个二维码")
    print(f"二维码保存在: {generator.save_dir}")


if __name__ == "__main__":
    main()