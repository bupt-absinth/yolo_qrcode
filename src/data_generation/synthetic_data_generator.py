"""
合成训练数据生成器
将背景图与各种二维码图片合成生成训练样本
"""

import os
import random
from PIL import Image, ImageEnhance
import numpy as np
from typing import List, Tuple


class SyntheticDataGenerator:
    """合成数据生成器"""
    
    def __init__(self,
                 background_dir: str = "data/backgrounds",
                 miniprogram_dir: str = "data/enhanced_miniprogram_codes",
                 square_qr_dir: str = "data/croped_square_qr_codes",
                 qr_dir: str = "data/qr_codes",
                 output_dir: str = "data/synthetic"):
        self.background_dir = background_dir
        self.miniprogram_dir = miniprogram_dir
        self.square_qr_dir = square_qr_dir
        self.qr_dir = qr_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
        
        # 获取各类图片文件
        self.background_files = self._get_image_files(background_dir)
        self.miniprogram_files = self._get_image_files(miniprogram_dir)
        self.square_qr_files = self._get_image_files(square_qr_dir)
        self.qr_files = self._get_image_files(qr_dir)
        
        print(f"背景图数量: {len(self.background_files)}")
        print(f"增强后小程序码数量: {len(self.miniprogram_files)}")
        print(f"裁剪后方形二维码数量: {len(self.square_qr_files)}")
        print(f"普通二维码数量: {len(self.qr_files)}")
    
    def _get_image_files(self, directory: str) -> List[str]:
        """获取目录中的图片文件"""
        if not os.path.exists(directory):
            return []
        return [f for f in os.listdir(directory) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def _random_transform(self, image: Image.Image) -> Image.Image:
        """对图片进行随机变换"""
        # 随机旋转 (-30到30度)
        angle = random.uniform(-30, 30)
        image = image.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
        
        # 随机缩放 (0.5到1.5倍)
        scale = random.uniform(0.5, 1.5)
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 随机调整亮度 (0.7到1.3倍)
        brightness = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        
        # 随机调整对比度 (0.8到1.2倍)
        contrast = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        return image
    
    def _place_image_on_background(self, background: Image.Image, 
                                 foreground: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """将前景图片放置在背景图片上"""
        bg_width, bg_height = background.size
        fg_width, fg_height = foreground.size
        
        # 确保前景图片不会太大
        max_size = min(bg_width, bg_height) // 3
        if fg_width > max_size or fg_height > max_size:
            scale = max_size / max(fg_width, fg_height)
            new_width = int(fg_width * scale)
            new_height = int(fg_height * scale)
            foreground = foreground.resize((new_width, new_height), Image.Resampling.LANCZOS)
            fg_width, fg_height = new_width, new_height
        
        # 随机选择放置位置
        max_x = bg_width - fg_width
        max_y = bg_height - fg_height
        
        if max_x > 0 and max_y > 0:
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
        else:
            # 如果背景图太小，将前景图放在中心
            x = (bg_width - fg_width) // 2
            y = (bg_height - fg_height) // 2
        
        # 粘贴前景图片到背景图片上
        if foreground.mode == 'RGBA':
            background.paste(foreground, (x, y), foreground)
        else:
            background.paste(foreground, (x, y))
        
        # 返回边界框坐标 (x1, y1, x2, y2)
        bbox = (x, y, x + fg_width, y + fg_height)
        return background, bbox
    
    def _convert_bbox_to_yolo_format(self, bbox: Tuple[int, int, int, int], 
                                   img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """将边界框转换为YOLO格式 (center_x, center_y, width, height) 归一化到0-1"""
        x1, y1, x2, y2 = bbox
        center_x = ((x1 + x2) / 2) / img_width
        center_y = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return (center_x, center_y, width, height)
    
    def generate_sample(self, sample_id: int) -> bool:
        """生成单个合成样本"""
        try:
            # 随机选择背景图片
            if not self.background_files:
                print("没有找到背景图片")
                return False
            
            background_file = random.choice(self.background_files)
            background_path = os.path.join(self.background_dir, background_file)
            background = Image.open(background_path).convert('RGBA')
            
            # 调整背景图片大小到固定尺寸 (640x640)
            background = background.resize((640, 640), Image.Resampling.LANCZOS)
            
            # 随机选择二维码类型
            qr_types = []
            if self.miniprogram_files:
                qr_types.append("miniprogram")
            if self.square_qr_files:
                qr_types.append("square_qr")
            if self.qr_files:
                qr_types.append("qr")
            
            if not qr_types:
                print("没有找到二维码图片")
                return False
            
            qr_type = random.choice(qr_types)
            
            # 根据类型选择二维码文件
            if qr_type == "miniprogram":
                qr_file = random.choice(self.miniprogram_files)
                qr_path = os.path.join(self.miniprogram_dir, qr_file)
                class_id = 0  # 小程序码类别ID
            elif qr_type == "square_qr":
                qr_file = random.choice(self.square_qr_files)
                qr_path = os.path.join(self.square_qr_dir, qr_file)
                class_id = 0  # 方形二维码类别ID (修改为0)
            else:  # qr
                qr_file = random.choice(self.qr_files)
                qr_path = os.path.join(self.qr_dir, qr_file)
                class_id = 0  # 普通二维码类别ID (修改为0)
            
            # 打开二维码图片
            qr_image = Image.open(qr_path).convert('RGBA')
            
            # 对二维码进行随机变换
            qr_image = self._random_transform(qr_image)
            
            # 将二维码放置在背景上
            combined_image, bbox = self._place_image_on_background(background, qr_image)
            
            # 保存合成图片
            image_filename = f"synthetic_{sample_id:06d}.png"
            image_path = os.path.join(self.output_dir, "images", image_filename)
            combined_image.convert('RGB').save(image_path, "PNG")
            
            # 保存标签文件 (YOLO格式)
            label_filename = f"synthetic_{sample_id:06d}.txt"
            label_path = os.path.join(self.output_dir, "labels", label_filename)
            
            # 转换边界框为YOLO格式
            yolo_bbox = self._convert_bbox_to_yolo_format(bbox, 640, 640)
            
            # 写入标签文件
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            
            return True
            
        except Exception as e:
            print(f"生成样本失败 {sample_id}: {e}")
            return False
    
    def generate_batch(self, count: int = 50000) -> int:
        """批量生成合成样本"""
        print(f"开始生成 {count} 个合成样本...")
        success_count = 0
        
        for i in range(count):
            if self.generate_sample(i):
                success_count += 1
            
            # 显示进度
            if (i + 1) % 1000 == 0:
                print(f"已生成 {i + 1}/{count} 个样本 (成功: {success_count})")
        
        print(f"批量生成完成，成功生成 {success_count}/{count} 个样本")
        return success_count


def main():
    """主函数"""
    print("=== 合成训练数据生成器 ===")
    
    # 创建生成器
    generator = SyntheticDataGenerator()
    
    # 生成工业级数量的合成样本 (50,000个)
    print("\n开始生成工业级合成训练数据...")
    success_count = generator.generate_batch(50000)
    
    print(f"\n=== 生成完成 ===")
    print(f"成功生成 {success_count} 个合成训练样本")
    print(f"合成数据保存在: {generator.output_dir}")
    print("目录结构:")
    print(f"  - 图片: {os.path.join(generator.output_dir, 'images')}")
    print(f"  - 标签: {os.path.join(generator.output_dir, 'labels')}")


if __name__ == "__main__":
    main()