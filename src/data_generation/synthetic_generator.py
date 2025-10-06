"""
自动化合成工具和BBox自动生成模块
"""

import os
import random
import json
import cv2
import numpy as np
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw
from utils.helpers import create_directory, save_json


class SyntheticGenerator:
    """自动化合成工具"""
    
    def __init__(self, 
                 backgrounds_dir: str = "data/backgrounds",
                 qr_codes_dir: str = "data/qr_codes",
                 mini_program_codes_dir: str = "data/mini_program_codes",
                 output_dir: str = "data/synthetic"):
        self.backgrounds_dir = backgrounds_dir
        self.qr_codes_dir = qr_codes_dir
        self.mini_program_codes_dir = mini_program_codes_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        create_directory(output_dir)
        create_directory(os.path.join(output_dir, "images"))
        create_directory(os.path.join(output_dir, "labels"))
        
        # 加载文件列表
        self.backgrounds = self._load_file_list(backgrounds_dir)
        self.qr_codes = self._load_file_list(qr_codes_dir)
        self.mini_program_codes = self._load_file_list(mini_program_codes_dir)
        
        print(f"加载背景图片: {len(self.backgrounds)} 张")
        print(f"加载二维码: {len(self.qr_codes)} 张")
        print(f"加载小程序码: {len(self.mini_program_codes)} 张")
    
    def _load_file_list(self, directory: str) -> List[str]:
        """加载目录中的文件列表"""
        if not os.path.exists(directory):
            return []
        return [f for f in os.listdir(directory) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def _random_scale_code(self, code_img: Image.Image) -> Image.Image:
        """随机缩放码图"""
        # 随机缩放因子 (0.1到1.0)
        scale_factor = random.uniform(0.1, 1.0)
        new_width = int(code_img.width * scale_factor)
        new_height = int(code_img.height * scale_factor)
        
        # 确保最小尺寸
        new_width = max(new_width, 20)
        new_height = max(new_height, 20)
        
        return code_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _random_rotate_code(self, code_img: Image.Image) -> Tuple[Image.Image, float]:
        """随机旋转码图"""
        angle = random.uniform(0, 360)
        return code_img.rotate(angle, expand=True), angle
    
    def _place_code_on_background(self, 
                                 background: Image.Image, 
                                 code_img: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """将码图粘贴到背景图上并返回边界框"""
        bg_width, bg_height = background.size
        code_width, code_height = code_img.size
        
        # 确保码图不超过背景图
        if code_width > bg_width or code_height > bg_height:
            scale_factor = min(bg_width / code_width, bg_height / code_height) * 0.9
            new_width = int(code_width * scale_factor)
            new_height = int(code_height * scale_factor)
            code_img = code_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            code_width, code_height = code_img.size
        
        # 随机位置 (确保码图完全在背景图内)
        max_x = bg_width - code_width
        max_y = bg_height - code_height
        
        if max_x <= 0 or max_y <= 0:
            # 如果背景图太小，调整码图大小
            scale_factor = min(bg_width / code_width, bg_height / code_height) * 0.9
            new_width = max(int(code_width * scale_factor), 1)
            new_height = max(int(code_height * scale_factor), 1)
            code_img = code_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            code_width, code_height = code_img.size
            max_x = bg_width - code_width
            max_y = bg_height - code_height
        
        x = random.randint(0, max(1, max_x))
        y = random.randint(0, max(1, max_y))
        
        # 粘贴码图到背景图
        background.paste(code_img, (x, y), code_img if code_img.mode == 'RGBA' else None)
        
        # 计算边界框 (xmin, ymin, xmax, ymax)
        bbox = (x, y, x + code_width, y + code_height)
        
        return background, bbox
    
    def _convert_bbox_to_yolo_format(self, 
                                   bbox: Tuple[int, int, int, int], 
                                   img_width: int, 
                                   img_height: int) -> List[float]:
        """将边界框转换为YOLO格式 (归一化的中心点和宽高)"""
        xmin, ymin, xmax, ymax = bbox
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        return [x_center, y_center, width, height]
    
    def _make_square_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """将边界框转换为正方形"""
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        side = max(width, height)
        
        # 保持中心点不变
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        
        new_xmin = center_x - side // 2
        new_ymin = center_y - side // 2
        new_xmax = new_xmin + side
        new_ymax = new_ymin + side
        
        return (new_xmin, new_ymin, new_xmax, new_ymax)
    
    def generate_synthetic_sample(self, 
                                sample_id: int,
                                apply_perspective: bool = True,
                                apply_occlusion: bool = True) -> bool:
        """生成单个合成样本"""
        try:
            # 随机选择背景图
            if not self.backgrounds:
                print("没有可用的背景图片")
                return False
                
            bg_filename = random.choice(self.backgrounds)
            bg_path = os.path.join(self.backgrounds_dir, bg_filename)
            background = Image.open(bg_path).convert('RGB')
            
            # 随机选择码图类型 (二维码或小程序码)
            use_qr = random.choice([True, False])
            code_files = self.qr_codes if use_qr else self.mini_program_codes
            
            if not code_files:
                print("没有可用的码图")
                return False
                
            code_filename = random.choice(code_files)
            code_path = os.path.join(
                self.qr_codes_dir if use_qr else self.mini_program_codes_dir, 
                code_filename
            )
            code_img = Image.open(code_path)
            
            # 随机缩放
            code_img = self._random_scale_code(code_img)
            
            # 随机旋转
            code_img, _ = self._random_rotate_code(code_img)
            
            # 将码图粘贴到背景图上
            background, bbox = self._place_code_on_background(background, code_img)
            
            # 应用透视变换
            if apply_perspective and random.random() < 0.6:  # 60%概率应用透视变换
                background, bbox = self._apply_perspective_transform(background, bbox)
            
            # 应用遮挡
            if apply_occlusion and random.random() < 0.2:  # 20%概率应用遮挡
                background = self._apply_occlusion(background)
            
            # 转换为正方形边界框
            square_bbox = self._make_square_bbox(bbox)
            
            # 保存合成图像
            output_img_path = os.path.join(self.output_dir, "images", f"synthetic_{sample_id:06d}.jpg")
            background.save(output_img_path, "JPEG", quality=95)
            
            # 保存YOLO格式标签
            img_width, img_height = background.size
            yolo_bbox = self._convert_bbox_to_yolo_format(square_bbox, img_width, img_height)
            
            # YOLO格式: class_id x_center y_center width height
            label_content = f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
            label_path = os.path.join(self.output_dir, "labels", f"synthetic_{sample_id:06d}.txt")
            
            with open(label_path, 'w') as f:
                f.write(label_content)
            
            # 保存JSON格式标签
            json_label = {
                "image": f"synthetic_{sample_id:06d}.jpg",
                "bbox": {
                    "xmin": int(square_bbox[0]),
                    "ymin": int(square_bbox[1]),
                    "xmax": int(square_bbox[2]),
                    "ymax": int(square_bbox[3])
                },
                "class_id": 0,
                "class_name": "code"
            }
            
            json_label_path = os.path.join(self.output_dir, "labels", f"synthetic_{sample_id:06d}.json")
            save_json(json_label, json_label_path)
            
            return True
            
        except Exception as e:
            print(f"生成合成样本失败 {sample_id}: {e}")
            return False
    
    def _apply_perspective_transform(self, 
                                   image: Image.Image, 
                                   bbox: Tuple[int, int, int, int]) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """应用透视变换"""
        # 转换为OpenCV格式
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = cv_image.shape[:2]
        
        # 定义源点和目标点
        src_points = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # 随机偏移目标点以创建透视效果
        offset = min(width, height) * 0.2
        dst_points = np.array([
            [random.uniform(0, offset), random.uniform(0, offset)],
            [width - random.uniform(0, offset), random.uniform(0, offset)],
            [width - random.uniform(0, offset), height - random.uniform(0, offset)],
            [random.uniform(0, offset), height - random.uniform(0, offset)]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用透视变换
        transformed = cv2.warpPerspective(cv_image, matrix, (width, height))
        
        # 转换回PIL格式
        transformed_pil = Image.fromarray(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
        
        # TODO: 更新边界框坐标以匹配透视变换
        # 这里简化处理，实际应用中需要计算变换后的边界框
        
        return transformed_pil, bbox
    
    def _apply_occlusion(self, image: Image.Image) -> Image.Image:
        """应用遮挡效果"""
        draw = ImageDraw.Draw(image, 'RGBA')
        
        # 随机遮挡类型
        occlusion_type = random.choice(['rectangle', 'circle', 'text'])
        
        img_width, img_height = image.size
        
        if occlusion_type == 'rectangle':
            # 随机矩形遮挡
            occlusion_width = random.randint(img_width // 20, img_width // 10)
            occlusion_height = random.randint(img_height // 20, img_height // 10)
            x = random.randint(0, img_width - occlusion_width)
            y = random.randint(0, img_height - occlusion_height)
            draw.rectangle([x, y, x + occlusion_width, y + occlusion_height], 
                          fill=(0, 0, 0, 180))
        
        elif occlusion_type == 'circle':
            # 随机圆形遮挡
            radius = random.randint(img_width // 40, img_width // 20)
            x = random.randint(radius, img_width - radius)
            y = random.randint(radius, img_height - radius)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                        fill=(255, 255, 255, 180))
        
        elif occlusion_type == 'text':
            # 随机文字遮挡
            from PIL import ImageFont
            try:
                # 尝试使用系统字体
                font = ImageFont.truetype("arial.ttf", random.randint(20, 40))
            except:
                # 使用默认字体
                font = ImageFont.load_default()
            
            text = random.choice(["FINGER", "HAND", "COVER", "OBSCURE"])
            x = random.randint(0, img_width - 100)
            y = random.randint(0, img_height - 50)
            draw.text((x, y), text, fill=(0, 0, 0, 180), font=font)
        
        return image
    
    def generate_batch(self, count: int = 50000) -> int:
        """批量生成合成样本"""
        success_count = 0
        
        for i in range(count):
            if self.generate_synthetic_sample(i):
                success_count += 1
                
            if (i + 1) % 1000 == 0:
                print(f"已生成 {i + 1}/{count} 个合成样本，成功 {success_count} 个")
        
        print(f"批量生成完成，总共生成 {success_count}/{count} 个合成样本")
        return success_count


if __name__ == "__main__":
    # 示例使用
    generator = SyntheticGenerator()
    # 生成100个合成样本作为示例
    success_count = generator.generate_batch(100)