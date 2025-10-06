"""
样本增强与形变处理模块
"""

import os
import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from PIL import Image
import json


class AugmentationPipeline:
    """增强处理管道"""
    
    def __init__(self, input_dir: str = "data/synthetic", output_dir: str = "data/augmented"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.images_dir = os.path.join(input_dir, "images")
        self.labels_dir = os.path.join(input_dir, "labels")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
        
        # 获取图像文件列表
        self.image_files = [f for f in os.listdir(self.images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"找到 {len(self.image_files)} 个待增强的图像")
    
    def _load_image_and_label(self, image_filename: str) -> Tuple[np.ndarray, Dict]:
        """加载图像和对应的标签"""
        # 加载图像
        image_path = os.path.join(self.images_dir, image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载标签
        label_filename = os.path.splitext(image_filename)[0] + ".json"
        label_path = os.path.join(self.labels_dir, label_filename)
        
        with open(label_path, 'r') as f:
            label = json.load(f)
        
        return image, label
    
    def _save_image_and_label(self, image: np.ndarray, label: Dict, filename: str):
        """保存增强后的图像和标签"""
        # 保存图像
        output_image_path = os.path.join(self.output_dir, "images", filename)
        cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # 保存标签
        output_label_path = os.path.join(self.output_dir, "labels", 
                                        os.path.splitext(filename)[0] + ".json")
        with open(output_label_path, 'w') as f:
            json.dump(label, f, indent=2)
        
        # 保存YOLO格式标签
        yolo_label_path = os.path.join(self.output_dir, "labels", 
                                      os.path.splitext(filename)[0] + ".txt")
        img_height, img_width = image.shape[:2]
        bbox = label['bbox']
        x_center = ((bbox['xmin'] + bbox['xmax']) / 2) / img_width
        y_center = ((bbox['ymin'] + bbox['ymax']) / 2) / img_height
        width = (bbox['xmax'] - bbox['xmin']) / img_width
        height = (bbox['ymax'] - bbox['ymin']) / img_height
        
        with open(yolo_label_path, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    def _apply_perspective_transform(self, image: np.ndarray, label: Dict) -> Tuple[np.ndarray, Dict]:
        """应用透视变换"""
        height, width = image.shape[:2]
        
        # 定义源点（图像四个角）
        src_points = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # 随机偏移目标点以创建透视效果
        offset = min(width, height) * 0.3
        dst_points = np.array([
            [random.uniform(0, offset), random.uniform(0, offset)],
            [width - random.uniform(0, offset), random.uniform(0, offset)],
            [width - random.uniform(0, offset), height - random.uniform(0, offset)],
            [random.uniform(0, offset), height - random.uniform(0, offset)]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用透视变换
        transformed = cv2.warpPerspective(image, matrix, (width, height))
        
        # 更新边界框坐标
        bbox = label['bbox']
        # 将边界框的四个角点进行变换
        corners = np.array([
            [bbox['xmin'], bbox['ymin']],
            [bbox['xmax'], bbox['ymin']],
            [bbox['xmax'], bbox['ymax']],
            [bbox['xmin'], bbox['ymax']]
        ], dtype=np.float32)
        
        # 添加齐次坐标
        corners = np.hstack([corners, np.ones((4, 1))])
        
        # 应用变换
        transformed_corners = np.dot(matrix, corners.T).T
        transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]
        
        # 计算新的边界框
        new_xmin = max(0, int(np.min(transformed_corners[:, 0])))
        new_xmax = min(width, int(np.max(transformed_corners[:, 0])))
        new_ymin = max(0, int(np.min(transformed_corners[:, 1])))
        new_ymax = min(height, int(np.max(transformed_corners[:, 1])))
        
        # 更新标签
        new_label = label.copy()
        new_label['bbox'] = {
            'xmin': new_xmin,
            'ymin': new_ymin,
            'xmax': new_xmax,
            'ymax': new_ymax
        }
        
        return transformed, new_label
    
    def _apply_lighting_effects(self, image: np.ndarray) -> np.ndarray:
        """应用光照效果增强"""
        # 随机亮度调整
        brightness = random.uniform(0.5, 1.5)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # 随机对比度调整
        contrast = random.uniform(0.8, 1.2)
        image = np.clip((image - 128) * contrast + 128, 0, 255).astype(np.uint8)
        
        # 随机饱和度调整
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = random.uniform(0.8, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 添加随机噪声
        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(5, 15), image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _apply_jpeg_compression(self, image: np.ndarray) -> np.ndarray:
        """应用JPEG压缩伪影模拟"""
        # 随机JPEG质量
        quality = random.randint(50, 95)
        
        # 编码为JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param)
        
        # 解码
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    
    def _apply_moire_pattern(self, image: np.ndarray) -> np.ndarray:
        """模拟摩尔纹效果"""
        height, width = image.shape[:2]
        
        # 创建摩尔纹图案
        pattern = np.zeros((height, width), dtype=np.float32)
        
        # 随机频率和方向
        frequency = random.uniform(0.01, 0.05)
        angle = random.uniform(0, 2 * np.pi)
        
        # 生成正弦波图案
        for i in range(height):
            for j in range(width):
                x = j - width/2
                y = i - height/2
                pattern[i, j] = np.sin(frequency * (x * np.cos(angle) + y * np.sin(angle)))
        
        # 归一化到0-1
        pattern = (pattern + 1) / 2
        
        # 应用到图像
        alpha = random.uniform(0.05, 0.15)
        image = image.astype(np.float32)
        for c in range(3):
            image[:, :, c] = image[:, :, c] * (1 - alpha * pattern) + \
                            128 * alpha * pattern
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def _apply_reflection(self, image: np.ndarray) -> np.ndarray:
        """模拟屏幕反光效果"""
        height, width = image.shape[:2]
        
        # 创建高光区域
        highlight = np.zeros((height, width), dtype=np.float32)
        
        # 随机高光位置和大小
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        radius = random.randint(width//4, width//2)
        
        # 创建径向渐变
        for i in range(height):
            for j in range(width):
                distance = np.sqrt((j - center_x)**2 + (i - center_y)**2)
                if distance < radius:
                    highlight[i, j] = 1 - (distance / radius)
        
        # 应用高光
        alpha = random.uniform(0.1, 0.3)
        image = image.astype(np.float32)
        image = image + 255 * alpha * highlight[:, :, np.newaxis]
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def augment_single_image(self, image_filename: str, augment_count: int = 5) -> int:
        """对单张图像进行增强处理"""
        try:
            # 加载原始图像和标签
            image, label = self._load_image_and_label(image_filename)
            
            success_count = 0
            
            # 保存原始图像
            self._save_image_and_label(image, label, f"aug_{image_filename}")
            success_count += 1
            
            # 进行多种增强
            for i in range(augment_count):
                augmented_image = image.copy()
                augmented_label = label.copy()
                
                # 随机应用不同的增强技术
                augmentation_types = []
                
                # 透视变换 (60%概率)
                if random.random() < 0.6:
                    augmented_image, augmented_label = self._apply_perspective_transform(
                        augmented_image, augmented_label)
                    augmentation_types.append("perspective")
                
                # 光照效果 (80%概率)
                if random.random() < 0.8:
                    augmented_image = self._apply_lighting_effects(augmented_image)
                    augmentation_types.append("lighting")
                
                # JPEG压缩 (50%概率)
                if random.random() < 0.5:
                    augmented_image = self._apply_jpeg_compression(augmented_image)
                    augmentation_types.append("jpeg")
                
                # 摩尔纹 (30%概率)
                if random.random() < 0.3:
                    augmented_image = self._apply_moire_pattern(augmented_image)
                    augmentation_types.append("moire")
                
                # 屏幕反光 (40%概率)
                if random.random() < 0.4:
                    augmented_image = self._apply_reflection(augmented_image)
                    augmentation_types.append("reflection")
                
                # 保存增强后的图像
                base_name = os.path.splitext(image_filename)[0]
                aug_filename = f"{base_name}_aug_{i}.jpg"
                self._save_image_and_label(augmented_image, augmented_label, aug_filename)
                success_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  增强进度: {i + 1}/{augment_count}")
            
            return success_count
            
        except Exception as e:
            print(f"增强图像失败 {image_filename}: {e}")
            return 0
    
    def augment_batch(self, count: Optional[int] = None, augment_per_image: int = 5) -> int:
        """批量增强图像"""
        if count is None:
            count = len(self.image_files)
        
        total_success = 0
        
        for i, image_file in enumerate(self.image_files[:count]):
            print(f"处理图像 {i + 1}/{min(count, len(self.image_files))}: {image_file}")
            success_count = self.augment_single_image(image_file, augment_per_image)
            total_success += success_count
            
            if (i + 1) % 10 == 0:
                print(f"批次进度: {i + 1}/{min(count, len(self.image_files))}, "
                      f"成功增强: {total_success} 个样本")
        
        print(f"批量增强完成，总共成功增强: {total_success} 个样本")
        return total_success


if __name__ == "__main__":
    # 示例使用
    pipeline = AugmentationPipeline()
    # 对前10张图像进行增强，每张图像生成5个增强版本
    success_count = pipeline.augment_batch(10, 5)