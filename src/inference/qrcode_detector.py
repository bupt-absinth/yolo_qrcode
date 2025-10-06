"""
二维码/小程序码检测推理模块
"""

import os
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from ultralytics import YOLO

from src.postprocessing.bbox_corrector import BBoxCorrector
from utils.helpers import load_json


class QRCodeDetector:
    """二维码/小程序码检测器"""
    
    def __init__(self, model_path: str, img_size: int = 640):
        self.model_path = model_path
        self.img_size = img_size
        
        # 加载模型
        self.model = YOLO(model_path)
        
        # 检查设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 将模型移动到指定设备
        self.model.to(self.device)
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """预处理图像"""
        original_height, original_width = image.shape[:2]
        
        # 调整图像大小
        resized_image = cv2.resize(image, (self.img_size, self.img_size))
        
        # 转换为张量
        image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
        
        return image_tensor, (original_width, original_height)
    
    def postprocess_detections(self, 
                             results, 
                             original_size: Tuple[int, int],
                             min_confidence: float = 0.5) -> List[Dict]:
        """后处理检测结果"""
        original_width, original_height = original_size
        
        # 提取检测结果
        detections = []
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                # 获取边界框坐标
                box = boxes.xyxy[i].cpu().numpy()
                confidence = boxes.conf[i].cpu().item()
                class_id = boxes.cls[i].cpu().item()
                
                if confidence >= min_confidence:
                    # 转换坐标到原始图像尺寸
                    xmin = int(box[0] * original_width / self.img_size)
                    ymin = int(box[1] * original_height / self.img_size)
                    xmax = int(box[2] * original_width / self.img_size)
                    ymax = int(box[3] * original_height / self.img_size)
                    
                    detection = {
                        'box_2d': [xmin, ymin, xmax, ymax],
                        'confidence': confidence,
                        'class_id': int(class_id),
                        'class_name': 'code'
                    }
                    detections.append(detection)
        
        # 应用后处理
        corrector = BBoxCorrector(original_width, original_height)
        corrected_detections = corrector.post_process_detections(detections)
        
        return corrected_detections
    
    def detect(self, 
              image_path: str, 
              min_confidence: float = 0.5) -> Dict:
        """检测图像中的二维码/小程序码"""
        start_time = time.time()
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 预处理
        image_tensor, original_size = self.preprocess_image(image)
        
        # 推理
        with torch.no_grad():
            results = self.model(image_tensor.to(self.device))
        
        # 后处理
        detections = self.postprocess_detections(results[0], original_size, min_confidence)
        
        # 计算处理时间
        elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 构建返回结果
        result = {
            'code_count': len(detections),
            'model_version': 'yolov8m',
            'time_ms': round(elapsed_time, 2),
            'detections': detections
        }
        
        return result
    
    def detect_from_array(self, 
                         image: np.ndarray, 
                         min_confidence: float = 0.5) -> Dict:
        """从numpy数组检测二维码/小程序码"""
        start_time = time.time()
        
        # 确保图像是RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
        else:
            raise ValueError("输入图像必须是RGB格式的numpy数组")
        
        # 预处理
        image_tensor, original_size = self.preprocess_image(image)
        
        # 推理
        with torch.no_grad():
            results = self.model(image_tensor.to(self.device))
        
        # 后处理
        detections = self.postprocess_detections(results[0], original_size, min_confidence)
        
        # 计算处理时间
        elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 构建返回结果
        result = {
            'code_count': len(detections),
            'model_version': 'yolov8m',
            'time_ms': round(elapsed_time, 2),
            'detections': detections
        }
        
        return result
    
    def visualize_detections(self, 
                           image_path: str, 
                           result: Dict, 
                           output_path: Optional[str] = None) -> np.ndarray:
        """可视化检测结果"""
        # 加载图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 绘制检测框
        for detection in result['detections']:
            box = detection['box_2d']
            confidence = detection['confidence']
            
            # 绘制边界框
            cv2.rectangle(image, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         (0, 255, 0), 2)
            
            # 绘制置信度
            label = f"Code: {confidence:.2f}"
            cv2.putText(image, label, 
                       (int(box[0]), int(box[1] - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 保存或返回结果
        if output_path:
            vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, vis_image)
        
        return image


if __name__ == "__main__":
    # 示例使用
    detector = QRCodeDetector("models/yolov8/yolov8m_qr_detector/weights/best.pt")
    
    # 检测图像
    result = detector.detect("data/test/sample.jpg")
    
    print("检测结果:")
    print(f"  检测到 {result['code_count']} 个码")
    print(f"  处理时间: {result['time_ms']} ms")
    print(f"  模型版本: {result['model_version']}")
    
    for i, det in enumerate(result['detections']):
        print(f"  码 {i+1}:")
        print(f"    位置: {det['box_2d']}")
        print(f"    置信度: {det['confidence']:.4f}")
        print(f"    是否正方形: {det.get('is_square', False)}")