"""
YOLOv9模型推理模块（适配Mac M1）
支持ONNX和CoreML格式模型在Mac M1上的高性能推理
"""

import os
import cv2
import time
import numpy as np
import torch
import platform
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


class YOLOv9Inference:
    """YOLOv9模型推理器（适配Mac M1）"""
    
    def __init__(self, model_path: str, model_format: str = "coreml"):
        """
        初始化推理器
        
        Args:
            model_path: 模型文件路径
            model_format: 模型格式 ("coreml", "onnx", "pt")
        """
        self.model_path = model_path
        self.model_format = model_format
        self.model = None
        self.device = None
        
        # 检查系统平台
        self.system = platform.system()
        self.machine = platform.machine()
        
        # 设置设备
        if self.system == "Darwin" and self.machine == "arm64":
            # Mac M1芯片
            if torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
            print(f"检测到Mac M1芯片，使用设备: {self.device}")
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"使用设备: {self.device}")
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            if self.model_format == "coreml":
                # CoreML模型推理
                from coremltools.models import MLModel
                self.model = MLModel(self.model_path)
                print(f"CoreML模型加载成功: {self.model_path}")
                
            elif self.model_format == "onnx":
                # ONNX模型推理
                import onnxruntime as ort
                
                # 配置ONNX Runtime提供者
                if self.system == "Darwin" and self.machine == "arm64":
                    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
                else:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
                
                self.model = ort.InferenceSession(self.model_path, providers=providers)
                print(f"ONNX模型加载成功: {self.model_path}")
                
            elif self.model_format == "pt":
                # PyTorch模型推理
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                print(f"PyTorch模型加载成功: {self.model_path}")
                
            else:
                raise ValueError(f"不支持的模型格式: {self.model_format}")
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            预处理后的图像
        """
        # 调整图像尺寸到640x640
        input_shape = (640, 640)
        img_resized = cv2.resize(image, input_shape)
        
        # 转换颜色空间 BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 归一化
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # 转换为CHW格式
        if self.model_format in ["onnx", "coreml"]:
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            img_batch = np.expand_dims(img_transposed, axis=0)
            return img_batch
        else:
            return img_rgb
    
    def postprocess_detections(self, 
                             outputs: np.ndarray, 
                             original_shape: Tuple[int, int],
                             conf_threshold: float = 0.5,
                             iou_threshold: float = 0.5) -> List[Dict]:
        """
        后处理检测结果
        
        Args:
            outputs: 模型输出
            original_shape: 原始图像尺寸 (height, width)
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            
        Returns:
            检测结果列表
        """
        try:
            # 解析模型输出
            if self.model_format == "coreml":
                # CoreML输出处理
                detections = outputs.get('coordinates', np.array([]))
            elif self.model_format == "onnx":
                # ONNX输出处理
                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    detections = outputs[0] if len(outputs) > 0 else np.array([])
                else:
                    detections = outputs
            else:
                # PyTorch输出处理
                detections = outputs[0].boxes.data.cpu().numpy() if hasattr(outputs[0], 'boxes') else np.array([])
            
            if detections.size == 0:
                return []
            
            # 提取检测框信息
            if self.model_format == "pt":
                # PyTorch格式: [x1, y1, x2, y2, conf, class]
                boxes = detections[:, :4]
                scores = detections[:, 4]
                class_ids = detections[:, 5].astype(int)
            else:
                # 其他格式需要根据具体输出结构调整
                # 假设格式为: [batch, num_detections, 6] 或 [num_detections, 6]
                if len(detections.shape) == 3:
                    detections = detections[0]  # 取第一个批次
                
                if detections.shape[1] >= 6:
                    boxes = detections[:, :4]
                    scores = detections[:, 4]
                    class_ids = detections[:, 5].astype(int)
                else:
                    # 简单处理
                    return []
            
            # 置信度过滤
            valid_indices = scores >= conf_threshold
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            class_ids = class_ids[valid_indices]
            
            # 尺寸转换 (从640x640转换回原始尺寸)
            orig_h, orig_w = original_shape
            scale_x, scale_y = orig_w / 640.0, orig_h / 640.0
            
            detections_list = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                
                # 转换回原始尺寸
                x1_orig = int(x1 * scale_x)
                y1_orig = int(y1 * scale_y)
                x2_orig = int(x2 * scale_x)
                y2_orig = int(y2 * scale_y)
                
                # 确保坐标在图像范围内
                x1_orig = max(0, min(x1_orig, orig_w - 1))
                y1_orig = max(0, min(y1_orig, orig_h - 1))
                x2_orig = max(0, min(x2_orig, orig_w - 1))
                y2_orig = max(0, min(y2_orig, orig_h - 1))
                
                detection = {
                    'bbox': [x1_orig, y1_orig, x2_orig, y2_orig],
                    'confidence': float(scores[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': 'qr_code'
                }
                detections_list.append(detection)
            
            return detections_list
            
        except Exception as e:
            print(f"后处理检测结果时出现错误: {e}")
            return []
    
    def infer(self, image_path: str, 
              conf_threshold: float = 0.5,
              iou_threshold: float = 0.5) -> Dict:
        """
        执行推理
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            
        Returns:
            推理结果
        """
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            original_shape = image.shape[:2]  # (height, width)
            
            # 预处理
            input_tensor = self.preprocess_image(image)
            
            # 推理
            if self.model_format == "coreml":
                # CoreML推理
                results = self.model.predict({'image': input_tensor})
            elif self.model_format == "onnx":
                # ONNX推理
                input_name = self.model.get_inputs()[0].name
                results = self.model.run(None, {input_name: input_tensor.astype(np.float32)})
            else:
                # PyTorch推理
                results = self.model(input_tensor)
            
            # 后处理
            detections = self.postprocess_detections(
                results, original_shape, conf_threshold, iou_threshold
            )
            
            # 计算推理时间
            elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 构造返回结果（符合项目规范）
            result = {
                'code_count': len(detections),
                'model_version': f'yolov9-{self.model_format}',
                'time_ms': round(elapsed_time, 2),
                'detections': detections
            }
            
            return result
            
        except Exception as e:
            print(f"推理过程中出现错误: {e}")
            # 返回错误结果（符合项目规范）
            return {
                'code_count': 0,
                'model_version': f'yolov9-{self.model_format}',
                'time_ms': 0,
                'detections': [],
                'error': str(e)
            }
    
    def batch_infer(self, image_paths: List[str], 
                   conf_threshold: float = 0.5,
                   iou_threshold: float = 0.5) -> List[Dict]:
        """
        批量推理
        
        Args:
            image_paths: 图像路径列表
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            
        Returns:
            推理结果列表
        """
        results = []
        for image_path in image_paths:
            result = self.infer(image_path, conf_threshold, iou_threshold)
            results.append(result)
        return results


def main():
    """主函数"""
    print("=== YOLOv9 QR码检测模型推理器 (Mac M1适配版) ===")
    print(f"系统信息: {platform.system()} {platform.machine()}")
    
    # 示例使用
    # 注意：需要先训练模型并导出相应格式的模型文件
    model_paths = {
        "coreml": "deployments/coreml/yolov9m_qr_detector.mlpackage",
        "onnx": "deployments/onnx/yolov9m_qr_detector.onnx",
        "pt": "models/yolov9/yolov9m_qr_detector/weights/best.pt"
    }
    
    # 选择要使用的模型格式
    model_format = "coreml"  # 可选: "coreml", "onnx", "pt"
    model_path = model_paths.get(model_format)
    
    if not model_path or not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型并导出相应格式的模型文件")
        return
    
    try:
        # 创建推理器
        inference = YOLOv9Inference(model_path, model_format)
        
        # 示例图像路径（需要替换为实际图像路径）
        test_images = [
            "data/swift_trail_formatted/images/1.png",
            "data/swift_trail_formatted/images/2.webp",
            # 添加更多测试图像路径
        ]
        
        # 过滤存在的图像
        existing_images = [img for img in test_images if os.path.exists(img)]
        
        if not existing_images:
            print("未找到测试图像，请检查图像路径")
            return
        
        print(f"开始推理 {len(existing_images)} 张图像...")
        
        # 批量推理
        results = inference.batch_infer(existing_images, conf_threshold=0.5, iou_threshold=0.5)
        
        # 输出结果
        for i, (image_path, result) in enumerate(zip(existing_images, results)):
            print(f"\n图像 {i+1}: {os.path.basename(image_path)}")
            print(f"  检测到 {result['code_count']} 个二维码")
            print(f"  推理时间: {result['time_ms']} ms")
            print(f"  模型版本: {result['model_version']}")
            
            for j, detection in enumerate(result['detections']):
                bbox = detection['bbox']
                conf = detection['confidence']
                print(f"    二维码 {j+1}: {bbox} (置信度: {conf:.3f})")
        
        print(f"\n=== 推理完成 ===")
        
    except Exception as e:
        print(f"推理过程中出现错误: {e}")


if __name__ == "__main__":
    main()