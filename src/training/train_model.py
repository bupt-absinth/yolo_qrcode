"""
模型训练主脚本
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.train_yolo import YOLOTrainer


def train_qr_detector():
    """训练二维码/小程序码检测模型"""
    print("开始训练二维码/小程序码检测模型...")
    
    # 创建训练器
    trainer = YOLOTrainer("configs/model_config.yaml")
    
    # 训练YOLOv8模型
    print("\n1. 训练YOLOv8模型...")
    yolov8_model = trainer.train_yolov8('m')
    
    # 验证YOLOv8模型
    print("\n2. 验证YOLOv8模型...")
    dataset_yaml = trainer._create_dataset_yaml()
    yolov8_results = trainer.validate_model(yolov8_model, dataset_yaml)
    
    # 导出YOLOv8模型
    print("\n3. 导出YOLOv8模型...")
    yolov8_onnx_path = trainer.export_model(yolov8_model, "onnx")
    
    print("\n训练完成!")
    print("YOLOv8模型指标:")
    for key, value in yolov8_results.items():
        print(f"  {key}: {value:.4f}")
    print(f"ONNX模型路径: {yolov8_onnx_path}")


if __name__ == "__main__":
    train_qr_detector()