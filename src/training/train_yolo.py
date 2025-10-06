"""
YOLOv8/v9模型训练模块
"""

import os
import sys
import yaml
import torch
from ultralytics import YOLO
from utils.helpers import load_json


class YOLOTrainer:
    """YOLO模型训练器"""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        self.config_path = config_path
        self.config = load_json(config_path) if config_path.endswith('.json') else self._load_yaml_config()
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs("models/yolov8", exist_ok=True)
        os.makedirs("models/yolov9", exist_ok=True)
    
    def _load_yaml_config(self) -> dict:
        """加载YAML配置文件"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_dataset_yaml(self, data_dir: str = "data") -> str:
        """创建YOLO格式的数据集配置文件"""
        dataset_config = {
            'path': os.path.abspath(data_dir),
            'train': 'augmented/images',  # 使用增强后的数据进行训练
            'val': 'test/images',         # 使用真实测试集进行验证
            'nc': 1,
            'names': ['code']
        }
        
        yaml_path = os.path.join(data_dir, "dataset.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        return yaml_path
    
    def train_yolov8(self, model_size: str = 'm') -> YOLO:
        """训练YOLOv8模型"""
        print(f"开始训练YOLOv8-{model_size}模型...")
        
        # 创建数据集配置文件
        dataset_yaml = self._create_dataset_yaml()
        
        # 初始化模型
        model_name = f"yolov8{model_size}.pt"
        model = YOLO(model_name)
        
        # 获取训练配置
        yolo_config = self.config.get('yolo', {})
        imgsz = yolo_config.get('imgsz', 640)
        epochs = yolo_config.get('epochs', 100)
        batch_size = yolo_config.get('batch_size', 16)
        learning_rate = yolo_config.get('learning_rate', 0.01)
        
        # 训练模型
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=learning_rate,
            device=self.device,
            project="models/yolov8",
            name=f"yolov8{model_size}_qr_detector",
            exist_ok=True
        )
        
        print(f"YOLOv8-{model_size}训练完成")
        return model
    
    def train_yolov9(self, model_size: str = 'm') -> YOLO:
        """训练YOLOv9模型 (使用YOLOv8作为基础，因为ultralytics暂不支持YOLOv9)"""
        print(f"开始训练YOLOv9-{model_size}模型...")
        
        # 创建数据集配置文件
        dataset_yaml = self._create_dataset_yaml()
        
        # 对于YOLOv9，我们使用最新的YOLOv8模型作为替代
        # 在实际应用中，如果有YOLOv9权重，可以替换这里的模型名称
        model_name = f"yolov8{model_size}.pt"
        model = YOLO(model_name)
        
        # 获取训练配置
        yolo_config = self.config.get('yolo', {})
        imgsz = yolo_config.get('imgsz', 640)
        epochs = yolo_config.get('epochs', 100)
        batch_size = yolo_config.get('batch_size', 16)
        learning_rate = yolo_config.get('learning_rate', 0.01)
        
        # 训练模型
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=learning_rate,
            device=self.device,
            project="models/yolov9",
            name=f"yolov9{model_size}_qr_detector",
            exist_ok=True
        )
        
        print(f"YOLOv9-{model_size}训练完成 (实际使用YOLOv8)")
        return model
    
    def export_model(self, model: YOLO, format: str = "onnx") -> str:
        """导出模型到指定格式"""
        print(f"导出模型到 {format} 格式...")
        
        if format == "onnx":
            export_path = model.export(format="onnx", opset=12)
        elif format == "pt":
            export_path = model.export(format="pt")
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        print(f"模型已导出到: {export_path}")
        return export_path
    
    def validate_model(self, model: YOLO, data_yaml: str) -> dict:
        """验证模型性能"""
        print("验证模型性能...")
        
        # 进行验证
        metrics = model.val(data=data_yaml)
        
        # 提取关键指标
        results = {
            'mAP50': metrics.box.map50,
            'mAP75': metrics.box.map75,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr
        }
        
        print("验证结果:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        return results


if __name__ == "__main__":
    # 示例使用
    trainer = YOLOTrainer()
    
    # 训练YOLOv8模型
    model = trainer.train_yolov8('m')
    
    # 验证模型
    dataset_yaml = trainer._create_dataset_yaml()
    results = trainer.validate_model(model, dataset_yaml)
    
    # 导出模型
    onnx_path = trainer.export_model(model, "onnx")