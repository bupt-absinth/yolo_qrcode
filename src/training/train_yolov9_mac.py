"""
YOLOv9模型训练模块（适配Mac M1）
使用指定的训练和测试数据集
"""

import os
import sys
import yaml
import torch
import platform
from ultralytics import YOLO
from pathlib import Path


class YOLOv9Trainer:
    """YOLOv9模型训练器（适配Mac M1）"""
    
    def __init__(self):
        # 检查系统平台
        self.system = platform.system()
        self.machine = platform.machine()
        
        # 设置设备
        if self.system == "Darwin" and self.machine == "arm64":
            # Mac M1芯片
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            print(f"检测到Mac M1芯片，使用设备: {self.device}")
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"使用设备: {self.device}")
        
        # 数据路径
        self.train_data_path = "data/synthetic"
        self.val_data_path = "data/swift_trail_formatted"
        
        # 创建输出目录
        os.makedirs("models/yolov9", exist_ok=True)
        os.makedirs("deployments/onnx", exist_ok=True)
        os.makedirs("deployments/coreml", exist_ok=True)
    
    def _create_dataset_yaml(self) -> str:
        """创建YOLO格式的数据集配置文件"""
        dataset_config = {
            'path': os.path.abspath('.'),  # 当前目录为根目录
            'train': self.train_data_path,  # 训练数据路径
            'val': self.val_data_path,      # 验证数据路径
            'nc': 3,                        # 类别数量 (修改为3)
            'names': ['miniprogram_code', 'square_qr', 'qr_code']  # 类别名称 (添加具体类别名称)
        }
        
        yaml_path = "configs/yolov9_dataset.yaml"
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"数据集配置文件已创建: {yaml_path}")
        return yaml_path
    
    def train_model(self, model_size: str = 'm', epochs: int = 100) -> YOLO:
        """训练YOLOv9模型"""
        print(f"开始训练YOLOv9-{model_size}模型...")
        print(f"训练数据: {self.train_data_path}")
        print(f"验证数据: {self.val_data_path}")
        
        # 创建数据集配置文件
        dataset_yaml = self._create_dataset_yaml()
        
        # 初始化模型（使用YOLOv8作为基础，因为ultralytics暂不支持YOLOv9）
        # 在实际应用中，如果有YOLOv9权重，可以替换这里的模型名称
        model_name = f"yolov8{model_size}.pt"
        print(f"使用预训练模型: {model_name}")
        
        try:
            model = YOLO(model_name)
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("尝试使用随机初始化模型...")
            model = YOLO(f"yolov8{model_size}.yaml")
        
        # 训练配置
        imgsz = 640
        batch_size = 8 if self.device == 'mps' else 16  # Mac M1上使用较小的批次
        learning_rate = 0.01
        
        print(f"训练配置:")
        print(f"  图像尺寸: {imgsz}")
        print(f"  训练轮数: {epochs}")
        print(f"  批次大小: {batch_size}")
        print(f"  学习率: {learning_rate}")
        print(f"  设备: {self.device}")
        
        # 训练模型
        try:
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                lr0=learning_rate,
                device=self.device,
                project="models/yolov9",
                name=f"yolov9{model_size}_qr_detector",
                exist_ok=True,
                patience=20,  # 早停机制
                optimizer='AdamW',  # 使用AdamW优化器
                verbose=True
            )
            print(f"YOLOv9-{model_size}训练完成")
            return model
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            raise
    
    def export_model(self, model: YOLO, formats: list = ["onnx", "coreml"]) -> list:
        """导出模型到多种格式以适配Mac M1"""
        exported_paths = []
        
        print(f"导出模型到格式: {formats}")
        
        for format in formats:
            try:
                print(f"正在导出到 {format} 格式...")
                
                if format == "onnx":
                    # 导出ONNX格式
                    export_path = model.export(
                        format="onnx", 
                        opset=12,
                        dynamic=False,  # Mac M1上使用静态形状
                        simplify=True
                    )
                    exported_paths.append(export_path)
                    print(f"ONNX模型已导出到: {export_path}")
                    
                elif format == "coreml":
                    # 导出CoreML格式（适配Mac M1）
                    export_path = model.export(
                        format="coreml",
                        nms=True,  # 包含NMS
                        verbose=True
                    )
                    exported_paths.append(export_path)
                    print(f"CoreML模型已导出到: {export_path}")
                    
                elif format == "pt":
                    # 导出PyTorch格式
                    export_path = model.export(format="pt")
                    exported_paths.append(export_path)
                    print(f"PyTorch模型已导出到: {export_path}")
                    
            except Exception as e:
                print(f"导出 {format} 格式时出现错误: {e}")
        
        return exported_paths
    
    def validate_model(self, model: YOLO) -> dict:
        """验证模型性能"""
        print("验证模型性能...")
        
        # 创建验证数据集配置
        dataset_yaml = self._create_dataset_yaml()
        
        try:
            # 进行验证
            metrics = model.val(
                data=dataset_yaml,
                device=self.device,
                verbose=True
            )
            
            # 提取关键指标
            results = {
                'mAP50': float(metrics.box.map50) if metrics.box.map50 is not None else 0.0,
                'mAP75': float(metrics.box.map75) if metrics.box.map75 is not None else 0.0,
                'mAP50-95': float(metrics.box.map) if metrics.box.map is not None else 0.0,
                'precision': float(metrics.box.mp) if metrics.box.mp is not None else 0.0,
                'recall': float(metrics.box.mr) if metrics.box.mr is not None else 0.0
            }
            
            print("验证结果:")
            for key, value in results.items():
                print(f"  {key}: {value:.4f}")
            
            return results
        except Exception as e:
            print(f"验证过程中出现错误: {e}")
            return {}


def main():
    """主函数"""
    print("=== YOLOv9 QR码检测模型训练器 (Mac M1适配版) ===")
    print(f"系统信息: {platform.system()} {platform.machine()}")
    
    # 创建训练器
    trainer = YOLOv9Trainer()
    
    # 训练模型
    try:
        model = trainer.train_model(model_size='m', epochs=50)
        
        # 验证模型
        print("\n=== 模型验证 ===")
        val_results = trainer.validate_model(model)
        
        # 检查验收标准
        print("\n=== 验收标准检查 ===")
        if val_results:
            map50 = val_results.get('mAP50', 0)
            recall = val_results.get('recall', 0)
            map75 = val_results.get('mAP75', 0)
            
            print(f"mAP@50: {map50:.4f} (要求≥0.95)")
            print(f"召回率: {recall:.4f} (要求≥0.98)")
            print(f"mAP@75: {map75:.4f} (要求≥0.80)")
            
            if map50 >= 0.95 and recall >= 0.98 and map75 >= 0.80:
                print("✓ 模型满足验收标准!")
            else:
                print("⚠ 模型未完全满足验收标准，建议继续训练优化")
        
        # 导出模型
        print("\n=== 模型导出 ===")
        exported_paths = trainer.export_model(model, formats=["onnx", "coreml"])
        
        print(f"\n=== 训练完成 ===")
        print("模型和导出文件位置:")
        print(f"  - 训练模型: models/yolov9/")
        for path in exported_paths:
            print(f"  - 导出模型: {path}")
        print("  - 验证数据: data/swift_trail_formatted/")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()