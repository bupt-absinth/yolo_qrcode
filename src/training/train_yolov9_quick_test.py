"""
YOLOv9快速训练验证脚本
使用1000张图像进行快速训练验证
"""

import os
import sys
import yaml
import torch
import platform
import shutil
import random
from ultralytics import YOLO
from pathlib import Path


class YOLOv9QuickTrainer:
    """YOLOv9快速训练验证器"""
    
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
        
        # 创建临时训练数据目录
        self.temp_train_dir = "data/temp_train"
        self.temp_images_dir = os.path.join(self.temp_train_dir, "images")
        self.temp_labels_dir = os.path.join(self.temp_train_dir, "labels")
        
        # 原始数据路径
        self.original_images_dir = "data/synthetic/images"
        self.original_labels_dir = "data/synthetic/labels"
        
        # 测试数据路径
        self.val_data_path = "data/swift_trail_formatted"
        
        # 创建输出目录
        os.makedirs("models/yolov9_quick", exist_ok=True)
    
    def prepare_training_data(self, sample_count: int = 1000) -> bool:
        """
        准备训练数据（随机选择指定数量的样本）
        
        Args:
            sample_count: 要选择的样本数量
            
        Returns:
            bool: 准备是否成功
        """
        try:
            print(f"准备 {sample_count} 张训练图像...")
            
            # 创建临时目录
            os.makedirs(self.temp_images_dir, exist_ok=True)
            os.makedirs(self.temp_labels_dir, exist_ok=True)
            
            # 获取所有图像文件
            all_images = [f for f in os.listdir(self.original_images_dir) if f.endswith('.png')]
            print(f"总图像数量: {len(all_images)}")
            
            # 随机选择指定数量的图像
            if len(all_images) < sample_count:
                selected_images = all_images
                print(f"图像数量不足，使用全部 {len(selected_images)} 张图像")
            else:
                selected_images = random.sample(all_images, sample_count)
                print(f"随机选择 {len(selected_images)} 张图像")
            
            # 复制选中的图像和标签文件
            for i, image_file in enumerate(selected_images):
                # 复制图像文件
                src_image = os.path.join(self.original_images_dir, image_file)
                dst_image = os.path.join(self.temp_images_dir, image_file)
                shutil.copy2(src_image, dst_image)
                
                # 复制对应的标签文件
                label_file = os.path.splitext(image_file)[0] + '.txt'
                src_label = os.path.join(self.original_labels_dir, label_file)
                dst_label = os.path.join(self.temp_labels_dir, label_file)
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                
                # 显示进度
                if (i + 1) % 100 == 0:
                    print(f"已复制 {i + 1}/{len(selected_images)} 个样本")
            
            print(f"训练数据准备完成，共 {len(selected_images)} 个样本")
            return True
            
        except Exception as e:
            print(f"准备训练数据时出现错误: {e}")
            return False
    
    def _create_dataset_yaml(self) -> str:
        """创建YOLO格式的数据集配置文件"""
        dataset_config = {
            'path': os.path.abspath('.'),  # 当前目录为根目录
            'train': self.temp_train_dir,  # 临时训练数据路径
            'val': self.val_data_path,     # 验证数据路径
            'nc': 3,                       # 类别数量
            'names': ['miniprogram_code', 'square_qr', 'qr_code']  # 类别名称
        }
        
        yaml_path = "configs/yolov9_quick_test.yaml"
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"数据集配置文件已创建: {yaml_path}")
        return yaml_path
    
    def train_model(self, epochs: int = 10) -> YOLO:
        """快速训练模型"""
        print(f"开始快速训练YOLOv9模型...")
        print(f"训练样本数: 1000")
        print(f"训练轮数: {epochs}")
        
        # 创建数据集配置文件
        dataset_yaml = self._create_dataset_yaml()
        
        # 初始化模型
        model_name = "yolov8s.pt"  # 使用较小的模型进行快速训练
        print(f"使用预训练模型: {model_name}")
        
        try:
            model = YOLO(model_name)
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("尝试使用随机初始化模型...")
            model = YOLO("yolov8s.yaml")
        
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
                project="models/yolov9_quick",
                name="yolov9s_qr_detector_quick",
                exist_ok=True,
                patience=5,  # 早停机制
                optimizer='AdamW',  # 使用AdamW优化器
                verbose=True
            )
            print(f"YOLOv9快速训练完成")
            return model
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            raise
    
    def export_model(self, model: YOLO, formats: list = ["onnx"]) -> list:
        """导出模型"""
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
                        dynamic=False,
                        simplify=True
                    )
                    exported_paths.append(export_path)
                    print(f"ONNX模型已导出到: {export_path}")
                    
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
    
    def cleanup(self):
        """清理临时文件"""
        try:
            if os.path.exists(self.temp_train_dir):
                shutil.rmtree(self.temp_train_dir)
                print(f"已清理临时训练数据: {self.temp_train_dir}")
        except Exception as e:
            print(f"清理临时文件时出现错误: {e}")


def main():
    """主函数"""
    print("=== YOLOv9快速训练验证器 ===")
    print(f"系统信息: {platform.system()} {platform.machine()}")
    
    # 创建训练器
    trainer = YOLOv9QuickTrainer()
    
    try:
        # 准备训练数据
        print("\n=== 准备训练数据 ===")
        if not trainer.prepare_training_data(sample_count=1000):
            print("准备训练数据失败")
            return
        
        # 训练模型
        print("\n=== 模型训练 ===")
        model = trainer.train_model(epochs=10)
        
        # 验证模型
        print("\n=== 模型验证 ===")
        val_results = trainer.validate_model(model)
        
        # 导出模型
        print("\n=== 模型导出 ===")
        exported_paths = trainer.export_model(model, formats=["onnx", "pt"])
        
        print(f"\n=== 训练完成 ===")
        print("模型和导出文件位置:")
        print(f"  - 训练模型: models/yolov9_quick/")
        for path in exported_paths:
            print(f"  - 导出模型: {path}")
        print("  - 验证数据: data/swift_trail_formatted/")
        
        # 显示验收标准检查
        if val_results:
            map50 = val_results.get('mAP50', 0)
            recall = val_results.get('recall', 0)
            map75 = val_results.get('mAP75', 0)
            
            print(f"\n=== 快速训练结果 ===")
            print(f"mAP@50: {map50:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"mAP@75: {map75:.4f}")
            print("此为快速训练结果，完整训练将获得更好性能")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        sys.exit(1)
    finally:
        # 清理临时文件
        trainer.cleanup()


if __name__ == "__main__":
    main()