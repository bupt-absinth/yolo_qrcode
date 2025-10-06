"""
YOLOv9超快速训练验证脚本
使用仅10张图像进行超快速训练验证，验证完整训练流程
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


class YOLOv9UltraQuickTrainer:
    """YOLOv9超快速训练验证器"""
    
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
            self.device = 'cpu'  # 超快速验证使用CPU即可
            print(f"使用设备: {self.device}")
        
        # 创建临时训练数据目录
        self.temp_train_dir = "data/temp_ultra_quick_train"
        self.temp_images_dir = os.path.join(self.temp_train_dir, "images")
        self.temp_labels_dir = os.path.join(self.temp_train_dir, "labels")
        
        # 原始数据路径
        self.original_images_dir = "data/synthetic/images"
        self.original_labels_dir = "data/synthetic/labels"
        
        # 测试数据路径（使用少量测试数据）
        self.val_data_path = "data/swift_trail_formatted"
        
        # 创建输出目录
        os.makedirs("models/yolov9_ultra_quick", exist_ok=True)
    
    def prepare_training_data(self, sample_count: int = 10) -> bool:
        """
        准备训练数据（随机选择指定数量的样本）
        
        Args:
            sample_count: 要选择的样本数量（默认10张）
            
        Returns:
            bool: 准备是否成功
        """
        try:
            print(f"准备 {sample_count} 张训练图像进行超快速验证...")
            
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
                print(f"已复制样本 {i + 1}/{len(selected_images)}: {image_file}")
            
            print(f"超快速训练数据准备完成，共 {len(selected_images)} 个样本")
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
            'nc': 1,                       # 类别数量（简化为1类）
            'names': ['qr_code']           # 类别名称
        }
        
        yaml_path = "configs/yolov9_ultra_quick_test.yaml"
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"数据集配置文件已创建: {yaml_path}")
        return yaml_path
    
    def train_model(self, epochs: int = 3) -> YOLO:
        """超快速训练模型"""
        print(f"开始超快速训练YOLOv9模型...")
        print(f"训练样本数: 10")
        print(f"训练轮数: {epochs}")
        
        # 创建数据集配置文件
        dataset_yaml = self._create_dataset_yaml()
        
        # 初始化模型（使用最小的模型）
        model_name = "yolov8n.pt"  # nano模型，最小最快
        print(f"使用预训练模型: {model_name}")
        
        try:
            model = YOLO(model_name)
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("尝试使用随机初始化模型...")
            model = YOLO("yolov8n.yaml")
        
        # 训练配置（超快速）
        imgsz = 320  # 更小的图像尺寸
        batch_size = 4  # 更小的批次
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
                project="models/yolov9_ultra_quick",
                name="yolov8n_qr_detector_ultra_quick",
                exist_ok=True,
                patience=1,  # 早停机制
                optimizer='AdamW',
                verbose=True,
                plots=False,  # 不生成图表加快速度
                save_period=-1  # 不定期保存加快速度
            )
            print(f"YOLOv9超快速训练完成")
            return model
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            raise
    
    def export_model(self, model: YOLO, formats: list = ["pt"]) -> list:
        """导出模型"""
        exported_paths = []
        
        print(f"导出模型到格式: {formats}")
        
        for format in formats:
            try:
                print(f"正在导出到 {format} 格式...")
                
                if format == "pt":
                    # 导出PyTorch格式
                    export_path = model.export(format="pt")
                    exported_paths.append(export_path)
                    print(f"PyTorch模型已导出到: {export_path}")
                    
            except Exception as e:
                print(f"导出 {format} 格式时出现错误: {e}")
        
        return exported_paths
    
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
    print("=== YOLOv9超快速训练验证器 ===")
    print(f"系统信息: {platform.system()} {platform.machine()}")
    
    # 创建训练器
    trainer = YOLOv9UltraQuickTrainer()
    
    try:
        # 准备训练数据
        print("\n=== 准备训练数据 ===")
        if not trainer.prepare_training_data(sample_count=10):
            print("准备训练数据失败")
            return
        
        # 训练模型
        print("\n=== 模型训练 ===")
        model = trainer.train_model(epochs=3)
        
        # 导出模型
        print("\n=== 模型导出 ===")
        exported_paths = trainer.export_model(model, formats=["pt"])
        
        print(f"\n=== 超快速训练完成 ===")
        print("模型和导出文件位置:")
        print(f"  - 训练模型: models/yolov9_ultra_quick/")
        for path in exported_paths:
            print(f"  - 导出模型: {path}")
        print("\n此为超快速训练验证，用于验证完整训练流程是否正常工作")
        print("完整训练将使用更多数据和更长训练时间以获得更好性能")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        sys.exit(1)
    finally:
        # 清理临时文件
        trainer.cleanup()


if __name__ == "__main__":
    main()