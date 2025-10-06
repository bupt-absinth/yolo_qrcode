"""
Mac M1平台推理优化模块
"""

import os
import torch
import coremltools as ct
from ultralytics import YOLO
import numpy as np
from typing import Dict, Any


class M1Optimizer:
    """Mac M1优化器"""
    
    def __init__(self):
        # 检查是否在Mac上运行
        self.is_mac = os.uname().sysname == 'Darwin'
        self.is_arm = os.uname().machine == 'arm64'
        
        # 检查MPS支持
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        print(f"系统信息: Mac={self.is_mac}, ARM={self.is_arm}, MPS={self.mps_available}")
    
    def convert_to_onnx(self, model_path: str, output_path: str, img_size: int = 640) -> str:
        """将PyTorch模型转换为ONNX格式"""
        print("正在将模型转换为ONNX格式...")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 导出ONNX模型
        onnx_path = model.export(format="onnx", opset=12)
        
        # 移动到指定位置
        os.rename(onnx_path, output_path)
        
        print(f"ONNX模型已保存到: {output_path}")
        return output_path
    
    def optimize_onnx_model(self, onnx_path: str, output_path: str) -> str:
        """优化ONNX模型"""
        print("正在优化ONNX模型...")
        
        # 注意：onnxruntime.tools.optimizer可能不可用，这里提供简化版本
        # 在实际应用中，您可能需要安装额外的工具或使用其他优化方法
        
        # 直接复制文件作为示例
        import shutil
        shutil.copy2(onnx_path, output_path)
        
        print(f"ONNX模型已保存到: {output_path}")
        return output_path
    
    def convert_to_coreml(self, onnx_path: str, output_path: str, quantize: bool = True) -> str:
        """将ONNX模型转换为Core ML格式"""
        if not self.is_mac:
            raise RuntimeError("Core ML转换仅支持在Mac上运行")
        
        print("正在将模型转换为Core ML格式...")
        
        try:
            # 转换为Core ML模型
            coreml_model = ct.converters.onnx.convert(
                onnx_path,
                minimum_ios_deployment_target='13'
            )
            
            # 设置模型元数据
            coreml_model.short_description = "二维码/小程序码检测模型"
            coreml_model.input_description["input"] = "输入图像 (RGB格式)"
            coreml_model.output_description["output"] = "检测结果"
            
            # 量化模型以减小大小
            if quantize:
                print("正在量化模型...")
                coreml_model = ct.models.neural_network.quantization_utils.quantize_weights(
                    coreml_model, 
                    nbits=16
                )
            
            # 保存Core ML模型
            coreml_model.save(output_path)
            
            print(f"Core ML模型已保存到: {output_path}")
            return output_path
        except Exception as e:
            print(f"Core ML转换失败: {e}")
            # 返回原始ONNX路径作为后备
            return onnx_path
    
    def benchmark_model(self, model_path: str, img_size: int = 640, iterations: int = 100) -> Dict[str, Any]:
        """基准测试模型性能"""
        print("正在进行模型基准测试...")
        
        # 初始化时间变量
        start_time = 0.0
        end_time = 0.0
        
        # 创建测试输入
        test_input = torch.randn(1, 3, img_size, img_size)
        
        # 根据模型类型选择推理引擎
        if model_path.endswith('.pt'):
            # PyTorch模型
            model = YOLO(model_path)
            model.to('mps' if self.mps_available else 'cpu')
            test_input = test_input.to('mps' if self.mps_available else 'cpu')
            
            # 预热
            for _ in range(10):
                _ = model(test_input)
            
            # 测试
            import time
            start_time = time.time()
            for _ in range(iterations):
                _ = model(test_input)
            end_time = time.time()
            
        elif model_path.endswith('.onnx'):
            # ONNX模型
            try:
                import onnxruntime as ort
                
                # 配置执行提供者
                providers = ['CPUExecutionProvider']
                if self.mps_available:
                    # 注意：CoreMLExecutionProvider可能不可用，需要检查
                    available_providers = ort.get_available_providers()
                    if 'CoreMLExecutionProvider' in available_providers:
                        providers = ['CoreMLExecutionProvider'] + providers
                    elif 'CPUExecutionProvider' in available_providers:
                        providers = ['CPUExecutionProvider']
                
                session = ort.InferenceSession(model_path, providers=providers)
                
                # 预热
                for _ in range(10):
                    _ = session.run(None, {'images': test_input.numpy()})
                
                # 测试
                import time
                start_time = time.time()
                for _ in range(iterations):
                    _ = session.run(None, {'images': test_input.numpy()})
                end_time = time.time()
            except ImportError:
                print("ONNX Runtime未安装，跳过ONNX模型测试")
                start_time = end_time = 0.0
        
        # 计算平均推理时间
        if end_time > start_time:
            avg_time_ms = (end_time - start_time) / iterations * 1000
        else:
            avg_time_ms = 0.0
        
        results = {
            'average_inference_time_ms': avg_time_ms,
            'fps': 1000 / avg_time_ms if avg_time_ms > 0 else 0,
            'model_path': model_path,
            'device': 'MPS' if self.mps_available else 'CPU'
        }
        
        print(f"基准测试结果:")
        print(f"  平均推理时间: {avg_time_ms:.2f} ms")
        print(f"  FPS: {results['fps']:.2f}")
        print(f"  设备: {results['device']}")
        
        return results


if __name__ == "__main__":
    # 示例使用
    optimizer = M1Optimizer()
    
    # 转换模型
    onnx_path = optimizer.convert_to_onnx(
        "models/yolov8/yolov8m_qr_detector/weights/best.pt",
        "deployments/onnx/yolov8m_qr_detector.onnx"
    )
    
    # 优化ONNX模型
    optimized_onnx_path = optimizer.optimize_onnx_model(
        onnx_path,
        "deployments/onnx/yolov8m_qr_detector_optimized.onnx"
    )
    
    # 转换为Core ML (仅在Mac上)
    if optimizer.is_mac:
        coreml_path = optimizer.convert_to_coreml(
            optimized_onnx_path,
            "deployments/coreml/yolov8m_qr_detector.mlmodel"
        )
    
    # 基准测试
    benchmark_results = optimizer.benchmark_model(optimized_onnx_path)