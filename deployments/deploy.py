"""
模型部署主脚本
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from deployments.m1_optimizer import M1Optimizer


def deploy_model():
    """部署模型到Mac M1平台"""
    print("开始部署二维码/小程序码检测模型到Mac M1平台...")
    
    # 创建优化器
    optimizer = M1Optimizer()
    
    # 检查模型文件是否存在
    model_path = "models/yolov8/yolov8m_qr_detector/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型或提供预训练模型")
        return
    
    # 1. 转换为ONNX格式
    print("\n1. 转换模型为ONNX格式...")
    onnx_path = optimizer.convert_to_onnx(
        model_path,
        "deployments/onnx/yolov8m_qr_detector.onnx"
    )
    
    # 2. 优化ONNX模型
    print("\n2. 优化ONNX模型...")
    optimized_onnx_path = optimizer.optimize_onnx_model(
        onnx_path,
        "deployments/onnx/yolov8m_qr_detector_optimized.onnx"
    )
    
    # 3. 转换为Core ML格式 (仅在Mac上)
    if optimizer.is_mac:
        print("\n3. 转换模型为Core ML格式...")
        coreml_path = optimizer.convert_to_coreml(
            optimized_onnx_path,
            "deployments/coreml/yolov8m_qr_detector.mlmodel"
        )
    
    # 4. 基准测试
    print("\n4. 进行模型基准测试...")
    benchmark_results = optimizer.benchmark_model(optimized_onnx_path)
    
    print("\n部署完成!")
    print("生成的模型文件:")
    print(f"  - ONNX模型: {optimized_onnx_path}")
    if optimizer.is_mac:
        print(f"  - Core ML模型: deployments/coreml/yolov8m_qr_detector.mlmodel")
    
    print("\n性能指标:")
    print(f"  - 平均推理时间: {benchmark_results['average_inference_time_ms']:.2f} ms")
    print(f"  - FPS: {benchmark_results['fps']:.2f}")


if __name__ == "__main__":
    deploy_model()