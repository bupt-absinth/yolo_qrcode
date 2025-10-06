# 二维码/小程序码目标检测系统

基于YOLOv8/v9的工业级二维码和小程序码检测系统，采用全合成数据训练，支持Mac M1平台高性能推理。

## 项目结构

```
qrcode_detect/
├── README.md
├── requirements.txt
├── data/
│   ├── backgrounds/           # 背景图库
│   ├── qr_codes/              # 二维码原图
│   ├── mini_program_codes/    # 小程序码原图
│   ├── synthetic/             # 合成数据集
│   └── test/                  # 真实测试集
├── models/
│   ├── yolov8/               # YOLOv8模型
│   └── yolov9/               # YOLOv9模型
├── src/
│   ├── data_generation/      # 数据生成模块
│   ├── augmentation/         # 数据增强模块
│   ├── training/             # 训练脚本
│   ├── inference/            # 推理模块
│   └── postprocessing/       # 后处理模块
├── utils/
│   └── helpers.py            # 辅助函数
├── configs/                  # 配置文件
└── deployments/              # 部署相关
    ├── onnx/                 # ONNX模型
    └── coreml/               # Core ML模型
```

## 系统特性

1. 高精度定位：精确框选码的正方形区域
2. 高召回率：识别各种复杂背景、尺寸、透视形变下的二维码/小程序码
3. 成本优化：100%自动化合成训练数据，无需人工标注
4. 高性能推理：Mac M1优化，推理时间≤100ms

## 安装依赖

```bash
pip install -r requirements.txt
```