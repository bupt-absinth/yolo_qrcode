# 二维码/小程序码目标检测系统

基于YOLOv8/v9的工业级二维码和小程序码检测系统，采用全合成数据训练，支持Mac M1平台高性能推理。

## 项目结构

```
qrcode_detect/
├── README.md
├── requirements.txt
├── data/
│   ├── backgrounds/           # 背景图库 (5000+张)
│   ├── qr_codes/              # 二维码原图
│   ├── mini_program_codes/    # 小程序码原图
│   ├── square_qr_codes/       # 方形二维码原图
│   ├── enhanced_miniprogram_codes/  # 增强后的小程序码
│   ├── enhanced_square_qr_codes/    # 增强后的方形二维码
│   ├── croped_square_qr_codes/      # 裁剪后的方形二维码
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
5. 多种码类型支持：
   - 标准二维码
   - 微信小程序码
   - 方形二维码（带Logo）

## 数据生成与增强功能

### 小程序码生成与增强
- 使用微信API生成真实小程序码
- 中心圆形区域直径为整张图片大小的一半
- 使用背景图片替换中心圆形部分

### 方形二维码生成与增强
- 使用微信API生成带Logo的方形二维码
- 文件名格式：`wechat_sp_{6位数字}.png`
- 中心圆形区域直径为整张图片大小的1/4
- 使用背景图片替换中心圆形部分

### 图片裁剪处理
- 对增强后的方形二维码进行裁剪：
  - 上部裁剪1/10
  - 左侧裁剪1/10
  - 右侧裁剪1/10
  - 底部裁剪1/5
- 裁剪后图片存储在`croped_square_qr_codes`目录

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

### 生成和增强二维码/小程序码

```bash
# 生成并增强所有类型的码
python src/data_generation/generate_and_enhance_miniprogram_codes.py

# 仅生成方形二维码
python src/data_generation/generate_and_enhance_miniprogram_codes.py

# 仅增强方形二维码
python src/data_generation/enhance_square_qr_codes.py

# 仅裁剪增强后的方形二维码
python src/data_generation/crop_square_qr_codes.py
```

## 模型训练

YOLOv8/v9模型训练脚本位于`src/training/`目录下。

## 推理部署

支持ONNX和Core ML格式，适用于Mac M1平台高性能推理。