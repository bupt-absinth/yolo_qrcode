# 二维码/小程序码目标检测系统

基于YOLOv8/v9的工业级二维码和小程序码检测系统，采用全合成数据训练，支持Mac M1平台高性能推理。

## 项目结构

```
qrcode_detect/
├── README.md
├── requirements.txt
├── data/
│   ├── backgrounds/           # 背景图库 (5000+张)
│   ├── qr_codes/              # 普通二维码原图
│   ├── logos/                 # Logo图库
│   ├── mini_program_codes/    # 小程序码原图
│   ├── enhanced_miniprogram_codes/  # 增强后的小程序码
│   ├── square_qr_codes/       # 方形二维码原图
│   ├── enhanced_square_qr_codes/    # 增强后的方形二维码
│   ├── croped_square_qr_codes/      # 裁剪后的方形二维码
│   ├── synthetic/             # 合成数据集 (50,000+张)
│   │   ├── images/           # 合成图片
│   │   └── labels/           # YOLO格式标签
│   ├── swift_trail/          # 人工标注的测试样本 (LabelImg格式)
│   ├── swift_trail_formatted/ # 标准化后的测试样本 (YOLO格式)
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

### 测试数据集标准化

支持将人工标注的测试样本（LabelImg格式）转换为标准YOLO格式：

1. **格式转换**：
   - 将LabelImg JSON格式转换为YOLO TXT格式
   - 自动归一化坐标到0-1范围
   - 统一类别ID（默认所有二维码类别ID为0）
   - 保持图片文件不变，仅转换标注格式

2. **目录结构**：
   - 输入：`data/swift_trail/`（LabelImg格式）
   - 输出：`data/swift_trail_formatted/`（YOLO格式）
     - `images/`：图片文件
     - `labels/`：YOLO格式标签文件

### 合成数据生成（工业级）

支持生成大量合成训练样本，满足工业级训练需求：

1. **数据合成**：
   - 将背景图与各种二维码图片合成
   - 支持随机变换（旋转、缩放、亮度、对比度调整）
   - 自动生成YOLO格式标签文件
   - 支持三类目标检测：
     - 类别0：小程序码
     - 类别1：方形二维码
     - 类别2：普通二维码

2. **数据规模**：
   - 100%自动化合成数据
   - 包含至少50,000张合成样本
   - 多样化的背景和二维码组合

### 普通二维码生成（工业级）

支持生成大量定制化二维码，满足工业级训练需求：

1. **基础生成**：
   - 使用Python qrcode库生成标准QR码
   - 随机化容错级别（L/M/Q/H）
   - 随机化点阵/背景色（HSL/RGB颜色）

2. **Logo嵌入**：
   - 自动从Logo库中随机抽取Logo
   - 根据容错级别智能确定Logo尺寸
   - 精确粘贴到二维码中心区域

3. **颜色/样式定制**：
   - 随机生成HSL或RGB颜色
   - 支持圆角点阵等样式调整
   - 模拟真实截屏中的金色点阵效果

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

# 生成工业级普通二维码（10,000个）
python src/data_generation/generate_industrial_qrcodes.py

# 生成工业级合成训练数据（50,000个）
python src/data_generation/synthetic_data_generator.py

# 将LabelImg格式测试数据转换为YOLO格式
python src/data_generation/convert_swift_trail_to_yolo.py

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