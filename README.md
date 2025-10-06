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

## 环境安装

### 基础依赖

```bash
# 安装基础Python依赖
pip install -r requirements.txt

# 安装YOLOv8/v9相关依赖
pip install ultralytics

# Mac M1芯片额外依赖
pip install coremltools
pip install onnxruntime-silicon  # Mac M1优化版本
```

### 依赖说明

- `ultralytics`: YOLOv8/v9模型训练和推理
- `coremltools`: Core ML模型转换和推理（Mac M1优化）
- `onnxruntime-silicon`: ONNX模型推理（Mac M1优化）
- `torch`: PyTorch深度学习框架
- `opencv-python`: 图像处理
- `numpy`: 数值计算
- `Pillow`: 图像处理

## 数据生成与增强功能

### 模型训练与推理（Mac M1适配）

支持在Mac M1芯片上进行高性能模型训练和推理：

1. **模型训练**：
   - 使用YOLOv9架构（基于YOLOv8实现）
   - 支持MPS（Metal Performance Shaders）加速
   - 自动适配Mac M1芯片优化

2. **模型导出**：
   - ONNX格式：跨平台兼容，支持Mac M1加速
   - CoreML格式：原生Mac M1优化，最高性能
   - PyTorch格式：原始模型格式

3. **模型推理**：
   - 支持多种模型格式
   - Mac M1芯片高性能推理（≤100ms）
   - 符合项目输出规范（JSON格式）

### 常见问题解决

#### 类别ID不匹配问题

**问题描述**：
```
train: /Users/bupt_absinth/Work/projects/qrcode_detect/data/synthetic/images/synthetic_019664.png: ignoring corrupt image/label: Label class 2 exceeds dataset class count 1. Possible class labels are 0-0
```

**问题原因**：
合成数据生成器为不同类型的二维码分配了不同的类别ID（0, 1, 2），但数据集配置文件只定义了一个类别。

**解决方案**：
1. 统一所有二维码类别ID为0（默认设置）
2. 或更新数据集配置以支持多类别（nc: 3）

**验证方法**：
```bash
# 生成测试样本验证修复
python src/data_generation/generate_test_samples.py

# 修复现有标签文件的类别ID
python src/data_generation/fix_label_class_ids.py
```

#### PNG图像ICC配置文件警告

**问题描述**：
```
libpng warning: iCCP: profile 'ICC Profile': 'wtpt': ICC profile tag start not a multiple of 4
libpng warning: iCCP: known incorrect sRGB profile
```

**问题原因**：
某些PNG图像包含不正确或损坏的ICC颜色配置文件，这通常是图像编辑软件生成的。

**影响评估**：
- **无实际影响**：这些警告仅涉及图像的颜色配置文件元数据
- **不影响训练**：图像数据本身完整，模型训练正常进行
- **不影响精度**：不会降低模型检测精度或性能

**解决方案**：
1. **自动抑制**：训练脚本已配置自动抑制这些警告
2. **手动修复**：使用PNG配置文件修复工具清理图像
   ```bash
   # 修复合成数据中的PNG图像
   python src/data_generation/fix_png_profiles.py
   ```

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

# 生成测试样本验证修复
python src/data_generation/generate_test_samples.py
```

### 模型训练（Mac M1适配）

```bash
# 训练YOLOv9模型并导出适配Mac M1的格式
python src/training/train_yolov9_mac.py

# 快速训练验证（使用1000张图像快速验证训练流程）
python src/training/train_yolov9_quick_test.py

# 原始YOLO训练脚本
python src/training/train_yolo.py
```

### 模型推理（Mac M1适配）

```bash
# 使用CoreML模型进行高性能推理（推荐）
python src/inference/yolov9_inference_mac.py
```

## 模型性能指标

- **mAP@50**：≥ 95%
- **召回率**：≥ 98%
- **定位精度mAP@75**：≥ 80%
- **Mac M1平台推理延迟**：≤ 100ms

## 输出格式规范

模型输出严格遵循JSON格式规范：
```json
{
  "code_count": 2,
  "model_version": "yolov9-coreml",
  "time_ms": 45.67,
  "detections": [
    {
      "bbox": [100, 120, 200, 220],
      "confidence": 0.987,
      "class_id": 0,
      "class_name": "qr_code"
    }
  ]
}
```

## 部署说明

### Mac M1部署优化

1. **CoreML格式**（推荐）：
   - 原生支持Mac M1芯片
   - 最高性能和最低功耗
   - 无需额外依赖

2. **ONNX格式**：
   - 跨平台兼容
   - 使用onnxruntime-silicon获得M1优化

3. **PyTorch格式**：
   - 原始模型格式
   - 支持MPS加速