"""
将swift_trail目录下的LabelImg格式标注文件转换为YOLO格式
"""

import os
import json
from PIL import Image


def convert_labelimg_to_yolo(json_file_path, image_file_path, output_dir):
    """
    将LabelImg格式的标注文件转换为YOLO格式
    
    Args:
        json_file_path: LabelImg格式的JSON文件路径
        image_file_path: 对应的图片文件路径
        output_dir: YOLO格式标签文件输出目录
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # 获取图片尺寸
        with Image.open(image_file_path) as img:
            img_width, img_height = img.size
        
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_file_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # 转换标注
        with open(output_file_path, 'w') as f:
            # 遍历所有标注
            for item in data:
                annotations = item.get('annotations', [])
                for annotation in annotations:
                    label = annotation.get('label', '')
                    coordinates = annotation.get('coordinates', {})
                    
                    # 获取坐标信息
                    x = coordinates.get('x', 0)
                    y = coordinates.get('y', 0)
                    width = coordinates.get('width', 0)
                    height = coordinates.get('height', 0)
                    
                    # 转换为YOLO格式（归一化坐标）
                    # YOLO格式: <class_id> <center_x> <center_y> <width> <height>
                    # 归一化到0-1范围
                    center_x = x / img_width
                    center_y = y / img_height
                    norm_width = width / img_width
                    norm_height = height / img_height
                    
                    # 确定类别ID（假设所有都是二维码，类别ID为0）
                    # 如果有多种类别，需要建立类别映射表
                    class_id = 0
                    
                    # 写入YOLO格式
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        return True
        
    except Exception as e:
        print(f"转换文件失败 {json_file_path}: {e}")
        return False


def process_swift_trail_dataset(input_dir="data/swift_trail", output_dir="data/swift_trail_formatted"):
    """
    处理swift_trail数据集，将LabelImg格式转换为YOLO格式
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个标注文件")
    
    success_count = 0
    
    # 处理每个JSON文件
    for json_file in json_files:
        json_file_path = os.path.join(input_dir, json_file)
        
        # 查找对应的图片文件
        base_name = os.path.splitext(json_file)[0]
        image_file = None
        
        # 尝试不同的图片扩展名
        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
            potential_image = os.path.join(input_dir, base_name + ext)
            if os.path.exists(potential_image):
                image_file = potential_image
                break
        
        if image_file is None:
            print(f"未找到对应的图片文件: {json_file}")
            continue
        
        # 转换标注文件
        label_output_dir = os.path.join(output_dir, "labels")
        if convert_labelimg_to_yolo(json_file_path, image_file, label_output_dir):
            # 复制图片文件到输出目录
            image_output_dir = os.path.join(output_dir, "images")
            output_image_path = os.path.join(image_output_dir, os.path.basename(image_file))
            
            # 复制图片文件
            try:
                from shutil import copy2
                copy2(image_file, output_image_path)
                success_count += 1
            except Exception as e:
                print(f"复制图片文件失败 {image_file}: {e}")
        else:
            print(f"转换标注文件失败: {json_file}")
        
        # 显示进度
        if (success_count % 10 == 0):
            print(f"已处理 {success_count}/{len(json_files)} 个文件")
    
    print(f"处理完成，成功转换 {success_count}/{len(json_files)} 个文件")
    print(f"标准化数据保存在: {output_dir}")
    print("目录结构:")
    print(f"  - 图片: {os.path.join(output_dir, 'images')}")
    print(f"  - 标签: {os.path.join(output_dir, 'labels')}")


def main():
    """主函数"""
    print("=== Swift Trail数据集格式转换器 ===")
    print("将LabelImg格式标注转换为YOLO格式")
    
    process_swift_trail_dataset()


if __name__ == "__main__":
    main()