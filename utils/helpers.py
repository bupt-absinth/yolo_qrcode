"""
辅助函数工具模块
"""

import os
import json
import numpy as np
import cv2
from typing import List, Tuple, Dict


def create_directory(dir_path: str) -> None:
    """创建目录"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_json(data: Dict, file_path: str) -> None:
    """保存JSON数据到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_path: str) -> Dict:
    """从文件加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_square_bbox(xmin: float, ymin: float, xmax: float, ymax: float) -> Tuple[float, float, float, float]:
    """
    将矩形边界框转换为正方形边界框
    
    Args:
        xmin, ymin, xmax, ymax: 原始矩形框坐标
        
    Returns:
        正方形框坐标 (xmin, ymin, xmax, ymax)
    """
    # 计算宽度和高度
    width = xmax - xmin
    height = ymax - ymin
    
    # 确定正方形边长
    side = max(width, height)
    
    # 计算中心点
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    
    # 计算正方形框的新坐标
    new_xmin = center_x - side / 2
    new_ymin = center_y - side / 2
    new_xmax = center_x + side / 2
    new_ymax = center_y + side / 2
    
    return new_xmin, new_ymin, new_xmax, new_ymax


def clip_bbox(bbox: Tuple[float, float, float, float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    将边界框裁剪到图像边界内
    
    Args:
        bbox: 边界框坐标 (xmin, ymin, xmax, ymax)
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        裁剪后的边界框坐标
    """
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(xmin, img_width))
    ymin = max(0, min(ymin, img_height))
    xmax = max(0, min(xmax, img_width))
    ymax = max(0, min(ymax, img_height))
    return xmin, ymin, xmax, ymax


def xywh2xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """将中心点+宽高格式转换为左上右下格式"""
    xmin = x - w / 2
    ymin = y - h / 2
    xmax = x + w / 2
    ymax = y + h / 2
    return xmin, ymin, xmax, ymax


def xyxy2xywh(xmin: float, ymin: float, xmax: float, ymax: float) -> Tuple[float, float, float, float]:
    """将左上右下格式转换为中心点+宽高格式"""
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return x, y, w, h