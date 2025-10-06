"""
边界框校正模块
"""

import numpy as np
from typing import List, Tuple, Dict
from utils.helpers import calculate_square_bbox, clip_bbox


class BBoxCorrector:
    """边界框校正器"""
    
    def __init__(self, img_width: int, img_height: int):
        self.img_width = img_width
        self.img_height = img_height
    
    def apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """应用非极大值抑制 (NMS)"""
        if len(detections) == 0:
            return detections
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while len(detections) > 0:
            # 保留置信度最高的检测框
            keep.append(detections[0])
            current = detections[0]
            
            # 计算当前框与其他框的IoU
            remaining = []
            for det in detections[1:]:
                iou = self._calculate_iou(current['box_2d'], det['box_2d'])
                if iou < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 计算并集
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def enforce_square_constraint(self, detections: List[Dict]) -> List[Dict]:
        """强制执行正方形约束"""
        corrected_detections = []
        
        for det in detections:
            # 获取原始边界框
            xmin, ymin, xmax, ymax = det['box_2d']
            
            # 转换为正方形边界框
            square_xmin, square_ymin, square_xmax, square_ymax = calculate_square_bbox(
                xmin, ymin, xmax, ymax
            )
            
            # 裁剪到图像边界内
            square_xmin, square_ymin, square_xmax, square_ymax = clip_bbox(
                (square_xmin, square_ymin, square_xmax, square_ymax),
                self.img_width, self.img_height
            )
            
            # 更新检测结果
            corrected_det = det.copy()
            corrected_det['box_2d'] = [square_xmin, square_ymin, square_xmax, square_ymax]
            corrected_det['is_square'] = True
            
            corrected_detections.append(corrected_det)
        
        return corrected_detections
    
    def filter_by_confidence(self, detections: List[Dict], min_confidence: float = 0.5) -> List[Dict]:
        """根据置信度过滤检测结果"""
        return [det for det in detections if det['confidence'] >= min_confidence]
    
    def post_process_detections(self, detections: List[Dict], 
                              apply_nms: bool = True,
                              nms_threshold: float = 0.5,
                              min_confidence: float = 0.5,
                              enforce_square: bool = True) -> List[Dict]:
        """完整的后处理流程"""
        # 1. 置信度过滤
        filtered_detections = self.filter_by_confidence(detections, min_confidence)
        
        # 2. 应用NMS
        if apply_nms and len(filtered_detections) > 1:
            filtered_detections = self.apply_nms(filtered_detections, nms_threshold)
        
        # 3. 强制正方形约束
        if enforce_square:
            filtered_detections = self.enforce_square_constraint(filtered_detections)
        
        return filtered_detections


if __name__ == "__main__":
    # 示例使用
    corrector = BBoxCorrector(640, 640)
    
    # 模拟检测结果
    detections = [
        {
            'box_2d': [100, 100, 200, 180],  # 矩形框
            'confidence': 0.95,
            'class_id': 0,
            'class_name': 'code'
        },
        {
            'box_2d': [300, 300, 400, 390],  # 矩形框
            'confidence': 0.87,
            'class_id': 0,
            'class_name': 'code'
        }
    ]
    
    # 后处理
    corrected_detections = corrector.post_process_detections(detections)
    
    print("原始检测结果:")
    for i, det in enumerate(detections):
        print(f"  检测 {i+1}: {det['box_2d']}")
    
    print("\n校正后检测结果:")
    for i, det in enumerate(corrected_detections):
        print(f"  检测 {i+1}: {det['box_2d']} (正方形: {det['is_square']})")