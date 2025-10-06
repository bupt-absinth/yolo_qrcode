"""
对所有数据进行增强处理的主脚本
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.augmentation.augmentation_pipeline import AugmentationPipeline


def augment_all_data():
    """对所有合成数据进行增强处理"""
    print("开始对所有合成数据进行增强处理...")
    
    # 创建增强处理管道
    pipeline = AugmentationPipeline(
        input_dir="data/synthetic",
        output_dir="data/augmented"
    )
    
    # 对所有数据进行增强 (每张图像生成5个增强版本)
    success_count = pipeline.augment_batch(None, 5)
    
    print(f"数据增强完成，总共生成 {success_count} 个增强样本")


if __name__ == "__main__":
    augment_all_data()