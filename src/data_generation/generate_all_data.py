"""
生成所有训练数据的主脚本
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_generation.qrcode_generator import QRCodeGenerator
from src.data_generation.mini_program_generator import MiniProgramCodeGenerator
from src.data_generation.background_downloader import BackgroundDownloader
from src.data_generation.synthetic_generator import SyntheticGenerator


def generate_all_data():
    """生成所有训练数据"""
    print("开始生成二维码/小程序码目标检测系统的全部训练数据...")
    
    # 1. 生成二维码
    print("\n1. 生成二维码...")
    qr_generator = QRCodeGenerator("data/qr_codes")
    qr_files = qr_generator.generate_batch(1000, "qr")
    print(f"生成二维码完成: {len(qr_files)} 个")
    
    # 2. 生成小程序码
    print("\n2. 生成小程序码...")
    mp_generator = MiniProgramCodeGenerator("data/mini_program_codes")
    mp_files = mp_generator.generate_batch(1000, "mp")
    print(f"生成小程序码完成: {len(mp_files)} 个")
    
    # 3. 下载背景图
    print("\n3. 下载背景图...")
    bg_downloader = BackgroundDownloader("data/backgrounds")
    bg_files = bg_downloader.download_sample_backgrounds(100)
    valid_bg_files = bg_downloader.validate_backgrounds()
    print(f"下载背景图完成: {len(valid_bg_files)} 个有效背景图")
    
    # 4. 生成合成数据
    print("\n4. 生成合成数据...")
    synthetic_generator = SyntheticGenerator()
    success_count = synthetic_generator.generate_batch(1000)  # 生成1000个作为示例
    print(f"生成合成数据完成: {success_count} 个")
    
    print("\n所有数据生成完成！")


if __name__ == "__main__":
    generate_all_data()