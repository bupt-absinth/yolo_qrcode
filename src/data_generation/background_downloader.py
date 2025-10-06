"""
背景图库下载模块
"""

import os
import requests
import random
from typing import List
from PIL import Image
from io import BytesIO


class BackgroundDownloader:
    """背景图库下载器"""
    
    def __init__(self, save_dir: str = "data/backgrounds"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 示例背景图片URL列表（实际应用中可以从公开数据集获取）
        self.sample_urls = [
            "https://picsum.photos/800/600",
            "https://picsum.photos/1024/768",
            "https://picsum.photos/1280/720",
            "https://picsum.photos/1920/1080"
        ]
    
    def download_image(self, url: str, filename: str) -> bool:
        """从URL下载单张图片"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # 打开图像并保存
            img = Image.open(BytesIO(response.content))
            filepath = os.path.join(self.save_dir, filename)
            img.save(filepath)
            return True
        except Exception as e:
            print(f"下载图片失败 {url}: {e}")
            return False
    
    def download_sample_backgrounds(self, count: int = 100) -> List[str]:
        """下载示例背景图片"""
        saved_files = []
        
        for i in range(count):
            # 随机选择一个URL模板
            url_template = random.choice(self.sample_urls)
            # 添加随机参数避免缓存
            url = f"{url_template}?random={random.randint(1000, 9999)}"
            
            filename = f"background_{i:04d}.jpg"
            
            if self.download_image(url, filename):
                saved_files.append(filename)
                print(f"已下载 {i + 1}/{count} 张背景图片")
            else:
                print(f"下载失败 {i + 1}/{count}")
        
        print(f"成功下载 {len(saved_files)} 张背景图片，保存在 {self.save_dir}")
        return saved_files
    
    def validate_backgrounds(self) -> List[str]:
        """验证背景图片并返回有效文件列表"""
        valid_files = []
        
        for filename in os.listdir(self.save_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(self.save_dir, filename)
                try:
                    img = Image.open(filepath)
                    img.verify()  # 验证图片完整性
                    valid_files.append(filename)
                except Exception:
                    print(f"无效图片文件: {filename}")
                    # 删除无效文件
                    os.remove(filepath)
        
        print(f"验证完成，有效背景图片: {len(valid_files)} 张")
        return valid_files


if __name__ == "__main__":
    # 示例使用
    downloader = BackgroundDownloader("data/backgrounds")
    # 下载10张示例背景图片
    files = downloader.download_sample_backgrounds(10)
    # 验证图片
    valid_files = downloader.validate_backgrounds()