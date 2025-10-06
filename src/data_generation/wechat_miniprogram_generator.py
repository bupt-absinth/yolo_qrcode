"""
微信小程序码生成器
使用真实的微信小程序凭证生成小程序码
"""

import os
import requests
import json
import time
from typing import List, Dict, Optional
from PIL import Image
from io import BytesIO


class WeChatMiniProgramCodeGenerator:
    """微信小程序码生成器"""
    
    def __init__(self, appid: str, appsecret: str, save_dir: str = "data/mini_program_codes"):
        self.appid = appid
        self.appsecret = appsecret
        self.save_dir = save_dir
        self.access_token = None
        self.token_expire_time = 0
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取access_token
        self._get_access_token()
    
    def _get_access_token(self) -> Optional[str]:
        """获取微信access_token"""
        # 检查token是否过期
        if self.access_token and time.time() < self.token_expire_time:
            return self.access_token
            
        url = "https://api.weixin.qq.com/cgi-bin/token"
        params = {
            "grant_type": "client_credential",
            "appid": self.appid,
            "secret": self.appsecret
        }
        
        try:
            response = requests.get(url, params=params)
            result = response.json()
            
            if "access_token" in result:
                self.access_token = result["access_token"]
                # access_token有效期为7200秒(2小时)
                self.token_expire_time = time.time() + 7000  # 提前200秒刷新
                print(f"成功获取access_token，有效期至: {time.ctime(self.token_expire_time)}")
                return self.access_token
            else:
                print(f"获取access_token失败: {result}")
                return None
        except Exception as e:
            print(f"获取access_token异常: {e}")
            return None
    
    def generate_unlimited_qrcode(self, 
                                scene: str, 
                                page: Optional[str] = None,
                                width: int = 430,
                                auto_color: bool = False,
                                line_color: Optional[Dict] = None,
                                is_hyaline: bool = False) -> Optional[bytes]:
        """生成无限制小程序码"""
        # 确保access_token有效
        if not self._get_access_token():
            return None
        
        url = "https://api.weixin.qq.com/wxa/getwxacodeunlimit"
        params = {
            "access_token": self.access_token
        }
        
        data = {
            "scene": scene,
            "width": width,
            "auto_color": auto_color,
            "is_hyaline": is_hyaline
        }
        
        if page:
            data["page"] = page
            
        if line_color:
            data["line_color"] = line_color
        
        try:
            response = requests.post(url, params=params, json=data)
            
            # 检查是否返回错误信息
            if response.headers.get('Content-Type', '').startswith('application/json'):
                error_result = response.json()
                if 'errcode' in error_result:
                    print(f"生成小程序码失败: {error_result}")
                    return None
            
            # 返回图像数据
            return response.content
        except Exception as e:
            print(f"生成小程序码异常: {e}")
            return None
    
    def generate_qr_code(self,
                        path: str,
                        width: int = 430) -> Optional[bytes]:
        """生成小程序二维码（方形二维码）
        
        Args:
            path: 扫码进入的小程序页面路径，最大长度 128 个字符，不能为空
            width: 二维码的宽度，单位 px。最小 280px，最大 1280px，默认是430
            
        Returns:
            二维码图像的二进制数据
        """
        # 确保access_token有效
        if not self._get_access_token():
            return None
        
        url = "https://api.weixin.qq.com/cgi-bin/wxaapp/createwxaqrcode"
        params = {
            "access_token": self.access_token
        }
        
        data = {
            "path": path,
            "width": width
        }
        
        try:
            response = requests.post(url, params=params, json=data)
            
            # 检查是否返回错误信息
            if response.headers.get('Content-Type', '').startswith('application/json'):
                error_result = response.json()
                if 'errcode' in error_result:
                    print(f"生成二维码失败: {error_result}")
                    return None
            
            # 返回图像数据
            return response.content
        except Exception as e:
            print(f"生成二维码异常: {e}")
            return None
    
    def generate_batch_qrcodes(self, count: int = 1000) -> int:
        """批量生成小程序码"""
        success_count = 0
        
        for i in range(count):
            # 生成不同的scene参数确保唯一性
            scene = f"qr_detect_{i:06d}"
            
            # 可选：随机化其他参数
            width = 400 + (i % 5) * 20  # 400-480像素之间变化
            auto_color = i % 3 == 0  # 1/3概率使用自动配色
            
            # 生成小程序码
            image_data = self.generate_unlimited_qrcode(
                scene=scene,
                width=width,
                auto_color=auto_color
            )
            
            if image_data:
                try:
                    # 保存图像
                    image = Image.open(BytesIO(image_data))
                    filename = f"wechat_mp_{i:06d}.png"
                    filepath = os.path.join(self.save_dir, filename)
                    image.save(filepath, "PNG")
                    success_count += 1
                    
                    print(f"已生成小程序码 {i+1}/{count}: {filename}")
                except Exception as e:
                    print(f"保存小程序码失败 {i+1}: {e}")
            else:
                print(f"生成小程序码失败 {i+1}")
            
            # 控制请求频率，避免触发限制
            if (i + 1) % 10 == 0:
                time.sleep(1)  # 每10个请求休息1秒
            
            # 每100个请求重新获取token（如果需要）
            if (i + 1) % 100 == 0:
                self._get_access_token()
        
        print(f"批量生成完成，成功生成 {success_count}/{count} 个小程序码")
        return success_count


if __name__ == "__main__":
    # 使用您提供的凭证
    APPID = "wx2feb1f6c99f73709"
    APPSECRET = "d5c822694247fcf18f72686812b07872"
    
    # 创建生成器
    generator = WeChatMiniProgramCodeGenerator(APPID, APPSECRET)
    
    # 生成100个小程序码作为示例
    success_count = generator.generate_batch_qrcodes(100)