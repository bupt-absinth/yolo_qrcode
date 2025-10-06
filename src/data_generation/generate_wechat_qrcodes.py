"""
使用微信凭证生成小程序码的脚本
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_generation.wechat_miniprogram_generator import WeChatMiniProgramCodeGenerator


def main():
    """主函数"""
    # 使用您提供的微信小程序凭证
    APPID = "wx2feb1f6c99f73709"
    APPSECRET = "d5c822694247fcf18f72686812b07872"
    
    print("开始生成微信小程序码...")
    print(f"AppID: {APPID}")
    
    # 创建生成器
    generator = WeChatMiniProgramCodeGenerator(APPID, APPSECRET)
    
    # 生成小程序码
    # 您可以根据需要调整生成数量
    count = 1000  # 生成1000个小程序码
    success_count = generator.generate_batch_qrcodes(count)
    
    print(f"\n生成完成!")
    print(f"请求生成数量: {count}")
    print(f"成功生成数量: {success_count}")
    print(f"保存路径: data/mini_program_codes/")


if __name__ == "__main__":
    main()