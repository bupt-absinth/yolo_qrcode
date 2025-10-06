"""
工业级二维码批量生成脚本
生成大量定制化二维码以满足工业级训练需求
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_generation.enhanced_qrcode_generator import EnhancedQRCodeGenerator


def main():
    """主函数"""
    print("=== 工业级二维码批量生成系统 ===")
    
    # 创建生成器
    generator = EnhancedQRCodeGenerator()
    
    # 生成工业级数量的二维码 (10,000个)
    print("\n开始生成工业级二维码...")
    print("预计生成10,000个定制化二维码...")
    
    # 分批生成以避免内存问题
    batch_size = 1000
    total_count = 10000
    generated_count = 0
    
    for batch in range(0, total_count, batch_size):
        batch_count = min(batch_size, total_count - batch)
        prefix = f"industrial_qr_{batch:06d}"
        
        print(f"\n生成批次 {batch//batch_size + 1}/{(total_count-1)//batch_size + 1}")
        print(f"批次大小: {batch_count}")
        
        try:
            files = generator.generate_batch(batch_count, prefix)
            generated_count += len(files)
            print(f"批次完成，成功生成 {len(files)} 个二维码")
        except Exception as e:
            print(f"批次生成失败: {e}")
        
        # 显示进度
        progress = (batch + batch_count) / total_count * 100
        print(f"总体进度: {generated_count}/{total_count} ({progress:.1f}%)")
    
    print(f"\n=== 生成完成 ===")
    print(f"总计生成 {generated_count} 个工业级二维码")
    print(f"二维码保存在: {generator.save_dir}")
    print(f"使用的Logo文件: {len(generator.logo_files)} 个")
    print(f"可选背景图片: {len(generator.background_files)} 张")


if __name__ == "__main__":
    main()