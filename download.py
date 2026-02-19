"""
下载 LimiX-2M 模型权重到指定位置。

使用方法：
    python download.py

如果环境没有网络，需要：
1. 在有网络的环境下运行此脚本
2. 或者手动从 HuggingFace 下载后放到指定位置
"""

from pathlib import Path
from huggingface_hub import hf_hub_download


def download_limix_model():
    """下载 LimiX-2M 模型权重到 /workspace/LimiX/cache/"""
    
    # 目标目录
    cache_dir = Path("/workspace/LimiX/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("开始下载 LimiX-2M 模型权重")
    print("=" * 60)
    print(f"目标目录: {cache_dir}")
    print(f"模型: stableai-org/LimiX-2M")
    print(f"文件名: LimiX-2M.ckpt")
    print()
    
    try:
        model_file = hf_hub_download(
            repo_id="stableai-org/LimiX-2M",
            filename="LimiX-2M.ckpt",
            local_dir=str(cache_dir),
        )
        
        print("=" * 60)
        print("✅ 下载成功！")
        print("=" * 60)
        print(f"模型文件路径: {model_file}")
        
        # 检查文件大小
        file_size = Path(model_file).stat().st_size / (1024 * 1024)  # MB
        print(f"文件大小: {file_size:.2f} MB")
        
        return model_file
        
    except Exception as e:
        print("=" * 60)
        print("❌ 下载失败！")
        print("=" * 60)
        print(f"错误信息: {e}")
        print()
        print("可能的原因：")
        print("1. 网络连接问题")
        print("2. HuggingFace 访问受限")
        print("3. 磁盘空间不足")
        print()
        print("解决方案：")
        print("1. 检查网络连接")
        print("2. 在有网络的环境下运行此脚本")
        print("3. 或手动从以下链接下载：")
        print("   https://huggingface.co/stableai-org/LimiX-2M")
        print(f"   然后放到: {cache_dir / 'LimiX-2M.ckpt'}")
        return None


if __name__ == "__main__":
    download_limix_model()

