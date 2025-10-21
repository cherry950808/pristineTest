#!/usr/bin/env python3
"""
视频格式转换工具
用于解决H.264解码警告问题
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def check_ffmpeg():
    """检查FFmpeg是否安装"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_video(input_path, output_path=None, quality='medium'):
    """
    转换视频格式，解决H.264解码问题
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径（可选）
        quality: 质量设置 ('low', 'medium', 'high')
    """
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在 {input_path}")
        return False
    
    if output_path is None:
        # 生成输出文件名
        base_name = Path(input_path).stem
        output_path = f"{base_name}_converted.mp4"
    
    # 质量参数设置
    quality_settings = {
        'low': {'preset': 'ultrafast', 'crf': '28'},
        'medium': {'preset': 'fast', 'crf': '23'},
        'high': {'preset': 'slow', 'crf': '18'}
    }
    
    settings = quality_settings.get(quality, quality_settings['medium'])
    
    # FFmpeg命令
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',           # H.264编码器
        '-preset', settings['preset'], # 编码速度预设
        '-crf', settings['crf'],      # 质量参数
        '-c:a', 'aac',               # 音频编码
        '-movflags', '+faststart',   # 优化网络播放
        '-y',                        # 覆盖输出文件
        output_path
    ]
    
    print(f"正在转换: {input_path} -> {output_path}")
    print(f"质量设置: {quality} (preset={settings['preset']}, crf={settings['crf']})")
    
    try:
        # 运行FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 转换成功: {output_path}")
            return True
        else:
            print(f"❌ 转换失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 转换错误: {e}")
        return False

def batch_convert(input_dir, output_dir=None, quality='medium', extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """
    批量转换目录中的视频文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可选）
        quality: 质量设置
        extensions: 要处理的文件扩展名
    """
    if not os.path.exists(input_dir):
        print(f"错误: 目录不存在 {input_dir}")
        return
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'converted')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找视频文件
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(extensions):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        print(f"在 {input_dir} 中未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"输出目录: {output_dir}")
    
    success_count = 0
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] 处理: {video_file}")
        
        # 生成输出路径
        rel_path = os.path.relpath(video_file, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        output_path = os.path.splitext(output_path)[0] + '_converted.mp4'
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if convert_video(video_file, output_path, quality):
            success_count += 1
    
    print(f"\n📊 转换完成: {success_count}/{len(video_files)} 成功")

def main():
    parser = argparse.ArgumentParser(description='视频格式转换工具')
    parser.add_argument('input', help='输入文件或目录路径')
    parser.add_argument('-o', '--output', help='输出文件或目录路径')
    parser.add_argument('-q', '--quality', choices=['low', 'medium', 'high'], 
                       default='medium', help='质量设置 (默认: medium)')
    parser.add_argument('--batch', action='store_true', help='批量转换模式')
    
    args = parser.parse_args()
    
    # 检查FFmpeg
    if not check_ffmpeg():
        print("❌ 错误: 未找到FFmpeg，请先安装FFmpeg")
        print("安装方法:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  CentOS/RHEL: sudo yum install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: 下载 https://ffmpeg.org/download.html")
        sys.exit(1)
    
    if args.batch:
        # 批量转换模式
        batch_convert(args.input, args.output, args.quality)
    else:
        # 单文件转换模式
        convert_video(args.input, args.output, args.quality)

if __name__ == "__main__":
    main()
