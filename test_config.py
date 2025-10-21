#!/usr/bin/env python3
"""
测试脚本：验证火灾检测项目配置是否正确
"""

import os
import sys
import torch
from pathlib import Path

def test_dataset_structure():
    """测试数据集结构"""
    print("🔍 检查数据集结构...")
    
    required_paths = [
        './dataset/FireMatch/Train',
        './dataset/FireMatch/Test'
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"❌ 路径不存在: {path}")
            return False
        else:
            print(f"✅ 路径存在: {path}")
            
            # 检查子文件夹
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            print(f"   子文件夹: {subdirs}")
            
            # 检查是否有PKL文件
            pkl_files = []
            for subdir in subdirs:
                subdir_path = os.path.join(path, subdir)
                pkl_count = len([f for f in os.listdir(subdir_path) if f.endswith('.pkl')])
                pkl_files.append(f"{subdir}: {pkl_count}个PKL文件")
            
            print(f"   PKL文件: {pkl_files}")
    
    return True

def test_model_config():
    """测试模型配置"""
    print("\n🔍 检查模型配置...")
    
    try:
        from all_model.mae2.videomae2 import VisionTransformer as vit
        
        # 测试创建模型
        model = vit(num_classes=2, embed_dim=384, img_size=160)
        print("✅ VideoMAE2模型创建成功")
        print(f"   类别数: 2")
        print(f"   嵌入维度: 384")
        print(f"   输入尺寸: 160")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   总参数数: {total_params/1e6:.2f}M")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型配置错误: {e}")
        return False

def test_data_loader():
    """测试数据加载器"""
    print("\n🔍 检查数据加载器...")
    
    try:
        from Load_Videos import Get_Dataloader
        
        # 测试数据加载器
        test_path = ['./dataset/FireMatch/Test']
        dataloader, label_dict = Get_Dataloader(test_path, 'test', 1)
        
        print("✅ 数据加载器创建成功")
        print(f"   标签字典: {label_dict}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器错误: {e}")
        return False

def test_gpu_availability():
    """测试GPU可用性"""
    print("\n🔍 检查GPU可用性...")
    
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   当前GPU: {torch.cuda.current_device()}")
    else:
        print("⚠️  GPU不可用，将使用CPU训练")
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始验证火灾检测项目配置...")
    print("=" * 50)
    
    tests = [
        test_dataset_structure,
        test_model_config,
        test_data_loader,
        test_gpu_availability
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目可以开始训练了！")
        print("\n📝 训练命令:")
        print("python SIAVC.py --num-classes 2 --dataset Fire --arch resnet --batch-size 2 --num-labeled 80")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
