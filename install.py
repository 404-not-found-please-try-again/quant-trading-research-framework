#!/usr/bin/env python3
"""
美股预测系统安装脚本
自动安装依赖和设置环境
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """安装依赖包"""
    print("📦 安装依赖包...")
    
    try:
        # 升级pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # 安装requirements.txt中的包
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ 依赖包安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False

def create_directories():
    """创建必要的目录"""
    print("📁 创建项目目录...")
    
    directories = [
        "data",
        "results",
        "results/plots",
        "results/models", 
        "results/backtesting",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ 创建目录: {directory}")
    
    print("✅ 目录创建完成")

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("   需要Python 3.8或更高版本")
        return False
    
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def test_imports():
    """测试关键模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import xgboost as xgb
        import matplotlib.pyplot as plt
        import seaborn as sns
        import ta
        print("✅ 所有关键模块导入成功")
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def main():
    """主安装函数"""
    print("🚀 美股预测系统安装程序")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 创建目录
    create_directories()
    
    # 安装依赖
    if not install_requirements():
        print("❌ 安装失败，请手动安装依赖包")
        sys.exit(1)
    
    # 测试导入
    if not test_imports():
        print("❌ 模块测试失败，请检查安装")
        sys.exit(1)
    
    print("\n🎉 安装完成！")
    print("=" * 50)
    print("📋 下一步:")
    print("  1. 运行 python example.py 查看示例")
    print("  2. 运行 python main.py 开始预测")
    print("  3. 打开 notebooks/stock_prediction_demo.ipynb 进行交互式分析")
    print("\n⚠️  注意: 首次运行需要下载数据，可能需要几分钟时间")

if __name__ == "__main__":
    main()


