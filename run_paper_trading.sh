#!/bin/bash
# Linux/Mac脚本 - 运行模拟盘交易

echo "========================================"
echo "模拟盘交易系统"
echo "========================================"
echo ""

# 检查Python是否安装
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python"
    exit 1
fi

# 运行模拟盘
echo "开始运行模拟盘交易..."
echo ""

python paper_trading_main.py --mode daily

echo ""
echo "========================================"
echo "模拟盘交易完成"
echo "========================================"





