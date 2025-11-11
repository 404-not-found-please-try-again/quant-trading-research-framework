@echo off
REM Windows批处理脚本 - 运行模拟盘交易

echo ========================================
echo 模拟盘交易系统
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)

REM 运行模拟盘
echo 开始运行模拟盘交易...
echo.

python paper_trading_main.py --mode daily

echo.
echo ========================================
echo 模拟盘交易完成
echo ========================================
pause




