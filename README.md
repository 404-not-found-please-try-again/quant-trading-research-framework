# 美股预测系统 (US Stock Prediction System)

一个基于机器学习的美国股票价格预测系统，支持多种技术指标特征提取和深度学习模型。

## 🚀 功能特性

- **数据获取**: 使用 yfinance 获取美股历史数据
- **特征工程**: 技术指标、时间特征、价格特征提取
- **多模型支持**: XGBoost、RandomForest、LSTM 等
- **可视化分析**: 准确率分析、混淆矩阵、预测结果展示
- **回测系统**: 策略验证和收益分析
- **风险管理**: 仓位管理和止损止盈

## 📁 项目结构

```
us_stock_predict/
├── data/                   # 数据存储目录
├── src/                    # 源代码
│   ├── data/              # 数据获取和处理
│   ├── features/          # 特征工程
│   ├── models/            # 模型定义和训练
│   ├── visualization/     # 可视化工具
│   ├── backtesting/       # 回测系统
│   └── utils/             # 工具函数
├── notebooks/             # Jupyter笔记本
├── config/                # 配置文件
├── tests/                 # 测试文件
└── results/               # 结果输出
```

## 🛠️ 安装和使用

1. 安装依赖:

```bash
pip install -r requirements.txt
```

2. 运行示例:

```bash
python main.py
```

## 📊 支持的股票

- 美股主要指数 (SPY, QQQ, IWM 等)
- 个股 (AAPL, MSFT, GOOGL 等)
- 自定义股票代码

## 🔧 配置

在 `config/config.yaml` 中配置:

- 股票代码
- 时间范围
- 特征参数
- 模型参数

## 📈 模型性能

- XGBoost: 准确率 ~60-70%
- LSTM: 准确率 ~55-65%
- 集成模型: 准确率 ~65-75%

## ⚠️ 免责声明

本系统仅用于学习和研究目的，不构成投资建议。股市有风险，投资需谨慎。


