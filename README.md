# Quantitative Trading Research Framework  
量化交易研究框架  

A modular research framework for short-term U.S. stock and ETF trend prediction.  
一个面向美股短期趋势预测的模块化量化研究框架。  

It provides a complete end-to-end pipeline — from data acquisition and feature engineering to model training, backtesting, and visualization.  
框架实现了从数据获取、特征工程、模型训练到回测与可视化的完整端到端流程。  

> ⚠️ **Disclaimer**: This project is for research and educational purposes only. It does not constitute financial advice or trading recommendations.  
> ⚠️ **免责声明**：本项目仅用于学习与研究，不构成任何投资建议或交易信号。  

---

## 🔧 Key Features / 核心特性  

### 📈 Data Pipeline / 数据处理流程  
- Fetches U.S. stock and ETF data using **`yfinance`**.  
  通过 **`yfinance`** 获取美股与 ETF 历史数据。  
- Fully configurable through YAML (`config/config.yaml`).  
  所有参数均可通过 YAML 文件配置（`config/config.yaml`）。  

---

### 🧠 Feature Engineering / 特征工程  
- Includes classical technical indicators: **SMA, EMA, RSI, MACD, Bollinger Bands, ATR**, etc.  
  支持多种经典技术指标（SMA、EMA、RSI、MACD、布林带、ATR 等）。  
- Adds advanced financial features such as volatility, resistance/support distance, and momentum.  
  融合了高级金融特征（波动率、支撑/阻力距离、动量指标等）。  
- Supports interaction features and category balancing (e.g., SMOTE).  
  支持特征交互与样本平衡（如 SMOTE 过采样）。  

---

### 🤖 Models / 模型模块  
- ✅ **XGBoost** — Primary model achieving strong backtesting performance.  
  主要模型，回测表现稳定优秀。  
- ✅ **Random Forest** — Used for comparison and ensemble learning.  
  用于对比与集成学习的辅助模型。  
- 🧪 **LSTM (Experimental)** — Temporarily removed due to limited data and sparse signals, but retained for research exploration.  
  实验性模块：因数据量不足与信号稀疏暂时停用，但代码保留用于后续研究。  

---

### 💹 Backtesting & Risk Management / 回测与风险控制  
- Supports position sizing, stop-loss/take-profit, and confidence filtering.  
  支持仓位控制、止损止盈与置信度过滤策略。  
- Calculates key metrics: **Return, Sharpe Ratio, Max Drawdown**, etc.  
  计算关键绩效指标（收益率、Sharpe 比率、最大回撤等）。  
- Fully modular design with configurable transaction costs and thresholds.  
  模块化设计，支持配置交易成本与风险阈值。  

---

### 📊 Visualization / 可视化  
- Generates ROC/PR curves, confusion matrices, feature importance, and backtest performance plots.  
  自动生成 ROC/PR 曲线、混淆矩阵、特征重要性与回测表现图。  
- All charts saved under `results/plots/`.  
  所有图表均保存在 `results/plots/` 目录下。  

---

## 📁 Project Structure / 项目结构  

\`\`\`bash
quant-trading-research-framework/
├── config/                # 全局配置（YAML 文件）
├── src/                   # 核心源码模块
│   ├── data/              # 数据下载与预处理
│   ├── features/          # 特征工程
│   ├── models/            # 模型训练与评估
│   ├── backtesting/       # 回测与风险管理
│   ├── visualization/     # 可视化绘图
│   └── utils/             # 工具函数
├── results/               # 模型结果与图表
├── main.py                # 主程序入口
├── requirements.txt        # 依赖包
└── README.md
\`\`\`

---

## 🚀 Quick Start / 快速开始  

\`\`\`bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 修改配置
vim config/config.yaml

# 3. 运行主程序（训练 + 评估 + 可视化）
python main.py
\`\`\`

---

## 📈 Example Results / 示例结果  

| Model | Annualized Return | Sharpe Ratio | Notes |
|--------|-------------------|---------------|--------|
| XGBoost | ~7.32% | 2.51 | Stable, strong signal quality |
| Random Forest | ~7.78% | 2.80 | Slightly higher Sharpe ratio |
| LSTM | 0.16% | — | Removed due to sparse signals and overfitting |

> These results are based on backtesting over QQQ and COO ETFs.  
> 实验结果基于 QQQ 与 COO 的回测区间，仅作研究参考。  

---

## 📚 Future Work / 后续方向  

- Extend dataset and retrain LSTM under higher data volume.  
  扩充时间序列数据以重新评估 LSTM 性能。  
- Implement ensemble stacking between tree-based and deep learning models.  
  尝试集成树模型与深度学习模型的堆叠结构。  
- Build a Streamlit dashboard for real-time visualization.  
  开发 Streamlit 实时可视化面板。  

---

## ⚖️ Disclaimer / 免责声明  

This project is intended for research and educational use only.  
All financial data used are public and anonymized.  
No content here constitutes investment advice or guarantees of performance.  

本项目仅供学习与研究使用。  
所使用的金融数据均来源于公开渠道。  
本项目不构成任何投资建议或收益保证。  



