# 美股预测系统 - 快速开始指南

## 🚀 快速开始

### 1. 环境要求

- Python 3.8 或更高版本
- Windows/Linux/macOS

### 2. 安装

#### 方法一：自动安装（推荐）

```bash
python install.py
```

#### 方法二：手动安装

```bash
# 安装依赖
pip install -r requirements.txt

# 创建目录
mkdir data results logs
mkdir results\plots results\models results\backtesting
```

### 3. 运行示例

#### 快速示例

```bash
python example.py
```

#### 完整预测

```bash
python main.py
```

#### Jupyter 笔记本

```bash
jupyter notebook notebooks/stock_prediction_demo.ipynb
```

## 📊 系统功能

### 数据获取

- 支持多只美股数据下载
- 自动处理股票分割和分红
- 数据缓存和更新

### 特征工程

- 50+ 技术指标
- 时间特征
- 价格特征
- 自动特征选择

### 模型训练

- XGBoost
- RandomForest
- LSTM（深度学习）
- 模型集成

### 回测分析

- 策略回测
- 风险指标计算
- 性能分析
- 可视化报告

## ⚙️ 配置

编辑 `config/config.yaml` 文件：

```yaml
# 股票代码
data:
  symbols:
    - "AAPL"
    - "MSFT"
    - "GOOGL"

  # 时间范围
  start_date: "2020-01-01"
  end_date: "2024-12-31"
```

## 📈 使用示例

### 基本使用

```python
from src.data.data_collector import DataCollector
from src.features.feature_engineer import FeatureEngineer
from src.models.model_trainer import ModelTrainer

# 加载配置
config = load_config('config/config.yaml')

# 数据收集
collector = DataCollector(config)
data = collector.collect_data()

# 特征工程
engineer = FeatureEngineer(config)
features = engineer.create_features(data)

# 模型训练
trainer = ModelTrainer(config)
models = trainer.train_models(features)
```

### 自定义股票

```python
# 修改配置文件
config['data']['symbols'] = ['TSLA', 'NVDA', 'AMD']

# 或者直接传递
collector = DataCollector(config)
data = collector.collect_data()
```

## 🔧 高级功能

### 自定义特征

```python
# 在 feature_engineer.py 中添加自定义特征
def _add_custom_features(self, data):
    # 你的自定义特征逻辑
    data['custom_feature'] = data['close'] / data['open']
    return data
```

### 自定义模型

```python
# 在 model_trainer.py 中添加自定义模型
def _train_custom_model(self, X_train, y_train):
    # 你的自定义模型
    model = YourCustomModel()
    model.fit(X_train, y_train)
    return model
```

## 📊 结果解读

### 模型性能指标

- **准确率**: 预测正确的比例
- **精确率**: 预测为上涨中实际上涨的比例
- **召回率**: 实际上涨中被预测为上涨的比例
- **F1 分数**: 精确率和召回率的调和平均

### 回测指标

- **总收益**: 策略总收益率
- **年化收益**: 年化收益率
- **夏普比率**: 风险调整后收益
- **最大回撤**: 最大亏损幅度
- **胜率**: 盈利交易比例

## 🚨 注意事项

1. **数据质量**: 确保网络连接稳定，数据下载完整
2. **过拟合**: 避免在训练集上过度优化
3. **风险控制**: 回测结果不代表未来表现
4. **实时更新**: 定期更新模型和数据

## 🆘 常见问题

### Q: 数据下载失败？

A: 检查网络连接，或尝试更换 VPN

### Q: 内存不足？

A: 减少股票数量或时间范围

### Q: 模型准确率低？

A: 尝试调整特征参数或模型参数

### Q: 回测结果不理想？

A: 检查交易成本设置，调整风险管理参数

## 📞 技术支持

如有问题，请查看：

1. 日志文件：`logs/prediction.log`
2. 错误信息：控制台输出
3. 配置文件：`config/config.yaml`

## 📚 更多资源

- [完整文档](README.md)
- [配置说明](config/config.yaml)
- [示例代码](example.py)
- [Jupyter 教程](notebooks/)


