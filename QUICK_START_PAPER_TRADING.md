# 模拟盘交易快速开始指南

## 🚀 三步开始模拟盘测试

### 步骤 1：确认模型已训练

```bash
# 如果还没有训练模型，先运行：
python main.py
```

### 步骤 2：运行模拟盘（Windows）

```bash
# 双击运行或命令行执行：
run_paper_trading.bat
```

### 步骤 3：运行模拟盘（Linux/Mac）

```bash
# 先添加执行权限：
chmod +x run_paper_trading.sh

# 然后运行：
./run_paper_trading.sh
```

或者直接使用 Python：

```bash
python paper_trading_main.py --mode daily
```

## 📊 查看结果

模拟盘运行后，结果保存在：

- **状态文件**: `results/paper_trading/paper_trading_state.json`
- **每日报告**: `results/paper_trading/daily_report_YYYYMMDD.json`
- **交易历史**: `results/paper_trading/trade_history.csv`

## ⏰ 建议运行时间

- **最佳时间**: 每个交易日收盘后（如 16:00 后）
- **运行频率**: 每个交易日运行一次
- **测试时长**: 建议至少运行 1-2 个月

## ✅ 验证清单

运行后检查：

- [ ] 日志文件正常生成（`logs/paper_trading.log`）
- [ ] 状态文件正常保存
- [ ] 交易信号正常生成
- [ ] 无错误信息

## 🔄 下一步

1. **每日运行**: 设置定时任务，每个交易日自动运行
2. **监控结果**: 每周查看一次投资组合表现
3. **对比回测**: 1 个月后对比模拟盘收益与回测收益
4. **准备实盘**: 如果表现稳定，考虑小资金实盘

---

**详细文档**: 请查看 `PAPER_TRADING_README.md`




