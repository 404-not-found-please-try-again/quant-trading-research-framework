#!/usr/bin/env python3
"""
美股预测系统示例脚本
演示如何使用系统进行股票预测
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_collector import DataCollector
from features.feature_engineer import FeatureEngineer
from models.model_trainer import ModelTrainer
from visualization.plotter import Plotter
from backtesting.backtester import Backtester
from utils.helpers import load_config, setup_logging

def main():
    """示例主函数"""
    print("美股预测系统示例")
    print("=" * 50)
    
    # 设置日志
    logger = setup_logging('INFO')
    
    try:
        # 加载配置
        config = load_config()
        logger.info("配置加载成功")
        
        # 1. 数据收集示例
        print("\n步骤1: 数据收集")
        print("-" * 30)
        
        data_collector = DataCollector(config)
        
        # 为了演示，我们创建一个简单的示例数据
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        sample_data = []
        
        for symbol in ['AAPL', 'MSFT']:
            # 生成模拟价格数据
            np.random.seed(42)
            price = 100
            for i, date in enumerate(dates):
                price += np.random.normal(0, 2)
                sample_data.append({
                    'Date': date,
                    'symbol': symbol,
                    'open': price,
                    'high': price + np.random.uniform(0, 5),
                    'low': price - np.random.uniform(0, 5),
                    'close': price,
                    'volume': np.random.randint(1000000, 10000000),
                    'adj_close': price
                })
        
        raw_data = pd.DataFrame(sample_data)
        print(f"生成了 {len(raw_data)} 条示例数据")
        
        # 2. 特征工程示例
        print("\n步骤2: 特征工程")
        print("-" * 30)
        
        feature_engineer = FeatureEngineer(config)
        features_data = feature_engineer.create_features(raw_data)
        print(f"特征工程完成，特征数量: {features_data.shape[1]}")
        
        # 3. 模型训练示例
        print("\n步骤3: 模型训练")
        print("-" * 30)
        
        model_trainer = ModelTrainer(config)
        models = model_trainer.train_models(features_data)
        print(f"模型训练完成，训练了 {len(models)} 个模型")
        
        # 4. 模型评估
        print("\n步骤4: 模型评估")
        print("-" * 30)
        
        evaluation_results = model_trainer.evaluate_models(models, features_data)
        
        for model_name, results in evaluation_results.items():
            print(f"  {model_name}: 准确率 = {results['accuracy']:.4f}")
        
        # 5. 可视化
        print("\n步骤5: 生成可视化")
        print("-" * 30)
        
        plotter = Plotter(config)
        plotter.create_plots(features_data, evaluation_results)
        print("可视化图表已生成")
        
        # 6. 回测
        print("\n步骤6: 回测分析")
        print("-" * 30)
        
        backtester = Backtester(config)
        backtest_results = backtester.run_backtest(features_data, models)
        
        for model_name, results in backtest_results.items():
            metrics = results['performance_metrics']
            print(f"  {model_name}: 总收益 = {metrics['total_return']:.2%}")
        
        print("\n示例运行完成！")
        print("=" * 50)
        print("结果文件保存在:")
        print("  - results/plots/ (可视化图表)")
        print("  - results/models/ (训练好的模型)")
        print("  - results/backtesting/ (回测结果)")
        
    except Exception as e:
        logger.error(f"示例运行出错: {str(e)}")
        print(f"错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
