#!/usr/bin/env python3
"""
美股预测系统主程序
US Stock Prediction System Main Program
"""

import os
import sys
import yaml
import logging
import numpy as np
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_collector import DataCollector
from features.feature_engineer import FeatureEngineer
from models.model_trainer import ModelTrainer
from visualization.plotter import Plotter
from backtesting.backtester import Backtester

def setup_logging(config):
    """设置日志配置"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建日志目录
    log_file = log_config.get('file', 'logs/prediction.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def main():
    """主函数"""
    print("启动美股预测系统...")
    
    # 加载配置
    config = load_config()
    
    # 设置日志
    logger = setup_logging(config)
    logger.info("系统启动")
    
    try:
        # 1. 数据收集
        logger.info("开始数据收集...")
        data_collector = DataCollector(config)
        raw_data = data_collector.collect_data()
        logger.info(f"数据收集完成，共获取 {len(raw_data)} 条记录")
        
        # 2. 特征工程
        logger.info("开始特征工程...")
        feature_engineer = FeatureEngineer(config)
        features_data = feature_engineer.create_features(raw_data)
        logger.info(f"特征工程完成，特征数量: {features_data.shape[1]}")
        
        # 3. 模型训练
        logger.info("开始模型训练...")
        model_trainer = ModelTrainer(config)
        models = model_trainer.train_models(features_data)
        logger.info("模型训练完成")
        
        # 4. 模型评估
        logger.info("开始模型评估...")
        evaluation_results = model_trainer.evaluate_models(models, features_data)
        logger.info("模型评估完成")
        
        # 5. 可视化
        logger.info("生成可视化图表...")
        plotter = Plotter(config)
        plotter.create_plots(features_data, evaluation_results)
        logger.info("可视化完成")
        
        # 6. 回测
        logger.info("开始回测...")
        backtester = Backtester(config)
        backtest_results = backtester.run_backtest(features_data, models)
        logger.info("回测完成")

        # 6.1 阈值扫描（扩展范围：0.10~0.90 step 0.05），含交易成本
        try:
            thresholds = [round(x/100, 2) for x in range(10, 91, 5)]  # 0.10到0.90，步长0.05
            logger.info(f"开始阈值扫描: {thresholds}")
            backtester.run_threshold_sweep(features_data, models, thresholds)
        except Exception as e:
            logger.warning(f"阈值扫描失败: {str(e)}")
        
        # 6.2 加权集成回测（优化3）
        try:
            logger.info("开始加权集成模型回测...")
            ensemble_results = backtester.run_ensemble_backtest(features_data, models)
            if ensemble_results:
                backtest_results.update(ensemble_results)
                logger.info("加权集成回测完成")
        except Exception as e:
            logger.warning(f"加权集成回测失败: {str(e)}")
        
        # 7. 输出结果
        logger.info("生成最终报告...")
        print("\n" + "="*50)
        print("美股预测系统运行完成！")
        print("="*50)
        print(f"数据记录数: {len(raw_data)}")
        print(f"特征数量: {features_data.shape[1]}")
        print(f"训练模型数: {len(models)}")
        print(f"最佳模型准确率: {max([result['accuracy'] for result in evaluation_results.values()]):.4f}")
        # 计算平均回测收益
        avg_return = np.mean([results['performance_metrics']['total_return'] for results in backtest_results.values()])
        avg_sharpe = np.mean([results['performance_metrics']['sharpe_ratio'] for results in backtest_results.values()])
        print(f"回测总收益: {avg_return:.2%}")
        print(f"夏普比率: {avg_sharpe:.4f}")
        print("="*50)
        
        logger.info("系统运行完成")
        
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
