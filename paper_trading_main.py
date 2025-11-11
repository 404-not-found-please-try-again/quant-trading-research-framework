"""
模拟盘交易主程序
用于实盘前的模拟交易测试
"""

import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time
import json

# 添加src目录到Python路径（与main.py保持一致）
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_collector import DataCollector
from features.feature_engineer import FeatureEngineer
from models.model_trainer import ModelTrainer
from backtesting.backtester import Backtester
from backtesting.paper_trading import PaperTrader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PaperTradingSystem:
    """模拟盘交易系统"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        初始化模拟盘交易系统
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.data_collector = DataCollector(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.backtester = Backtester(self.config)
        self.paper_trader = PaperTrader(self.config)
        
        # 加载已保存的状态
        self.paper_trader.load_state()
        
        # 加载模型
        self.models = self._load_models()
        
        # 创建结果目录
        self.results_path = "results/paper_trading"
        os.makedirs(self.results_path, exist_ok=True)
        
    def _load_models(self) -> dict:
        """加载训练好的模型"""
        models = {}
        models_path = "results/models"
        
        # 查找最新的模型文件
        model_files = {}
        for model_name in ['xgboost', 'random_forest', 'lstm']:
            if model_name == 'lstm':
                # LSTM模型目录
                model_dirs = [d for d in os.listdir(models_path) 
                            if d.startswith(f'{model_name}_') and os.path.isdir(os.path.join(models_path, d))]
            else:
                # 其他模型文件
                model_files_list = [f for f in os.listdir(models_path) 
                                   if f.startswith(f'{model_name}_') and f.endswith('.joblib')]
                model_dirs = model_files_list
            
            if model_dirs:
                # 按时间排序，取最新的
                model_dirs.sort(reverse=True)
                model_files[model_name] = model_dirs[0]
        
        # 加载模型
        try:
            import joblib
            
            # 加载XGBoost
            if 'xgboost' in model_files:
                xgb_path = os.path.join(models_path, model_files['xgboost'])
                xgb_model = joblib.load(xgb_path)
                
                # 尝试从模型获取特征名称
                feature_names = None
                if hasattr(xgb_model, 'feature_names_in_'):
                    # 确保转换为字符串列表
                    feature_names = [str(f) for f in xgb_model.feature_names_in_]
                elif hasattr(xgb_model, 'get_booster'):
                    try:
                        feature_names = xgb_model.get_booster().feature_names
                    except:
                        pass
                
                models['xgboost'] = {
                    'model': xgb_model,
                    'feature_names': feature_names
                }
                self.logger.info(f"加载XGBoost模型: {xgb_path}")
                if feature_names:
                    self.logger.info(f"XGBoost模型特征数: {len(feature_names)}")
            
            # 加载Random Forest
            if 'random_forest' in model_files:
                rf_path = os.path.join(models_path, model_files['random_forest'])
                rf_model = joblib.load(rf_path)
                
                # 尝试从模型获取特征名称
                feature_names = None
                if hasattr(rf_model, 'feature_names_in_'):
                    # 确保转换为字符串列表
                    feature_names = [str(f) for f in rf_model.feature_names_in_]
                
                models['random_forest'] = {
                    'model': rf_model,
                    'feature_names': feature_names
                }
                self.logger.info(f"加载Random Forest模型: {rf_path}")
                if feature_names:
                    self.logger.info(f"Random Forest模型特征数: {len(feature_names)}")
            
            # 加载LSTM
            if 'lstm' in model_files:
                from models.lstm_model import LSTMModel
                lstm_path = os.path.join(models_path, model_files['lstm'])
                lstm_model = LSTMModel(self.config)
                lstm_model.load_model(lstm_path)
                models['lstm'] = {
                    'model': lstm_model,
                    'feature_names': None
                }
                self.logger.info(f"加载LSTM模型: {lstm_path}")
            
            # 优先使用模型内部的特征名称（这是训练时实际使用的）
            # 如果模型没有，再尝试从文件加载
            for model_name in models:
                if not models[model_name].get('feature_names'):
                    # 尝试从特征选择文件加载（作为备用）
                    feature_files = [f for f in os.listdir(models_path) 
                                   if f.startswith('selected_features_') and f.endswith('.txt')]
                    if feature_files:
                        feature_files.sort(reverse=True)
                        feature_names_file = os.path.join(models_path, feature_files[0])
                        try:
                            with open(feature_names_file, 'r', encoding='utf-8') as f:
                                feature_names = [line.strip() for line in f.readlines() if line.strip()]
                            models[model_name]['feature_names'] = feature_names
                            self.logger.info(f"{model_name} 从文件加载特征名称: {len(feature_names)} 个特征")
                        except Exception as e:
                            self.logger.warning(f"{model_name} 加载特征名称失败: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            raise
        
        return models
    
    def get_latest_data(self, symbols: list, days: int = 90) -> pd.DataFrame:
        """
        获取最新的股票数据
        
        Args:
            symbols: 股票代码列表
            days: 获取最近N天的数据
            
        Returns:
            股票数据DataFrame
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        self.logger.info(f"获取最新数据: {symbols}, {start_date} 到 {end_date}")
        
        all_data = []
        for symbol in symbols:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if data.empty:
                    self.logger.warning(f"{symbol} 数据为空")
                    continue
                
                # 重置索引，将Date作为列
                data = data.reset_index()
                data.columns = [col.lower() for col in data.columns]
                
                if 'date' in data.columns:
                    data['Date'] = pd.to_datetime(data['date'])
                    data = data.drop('date', axis=1)
                
                data['symbol'] = symbol
                all_data.append(data)
                
                self.logger.info(f"{symbol} 数据获取成功: {len(data)} 条记录")
                
            except Exception as e:
                self.logger.error(f"获取 {symbol} 数据失败: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("没有成功获取到任何数据")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values(['Date', 'symbol']).reset_index(drop=True)
        
        return combined_data
    
    def _remove_highly_correlated_features(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        删除高相关特征（与训练时相同的逻辑）
        """
        if len(X.columns) <= 1:
            return X
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = []
        for column in upper_triangle.columns:
            high_corr_features = upper_triangle.index[upper_triangle[column] > threshold].tolist()
            if high_corr_features:
                for feat in high_corr_features:
                    if feat not in to_drop:
                        to_drop.append(feat)
        if to_drop:
            self.logger.info(f"删除 {len(to_drop)} 个高相关特征（阈值>{threshold}）")
            X = X.drop(columns=to_drop)
        return X
    
    def generate_signals(self, data: pd.DataFrame) -> dict:
        """
        生成交易信号
        
        Args:
            data: 特征数据
            
        Returns:
            信号字典 {symbol: {'action': 'buy'/'sell', 'confidence': float, 'signal_strength': float}}
        """
        signals = {}
        
        # 准备回测数据（用于获取价格等信息）
        backtest_data = self.backtester._prepare_backtest_data(data)
        
        # 获取集成模型的预测
        ensemble_config = self.config['backtesting'].get('ensemble', {})
        weights = ensemble_config.get('weights', {})
        
        all_predictions = {}
        for model_name, model_info in self.models.items():
            if model_name not in weights or weights.get(model_name, 0) == 0:
                continue
            
            try:
                # 使用包含特征的数据进行预测（而不是backtest_data）
                # data包含所有特征，backtest_data只包含价格信息
                model_signals = self.backtester._generate_signals(
                    data, model_info, threshold=0.5, model_name=model_name
                )
                
                # 检查预测概率
                proba = model_signals['pred_proba'].values
                self.logger.info(f"{model_name} 预测概率统计: min={proba.min():.3f}, max={proba.max():.3f}, mean={proba.mean():.3f}, median={np.median(proba):.3f}")
                
                # 保存完整的signals_data，包含symbol信息，方便后续按symbol分组
                all_predictions[model_name] = {
                    'signals_data': model_signals,  # 保存完整数据，包含symbol和pred_proba
                    'pred_proba': proba,
                    'confidence': model_signals.get('confidence', 
                        np.abs(proba - 0.5) * 2).values if hasattr(model_signals.get('confidence', None), 'values') else np.abs(proba - 0.5) * 2
                }
            except Exception as e:
                self.logger.warning(f"生成 {model_name} 信号失败: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
        
        if not all_predictions:
            self.logger.warning("没有生成任何预测")
            return signals
        
        # 按symbol分组处理（支持多只基金）
        latest_data = backtest_data.groupby('symbol').tail(1).reset_index(drop=True)
        
        # 为每只基金分别生成信号
        for idx, row in latest_data.iterrows():
            symbol = row['symbol']
            
            # 获取该symbol的所有数据（从backtest_data中）
            symbol_mask = backtest_data['symbol'] == symbol
            symbol_data = backtest_data[symbol_mask].reset_index(drop=True)
            if len(symbol_data) == 0:
                continue
            
            # 为这只基金计算加权平均概率
            # 需要从每个模型的signals_data中提取该symbol的数据
            weighted_proba = np.zeros(len(symbol_data))
            total_weight = 0
            
            for model_name, pred_data in all_predictions.items():
                weight = weights.get(model_name, 0)
                if weight > 0:
                    # 从signals_data中提取该symbol的预测概率
                    signals_data = pred_data['signals_data']
                    symbol_signals = signals_data[symbol_mask].reset_index(drop=True)
                    if len(symbol_signals) > 0:
                        symbol_proba = symbol_signals['pred_proba'].values
                        # 确保长度匹配
                        min_len = min(len(weighted_proba), len(symbol_proba))
                        weighted_proba[:min_len] += symbol_proba[:min_len] * weight
                        total_weight += weight
            
            if total_weight > 0:
                weighted_proba = weighted_proba / total_weight
            
            # 应用动态阈值和置信度滤波
            signal_config = self.config['backtesting'].get('signal_strategy', {})
            dynamic_threshold_config = signal_config.get('dynamic_threshold', {})
            confidence_filter_config = signal_config.get('confidence_filter', {})
            
            base_threshold = dynamic_threshold_config.get('base_threshold', 0.55) if dynamic_threshold_config.get('enable', False) else 0.5
            
            # 计算置信度
            confidence = np.abs(weighted_proba - 0.5) * 2
            
            # 应用置信度滤波
            if confidence_filter_config.get('enable', False):
                top_percentile = confidence_filter_config.get('top_percentile', 0.35)
                min_confidence = confidence_filter_config.get('min_confidence', 0.45)
                confidence_threshold = np.percentile(confidence, (1 - top_percentile) * 100)
                confidence_threshold = max(confidence_threshold, min_confidence)
                
                prob_threshold_high = 0.5 + confidence_threshold / 2
                prob_threshold_low = 0.5 - confidence_threshold / 2
                
                mask_high_conf = (
                    (weighted_proba >= prob_threshold_high) | 
                    (weighted_proba <= prob_threshold_low)
                )
            else:
                mask_high_conf = np.ones(len(symbol_data), dtype=bool)
            
            # 生成信号（只检查最后一天）
            last_proba = weighted_proba[-1]
            last_conf = confidence[-1]
            last_mask = mask_high_conf[-1]
            
            if last_proba > base_threshold and last_mask:
                current_price = row['close']
                
                signals[symbol] = {
                    'action': 'buy',
                    'confidence': last_conf,
                    'signal_strength': last_proba,
                    'price': current_price,
                    'pred_proba': last_proba
                }
                
                self.logger.info(f"生成买入信号: {symbol}, 价格={current_price:.2f}, "
                               f"置信度={last_conf:.3f}, 概率={last_proba:.3f}")
        
        return signals
    
    def run_daily_trading(self):
        """运行每日交易"""
        self.logger.info("=" * 60)
        self.logger.info("开始每日模拟盘交易")
        self.logger.info("=" * 60)
        
        try:
            # 获取股票列表
            symbols = self.config['data']['symbols']
            
            # 获取最新数据（至少250天，确保所有特征都能计算，包括200日均线等）
            latest_data = self.get_latest_data(symbols, days=250)
            
            # 特征工程
            self.logger.info("开始特征工程...")
            features_data = self.feature_engineer.create_features(latest_data)
            
            # 直接使用训练时保存的特征列表（确保完全一致）
            # 不重新进行特征选择，因为数据量不同会导致选择结果不同
            self.logger.info("使用训练时保存的特征列表（确保特征完全一致）...")
            
            # 从模型获取特征名称（这是训练时实际使用的特征）
            model_feature_names = None
            for model_name, model_info in self.models.items():
                if model_info.get('feature_names'):
                    model_feature_names = model_info['feature_names']
                    self.logger.info(f"使用 {model_name} 模型的特征列表: {len(model_feature_names)} 个特征")
                    break
            
            if model_feature_names:
                # 确保所有必需特征都存在（缺失的用0填充）
                missing_features = [f for f in model_feature_names if f not in features_data.columns]
                if missing_features:
                    self.logger.warning(f"缺失 {len(missing_features)} 个特征，将用0填充")
                    for feat in missing_features:
                        features_data[feat] = 0.0
                
                # 只保留模型需要的特征
                keep_cols = ['Date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'target'] + model_feature_names
                features_data = features_data[[col for col in keep_cols if col in features_data.columns]]
                
                self.logger.info(f"特征对齐完成：使用 {len(model_feature_names)} 个特征")
            else:
                self.logger.warning("无法获取模型特征列表，将使用所有生成的特征")
            
            # 生成交易信号
            self.logger.info("生成交易信号...")
            signals = self.generate_signals(features_data)
            
            # 获取当前价格
            current_prices = {}
            for symbol in symbols:
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    current_info = ticker.history(period='1d', interval='1m')
                    if not current_info.empty:
                        current_prices[symbol] = current_info['Close'].iloc[-1]
                    else:
                        # 如果1分钟数据不可用，使用日线数据
                        daily_data = ticker.history(period='1d')
                        if not daily_data.empty:
                            current_prices[symbol] = daily_data['Close'].iloc[-1]
                except Exception as e:
                    self.logger.warning(f"获取 {symbol} 当前价格失败: {str(e)}")
                    # 使用最新历史数据的收盘价
                    symbol_data = latest_data[latest_data['symbol'] == symbol]
                    if not symbol_data.empty:
                        current_prices[symbol] = symbol_data['close'].iloc[-1]
            
            # 处理交易信号
            if signals:
                self.logger.info(f"处理 {len(signals)} 个交易信号...")
                self.paper_trader.process_signals(signals, current_prices)
            else:
                self.logger.info("今日无交易信号")
            
            # 检查现有持仓是否需要平仓
            portfolio_summary = self.paper_trader.get_portfolio_summary(current_prices)
            
            # 保存每日报告
            self.paper_trader.save_daily_report(current_prices)
            
            # 打印投资组合摘要
            self.logger.info("=" * 60)
            self.logger.info("投资组合摘要")
            self.logger.info("=" * 60)
            self.logger.info(f"现金: ${portfolio_summary['cash']:.2f}")
            self.logger.info(f"总价值: ${portfolio_summary['total_value']:.2f}")
            self.logger.info(f"初始资金: ${portfolio_summary['initial_capital']:.2f}")
            self.logger.info(f"总收益: {portfolio_summary['total_return']:.2f}%")
            self.logger.info(f"持仓数: {portfolio_summary['num_positions']}")
            self.logger.info(f"交易次数: {portfolio_summary['num_trades']}")
            
            if portfolio_summary['positions']:
                self.logger.info("\n当前持仓:")
                for pos in portfolio_summary['positions']:
                    self.logger.info(f"  {pos['symbol']}: {pos['shares']}股 @ ${pos['avg_price']:.2f}, "
                                   f"当前价=${pos['current_price']:.2f}, "
                                   f"盈亏={pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_pct']:.2f}%)")
            
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"每日交易失败: {str(e)}", exc_info=True)
            raise
    
    def run_simulation(self, days: int = 30):
        """
        运行模拟盘测试
        
        Args:
            days: 模拟天数
        """
        self.logger.info(f"开始模拟盘测试，将持续 {days} 天")
        
        for day in range(days):
            self.logger.info(f"\n第 {day + 1}/{days} 天")
            try:
                self.run_daily_trading()
            except Exception as e:
                self.logger.error(f"第 {day + 1} 天交易失败: {str(e)}")
                continue
            
            # 如果不是最后一天，等待到下一个交易日
            if day < days - 1:
                # 简单等待（实际应该等待到下一个交易日）
                self.logger.info("等待下一个交易日...")
                time.sleep(60)  # 实际应该使用交易日历
        
        self.logger.info("模拟盘测试完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模拟盘交易系统')
    parser.add_argument('--mode', type=str, default='daily', 
                       choices=['daily', 'simulation'],
                       help='运行模式: daily=每日运行, simulation=模拟运行')
    parser.add_argument('--days', type=int, default=30,
                       help='模拟天数（仅simulation模式）')
    
    args = parser.parse_args()
    
    try:
        system = PaperTradingSystem()
        
        if args.mode == 'daily':
            # 每日运行模式
            system.run_daily_trading()
        elif args.mode == 'simulation':
            # 模拟运行模式
            system.run_simulation(days=args.days)
    
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"程序错误: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

