"""
回测模块
实现策略回测和性能分析
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime, timedelta

class Backtester:
    """回测器类"""
    
    def __init__(self, config: dict):
        """
        初始化回测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.backtesting_config = config['backtesting']
        self.logger = logging.getLogger(__name__)
        
        # 回测参数
        self.initial_capital = self.backtesting_config['initial_capital']
        self.commission = self.backtesting_config['commission']
        self.slippage = self.backtesting_config['slippage']
        self.max_position_size = self.backtesting_config['max_position_size']
        self.stop_loss = self.backtesting_config['stop_loss']
        self.take_profit = self.backtesting_config['take_profit']
        
        # 创建结果保存目录
        self.results_path = "results/backtesting"
        os.makedirs(self.results_path, exist_ok=True)
        
    def run_backtest(self, data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            data: 特征数据
            models: 训练好的模型
            
        Returns:
            回测结果字典
        """
        self.logger.info("开始回测...")
        
        # 准备回测数据
        backtest_data = self._prepare_backtest_data(data)
        
        # 运行不同模型的回测
        backtest_results = {}
        
        for model_name, model_info in models.items():
            self.logger.info(f"运行 {model_name} 模型回测...")
            
            # 生成交易信号（使用包含特征的数据）
            signals = self._generate_signals(data, model_info, model_name=model_name)
            
            # 执行回测
            portfolio_values, trades = self._execute_backtest(backtest_data, signals)
            
            # 计算性能指标
            performance_metrics = self._calculate_performance_metrics(
                portfolio_values, trades, backtest_data
            )
            
            backtest_results[model_name] = {
                'portfolio_values': portfolio_values,
                'trades': trades,
                'performance_metrics': performance_metrics
            }
            
            self.logger.info(f"{model_name} 回测完成，总收益: {performance_metrics['total_return']:.2%}")
        
        # 保存回测结果
        self._save_backtest_results(backtest_results)
        
        # 创建回测报告
        self._create_backtest_report(backtest_results)
        
        self.logger.info("回测完成")
        return backtest_results
    
    def _prepare_backtest_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备回测数据
        
        Args:
            data: 特征数据
            
        Returns:
            回测数据
        """
        # 选择回测需要的列
        backtest_cols = ['Date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'target']
        backtest_data = data[backtest_cols].copy()
        
        # 按日期排序
        backtest_data = backtest_data.sort_values(['Date', 'symbol']).reset_index(drop=True)
        
        # 添加价格变化
        backtest_data['price_change'] = backtest_data['close'].pct_change()
        
        return backtest_data
    
    def _generate_signals(self, data: pd.DataFrame, model_info: Any, threshold: float = 0.5, model_name: str = None) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 回测数据（包含特征）
            model_info: 模型信息字典，包含 'model' 和 'feature_names'
            threshold: 预测阈值（默认0.5）
            model_name: 模型名称（用于日志）
            
        Returns:
            包含交易信号和预测概率的数据
        """
        signals_data = data.copy()
        
        # 提取模型和特征名称
        if isinstance(model_info, dict):
            model = model_info.get('model')
            feature_names = model_info.get('feature_names')
        else:
            # 兼容旧版本：直接传入模型对象
            model = model_info
            feature_names = None
        
        # 准备特征数据
        if feature_names:
            # 使用模型保存的特征名称
            feature_cols = [col for col in feature_names if col in data.columns]
            missing_features = [f for f in feature_names if f not in data.columns]
            if missing_features:
                self.logger.warning(f"缺失 {len(missing_features)} 个特征，将用0填充")
                for feat in missing_features:
                    data[feat] = 0.0
                    feature_cols.append(feat)
        else:
            # 自动识别特征列
            feature_cols = [col for col in data.columns if col not in ['Date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'target', 'price_change']]
        
        if not feature_cols:
            self.logger.warning("没有找到特征列，使用简化信号生成")
            signals_data['pred_proba'] = 0.5
            signals_data['signal'] = 0
            signals_data['confidence'] = 0.0
            return signals_data
        
        # 提取特征数据
        X = data[feature_cols].copy()
        
        # 确保特征名称是字符串类型（某些模型要求）
        X.columns = X.columns.astype(str)
        if feature_names:
            # 确保特征顺序与模型训练时一致
            X = X[[str(f) for f in feature_names if str(f) in X.columns]]
        
        # 使用模型进行预测
        try:
            if model_name == 'lstm':
                # LSTM模型需要特殊处理
                if hasattr(model, 'predict'):
                    # 这里需要根据LSTM模型的实际接口调整
                    pred_proba = model.predict(X) if hasattr(model, 'predict') else np.full(len(X), 0.5)
                    if pred_proba.ndim > 1:
                        pred_proba = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba[:, 0]
                else:
                    pred_proba = np.full(len(X), 0.5)
            else:
                # XGBoost和Random Forest
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[:, 1]  # 获取正类概率
                elif hasattr(model, 'predict'):
                    pred = model.predict(X)
                    pred_proba = pred.astype(float)  # 如果只有predict，使用预测值作为概率
                else:
                    pred_proba = np.full(len(X), 0.5)
        except Exception as e:
            self.logger.error(f"模型预测失败: {str(e)}")
            pred_proba = np.full(len(X), 0.5)
        
        # 添加预测概率和信号
        signals_data['pred_proba'] = pred_proba
        signals_data['confidence'] = np.abs(pred_proba - 0.5) * 2  # 置信度：距离0.5的距离
        signals_data['signal'] = 0  # 0: 持有, 1: 买入, -1: 卖出
        
        # 基于预测概率生成信号
        signals_data.loc[pred_proba > threshold, 'signal'] = 1   # 买入信号
        signals_data.loc[pred_proba < (1 - threshold), 'signal'] = -1  # 卖出信号
        
        return signals_data
    
    def _execute_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        执行回测
        
        Args:
            data: 回测数据
            signals: 交易信号
            
        Returns:
            投资组合价值和交易记录
        """
        # 初始化投资组合
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = {}  # {symbol: {'shares': int, 'avg_price': float}}
        
        portfolio_values = []
        trades = []
        
        # 按日期遍历
        for date, group in signals.groupby('Date'):
            daily_portfolio_value = cash
            
            # 计算当前持仓价值
            for symbol, position in positions.items():
                if symbol in group['symbol'].values:
                    current_price = group[group['symbol'] == symbol]['close'].iloc[0]
                    daily_portfolio_value += position['shares'] * current_price
            
            # 处理交易信号
            for _, row in group.iterrows():
                symbol = row['symbol']
                signal = row['signal']
                price = row['close']
                signal_strength = row['signal_strength']
                
                # 计算目标仓位
                target_position_value = portfolio_value * self.max_position_size * signal_strength
                target_shares = int(target_position_value / price)
                
                # 获取当前持仓
                current_shares = positions.get(symbol, {'shares': 0})['shares']
                shares_to_trade = target_shares - current_shares
                
                # 执行交易
                if shares_to_trade != 0:
                    trade_value = abs(shares_to_trade) * price
                    commission_cost = trade_value * self.commission
                    slippage_cost = trade_value * self.slippage
                    total_cost = trade_value + commission_cost + slippage_cost
                    
                    if shares_to_trade > 0:  # 买入
                        if total_cost <= cash:
                            cash -= total_cost
                            if symbol in positions:
                                # 更新平均价格
                                old_value = positions[symbol]['shares'] * positions[symbol]['avg_price']
                                new_value = shares_to_trade * price
                                total_shares = positions[symbol]['shares'] + shares_to_trade
                                positions[symbol]['avg_price'] = (old_value + new_value) / total_shares
                                positions[symbol]['shares'] = total_shares
                            else:
                                positions[symbol] = {'shares': shares_to_trade, 'avg_price': price}
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'BUY',
                                'shares': shares_to_trade,
                                'price': price,
                                'value': trade_value,
                                'commission': commission_cost,
                                'slippage': slippage_cost
                            })
                    
                    elif shares_to_trade < 0:  # 卖出
                        shares_to_sell = min(abs(shares_to_trade), positions[symbol]['shares'])
                        if shares_to_sell > 0:
                            trade_value = shares_to_sell * price
                            commission_cost = trade_value * self.commission
                            slippage_cost = trade_value * self.slippage
                            total_cost = commission_cost + slippage_cost
                            
                            cash += trade_value - total_cost
                            positions[symbol]['shares'] -= shares_to_sell
                            
                            if positions[symbol]['shares'] == 0:
                                del positions[symbol]
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'SELL',
                                'shares': shares_to_sell,
                                'price': price,
                                'value': trade_value,
                                'commission': commission_cost,
                                'slippage': slippage_cost
                            })
            
            # 记录投资组合价值
            portfolio_values.append({
                'date': date,
                'portfolio_value': daily_portfolio_value,
                'cash': cash,
                'positions_value': daily_portfolio_value - cash,
                'num_positions': len(positions)
            })
        
        portfolio_df = pd.DataFrame(portfolio_values)
        trades_df = pd.DataFrame(trades)
        
        return portfolio_df, trades_df
    
    def _calculate_performance_metrics(self, portfolio_values: pd.DataFrame, 
                                     trades: pd.DataFrame, data: pd.DataFrame) -> Dict[str, float]:
        """
        计算性能指标
        
        Args:
            portfolio_values: 投资组合价值
            trades: 交易记录
            data: 原始数据
            
        Returns:
            性能指标字典
        """
        if portfolio_values.empty:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        # 基本指标
        initial_value = portfolio_values['portfolio_value'].iloc[0]
        final_value = portfolio_values['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # 计算日收益率
        portfolio_values['daily_return'] = portfolio_values['portfolio_value'].pct_change()
        
        # 夏普比率
        daily_returns = portfolio_values['daily_return'].dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        portfolio_values['cumulative_max'] = portfolio_values['portfolio_value'].cummax()
        portfolio_values['drawdown'] = (portfolio_values['portfolio_value'] - portfolio_values['cumulative_max']) / portfolio_values['cumulative_max']
        max_drawdown = portfolio_values['drawdown'].min()
        
        # 胜率
        if not trades.empty:
            winning_trades = trades[trades['action'] == 'SELL']
            if not winning_trades.empty:
                # 这里需要更复杂的逻辑来计算实际盈亏
                win_rate = 0.5  # 简化处理
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        # 年化收益率
        days = (portfolio_values['date'].iloc[-1] - portfolio_values['date'].iloc[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'final_value': final_value
        }
    
    def _save_backtest_results(self, backtest_results: Dict[str, Any]):
        """
        保存回测结果
        
        Args:
            backtest_results: 回测结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, results in backtest_results.items():
            # 保存投资组合价值
            portfolio_path = os.path.join(self.results_path, f"{model_name}_portfolio_{timestamp}.csv")
            results['portfolio_values'].to_csv(portfolio_path, index=False)
            
            # 保存交易记录
            if not results['trades'].empty:
                trades_path = os.path.join(self.results_path, f"{model_name}_trades_{timestamp}.csv")
                results['trades'].to_csv(trades_path, index=False)
            
            # 保存性能指标
            metrics_path = os.path.join(self.results_path, f"{model_name}_metrics_{timestamp}.csv")
            pd.DataFrame([results['performance_metrics']]).to_csv(metrics_path, index=False)
            
            self.logger.info(f"{model_name} 回测结果已保存")
    
    def _create_backtest_report(self, backtest_results: Dict[str, Any]):
        """
        创建回测报告
        
        Args:
            backtest_results: 回测结果
        """
        report_data = []
        
        for model_name, results in backtest_results.items():
            metrics = results['performance_metrics']
            report_data.append({
                'Model': model_name,
                'Total Return': f"{metrics['total_return']:.2%}",
                'Annualized Return': f"{metrics['annualized_return']:.2%}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.4f}",
                'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                'Win Rate': f"{metrics['win_rate']:.2%}",
                'Number of Trades': metrics['num_trades'],
                'Final Value': f"${metrics['final_value']:,.2f}"
            })
        
        report_df = pd.DataFrame(report_data)
        report_path = os.path.join(self.results_path, 'backtest_report.csv')
        report_df.to_csv(report_path, index=False)
        
        self.logger.info(f"回测报告已保存到: {report_path}")
    
    def plot_backtest_results(self, backtest_results: Dict[str, Any]):
        """
        绘制回测结果图表
        
        Args:
            backtest_results: 回测结果
        """
        import matplotlib.pyplot as plt
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('回测结果分析', fontsize=16)
        
        # 1. 投资组合价值走势
        ax1 = axes[0, 0]
        for model_name, results in backtest_results.items():
            portfolio_values = results['portfolio_values']
            ax1.plot(portfolio_values['date'], portfolio_values['portfolio_value'], 
                    label=model_name, linewidth=2)
        ax1.set_title('投资组合价值走势')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('投资组合价值 ($)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 回撤分析
        ax2 = axes[0, 1]
        for model_name, results in backtest_results.items():
            portfolio_values = results['portfolio_values']
            if 'drawdown' in portfolio_values.columns:
                ax2.fill_between(portfolio_values['date'], portfolio_values['drawdown'], 0, 
                               alpha=0.3, label=model_name)
        ax2.set_title('回撤分析')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('回撤 (%)')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 性能指标对比
        ax3 = axes[1, 0]
        models = list(backtest_results.keys())
        returns = [results['performance_metrics']['total_return'] for results in backtest_results.values()]
        bars = ax3.bar(models, returns, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax3.set_title('总收益率对比')
        ax3.set_xlabel('模型')
        ax3.set_ylabel('总收益率')
        
        # 添加数值标签
        for bar, ret in zip(bars, returns):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{ret:.2%}', ha='center', va='bottom')
        
        # 4. 夏普比率对比
        ax4 = axes[1, 1]
        sharpe_ratios = [results['performance_metrics']['sharpe_ratio'] for results in backtest_results.values()]
        bars = ax4.bar(models, sharpe_ratios, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax4.set_title('夏普比率对比')
        ax4.set_xlabel('模型')
        ax4.set_ylabel('夏普比率')
        
        # 添加数值标签
        for bar, ratio in zip(bars, sharpe_ratios):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{ratio:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'backtest_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

