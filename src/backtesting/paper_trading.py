"""
模拟盘交易模块
用于实盘前的模拟交易测试
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import os
import json
import time

class PaperTrader:
    """模拟盘交易器"""
    
    def __init__(self, config: dict):
        """
        初始化模拟盘交易器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.backtesting_config = config['backtesting']
        self.logger = logging.getLogger(__name__)
        
        # 交易参数
        self.initial_capital = self.backtesting_config['initial_capital']
        self.commission = self.backtesting_config['commission']
        self.slippage = self.backtesting_config['slippage']
        self.max_position_size = self.backtesting_config['max_position_size']
        
        # 模拟盘状态
        self.cash = self.initial_capital
        self.positions = {}  # {symbol: {'shares': int, 'avg_price': float, 'entry_date': datetime}}
        self.trade_history = []
        self.portfolio_value_history = []
        
        # 创建结果保存目录
        self.results_path = "results/paper_trading"
        os.makedirs(self.results_path, exist_ok=True)
        
        # 状态文件
        self.state_file = os.path.join(self.results_path, "paper_trading_state.json")
        
    def load_state(self):
        """加载模拟盘状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.cash = state.get('cash', self.initial_capital)
                    self.positions = state.get('positions', {})
                    self.logger.info(f"加载模拟盘状态: 现金={self.cash:.2f}, 持仓数={len(self.positions)}")
            except Exception as e:
                self.logger.warning(f"加载状态失败: {str(e)}，使用初始状态")
    
    def save_state(self):
        """保存模拟盘状态"""
        try:
            state = {
                'cash': self.cash,
                'positions': self.positions,
                'last_update': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"保存状态失败: {str(e)}")
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        计算当前投资组合价值
        
        Args:
            current_prices: {symbol: current_price}
            
        Returns:
            投资组合总价值
        """
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, 0)
            if current_price > 0:
                position_value = position['shares'] * current_price
                total_value += position_value
        
        return total_value
    
    def execute_trade(self, symbol: str, action: str, price: float, 
                     confidence: float, signal_strength: float = 1.0) -> Dict[str, Any]:
        """
        执行交易
        
        Args:
            symbol: 股票代码
            action: 'buy' 或 'sell'
            price: 交易价格
            confidence: 置信度
            signal_strength: 信号强度（用于仓位管理）
            
        Returns:
            交易结果字典
        """
        trade_result = {
            'success': False,
            'message': '',
            'shares': 0,
            'value': 0,
            'commission': 0,
            'slippage': 0
        }
        
        try:
            # 应用滑点
            if action == 'buy':
                execution_price = price * (1 + self.slippage)
            else:
                execution_price = price * (1 - self.slippage)
            
            if action == 'buy':
                # 买入逻辑
                if symbol in self.positions:
                    # 已有持仓，加仓
                    current_shares = self.positions[symbol]['shares']
                    max_shares = int((self.cash * self.max_position_size * signal_strength) / execution_price)
                    
                    if max_shares <= 0:
                        trade_result['message'] = "资金不足"
                        return trade_result
                    
                    # 计算新平均价格
                    total_cost = current_shares * self.positions[symbol]['avg_price'] + max_shares * execution_price
                    total_shares = current_shares + max_shares
                    new_avg_price = total_cost / total_shares
                    
                    trade_value = max_shares * execution_price
                    commission_cost = trade_value * self.commission
                    
                    if trade_value + commission_cost > self.cash:
                        trade_result['message'] = "资金不足"
                        return trade_result
                    
                    self.cash -= (trade_value + commission_cost)
                    self.positions[symbol] = {
                        'shares': total_shares,
                        'avg_price': new_avg_price,
                        'entry_date': self.positions[symbol]['entry_date']
                    }
                    
                    trade_result.update({
                        'success': True,
                        'shares': max_shares,
                        'value': trade_value,
                        'commission': commission_cost,
                        'message': f"加仓 {max_shares} 股，平均价格 {new_avg_price:.2f}"
                    })
                else:
                    # 新建仓
                    max_shares = int((self.cash * self.max_position_size * signal_strength) / execution_price)
                    
                    if max_shares <= 0:
                        trade_result['message'] = "资金不足"
                        return trade_result
                    
                    trade_value = max_shares * execution_price
                    commission_cost = trade_value * self.commission
                    
                    if trade_value + commission_cost > self.cash:
                        trade_result['message'] = "资金不足"
                        return trade_result
                    
                    self.cash -= (trade_value + commission_cost)
                    self.positions[symbol] = {
                        'shares': max_shares,
                        'avg_price': execution_price,
                        'entry_date': datetime.now().isoformat()
                    }
                    
                    trade_result.update({
                        'success': True,
                        'shares': max_shares,
                        'value': trade_value,
                        'commission': commission_cost,
                        'message': f"买入 {max_shares} 股"
                    })
            
            elif action == 'sell':
                # 卖出逻辑
                if symbol not in self.positions or self.positions[symbol]['shares'] <= 0:
                    trade_result['message'] = "无持仓"
                    return trade_result
                
                current_shares = self.positions[symbol]['shares']
                shares_to_sell = current_shares  # 全部卖出
                
                trade_value = shares_to_sell * execution_price
                commission_cost = trade_value * self.commission
                
                self.cash += (trade_value - commission_cost)
                
                # 计算盈亏
                avg_price = self.positions[symbol]['avg_price']
                pnl = (execution_price - avg_price) * shares_to_sell - commission_cost
                pnl_pct = (execution_price - avg_price) / avg_price * 100
                
                del self.positions[symbol]
                
                trade_result.update({
                    'success': True,
                    'shares': shares_to_sell,
                    'value': trade_value,
                    'commission': commission_cost,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'message': f"卖出 {shares_to_sell} 股，盈亏 {pnl:.2f} ({pnl_pct:.2f}%)"
                })
            
            # 记录交易历史
            self.trade_history.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': action,
                'price': execution_price,
                'shares': trade_result['shares'],
                'value': trade_result['value'],
                'commission': trade_result['commission'],
                'confidence': confidence,
                'signal_strength': signal_strength
            })
            
            # 保存状态
            self.save_state()
            
        except Exception as e:
            self.logger.error(f"执行交易失败: {str(e)}")
            trade_result['message'] = f"交易失败: {str(e)}"
        
        return trade_result
    
    def process_signals(self, signals: Dict[str, Dict[str, Any]], 
                       current_prices: Dict[str, float]):
        """
        处理交易信号
        
        Args:
            signals: {symbol: {'action': 'buy'/'sell', 'confidence': float, 'signal_strength': float}}
            current_prices: {symbol: current_price}
        """
        for symbol, signal_info in signals.items():
            action = signal_info.get('action')
            confidence = signal_info.get('confidence', 0.5)
            signal_strength = signal_info.get('signal_strength', 1.0)
            current_price = current_prices.get(symbol)
            
            if not current_price or current_price <= 0:
                self.logger.warning(f"{symbol} 价格无效，跳过")
                continue
            
            # 执行交易
            trade_result = self.execute_trade(
                symbol, action, current_price, confidence, signal_strength
            )
            
            if trade_result['success']:
                self.logger.info(f"{symbol} {action} {trade_result['message']}")
            else:
                self.logger.warning(f"{symbol} {action} 失败: {trade_result['message']}")
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        获取投资组合摘要
        
        Args:
            current_prices: {symbol: current_price}
            
        Returns:
            投资组合摘要字典
        """
        total_value = self.calculate_portfolio_value(current_prices)
        total_return = (total_value - self.initial_capital) / self.initial_capital * 100
        
        position_details = []
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, 0)
            if current_price > 0:
                position_value = position['shares'] * current_price
                unrealized_pnl = (current_price - position['avg_price']) * position['shares']
                unrealized_pnl_pct = (current_price - position['avg_price']) / position['avg_price'] * 100
                
                position_details.append({
                    'symbol': symbol,
                    'shares': position['shares'],
                    'avg_price': position['avg_price'],
                    'current_price': current_price,
                    'value': position_value,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct
                })
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'cash': self.cash,
            'total_value': total_value,
            'initial_capital': self.initial_capital,
            'total_return': total_return,
            'positions': position_details,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }
        
        return summary
    
    def save_daily_report(self, current_prices: Dict[str, float]):
        """保存每日报告"""
        summary = self.get_portfolio_summary(current_prices)
        self.portfolio_value_history.append(summary)
        
        # 保存到CSV
        report_file = os.path.join(self.results_path, f"daily_report_{datetime.now().strftime('%Y%m%d')}.json")
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 保存交易历史
        if self.trade_history:
            trades_file = os.path.join(self.results_path, "trade_history.csv")
            trades_df = pd.DataFrame(self.trade_history)
            trades_df.to_csv(trades_file, index=False)
        
        self.logger.info(f"投资组合价值: {summary['total_value']:.2f}, 收益率: {summary['total_return']:.2f}%")




