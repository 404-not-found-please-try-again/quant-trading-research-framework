"""
风险管理模块
实现止损、仓位管理、波动率控制等风险管理功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

class RiskManager:
    """风险管理器类"""
    
    def __init__(self, config: dict):
        """
        初始化风险管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.risk_config = config.get('risk_management', {})
        self.logger = logging.getLogger(__name__)
        
        # 风险参数
        self.max_position_size = self.risk_config.get('max_position_size', 0.1)
        self.stop_loss = self.risk_config.get('stop_loss', 0.05)
        self.take_profit = self.risk_config.get('take_profit', 0.1)
        self.max_drawdown = self.risk_config.get('max_drawdown', 0.15)
        self.volatility_threshold = self.risk_config.get('volatility_threshold', 0.3)
        
    def calculate_position_size(self, signal_strength: float, volatility: float, 
                              account_value: float, current_price: float) -> float:
        """
        计算仓位大小
        
        Args:
            signal_strength: 信号强度 (0-1)
            volatility: 当前波动率
            account_value: 账户价值
            current_price: 当前价格
            
        Returns:
            建议仓位大小（股数）
        """
        # 基础仓位大小
        base_position = account_value * self.max_position_size * signal_strength
        
        # 根据波动率调整
        volatility_adjustment = max(0.1, 1 - volatility / self.volatility_threshold)
        adjusted_position = base_position * volatility_adjustment
        
        # 转换为股数
        shares = int(adjusted_position / current_price)
        
        self.logger.debug(f"仓位计算: 信号强度={signal_strength:.3f}, 波动率={volatility:.3f}, 建议股数={shares}")
        
        return shares
    
    def check_stop_loss(self, entry_price: float, current_price: float, 
                       position_type: str) -> bool:
        """
        检查是否触发止损
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            position_type: 仓位类型 ('long' 或 'short')
            
        Returns:
            是否触发止损
        """
        if position_type == 'long':
            loss_pct = (entry_price - current_price) / entry_price
            return loss_pct >= self.stop_loss
        else:  # short
            loss_pct = (current_price - entry_price) / entry_price
            return loss_pct >= self.stop_loss
    
    def check_take_profit(self, entry_price: float, current_price: float, 
                         position_type: str) -> bool:
        """
        检查是否触发止盈
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            position_type: 仓位类型 ('long' 或 'short')
            
        Returns:
            是否触发止盈
        """
        if position_type == 'long':
            profit_pct = (current_price - entry_price) / entry_price
            return profit_pct >= self.take_profit
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price
            return profit_pct >= self.take_profit
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        计算风险价值(VaR)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            
        Returns:
            VaR值
        """
        if len(returns) == 0:
            return 0
        
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        计算期望损失(ES)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            
        Returns:
            ES值
        """
        if len(returns) == 0:
            return 0
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def check_max_drawdown(self, portfolio_values: pd.Series) -> Tuple[bool, float]:
        """
        检查最大回撤
        
        Args:
            portfolio_values: 投资组合价值序列
            
        Returns:
            (是否超过最大回撤限制, 当前回撤)
        """
        if len(portfolio_values) == 0:
            return False, 0
        
        cumulative_max = portfolio_values.cummax()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        current_drawdown = drawdown.min()
        
        return current_drawdown <= -self.max_drawdown, current_drawdown
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            
        Returns:
            夏普比率
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        计算索提诺比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            
        Returns:
            索提诺比率
        """
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    
    def calculate_calmar_ratio(self, returns: pd.Series, portfolio_values: pd.Series) -> float:
        """
        计算卡尔玛比率
        
        Args:
            returns: 收益率序列
            portfolio_values: 投资组合价值序列
            
        Returns:
            卡尔玛比率
        """
        if len(returns) == 0 or len(portfolio_values) == 0:
            return 0
        
        annual_return = returns.mean() * 252
        _, max_dd = self.check_max_drawdown(portfolio_values)
        max_dd_abs = abs(max_dd)
        
        if max_dd_abs == 0:
            return 0
        
        return annual_return / max_dd_abs
    
    def generate_risk_report(self, trades: pd.DataFrame, portfolio_values: pd.Series) -> Dict[str, Any]:
        """
        生成风险报告
        
        Args:
            trades: 交易记录
            portfolio_values: 投资组合价值序列
            
        Returns:
            风险报告字典
        """
        if trades.empty or len(portfolio_values) == 0:
            return {}
        
        # 计算收益率
        returns = portfolio_values.pct_change().dropna()
        
        # 基本风险指标
        var_95 = self.calculate_var(returns, 0.05)
        var_99 = self.calculate_var(returns, 0.01)
        es_95 = self.calculate_expected_shortfall(returns, 0.05)
        
        # 回撤分析
        max_dd_exceeded, current_dd = self.check_max_drawdown(portfolio_values)
        
        # 风险调整收益指标
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        calmar = self.calculate_calmar_ratio(returns, portfolio_values)
        
        # 交易分析
        if 'profit' in trades.columns:
            win_rate = (trades['profit'] > 0).mean()
            avg_win = trades[trades['profit'] > 0]['profit'].mean() if (trades['profit'] > 0).any() else 0
            avg_loss = trades[trades['profit'] < 0]['profit'].mean() if (trades['profit'] < 0).any() else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        risk_report = {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'max_drawdown': current_dd,
            'max_drawdown_exceeded': max_dd_exceeded,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'volatility': returns.std() * np.sqrt(252)
        }
        
        return risk_report
    
    def should_trade(self, signal_strength: float, volatility: float, 
                    current_drawdown: float) -> bool:
        """
        判断是否应该交易
        
        Args:
            signal_strength: 信号强度
            volatility: 当前波动率
            current_drawdown: 当前回撤
            
        Returns:
            是否应该交易
        """
        # 信号强度检查
        if signal_strength < 0.3:
            return False
        
        # 波动率检查
        if volatility > self.volatility_threshold:
            return False
        
        # 回撤检查
        if current_drawdown < -self.max_drawdown:
            return False
        
        return True


