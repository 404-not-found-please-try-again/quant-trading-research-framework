"""
工具函数模块
提供各种辅助函数
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import os
from datetime import datetime, timedelta
import yaml

def setup_logging(log_level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        
    Returns:
        配置好的logger
    """
    # 创建日志目录
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 配置处理器
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        raise

def save_config(config: Dict[str, Any], config_path: str):
    """
    保存配置文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        logging.error(f"保存配置文件失败: {str(e)}")
        raise

def create_directories(directories: List[str]):
    """
    创建目录
    
    Args:
        directories: 目录路径列表
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    计算收益率
    
    Args:
        prices: 价格序列
        method: 计算方法 ('simple' 或 'log')
        
    Returns:
        收益率序列
    """
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError("method 必须是 'simple' 或 'log'")

def calculate_volatility(returns: pd.Series, window: int = 20, annualized: bool = True) -> pd.Series:
    """
    计算波动率
    
    Args:
        returns: 收益率序列
        window: 滚动窗口大小
        annualized: 是否年化
        
    Returns:
        波动率序列
    """
    volatility = returns.rolling(window=window).std()
    
    if annualized:
        volatility = volatility * np.sqrt(252)
    
    return volatility

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        
    Returns:
        夏普比率
    """
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / returns.std() * np.sqrt(252)

def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    计算最大回撤
    
    Args:
        prices: 价格序列
        
    Returns:
        最大回撤
    """
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    return drawdown.min()

def calculate_calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
    """
    计算卡尔玛比率
    
    Args:
        returns: 收益率序列
        prices: 价格序列
        
    Returns:
        卡尔玛比率
    """
    annual_return = returns.mean() * 252
    max_dd = abs(calculate_max_drawdown(prices))
    return annual_return / max_dd if max_dd > 0 else 0

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    计算索提诺比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        
    Returns:
        索提诺比率
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    
    return excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    计算信息比率
    
    Args:
        returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        
    Returns:
        信息比率
    """
    excess_returns = returns - benchmark_returns
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

def calculate_tracking_error(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    计算跟踪误差
    
    Args:
        returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        
    Returns:
        跟踪误差
    """
    excess_returns = returns - benchmark_returns
    return excess_returns.std() * np.sqrt(252)

def calculate_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    计算贝塔系数
    
    Args:
        returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        
    Returns:
        贝塔系数
    """
    covariance = np.cov(returns.dropna(), benchmark_returns.dropna())[0, 1]
    benchmark_variance = np.var(benchmark_returns.dropna())
    return covariance / benchmark_variance if benchmark_variance > 0 else 0

def calculate_alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    计算阿尔法系数
    
    Args:
        returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        risk_free_rate: 无风险利率
        
    Returns:
        阿尔法系数
    """
    beta = calculate_beta(returns, benchmark_returns)
    alpha = returns.mean() - risk_free_rate / 252 - beta * (benchmark_returns.mean() - risk_free_rate / 252)
    return alpha * 252

def calculate_win_rate(trades: pd.DataFrame) -> float:
    """
    计算胜率
    
    Args:
        trades: 交易记录DataFrame
        
    Returns:
        胜率
    """
    if trades.empty:
        return 0
    
    # 这里需要根据实际的交易记录结构来计算
    # 假设有 'profit' 列表示每笔交易的盈亏
    if 'profit' in trades.columns:
        winning_trades = trades[trades['profit'] > 0]
        return len(winning_trades) / len(trades)
    else:
        return 0.5  # 默认值

def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """
    计算盈利因子
    
    Args:
        trades: 交易记录DataFrame
        
    Returns:
        盈利因子
    """
    if trades.empty or 'profit' not in trades.columns:
        return 1
    
    gross_profit = trades[trades['profit'] > 0]['profit'].sum()
    gross_loss = abs(trades[trades['profit'] < 0]['profit'].sum())
    
    return gross_profit / gross_loss if gross_loss > 0 else float('inf')

def calculate_average_trade(trades: pd.DataFrame) -> float:
    """
    计算平均交易盈亏
    
    Args:
        trades: 交易记录DataFrame
        
    Returns:
        平均交易盈亏
    """
    if trades.empty or 'profit' not in trades.columns:
        return 0
    
    return trades['profit'].mean()

def calculate_largest_win(trades: pd.DataFrame) -> float:
    """
    计算最大单笔盈利
    
    Args:
        trades: 交易记录DataFrame
        
    Returns:
        最大单笔盈利
    """
    if trades.empty or 'profit' not in trades.columns:
        return 0
    
    return trades['profit'].max()

def calculate_largest_loss(trades: pd.DataFrame) -> float:
    """
    计算最大单笔亏损
    
    Args:
        trades: 交易记录DataFrame
        
    Returns:
        最大单笔亏损
    """
    if trades.empty or 'profit' not in trades.columns:
        return 0
    
    return trades['profit'].min()

def calculate_consecutive_wins(trades: pd.DataFrame) -> int:
    """
    计算最大连续盈利次数
    
    Args:
        trades: 交易记录DataFrame
        
    Returns:
        最大连续盈利次数
    """
    if trades.empty or 'profit' not in trades.columns:
        return 0
    
    profits = trades['profit'].values
    max_consecutive = 0
    current_consecutive = 0
    
    for profit in profits:
        if profit > 0:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive

def calculate_consecutive_losses(trades: pd.DataFrame) -> int:
    """
    计算最大连续亏损次数
    
    Args:
        trades: 交易记录DataFrame
        
    Returns:
        最大连续亏损次数
    """
    if trades.empty or 'profit' not in trades.columns:
        return 0
    
    profits = trades['profit'].values
    max_consecutive = 0
    current_consecutive = 0
    
    for profit in profits:
        if profit < 0:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive

def format_currency(amount: float, currency: str = 'USD') -> str:
    """
    格式化货币显示
    
    Args:
        amount: 金额
        currency: 货币符号
        
    Returns:
        格式化后的货币字符串
    """
    if currency == 'USD':
        return f"${amount:,.2f}"
    elif currency == 'CNY':
        return f"¥{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    格式化百分比显示
    
    Args:
        value: 数值
        decimals: 小数位数
        
    Returns:
        格式化后的百分比字符串
    """
    return f"{value:.{decimals}%}"

def get_trading_days(start_date: str, end_date: str) -> List[str]:
    """
    获取交易日列表（简化版本，实际应用中需要更复杂的逻辑）
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        交易日列表
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # 生成日期范围
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    # 过滤掉周末（简化处理）
    trading_days = [date.strftime('%Y-%m-%d') for date in date_range if date.weekday() < 5]
    
    return trading_days

def validate_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    验证数据完整性
    
    Args:
        data: 数据DataFrame
        required_columns: 必需的列名列表
        
    Returns:
        验证是否通过
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logging.error(f"缺少必需的列: {missing_columns}")
        return False
    
    if data.empty:
        logging.error("数据为空")
        return False
    
    return True

def clean_data(data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
    """
    清理数据
    
    Args:
        data: 原始数据
        method: 清理方法
        
    Returns:
        清理后的数据
    """
    cleaned_data = data.copy()
    
    if method == 'forward_fill':
        cleaned_data = cleaned_data.fillna(method='ffill')
    elif method == 'backward_fill':
        cleaned_data = cleaned_data.fillna(method='bfill')
    elif method == 'drop':
        cleaned_data = cleaned_data.dropna()
    elif method == 'median':
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(cleaned_data[numeric_cols].median())
    
    return cleaned_data

