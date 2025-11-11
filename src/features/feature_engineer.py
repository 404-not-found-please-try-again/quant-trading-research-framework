"""
特征工程模块
创建技术指标、时间特征和价格特征
"""

import pandas as pd
import numpy as np
import ta
import logging
from typing import List, Dict, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, config: dict):
        """
        初始化特征工程器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.features_config = config['features']
        self.labels_config = config['labels']
        self.logger = logging.getLogger(__name__)
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建所有特征
        
        Args:
            data: 原始数据
            
        Returns:
            包含所有特征的DataFrame
        """
        self.logger.info("开始特征工程...")
        
        # 复制数据避免修改原始数据
        features_data = data.copy()
        
        # 按股票分组处理
        grouped_data = []
        for symbol, group in features_data.groupby('symbol'):
            self.logger.info(f"处理 {symbol} 的特征...")
            
            # 创建技术指标
            group = self._create_technical_indicators(group)
            
            # 创建时间特征
            group = self._create_time_features(group)
            
            # 创建价格特征
            group = self._create_price_features(group)
            
            # 优先级1：创建交互特征（增强特征区分能力）
            group = self._create_interaction_features(group)
            
            # 创建标签
            group = self._create_labels(group)
            
            # 清理数据
            group = self._clean_data(group)
            
            grouped_data.append(group)
        
        # 合并所有数据
        features_data = pd.concat(grouped_data, ignore_index=True)
        
        # 最终数据清理
        features_data = self._final_cleanup(features_data)
        
        self.logger.info(f"特征工程完成，最终特征数量: {features_data.shape[1]}")
        
        return features_data
    
    def _create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建技术指标特征
        
        Args:
            data: 单只股票的数据
            
        Returns:
            添加技术指标后的数据
        """
        # 首先添加基本特征
        data = self._add_basic_features(data)
        
        # 移动平均线
        data = self._add_moving_averages(data)
        
        # RSI
        data = self._add_rsi(data)
        
        # MACD
        data = self._add_macd(data)
        
        # 布林带
        data = self._add_bollinger_bands(data)
        
        # ATR
        data = self._add_atr(data)
        
        # 成交量指标
        data = self._add_volume_indicators(data)
        
        # 价格动量指标
        data = self._add_momentum_indicators(data)
        
        # 高级技术指标
        data = self._add_advanced_indicators(data)
        
        # 中长期技术指标
        data = self._add_long_term_indicators(data)
        
        # 优化4：添加更多金融领域特征
        data = self._add_financial_domain_features(data)
        
        return data
    
    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加基本的价格特征
        
        Args:
            data: 原始价格数据
            
        Returns:
            添加基本特征后的数据
        """
        # 计算日收益率
        data['daily_return'] = data['close'].pct_change()
        
        # 计算对数收益率
        data['log_return'] = np.log(data['close'] / data['close'].shift(1))
        
        # 计算价格变化
        data['price_change'] = data['close'] - data['open']
        
        # 计算价格变化百分比
        data['price_change_pct'] = data['price_change'] / data['open']
        
        # 计算高低价差
        data['high_low_spread'] = data['high'] - data['low']
        
        # 计算高低价差百分比
        data['high_low_spread_pct'] = data['high_low_spread'] / data['close']
        
        # 计算开盘收盘价差
        data['open_close_spread'] = data['close'] - data['open']
        
        # 计算开盘收盘价差百分比
        data['open_close_spread_pct'] = data['open_close_spread'] / data['open']
        
        # 计算成交量变化
        data['volume_change'] = data['volume'].pct_change()
        
        # 计算成交量移动平均
        data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
        data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
        
        return data
    
    def _add_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加移动平均线"""
        sma_periods = self.features_config['technical_indicators']['sma_periods']
        ema_periods = self.features_config['technical_indicators']['ema_periods']
        
        # 简单移动平均
        for period in sma_periods:
            data[f'sma_{period}'] = ta.trend.sma_indicator(data['close'], window=period)
            data[f'sma_{period}_ratio'] = data['close'] / data[f'sma_{period}']
        
        # 指数移动平均
        for period in ema_periods:
            data[f'ema_{period}'] = ta.trend.ema_indicator(data['close'], window=period)
            data[f'ema_{period}_ratio'] = data['close'] / data[f'ema_{period}']
        
        return data
    
    def _add_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加RSI指标"""
        rsi_period = self.features_config['technical_indicators']['rsi_period']
        data['rsi'] = ta.momentum.rsi(data['close'], window=rsi_period)
        
        # RSI分类特征
        data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
        data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
        data['rsi_neutral'] = ((data['rsi'] >= 30) & (data['rsi'] <= 70)).astype(int)
        
        return data
    
    def _add_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加MACD指标"""
        macd_config = self.features_config['technical_indicators']
        fast = macd_config['macd_fast']
        slow = macd_config['macd_slow']
        signal = macd_config['macd_signal']
        
        data['macd'] = ta.trend.macd(data['close'], window_fast=fast, window_slow=slow)
        data['macd_signal'] = ta.trend.macd_signal(data['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        data['macd_histogram'] = ta.trend.macd_diff(data['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        
        # MACD信号
        data['macd_bullish'] = (data['macd'] > data['macd_signal']).astype(int)
        data['macd_bearish'] = (data['macd'] < data['macd_signal']).astype(int)
        
        return data
    
    def _add_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加布林带指标"""
        bb_config = self.features_config['technical_indicators']
        period = bb_config['bollinger_period']
        std = bb_config['bollinger_std']
        
        data['bb_upper'] = ta.volatility.bollinger_hband(data['close'], window=period, window_dev=std)
        data['bb_lower'] = ta.volatility.bollinger_lband(data['close'], window=period, window_dev=std)
        data['bb_middle'] = ta.volatility.bollinger_mavg(data['close'], window=period)
        
        # 布林带位置
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # 布林带突破
        data['bb_upper_break'] = (data['close'] > data['bb_upper']).astype(int)
        data['bb_lower_break'] = (data['close'] < data['bb_lower']).astype(int)
        
        return data
    
    def _add_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加ATR指标"""
        atr_period = self.features_config['technical_indicators']['atr_period']
        data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=atr_period)
        
        # ATR比率
        data['atr_ratio'] = data['atr'] / data['close']
        
        return data
    
    def _add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加成交量指标"""
        # 成交量移动平均
        data['volume_sma_5'] = data['volume'].rolling(window=5).mean()
        data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
        
        # 成交量比率
        data['volume_ratio_5'] = data['volume'] / data['volume_sma_5']
        data['volume_ratio_20'] = data['volume'] / data['volume_sma_20']
        
        # 成交量加权平均价格
        data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()
        data['vwap_ratio'] = data['close'] / data['vwap']
        
        # 成交量价格趋势
        data['vpt'] = (data['volume'] * data['daily_return']).cumsum()
        
        return data
    
    def _add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加动量指标"""
        # 价格动量
        for period in [5, 10, 20]:
            data[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
            data[f'roc_{period}'] = ta.momentum.roc(data['close'], window=period)
        
        # 威廉指标
        data['williams_r'] = ta.momentum.williams_r(data['high'], data['low'], data['close'])
        
        # 随机指标
        data['stoch_k'] = ta.momentum.stoch(data['high'], data['low'], data['close'])
        data['stoch_d'] = ta.momentum.stoch_signal(data['high'], data['low'], data['close'])
        
        # 商品通道指数
        data['cci'] = ta.trend.cci(data['high'], data['low'], data['close'])
        
        return data
    
    def _add_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加高级技术指标"""
        # 布林带宽度变化率
        if 'bb_width' in data.columns:
            data['bb_width_change'] = data['bb_width'].pct_change()
            data['bb_width_ma'] = data['bb_width'].rolling(window=10).mean()
            data['bb_width_ratio'] = data['bb_width'] / data['bb_width_ma']
        
        # RSI动量
        if 'rsi' in data.columns:
            data['rsi_momentum'] = data['rsi'].diff()
            data['rsi_ma'] = data['rsi'].rolling(window=5).mean()
            data['rsi_deviation'] = data['rsi'] - data['rsi_ma']
        
        # MACD动量
        if 'macd' in data.columns:
            data['macd_momentum'] = data['macd'].diff()
            data['macd_signal_momentum'] = data['macd_signal'].diff()
            data['macd_histogram_momentum'] = data['macd_histogram'].diff()
        
        # 价格与移动平均线的距离
        for period in [5, 10, 20, 50]:
            if f'sma_{period}' in data.columns:
                data[f'price_sma_{period}_distance'] = (data['close'] - data[f'sma_{period}']) / data[f'sma_{period}']
                data[f'price_sma_{period}_distance_ma'] = data[f'price_sma_{period}_distance'].rolling(window=5).mean()
        
        # 波动率指标的变化率
        for period in [5, 10, 20]:
            if f'volatility_{period}d' in data.columns:
                data[f'volatility_{period}d_change'] = data[f'volatility_{period}d'].pct_change()
                data[f'volatility_{period}d_ma'] = data[f'volatility_{period}d'].rolling(window=5).mean()
        
        # 成交量价格趋势的动量
        if 'vpt' in data.columns:
            data['vpt_momentum'] = data['vpt'].diff()
            data['vpt_ma'] = data['vpt'].rolling(window=10).mean()
            data['vpt_ratio'] = data['vpt'] / data['vpt_ma']
        
        # 价格加速度（二阶导数）
        data['price_acceleration'] = data['daily_return'].diff()
        
        # 成交量加速度
        data['volume_acceleration'] = data['volume_change'].diff()
        
        # 价格波动性指标
        data['price_volatility_ratio'] = data['daily_return'].rolling(window=5).std() / data['daily_return'].rolling(window=20).std()
        
        # 相对强弱指标
        data['relative_strength'] = data['close'] / data['close'].rolling(window=20).mean()
        
        # 价格动量指标
        data['price_momentum_5'] = data['close'] / data['close'].shift(5) - 1
        data['price_momentum_10'] = data['close'] / data['close'].shift(10) - 1
        data['price_momentum_20'] = data['close'] / data['close'].shift(20) - 1
        
        return data
    
    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建时间特征
        
        Args:
            data: 数据
            
        Returns:
            添加时间特征后的数据
        """
        # 确保Date列是datetime类型
        data['Date'] = pd.to_datetime(data['Date'])
        
        # 基本时间特征
        data['year'] = data['Date'].dt.year
        data['month'] = data['Date'].dt.month
        data['day'] = data['Date'].dt.day
        data['day_of_week'] = data['Date'].dt.dayofweek
        data['day_of_year'] = data['Date'].dt.dayofyear
        data['week_of_year'] = data['Date'].dt.isocalendar().week
        data['quarter'] = data['Date'].dt.quarter
        
        # 周期性特征
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # 特殊日期特征
        data['is_month_end'] = data['Date'].dt.is_month_end.astype(int)
        data['is_quarter_end'] = data['Date'].dt.is_quarter_end.astype(int)
        data['is_year_end'] = data['Date'].dt.is_year_end.astype(int)
        
        # 交易日特征
        data['days_since_start'] = (data['Date'] - data['Date'].min()).dt.days
        data['trading_day_of_month'] = data.groupby(data['Date'].dt.to_period('M')).cumcount() + 1
        
        return data
    
    def _create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建价格特征
        
        Args:
            data: 数据
            
        Returns:
            添加价格特征后的数据
        """
        # 价格位置特征
        for period in [5, 10, 20, 50]:
            high_max = data['high'].rolling(window=period).max()
            low_min = data['low'].rolling(window=period).min()
            data[f'price_position_{period}'] = (data['close'] - low_min) / (high_max - low_min)
        
        # 价格变化特征
        for period in [1, 2, 3, 5, 10]:
            data[f'price_change_{period}d'] = data['close'].pct_change(period)
            data[f'high_change_{period}d'] = data['high'].pct_change(period)
            data[f'low_change_{period}d'] = data['low'].pct_change(period)
        
        # 波动率特征
        for period in [5, 10, 20]:
            data[f'volatility_{period}d'] = data['daily_return'].rolling(window=period).std()
            data[f'volatility_{period}d_annualized'] = data[f'volatility_{period}d'] * np.sqrt(252)
        
        # 价格缺口
        data['gap_up'] = (data['open'] > data['high'].shift(1)).astype(int)
        data['gap_down'] = (data['open'] < data['low'].shift(1)).astype(int)
        data['gap_size'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        
        return data
    
    def _create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建预测标签 - 优化版本
        
        Args:
            data: 数据
            
        Returns:
            添加标签后的数据
        """
        # 获取分类阈值
        up_threshold = self.labels_config['classification_thresholds']['up_threshold']
        down_threshold = self.labels_config['classification_thresholds']['down_threshold']
        neutral_threshold = self.labels_config['classification_thresholds'].get('neutral_threshold', 0.002)
        
        # 计算未来收益率
        for window in self.labels_config['prediction_windows']:
            # 计算未来N天的平均收益率（平滑处理）
            future_returns = []
            for i in range(1, window + 1):
                future_returns.append(data['close'].shift(-i) / data['close'] - 1)
            
            # 计算平均收益率
            data[f'future_return_{window}d'] = pd.concat(future_returns, axis=1).mean(axis=1)
        
        # 创建分类标签
        classification_type = self.labels_config.get('classification_type', 'binary')
        
        for window in self.labels_config['prediction_windows']:
            future_return = data[f'future_return_{window}d']
            
            if classification_type == 'binary':
                # 二分类：上涨(1)/下跌(0) - 使用阈值过滤噪声
                data[f'label_{window}d'] = (future_return > up_threshold).astype(int)
                data[f'label_binary_{window}d'] = data[f'label_{window}d']
                
                # 添加置信度标签（过滤掉中性区间）
                data[f'label_confidence_{window}d'] = (
                    (future_return > up_threshold) | (future_return < down_threshold)
                ).astype(int)
            else:
                # 三分类：上涨、下跌、横盘
                data[f'label_{window}d'] = 0  # 横盘
                data.loc[future_return > up_threshold, f'label_{window}d'] = 1  # 上涨
                data.loc[future_return < down_threshold, f'label_{window}d'] = -1  # 下跌
                
                # 二分类：上涨/下跌
                data[f'label_binary_{window}d'] = (future_return > 0).astype(int)
        
        # 标签平滑处理
        if self.labels_config.get('smoothing', {}).get('enable', False):
            data = self._smooth_labels(data)
        
        # 主要预测目标（3天）
        data['target'] = data['label_3d']
        data['target_binary'] = data['label_binary_3d']
        data['target_return'] = data['future_return_3d']
        # 移除 target_confidence，以避免潜在的信息泄露
        
        return data
    
    def _smooth_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        平滑标签以减少噪声
        
        Args:
            data: 数据
            
        Returns:
            平滑后的数据
        """
        smoothing_config = self.labels_config['smoothing']
        window = smoothing_config['window']
        min_samples = smoothing_config['min_samples']
        
        for window_name in self.labels_config['prediction_windows']:
            label_col = f'label_{window_name}d'
            if label_col in data.columns:
                # 使用移动平均平滑标签
                smoothed = data[label_col].rolling(window=window, min_periods=min_samples).mean()
                data[f'{label_col}_smoothed'] = smoothed
                
                # 更新二分类标签
                data[f'label_binary_{window_name}d'] = (smoothed > 0.5).astype(int)
        
        return data
    
    def _add_long_term_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加中长期技术指标
        """
        self.logger.info("添加中长期技术指标...")
        
        # 获取长期指标配置
        long_term_config = self.features_config['technical_indicators'].get('long_term_indicators', {})
        
        # 长期移动平均线
        for period in long_term_config.get('sma_periods', [30, 60, 90, 120]):
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            data[f'sma_{period}_ratio'] = data['close'] / data[f'sma_{period}']
            data[f'sma_{period}_distance'] = (data['close'] - data[f'sma_{period}']) / data[f'sma_{period}']
        
        for period in long_term_config.get('ema_periods', [30, 60, 90]):
            data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            data[f'ema_{period}_ratio'] = data['close'] / data[f'ema_{period}']
            data[f'ema_{period}_distance'] = (data['close'] - data[f'ema_{period}']) / data[f'ema_{period}']
        
        # 长期动量指标
        for period in long_term_config.get('momentum_periods', [20, 30, 60]):
            data[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
            data[f'price_change_{period}d'] = data['close'].pct_change(period)
            data[f'high_low_ratio_{period}d'] = data['high'].rolling(period).max() / data['low'].rolling(period).min()
        
        # 长期波动率指标
        for period in long_term_config.get('volatility_periods', [20, 30, 60]):
            data[f'volatility_{period}d'] = data['close'].pct_change().rolling(window=period).std()
            data[f'volatility_{period}d_annualized'] = data[f'volatility_{period}d'] * np.sqrt(252)
            data[f'volatility_{period}d_ma'] = data[f'volatility_{period}d'].rolling(window=5).mean()
        
        # 长期成交量指标
        for period in long_term_config.get('volume_periods', [20, 30, 60]):
            data[f'volume_ma_{period}'] = data['volume'].rolling(window=period).mean()
            data[f'volume_ratio_{period}'] = data['volume'] / data[f'volume_ma_{period}']
            data[f'volume_std_{period}'] = data['volume'].rolling(window=period).std()
            data[f'volume_cv_{period}'] = data[f'volume_std_{period}'] / data[f'volume_ma_{period}']  # 变异系数
        
        # MACD信号交叉
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            data['macd_signal_cross'] = 0
            data.loc[data['macd'] > data['macd_signal'], 'macd_signal_cross'] = 1
            data.loc[data['macd'] < data['macd_signal'], 'macd_signal_cross'] = -1
            
            # MACD金叉死叉
            data['macd_golden_cross'] = ((data['macd'] > data['macd_signal']) & 
                                       (data['macd'].shift(1) <= data['macd_signal'].shift(1))).astype(int)
            data['macd_death_cross'] = ((data['macd'] < data['macd_signal']) & 
                                      (data['macd'].shift(1) >= data['macd_signal'].shift(1))).astype(int)
        
        # RSI信号
        if 'rsi' in data.columns:
            data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
            data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
            data['rsi_neutral'] = ((data['rsi'] >= 30) & (data['rsi'] <= 70)).astype(int)
        
        # 布林带信号
        if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
            data['bb_squeeze'] = (data['bb_upper'] - data['bb_lower']) / data['close'] < 0.1  # 布林带收缩
            data['bb_expansion'] = (data['bb_upper'] - data['bb_lower']) / data['close'] > 0.2  # 布林带扩张
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # 趋势强度指标
        data['trend_strength_20'] = data['close'].rolling(20).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) == 20 else np.nan
        )
        data['trend_strength_60'] = data['close'].rolling(60).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) == 60 else np.nan
        )
        
        # 价格通道
        for period in [20, 30, 60]:
            data[f'price_channel_high_{period}'] = data['high'].rolling(period).max()
            data[f'price_channel_low_{period}'] = data['low'].rolling(period).min()
            data[f'price_channel_position_{period}'] = (data['close'] - data[f'price_channel_low_{period}']) / \
                                                      (data[f'price_channel_high_{period}'] - data[f'price_channel_low_{period}'])
        
        # 移动平均线交叉
        if 'sma_20' in data.columns and 'sma_60' in data.columns:
            data['sma_cross_20_60'] = 0
            data.loc[data['sma_20'] > data['sma_60'], 'sma_cross_20_60'] = 1
            data.loc[data['sma_20'] < data['sma_60'], 'sma_cross_20_60'] = -1
            
            # 金叉死叉
            data['sma_golden_cross'] = ((data['sma_20'] > data['sma_60']) & 
                                      (data['sma_20'].shift(1) <= data['sma_60'].shift(1))).astype(int)
            data['sma_death_cross'] = ((data['sma_20'] < data['sma_60']) & 
                                     (data['sma_20'].shift(1) >= data['sma_60'].shift(1))).astype(int)
        
        # 相对强弱指标（相对于大盘）
        if 'sma_200' in data.columns:
            data['relative_strength_200'] = data['close'] / data['sma_200']
            data['relative_strength_200_ma'] = data['relative_strength_200'].rolling(10).mean()
        
        self.logger.info("中长期技术指标添加完成")
        return data
    
    def _add_financial_domain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        优化4：添加更多金融领域特征
        
        包括：
        1. ADX（平均趋向指标）- 趋势强度
        2. 波动率特征增强
        3. 价格形态特征（支撑/阻力）
        4. 资金流向特征
        5. 相对强度特征增强
        """
        self.logger.info("添加金融领域特征（优化4）...")
        
        # 1. ADX（平均趋向指标）- 趋势强度指标
        try:
            # 使用ta库计算ADX
            data['adx'] = ta.trend.adx(data['high'], data['low'], data['close'], window=14)
            data['adx_strong_trend'] = (data['adx'] > 25).astype(int)  # 强趋势
            data['adx_weak_trend'] = (data['adx'] < 20).astype(int)  # 弱趋势
            data['adx_momentum'] = data['adx'].diff()
            data['adx_ma'] = data['adx'].rolling(window=5).mean()
        except Exception as e:
            self.logger.warning(f"ADX计算失败: {str(e)}")
        
        # 2. 波动率特征增强
        # 历史波动率（滚动标准差）
        volatility_periods = [5, 10, 20, 30]
        for period in volatility_periods:
            data[f'historical_volatility_{period}d'] = data['close'].pct_change().rolling(window=period).std()
        
        # 波动率比率（短期/长期）- 在所有波动率特征创建后计算
        for period in [10, 20, 30]:
            short_period = period // 2
            if f'historical_volatility_{short_period}d' in data.columns:
                data[f'volatility_ratio_{period}d'] = (
                    data[f'historical_volatility_{short_period}d'] / 
                    (data[f'historical_volatility_{period}d'] + 1e-6)  # 避免除零
                )
        
        # 已实现波动率（基于高低价差）
        for period in [5, 10, 20]:
            high_low_vol = np.log(data['high'] / data['low']).rolling(window=period).std()
            data[f'realized_volatility_{period}d'] = high_low_vol
            data[f'volatility_premium_{period}d'] = (
                data[f'historical_volatility_{period}d'] - 
                data[f'realized_volatility_{period}d']
            )
        
        # 3. 价格形态特征（支撑位和阻力位）
        # 局部最高点和最低点
        for period in [5, 10, 20]:
            local_high = data['high'].rolling(window=period, center=True).max()
            local_low = data['low'].rolling(window=period, center=True).min()
            
            # 距离支撑位和阻力位的距离
            data[f'resistance_distance_{period}'] = (local_high - data['close']) / data['close']
            data[f'support_distance_{period}'] = (data['close'] - local_low) / data['close']
            
            # 是否接近支撑/阻力位（5%以内）
            data[f'near_resistance_{period}'] = (data[f'resistance_distance_{period}'] < 0.05).astype(int)
            data[f'near_support_{period}'] = (data[f'support_distance_{period}'] < 0.05).astype(int)
        
        # 4. 资金流向特征
        # 价格-成交量关系（OBV的变体）
        data['price_volume_trend'] = (data['close'].pct_change() * data['volume']).cumsum()
        data['pvt_momentum'] = data['price_volume_trend'].diff()
        data['pvt_ma'] = data['price_volume_trend'].rolling(window=10).mean()
        data['pvt_ratio'] = data['price_volume_trend'] / data['pvt_ma']
        
        # 资金流动指数（MFI - Money Flow Index）
        try:
            data['mfi'] = ta.volume.money_flow_index(
                data['high'], data['low'], data['close'], data['volume'], window=14
            )
            data['mfi_oversold'] = (data['mfi'] < 20).astype(int)
            data['mfi_overbought'] = (data['mfi'] > 80).astype(int)
        except Exception as e:
            self.logger.warning(f"MFI计算失败: {str(e)}")
        
        # 成交量加权价格趋势
        if 'vwap' in data.columns:
            data['vwap_distance'] = (data['close'] - data['vwap']) / data['vwap']
            data['vwap_above'] = (data['close'] > data['vwap']).astype(int)
            data['vwap_momentum'] = data['vwap'].pct_change()
        
        # 5. 相对强度特征增强
        # 相对强度指数（相对于移动平均）
        for period in [20, 50, 200]:
            if f'sma_{period}' in data.columns:
                data[f'relative_strength_sma_{period}'] = data['close'] / data[f'sma_{period}']
                data[f'relative_strength_sma_{period}_ma'] = data[f'relative_strength_sma_{period}'].rolling(window=5).mean()
                data[f'relative_strength_sma_{period}_momentum'] = data[f'relative_strength_sma_{period}'].diff()
        
        # 相对价格位置（在最近N天价格区间中的位置）
        for period in [10, 20, 30]:
            period_high = data['high'].rolling(window=period).max()
            period_low = data['low'].rolling(window=period).min()
            data[f'price_position_in_range_{period}'] = (
                (data['close'] - period_low) / (period_high - period_low)
            )
        
        # 6. 价格加速度和急动度（三阶导数）
        data['price_jerk'] = data['price_acceleration'].diff()  # 价格急动度
        
        # 7. 成交量价格背离特征
        # 价格创新高但成交量下降（可能的背离）
        price_new_high = (data['close'] == data['close'].rolling(window=20).max()).astype(int)
        volume_declining = (data['volume'] < data['volume'].rolling(window=20).mean()).astype(int)
        data['price_volume_divergence'] = (price_new_high & volume_declining).astype(int)
        
        # 8. 价格波动率特征（基于真实波动范围）
        if 'atr' in data.columns:
            # ATR百分比
            data['atr_pct'] = data['atr'] / data['close']
            # ATR比率（当前ATR vs 平均ATR）
            data['atr_ratio'] = data['atr'] / data['atr'].rolling(window=20).mean()
            # ATR动量
            data['atr_momentum'] = data['atr'].pct_change()
        
        # 9. 趋势持续性特征
        # 连续上涨/下跌天数
        data['consecutive_up'] = (data['close'] > data['close'].shift(1)).astype(int)
        data['consecutive_down'] = (data['close'] < data['close'].shift(1)).astype(int)
        
        # 计算连续天数
        def count_consecutive(series):
            result = []
            count = 0
            for val in series:
                if pd.isna(val):
                    result.append(0)
                    count = 0
                elif val:
                    count += 1
                    result.append(count)
                else:
                    count = 0
                    result.append(0)
            return pd.Series(result, index=series.index)
        
        data['consecutive_up_days'] = count_consecutive(data['consecutive_up'])
        data['consecutive_down_days'] = count_consecutive(data['consecutive_down'])
        
        # 10. 价格动量比率（短期/长期动量）
        if 'price_momentum_5' in data.columns and 'price_momentum_20' in data.columns:
            data['momentum_ratio_5_20'] = data['price_momentum_5'] / (data['price_momentum_20'] + 1e-6)
        
        self.logger.info("金融领域特征添加完成")
        return data
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建交互特征（优先级2：基于重要性的智能交互特征选择）
        
        交互特征可以捕获特征之间的非线性关系，提高模型的区分能力
        本次优化：基于特征与目标的相关性来选择高重要性特征进行交互
        
        Args:
            data: 数据
            
        Returns:
            添加交互特征后的数据
        """
        interaction_config = self.features_config.get('interaction_features', {})
        
        if not interaction_config.get('enable', False):
            self.logger.info("交互特征未启用，跳过")
            return data
        
        self.logger.info("开始创建交互特征（基于重要性选择）...")
        initial_feature_count = len([col for col in data.columns if col not in ['Date', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
        
        # 优先级2：计算特征重要性（如果目标列存在）
        target_cols = ['target', 'target_binary', 'target_return']
        available_target = None
        for tc in target_cols:
            if tc in data.columns:
                available_target = tc
                break
        
        feature_importance_map = {}
        if available_target and len(data[available_target].dropna()) > 10:
            # 计算特征与目标的相关性
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in target_cols + ['Date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            if len(feature_cols) > 0:
                try:
                    correlations = data[feature_cols + [available_target]].corr()[available_target].abs()
                    correlations = correlations.drop(available_target, errors='ignore')
                    feature_importance_map = correlations.to_dict()
                    self.logger.info(f"基于目标{available_target}计算了{len(feature_importance_map)}个特征的重要性")
                except Exception as e:
                    self.logger.warning(f"计算特征重要性失败: {str(e)}，将使用无排序选择")
        
        # 预先缓存特征列表（避免重复搜索，提升性能）
        all_cols = data.columns.tolist()
        vol_cols = [col for col in all_cols if 'volume_ratio' in col.lower() or 'vpt' in col.lower()]
        momentum_cols = [col for col in all_cols if 'momentum' in col.lower()]
        trend_cols = [col for col in all_cols if 'trend_strength' in col.lower()]
        price_change_cols = [col for col in all_cols if 'price_change' in col.lower() or 'daily_return' in col.lower()]
        volatility_cols = [col for col in all_cols if 'volatility' in col.lower()]
        tech_cols = [col for col in all_cols if col.lower() in ['rsi', 'macd', 'macd_signal', 'bb_position', 'vpt']]
        
        # 优先级2：基于重要性排序特征（如果可用）
        def sort_by_importance(cols, importance_map):
            """根据重要性排序特征列表"""
            if not importance_map:
                return cols
            return sorted(cols, key=lambda x: importance_map.get(x, 0), reverse=True)
        
        vol_cols = sort_by_importance(vol_cols, feature_importance_map)[:4]
        momentum_cols = sort_by_importance(momentum_cols, feature_importance_map)[:4]
        trend_cols = sort_by_importance(trend_cols, feature_importance_map)[:3]
        price_change_cols = sort_by_importance(price_change_cols, feature_importance_map)[:3]
        volatility_cols = sort_by_importance(volatility_cols, feature_importance_map)[:3]
        
        interaction_features = []
        interaction_candidates = []  # 存储候选交互特征及其评分
        max_interactions = interaction_config.get('max_interactions', 50)
        
        # 1. 技术指标与成交量的交互（基于重要性）
        if interaction_config.get('tech_volume_interactions', True):
            if 'rsi' in all_cols:
                rsi_importance = feature_importance_map.get('rsi', 0)
                for vol_col in vol_cols[:3]:
                    vol_importance = feature_importance_map.get(vol_col, 0)
                    score = (rsi_importance + vol_importance) / 2
                    feature_name = f'rsi_volume_interaction_{vol_col}'
                    interaction_candidates.append((feature_name, score, 'rsi', vol_col))
            
            if 'macd' in all_cols:
                macd_importance = feature_importance_map.get('macd', 0)
                for vol_col in vol_cols[:2]:
                    vol_importance = feature_importance_map.get(vol_col, 0)
                    score = (macd_importance + vol_importance) / 2
                    feature_name = f'macd_volume_interaction_{vol_col}'
                    interaction_candidates.append((feature_name, score, 'macd', vol_col))
            
            if 'rsi' in all_cols and 'vpt' in all_cols:
                rsi_importance = feature_importance_map.get('rsi', 0)
                vpt_importance = feature_importance_map.get('vpt', 0)
                score = (rsi_importance + vpt_importance) / 2
                interaction_candidates.append(('rsi_vpt_interaction', score, 'rsi', 'vpt'))
        
        # 2. 技术指标之间的交互（基于重要性）
        if interaction_config.get('tech_tech_interactions', True):
            if 'rsi' in all_cols and 'macd' in all_cols:
                rsi_importance = feature_importance_map.get('rsi', 0)
                macd_importance = feature_importance_map.get('macd', 0)
                score = (rsi_importance + macd_importance) / 2
                interaction_candidates.append(('rsi_macd_interaction', score, 'rsi', 'macd'))
            
            # RSI × Momentum
            if 'rsi' in all_cols:
                rsi_importance = feature_importance_map.get('rsi', 0)
                for mom_col in momentum_cols[:2]:
                    mom_importance = feature_importance_map.get(mom_col, 0)
                    score = (rsi_importance + mom_importance) / 2
                    feature_name = f'rsi_momentum_interaction_{mom_col}'
                    interaction_candidates.append((feature_name, score, 'rsi', mom_col))
            
            # MACD × Momentum
            if 'macd' in all_cols:
                macd_importance = feature_importance_map.get('macd', 0)
                for mom_col in momentum_cols[:2]:
                    mom_importance = feature_importance_map.get(mom_col, 0)
                    score = (macd_importance + mom_importance) / 2
                    feature_name = f'macd_momentum_interaction_{mom_col}'
                    interaction_candidates.append((feature_name, score, 'macd', mom_col))
            
            # RSI × Trend Strength
            if 'rsi' in all_cols:
                rsi_importance = feature_importance_map.get('rsi', 0)
                for trend_col in trend_cols[:2]:
                    trend_importance = feature_importance_map.get(trend_col, 0)
                    score = (rsi_importance + trend_importance) / 2
                    feature_name = f'rsi_trend_interaction_{trend_col}'
                    interaction_candidates.append((feature_name, score, 'rsi', trend_col))
        
        # 3. 价格变化与成交量的交互（基于重要性）
        if interaction_config.get('price_volume_interactions', True):
            for price_change_col in price_change_cols[:2]:
                price_importance = feature_importance_map.get(price_change_col, 0)
                for vol_col in vol_cols[:2]:
                    vol_importance = feature_importance_map.get(vol_col, 0)
                    score = (price_importance + vol_importance) / 2
                    feature_name = f'price_vol_interaction_{price_change_col}_{vol_col}'
                    interaction_candidates.append((feature_name, score, price_change_col, vol_col))
            
            if 'high_low_spread_pct' in all_cols:
                hlspread_importance = feature_importance_map.get('high_low_spread_pct', 0)
                for vol_col in vol_cols[:2]:
                    vol_importance = feature_importance_map.get(vol_col, 0)
                    score = (hlspread_importance + vol_importance) / 2
                    feature_name = f'hlspread_vol_interaction_{vol_col}'
                    interaction_candidates.append((feature_name, score, 'high_low_spread_pct', vol_col))
        
        # 4. 波动率与动量的交互（基于重要性）
        if interaction_config.get('volatility_momentum_interactions', True):
            for vol_col in volatility_cols[:2]:
                vol_importance = feature_importance_map.get(vol_col, 0)
                for mom_col in momentum_cols[:2]:
                    mom_importance = feature_importance_map.get(mom_col, 0)
                    score = (vol_importance + mom_importance) / 2
                    feature_name = f'vol_mom_interaction_{vol_col}_{mom_col}'
                    interaction_candidates.append((feature_name, score, vol_col, mom_col))
            
            # Volatility × RSI
            if 'rsi' in all_cols:
                rsi_importance = feature_importance_map.get('rsi', 0)
                for vol_col in volatility_cols[:2]:
                    vol_importance = feature_importance_map.get(vol_col, 0)
                    score = (vol_importance + rsi_importance) / 2
                    feature_name = f'vol_rsi_interaction_{vol_col}'
                    interaction_candidates.append((feature_name, score, vol_col, 'rsi'))
        
        # 5. 布林带位置与RSI的交互
        if 'bb_position' in all_cols and 'rsi' in all_cols:
            bb_importance = feature_importance_map.get('bb_position', 0)
            rsi_importance = feature_importance_map.get('rsi', 0)
            score = (bb_importance + rsi_importance) / 2
            interaction_candidates.append(('bb_rsi_interaction', score, 'bb_position', 'rsi'))
        
        # 6. MACD信号强度与波动率的交互
        if 'macd' in all_cols and 'macd_signal' in all_cols:
            macd_importance = feature_importance_map.get('macd', 0)
            signal_importance = feature_importance_map.get('macd_signal', 0)
            signal_strength_importance = (macd_importance + signal_importance) / 2
            for vol_col in volatility_cols[:2]:
                vol_importance = feature_importance_map.get(vol_col, 0)
                score = (signal_strength_importance + vol_importance) / 2
                feature_name = f'macd_signal_vol_interaction_{vol_col}'
                interaction_candidates.append((feature_name, score, 'macd_signal_strength', vol_col))
        
        # 优先级2：基于重要性排序并选择top K交互特征
        if interaction_candidates:
            # 按评分排序（降序）
            interaction_candidates.sort(key=lambda x: x[1], reverse=True)
            selected_candidates = interaction_candidates[:max_interactions]
            
            self.logger.info(f"基于重要性选择了 {len(selected_candidates)} 个交互特征（从 {len(interaction_candidates)} 个候选中）")
            
            # 生成选中的交互特征
            for feature_name, score, col1, col2 in selected_candidates:
                if col1 == 'macd_signal_strength':
                    # MACD信号强度特殊处理
                    data[feature_name] = (data['macd'] - data['macd_signal']) * data[col2]
                else:
                    data[feature_name] = data[col1] * data[col2]
                interaction_features.append(feature_name)
            
            if len(interaction_candidates) > max_interactions:
                self.logger.info(f"交互特征重要性评分范围: {selected_candidates[0][1]:.4f} ~ {selected_candidates[-1][1]:.4f}")
        else:
            self.logger.warning("未生成任何交互特征候选")
        
        final_feature_count = len([col for col in data.columns if col not in ['Date', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
        self.logger.info(f"交互特征创建完成，新增 {len(interaction_features)} 个交互特征（总特征数: {initial_feature_count} -> {final_feature_count}）")
        
        return data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清理数据
        
        Args:
            data: 数据
            
        Returns:
            清理后的数据
        """
        # 删除包含无穷大值的行
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # 删除全为NaN的列
        data = data.dropna(axis=1, how='all')
        
        return data
    
    def _final_cleanup(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        最终数据清理
        
        Args:
            data: 数据
            
        Returns:
            最终清理后的数据
        """
        # 删除包含未来信息的特征列
        future_cols = [col for col in data.columns if col.startswith(('future_', 'label_'))]
        if future_cols:
            self.logger.info(f"删除未来信息特征: {future_cols}")
            data = data.drop(columns=future_cols)
        
        # 安全起见，确保不包含 target_confidence（即使外部数据带入）
        if 'target_confidence' in data.columns:
            self.logger.info("删除潜在泄露特征: target_confidence")
            data = data.drop(columns=['target_confidence'])
        
        # 删除包含NaN的行（这些行无法用于训练）
        initial_rows = len(data)
        data = data.dropna()
        final_rows = len(data)
        
        if initial_rows != final_rows:
            self.logger.info(f"删除了 {initial_rows - final_rows} 行包含NaN的数据")
        
        # 重置索引
        data = data.reset_index(drop=True)
        
        return data
    
    def get_feature_importance(self, data: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """
        获取特征重要性（基于相关性）
        
        Args:
            data: 数据
            target_col: 目标列名
            
        Returns:
            特征重要性DataFrame
        """
        # 选择数值特征
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['target', 'target_binary', 'target_return']]
        
        # 计算与目标的相关性
        correlations = data[feature_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': correlations.index[1:],  # 排除目标列本身
            'correlation': correlations.values[1:],
            'abs_correlation': correlations.values[1:]
        })
        
        return importance_df
