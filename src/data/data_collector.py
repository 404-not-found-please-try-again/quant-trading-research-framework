"""
数据收集模块
使用yfinance获取美股历史数据
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os

class DataCollector:
    """数据收集器类"""
    
    def __init__(self, config: dict):
        """
        初始化数据收集器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.data_config = config['data']
        self.logger = logging.getLogger(__name__)
        
        # 创建数据存储目录
        self.data_path = self.data_config['data_path']
        os.makedirs(self.data_path, exist_ok=True)
        
    def collect_data(self) -> pd.DataFrame:
        """
        收集股票数据
        
        Returns:
            包含所有股票数据的DataFrame
        """
        symbols = self.data_config['symbols']
        start_date = self.data_config['start_date']
        end_date = self.data_config['end_date']
        
        self.logger.info(f"开始收集 {len(symbols)} 只股票的数据")
        self.logger.info(f"时间范围: {start_date} 到 {end_date}")
        
        all_data = []
        
        for symbol in symbols:
            try:
                self.logger.info(f"正在收集 {symbol} 的数据...")
                data = self._download_stock_data(symbol, start_date, end_date)
                
                if data is not None and not data.empty:
                    # 添加股票代码列
                    data['symbol'] = symbol
                    all_data.append(data)
                    self.logger.info(f"{symbol} 数据收集成功，共 {len(data)} 条记录")
                else:
                    self.logger.warning(f"{symbol} 数据为空，跳过")
                    
            except Exception as e:
                self.logger.error(f"收集 {symbol} 数据时出错: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("没有成功收集到任何数据")
        
        # 合并所有数据
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 按日期和股票代码排序
        combined_data = combined_data.sort_values(['Date', 'symbol']).reset_index(drop=True)
        
        self.logger.info(f"数据收集完成，总共 {len(combined_data)} 条记录")
        
        # 保存原始数据
        self._save_raw_data(combined_data)
        
        return combined_data
    
    def _download_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        下载单只股票的数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            股票数据DataFrame
        """
        try:
            # 创建ticker对象
            ticker = yf.Ticker(symbol)
            
            # 下载历史数据
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                return None
            
            # 重置索引，将Date作为列
            data = data.reset_index()
            
            # 重命名列以保持一致性
            data.columns = [col.lower() for col in data.columns]
            
            # 确保Date列是datetime类型
            if 'date' in data.columns:
                data['Date'] = pd.to_datetime(data['date'])
                data = data.drop('date', axis=1)
            else:
                # 如果没有date列，使用索引
                data['Date'] = data.index
            
            # 添加基本的价格特征
            data = self._add_basic_features(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"下载 {symbol} 数据失败: {str(e)}")
            return None
    
    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加基本的价格特征
        
        Args:
            data: 原始价格数据
            
        Returns:
            添加特征后的数据
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
    
    def _save_raw_data(self, data: pd.DataFrame):
        """
        保存原始数据到文件
        
        Args:
            data: 要保存的数据
        """
        try:
            # 保存为CSV格式
            csv_path = os.path.join(self.data_path, 'raw_data.csv')
            data.to_csv(csv_path, index=False)
            self.logger.info(f"原始数据已保存到: {csv_path}")
            
            # 保存为Parquet格式（更高效）
            parquet_path = os.path.join(self.data_path, 'raw_data.parquet')
            data.to_parquet(parquet_path, index=False)
            self.logger.info(f"原始数据已保存到: {parquet_path}")
            
        except Exception as e:
            self.logger.error(f"保存原始数据失败: {str(e)}")
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        从文件加载数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            加载的数据DataFrame
        """
        if file_path is None:
            file_path = os.path.join(self.data_path, 'raw_data.parquet')
        
        try:
            if file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                data = pd.read_csv(file_path)
            
            # 确保Date列是datetime类型
            data['Date'] = pd.to_datetime(data['Date'])
            
            self.logger.info(f"成功加载数据: {len(data)} 条记录")
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            raise
    
    def get_data_info(self, data: pd.DataFrame) -> Dict:
        """
        获取数据基本信息
        
        Args:
            data: 数据DataFrame
            
        Returns:
            数据信息字典
        """
        info = {
            'total_records': len(data),
            'date_range': {
                'start': data['Date'].min(),
                'end': data['Date'].max()
            },
            'symbols': data['symbol'].unique().tolist(),
            'columns': data.columns.tolist(),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }
        
        return info
