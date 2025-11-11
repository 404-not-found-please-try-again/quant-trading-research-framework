"""
特征选择模块
从大量特征中选择最重要的特征
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import xgboost as xgb

class FeatureSelector:
    """特征选择器类"""
    
    def __init__(self, config: dict):
        """
        初始化特征选择器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'combined', n_features: int = 50) -> Dict[str, Any]:
        """
        选择最重要的特征
        
        Args:
            X: 特征矩阵
            y: 目标变量
            method: 选择方法 ('correlation', 'mutual_info', 'rfe', 'combined')
            n_features: 选择的特征数量
            
        Returns:
            特征选择结果
        """
        self.logger.info(f"开始特征选择，方法: {method}, 目标特征数: {n_features}")
        
        if method == 'correlation':
            return self._correlation_selection(X, y, n_features)
        elif method == 'mutual_info':
            return self._mutual_info_selection(X, y, n_features)
        elif method == 'rfe':
            return self._rfe_selection(X, y, n_features)
        elif method == 'combined':
            return self._combined_selection(X, y, n_features)
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
    
    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> Dict[str, Any]:
        """基于相关性的特征选择"""
        # 计算每个特征与目标的相关性
        correlations = []
        for col in X.columns:
            corr = X[col].corr(y)
            correlations.append((col, abs(corr)))
        
        # 按相关性排序
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前n_features个特征
        selected_features = [feat[0] for feat in correlations[:n_features]]
        
        self.logger.info(f"相关性选择完成，选择了 {len(selected_features)} 个特征")
        
        return {
            'selected_features': selected_features,
            'feature_scores': dict(correlations[:n_features]),
            'method': 'correlation'
        }
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> Dict[str, Any]:
        """基于互信息的特征选择"""
        # 处理缺失值
        X_clean = X.fillna(X.median())
        
        # 计算互信息
        mi_scores = mutual_info_classif(X_clean, y, random_state=42)
        
        # 创建特征分数字典
        feature_scores = dict(zip(X.columns, mi_scores))
        
        # 按分数排序
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前n_features个特征
        selected_features = [feat[0] for feat in sorted_features[:n_features]]
        
        self.logger.info(f"互信息选择完成，选择了 {len(selected_features)} 个特征")
        
        return {
            'selected_features': selected_features,
            'feature_scores': dict(sorted_features[:n_features]),
            'method': 'mutual_info'
        }
    
    def _rfe_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> Dict[str, Any]:
        """基于递归特征消除的特征选择"""
        # 使用RandomForest作为基础估计器
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # 创建RFE选择器
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        
        # 处理缺失值
        X_clean = X.fillna(X.median())
        
        # 拟合RFE
        rfe.fit(X_clean, y)
        
        # 获取选择的特征
        selected_features = X.columns[rfe.support_].tolist()
        
        # 获取特征排名
        feature_ranking = dict(zip(X.columns, rfe.ranking_))
        
        self.logger.info(f"RFE选择完成，选择了 {len(selected_features)} 个特征")
        
        return {
            'selected_features': selected_features,
            'feature_ranking': feature_ranking,
            'method': 'rfe'
        }
    
    def _combined_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> Dict[str, Any]:
        """组合多种方法进行特征选择"""
        self.logger.info("开始组合特征选择...")
        
        # 1. 相关性选择
        corr_result = self._correlation_selection(X, y, n_features * 2)
        
        # 2. 互信息选择
        mi_result = self._mutual_info_selection(X, y, n_features * 2)
        
        # 3. RFE选择
        rfe_result = self._rfe_selection(X, y, n_features * 2)
        
        # 计算特征投票分数
        feature_votes = {}
        feature_scores = {}
        
        # 相关性分数 (权重: 0.3)
        for feat, score in corr_result['feature_scores'].items():
            feature_votes[feat] = feature_votes.get(feat, 0) + 0.3
            feature_scores[feat] = feature_scores.get(feat, 0) + score * 0.3
        
        # 互信息分数 (权重: 0.4)
        for feat, score in mi_result['feature_scores'].items():
            feature_votes[feat] = feature_votes.get(feat, 0) + 0.4
            feature_scores[feat] = feature_scores.get(feat, 0) + score * 0.4
        
        # RFE分数 (权重: 0.3)
        for feat, ranking in rfe_result['feature_ranking'].items():
            if feat in corr_result['selected_features'] or feat in mi_result['selected_features']:
                feature_votes[feat] = feature_votes.get(feat, 0) + 0.3
                feature_scores[feat] = feature_scores.get(feat, 0) + (1.0 / ranking) * 0.3
        
        # 按投票分数排序
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前n_features个特征
        selected_features = [feat[0] for feat in sorted_features[:n_features]]
        
        self.logger.info(f"组合选择完成，选择了 {len(selected_features)} 个特征")
        
        return {
            'selected_features': selected_features,
            'feature_scores': dict(sorted_features[:n_features]),
            'feature_votes': dict(sorted_features[:n_features]),
            'method': 'combined',
            'correlation_result': corr_result,
            'mutual_info_result': mi_result,
            'rfe_result': rfe_result
        }
    
    def evaluate_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                                  selected_features: List[str]) -> Dict[str, Any]:
        """
        评估特征重要性
        
        Args:
            X: 特征矩阵
            y: 目标变量
            selected_features: 选择的特征列表
            
        Returns:
            特征重要性评估结果
        """
        self.logger.info("评估特征重要性...")
        
        # 使用选择的特征
        X_selected = X[selected_features]
        
        # 处理缺失值
        X_clean = X_selected.fillna(X_selected.median())
        
        # 使用XGBoost评估特征重要性
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(X_clean, y)
        
        # 获取特征重要性
        importance_scores = xgb_model.feature_importances_
        feature_importance = dict(zip(selected_features, importance_scores))
        
        # 按重要性排序
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 交叉验证评估
        cv_scores = cross_val_score(xgb_model, X_clean, y, cv=5, scoring='accuracy')
        
        self.logger.info(f"特征重要性评估完成，交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'feature_importance': dict(sorted_importance),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def create_feature_report(self, selection_result: Dict[str, Any], 
                            importance_result: Dict[str, Any]) -> pd.DataFrame:
        """
        创建特征选择报告
        
        Args:
            selection_result: 特征选择结果
            importance_result: 特征重要性结果
            
        Returns:
            特征选择报告DataFrame
        """
        report_data = []
        
        for feat in selection_result['selected_features']:
            report_data.append({
                'feature': feat,
                'selection_score': selection_result['feature_scores'].get(feat, 0),
                'importance_score': importance_result['feature_importance'].get(feat, 0),
                'vote_score': selection_result.get('feature_votes', {}).get(feat, 0)
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('importance_score', ascending=False)
        
        return report_df


