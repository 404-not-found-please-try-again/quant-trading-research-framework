"""
模型集成模块
结合多个模型的预测结果
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

class ModelEnsemble:
    """模型集成器类"""
    
    def __init__(self, config: dict):
        """
        初始化模型集成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        创建集成模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            
        Returns:
            集成模型和相关信息
        """
        self.logger.info("创建模型集成...")
        
        # 获取模型配置
        xgb_config = self.config['models']['xgboost']
        rf_config = self.config['models']['random_forest']
        
        # 创建基础模型
        xgb_model = xgb.XGBClassifier(
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            subsample=xgb_config['subsample'],
            colsample_bytree=xgb_config['colsample_bytree'],
            reg_alpha=xgb_config.get('reg_alpha', 0),
            reg_lambda=xgb_config.get('reg_lambda', 1),
            random_state=xgb_config['random_state'],
            eval_metric='mlogloss'
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            min_samples_split=rf_config.get('min_samples_split', 2),
            min_samples_leaf=rf_config.get('min_samples_leaf', 1),
            max_features=rf_config.get('max_features', 'auto'),
            random_state=rf_config['random_state'],
            n_jobs=-1
        )
        
        # 创建逻辑回归模型作为元学习器
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        # 创建投票分类器
        voting_classifier = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('lr', lr_model)
            ],
            voting='soft'  # 使用概率投票
        )
        
        # 训练集成模型
        voting_classifier.fit(X_train, y_train)
        
        # 获取各个子模型的性能
        individual_scores = {}
        for name, model in voting_classifier.named_estimators_.items():
            score = model.score(X_train, y_train)
            individual_scores[name] = score
            self.logger.info(f"{name} 训练准确率: {score:.4f}")
        
        return {
            'model': voting_classifier,
            'individual_scores': individual_scores,
            'model_type': 'ensemble'
        }
    
    def predict_ensemble(self, ensemble_model: Dict[str, Any], X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用集成模型进行预测
        
        Args:
            ensemble_model: 集成模型
            X: 特征数据
            
        Returns:
            预测结果和概率
        """
        model = ensemble_model['model']
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return predictions, probabilities
    
    def get_feature_importance_ensemble(self, ensemble_model: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
        """
        获取集成模型的特征重要性
        
        Args:
            ensemble_model: 集成模型
            feature_names: 特征名称列表
            
        Returns:
            特征重要性DataFrame
        """
        model = ensemble_model['model']
        
        # 计算各子模型特征重要性的平均值
        importance_dict = {}
        
        for name, estimator in model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importance_dict[name] = estimator.feature_importances_
            elif hasattr(estimator, 'coef_'):
                # 对于逻辑回归，使用系数的绝对值
                importance_dict[name] = np.abs(estimator.coef_[0])
        
        if importance_dict:
            # 计算平均重要性
            avg_importance = np.mean(list(importance_dict.values()), axis=0)
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return pd.DataFrame({'feature': feature_names, 'importance': [0] * len(feature_names)})
    
    def create_weighted_ensemble(self, models: Dict[str, Any], weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        创建加权集成模型
        
        Args:
            models: 训练好的模型字典
            weights: 模型权重字典
            
        Returns:
            加权集成模型
        """
        if weights is None:
            # 默认等权重
            weights = {name: 1.0 / len(models) for name in models.keys()}
        
        self.logger.info(f"创建加权集成模型，权重: {weights}")
        
        def weighted_predict(X):
            """加权预测函数"""
            predictions = []
            probabilities = []
            
            for name, model_info in models.items():
                model = model_info['model']
                pred = model.predict(X)
                prob = model.predict_proba(X)
                
                predictions.append(pred * weights[name])
                probabilities.append(prob * weights[name])
            
            # 计算加权平均
            weighted_pred = np.sum(predictions, axis=0)
            weighted_prob = np.sum(probabilities, axis=0)
            
            # 对于分类，需要转换为类别
            if weighted_pred.ndim == 1:
                weighted_pred = np.round(weighted_pred).astype(int)
            else:
                weighted_pred = np.argmax(weighted_pred, axis=1)
            
            return weighted_pred, weighted_prob
        
        return {
            'predict_func': weighted_predict,
            'weights': weights,
            'model_type': 'weighted_ensemble'
        }


