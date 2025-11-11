"""
超参数调优模块
使用网格搜索和随机搜索优化模型参数
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from datetime import datetime

class HyperparameterTuner:
    """超参数调优器类"""
    
    def __init__(self, config: dict):
        """
        初始化超参数调优器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 创建调优结果保存目录
        self.tuning_save_path = "results/hyperparameter_tuning"
        os.makedirs(self.tuning_save_path, exist_ok=True)
        
    def tune_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        调优XGBoost参数
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            调优结果
        """
        self.logger.info("开始XGBoost超参数调优...")
        
        # 定义更聚焦的参数网格（关注不平衡与核心结构）
        # 优先级6：基于用户反馈：max_depth=5, eta=0.05
        param_grid = {
            'n_estimators': [300, 500, 800, 1000],
            'max_depth': [3, 4, 5],  # 包含5（用户建议）
            'min_child_weight': [1, 3, 5],
            'learning_rate': [0.03, 0.05, 0.1],  # 包含0.05（用户建议的eta）
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1.0, 2.0, 3.0]
        }
        
        # 创建XGBoost分类器
        # 计算类别不平衡的scale_pos_weight（基于合并集）
        X_combined = pd.concat([X_train, X_val], ignore_index=True)
        y_combined = pd.concat([y_train, y_val], ignore_index=True)
        neg_count = (y_combined == 0).sum()
        pos_count = (y_combined == 1).sum()
        scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0
        
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='aucpr',
            scale_pos_weight=scale_pos_weight,
            tree_method='hist',
            n_jobs=1  # Windows修复：避免多线程问题
        )
        
        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)
        
        # 使用随机搜索（因为参数空间很大）
        # 改用f1_macro评分，更适合不平衡数据评估
        # Windows兼容性：n_jobs=1 避免多进程访问冲突
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=60,  # 增加搜索次数以找到更好的参数
            cv=tscv,
            scoring='f1_macro',  # 使用macro F1更适合不平衡数据
            n_jobs=1,  # Windows修复：单进程避免访问冲突
            random_state=42,
            verbose=1
        )
        
        # 执行搜索
        # 使用平均精度（PR-AUC）作为评分，贴合不平衡评估
        random_search.fit(X_combined, y_combined)
        
        # 获取最佳参数
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        self.logger.info(f"XGBoost最佳参数: {best_params}")
        self.logger.info(f"XGBoost最佳分数: {best_score:.4f}")
        
        # 使用最佳参数训练模型
        best_model = xgb.XGBClassifier(
            **best_params,
            random_state=42,
            eval_metric='aucpr',
            scale_pos_weight=scale_pos_weight,
            tree_method='hist',
            n_jobs=1  # Windows修复：避免多线程问题
        )
        best_model.fit(X_train, y_train)
        
        # 在验证集上评估
        val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1_weighted = f1_score(y_val, val_pred, average='weighted')
        val_f1_macro = f1_score(y_val, val_pred, average='macro')
        
        self.logger.info(f"XGBoost验证集准确率: {val_accuracy:.4f}")
        self.logger.info(f"XGBoost验证集F1(weighted): {val_f1_weighted:.4f}")
        self.logger.info(f"XGBoost验证集F1(macro): {val_f1_macro:.4f}")
        
        return {
            'model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1_macro,  # 使用macro F1
            'val_f1_weighted': val_f1_weighted,
            'search_results': random_search.cv_results_
        }
    
    def tune_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        调优RandomForest参数
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            调优结果
        """
        self.logger.info("开始RandomForest超参数调优...")
        
        # 定义小网格（加强类别平衡与核心深度/叶子数/树数的组合）
        # 基于用户反馈：n_estimators增加到1000，max_depth到15，min_samples_leaf到3
        param_grid = {
            'n_estimators': [500, 800, 1000, 1500],  # 增加树的数量
            'max_depth': [10, 15, 20],  # 增加到15
            'min_samples_leaf': [2, 3, 5],  # 调整到3附近
            'class_weight': ['balanced', 'balanced_subsample'],
            # 固定常见的健壮配置，减少搜索维度
            'max_features': ['sqrt'],
            'bootstrap': [True]
        }
        
        # 创建RandomForest分类器
        # Windows修复：n_jobs=1 避免多进程问题
        rf_model = RandomForestClassifier(random_state=42, n_jobs=1)
        
        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)
        
        # 使用随机搜索（小网格下仍采用随机搜索以节省时间）
        # 改用f1_macro评分，更适合不平衡数据评估
        # Windows兼容性：n_jobs=1 避免多进程访问冲突
        random_search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_grid,
            n_iter=30,  # 增加搜索次数
            cv=tscv,
            scoring='f1_macro',  # 使用macro F1更适合不平衡数据
            n_jobs=1,  # Windows修复：单进程避免访问冲突
            random_state=42,
            verbose=1
        )
        
        # 合并训练和验证数据
        X_combined = pd.concat([X_train, X_val], ignore_index=True)
        y_combined = pd.concat([y_train, y_val], ignore_index=True)
        
        # 执行搜索
        random_search.fit(X_combined, y_combined)
        
        # 获取最佳参数
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        self.logger.info(f"RandomForest最佳参数: {best_params}")
        self.logger.info(f"RandomForest最佳分数: {best_score:.4f}")
        
        # 使用最佳参数训练模型
        # Windows修复：n_jobs=1 避免多进程问题
        best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=1)
        best_model.fit(X_train, y_train)
        
        # 在验证集上评估
        val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1_weighted = f1_score(y_val, val_pred, average='weighted')
        val_f1_macro = f1_score(y_val, val_pred, average='macro')
        
        self.logger.info(f"RandomForest验证集准确率: {val_accuracy:.4f}")
        self.logger.info(f"RandomForest验证集F1(weighted): {val_f1_weighted:.4f}")
        self.logger.info(f"RandomForest验证集F1(macro): {val_f1_macro:.4f}")
        
        return {
            'model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1_macro,  # 使用macro F1
            'val_f1_weighted': val_f1_weighted,
            'search_results': random_search.cv_results_
        }
    
    def tune_ensemble_weights(self, models: Dict[str, Any], X_val: pd.DataFrame, 
                             y_val: pd.Series) -> Dict[str, Any]:
        """
        调优集成模型权重
        
        Args:
            models: 训练好的模型字典
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            最佳权重配置
        """
        self.logger.info("开始调优集成模型权重...")
        
        # 获取各模型的预测概率
        model_predictions = {}
        for name, model_info in models.items():
            model = model_info['model']
            pred_proba = model.predict_proba(X_val)
            model_predictions[name] = pred_proba
        
        # 定义权重搜索空间
        weight_ranges = {
            'xgboost': (0.1, 0.6),
            'random_forest': (0.1, 0.6)
        }
        
        best_score = 0
        best_weights = {}
        
        # 简单的网格搜索
        for xgb_weight in np.arange(0.1, 0.6, 0.1):
            for rf_weight in np.arange(0.1, 0.6, 0.1):
                # 确保权重和为1
                total_weight = xgb_weight + rf_weight
                if total_weight > 1.0:
                    continue
                
                # 归一化权重
                xgb_norm = xgb_weight / total_weight
                rf_norm = rf_weight / total_weight
                
                # 计算加权预测
                weighted_pred = (xgb_norm * model_predictions['xgboost'] + 
                               rf_norm * model_predictions['random_forest'])
                
                # 转换为预测类别
                pred_classes = np.argmax(weighted_pred, axis=1)
                
                # 计算F1分数
                f1 = f1_score(y_val, pred_classes, average='weighted')
                
                if f1 > best_score:
                    best_score = f1
                    best_weights = {
                        'xgboost': xgb_norm,
                        'random_forest': rf_norm
                    }
        
        self.logger.info(f"最佳权重: {best_weights}")
        self.logger.info(f"最佳F1分数: {best_score:.4f}")
        
        return {
            'best_weights': best_weights,
            'best_score': best_score
        }
    
    def save_tuning_results(self, xgb_result: Dict[str, Any], rf_result: Dict[str, Any],
                           ensemble_result: Dict[str, Any] = None):
        """
        保存调优结果
        
        Args:
            xgb_result: XGBoost调优结果
            rf_result: RandomForest调优结果
            ensemble_result: 集成模型调优结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存XGBoost结果
        xgb_summary = {
            'model': 'XGBoost',
            'best_params': xgb_result['best_params'],
            'best_score': xgb_result['best_score'],
            'val_accuracy': xgb_result['val_accuracy'],
            'val_f1': xgb_result['val_f1']
        }
        
        xgb_df = pd.DataFrame([xgb_summary])
        xgb_path = os.path.join(self.tuning_save_path, f"xgboost_tuning_{timestamp}.csv")
        xgb_df.to_csv(xgb_path, index=False)
        
        # 保存RandomForest结果
        rf_summary = {
            'model': 'RandomForest',
            'best_params': rf_result['best_params'],
            'best_score': rf_result['best_score'],
            'val_accuracy': rf_result['val_accuracy'],
            'val_f1': rf_result['val_f1']
        }
        
        rf_df = pd.DataFrame([rf_summary])
        rf_path = os.path.join(self.tuning_save_path, f"random_forest_tuning_{timestamp}.csv")
        rf_df.to_csv(rf_path, index=False)
        
        # 保存集成模型结果
        if ensemble_result:
            ensemble_df = pd.DataFrame([ensemble_result])
            ensemble_path = os.path.join(self.tuning_save_path, f"ensemble_tuning_{timestamp}.csv")
            ensemble_df.to_csv(ensemble_path, index=False)
        
        self.logger.info(f"调优结果已保存到: {self.tuning_save_path}")
    
    def create_tuning_report(self, xgb_result: Dict[str, Any], rf_result: Dict[str, Any],
                           ensemble_result: Dict[str, Any] = None) -> pd.DataFrame:
        """
        创建调优报告
        
        Args:
            xgb_result: XGBoost调优结果
            rf_result: RandomForest调优结果
            ensemble_result: 集成模型调优结果
            
        Returns:
            调优报告DataFrame
        """
        report_data = [
            {
                'Model': 'XGBoost',
                'Best_Score': xgb_result['best_score'],
                'Val_Accuracy': xgb_result['val_accuracy'],
                'Val_F1': xgb_result['val_f1'],
                'Best_Params': str(xgb_result['best_params'])
            },
            {
                'Model': 'RandomForest',
                'Best_Score': rf_result['best_score'],
                'Val_Accuracy': rf_result['val_accuracy'],
                'Val_F1': rf_result['val_f1'],
                'Best_Params': str(rf_result['best_params'])
            }
        ]
        
        if ensemble_result:
            report_data.append({
                'Model': 'Ensemble',
                'Best_Score': ensemble_result['best_score'],
                'Val_Accuracy': 'N/A',
                'Val_F1': ensemble_result['best_score'],
                'Best_Params': str(ensemble_result['best_weights'])
            })
        
        return pd.DataFrame(report_data)

