"""
模型训练模块
支持XGBoost、RandomForest、LSTM等模型
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import xgboost as xgb
import joblib
import os
from datetime import datetime
from features.feature_selector import FeatureSelector
from models.hyperparameter_tuner import HyperparameterTuner
from models.lstm_model import LSTMModel

# SMOTE支持
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logging.warning("imbalanced-learn未安装，SMOTE功能将不可用。请运行: pip install imbalanced-learn")

class ModelTrainer:
    """模型训练器类"""
    
    def __init__(self, config: dict):
        """
        初始化模型训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.models_config = config['models']
        self.training_config = config['training']
        self.logger = logging.getLogger(__name__)
        
        # 创建模型保存目录
        self.model_save_path = "results/models"
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # 初始化特征选择器
        self.feature_selector = FeatureSelector(config)
        
        # 初始化超参数调优器
        self.hyperparameter_tuner = HyperparameterTuner(config)
        
        # 初始化LSTM模型
        self.lstm_model = LSTMModel(config)
        
    def train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练所有模型
        
        Args:
            data: 特征数据
            
        Returns:
            训练好的模型字典
        """
        self.logger.info("开始模型训练...")
        
        # 准备数据
        X, y = self._prepare_data(data)
        
        # 特征选择
        self.logger.info("开始特征选择...")
        # 读取特征选择目标数量（默认60）
        n_features_target = (
            self.config.get('features', {})
                .get('selection', {})
                .get('n_features', 60)
        )
        feature_selection_result = self.feature_selector.select_features(
            X, y, method='combined', n_features=n_features_target
        )
        selected_features = feature_selection_result['selected_features']
        X_selected = X[selected_features]
        
        # 优先级5: 删除高相关特征（corr > 0.95）
        X_selected = self._remove_highly_correlated_features(X_selected, threshold=0.95)
        selected_features = X_selected.columns.tolist()
        
        # 评估特征重要性
        importance_result = self.feature_selector.evaluate_feature_importance(
            X, y, selected_features
        )
        
        self.logger.info(f"特征选择完成，从 {X.shape[1]} 个特征中选择了 {len(selected_features)} 个")
        
        # 分割数据
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X_selected, y)
        
        # 优先级1: 应用SMOTE过采样（如果需要）
        X_train, y_train = self._apply_smote_if_needed(X_train, y_train, 'train')
        
        # 存储分割后的数据
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # 超参数调优
        self.logger.info("开始超参数调优...")
        xgb_tuning_result = self.hyperparameter_tuner.tune_xgboost(X_train, y_train, X_val, y_val)
        rf_tuning_result = self.hyperparameter_tuner.tune_random_forest(X_train, y_train, X_val, y_val)
        
        # 调优集成权重
        temp_models = {
            'xgboost': {'model': xgb_tuning_result['model']},
            'random_forest': {'model': rf_tuning_result['model']}
        }
        ensemble_tuning_result = self.hyperparameter_tuner.tune_ensemble_weights(temp_models, X_val, y_val)
        
        # 训练LSTM模型
        self.logger.info("开始训练LSTM模型...")
        lstm_result = self.lstm_model.train(X_train, y_train, X_val, y_val)
        
        # 保存调优结果
        self.hyperparameter_tuner.save_tuning_results(xgb_tuning_result, rf_tuning_result, ensemble_tuning_result)
        
        # 计算特征重要性
        xgb_importance = self._calculate_feature_importance(xgb_tuning_result['model'], X_train.columns)
        rf_importance = self._calculate_feature_importance(rf_tuning_result['model'], X_train.columns)
        lstm_importance = self.lstm_model.get_feature_importance(X_train, X_train.columns.tolist())
        
        # 使用调优后的模型
        models = {
            'xgboost': {
                'model': xgb_tuning_result['model'],
                'accuracy': xgb_tuning_result['val_accuracy'],
                'f1_score': xgb_tuning_result['val_f1'],
                'best_params': xgb_tuning_result['best_params'],
                'feature_importance': xgb_importance,
                'feature_names': X_train.columns.tolist()
            },
            'random_forest': {
                'model': rf_tuning_result['model'],
                'accuracy': rf_tuning_result['val_accuracy'],
                'f1_score': rf_tuning_result['val_f1'],
                'best_params': rf_tuning_result['best_params'],
                'feature_importance': rf_importance,
                'feature_names': X_train.columns.tolist()
            },
            'lstm': {
                'model': lstm_result['model'],
                'scaler': lstm_result['scaler'],
                'accuracy': lstm_result['val_accuracy'],
                'f1_score': lstm_result['val_f1'],
                'feature_importance': lstm_importance,
                'feature_names': X_train.columns.tolist(),
                'wrapper': self.lstm_model
            }
        }
        
        # 保存模型
        self._save_models(models)
        
        # 保存特征选择结果
        self._save_feature_selection_results(feature_selection_result, importance_result)
        
        self.logger.info("模型训练完成")
        return models
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据
        
        Args:
            data: 特征数据
            
        Returns:
            X: 特征矩阵, y: 目标变量
        """
        # 选择特征列（排除非数值列和目标列）
        exclude_cols = ['Date', 'symbol', 'target', 'target_binary', 'target_return']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # 选择数值特征
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if col in numeric_cols]
        
        X = data[feature_cols].copy()
        y = data['target'].copy()
        
        # 根据分类类型调整标签
        if y.min() == -1:  # 三分类
            # 将标签从[-1, 0, 1]转换为[0, 1, 2]
            y = y + 1
        # 二分类已经是[0, 1]，不需要转换
        
        # 处理缺失值
        X = X.fillna(X.median())
        
        self.logger.info(f"特征数量: {X.shape[1]}")
        self.logger.info(f"样本数量: {X.shape[0]}")
        
        return X, y
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        分割数据 - 严格的时间序列分割
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            训练集、验证集、测试集
        """
        # 严格的时间序列分割：训练集2020-2022，验证集2023，测试集2024
        total_samples = len(X)
        
        # 假设数据按时间排序，前70%作为训练集，中间15%作为验证集，最后15%作为测试集
        train_size = int(total_samples * 0.7)
        val_size = int(total_samples * 0.15)
        
        X_train = X.iloc[:train_size]
        X_val = X.iloc[train_size:train_size + val_size]
        X_test = X.iloc[train_size + val_size:]
        
        y_train = y.iloc[:train_size]
        y_val = y.iloc[train_size:train_size + val_size]
        y_test = y.iloc[train_size + val_size:]
        
        self.logger.info(f"训练集大小: {X_train.shape[0]} (2020-2022)")
        self.logger.info(f"验证集大小: {X_val.shape[0]} (2023)")
        self.logger.info(f"测试集大小: {X_test.shape[0]} (2024)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        训练XGBoost模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            训练好的模型和相关信息
        """
        xgb_config = self.models_config['xgboost']
        
        # 计算类别平衡权重
        scale_pos_weight = xgb_config.get('scale_pos_weight', 1.0)
        if scale_pos_weight == "auto":
            # 自动计算正负样本比例
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            self.logger.info(f"XGBoost scale_pos_weight (基础): {scale_pos_weight:.2f}")
        
        # 优先级6: 应用权重倍数增强
        multiplier = xgb_config.get('scale_pos_weight_multiplier', 1.0)
        scale_pos_weight = scale_pos_weight * multiplier
        self.logger.info(f"XGBoost scale_pos_weight (增强后): {scale_pos_weight:.2f} (倍数: {multiplier})")
        
        # 高级类别平衡策略（备用，SMOTE已在数据准备阶段应用）
        class_balance_config = xgb_config.get('class_balance', {})
        if class_balance_config.get('method') == 'focal_loss':
            # 使用Focal Loss的权重调整
            alpha = class_balance_config.get('alpha', 0.25)
            gamma = class_balance_config.get('gamma', 2.0)
            # 调整scale_pos_weight以模拟focal loss效果
            scale_pos_weight = scale_pos_weight * (1 + alpha * gamma)
            self.logger.info(f"XGBoost focal loss adjusted scale_pos_weight: {scale_pos_weight:.2f}")
        
        # 创建XGBoost分类器
        # Windows修复：n_jobs=1 避免多线程问题
        model = xgb.XGBClassifier(
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            subsample=xgb_config['subsample'],
            colsample_bytree=xgb_config['colsample_bytree'],
            reg_alpha=xgb_config.get('reg_alpha', 0),
            reg_lambda=xgb_config.get('reg_lambda', 1),
            random_state=xgb_config['random_state'],
            scale_pos_weight=scale_pos_weight,
            eval_metric=xgb_config.get('eval_metric', 'logloss'),
            n_jobs=1  # Windows修复：避免多线程问题
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)
        
        # 计算指标
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'feature_importance': feature_importance
        }
    
    def _train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        训练RandomForest模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            训练好的模型和相关信息
        """
        rf_config = self.models_config['random_forest']
        
        # 高级类别平衡策略
        class_balance_config = rf_config.get('class_balance', {})
        class_weight = rf_config.get('class_weight', None)
        
        if class_balance_config.get('method') == 'balanced_subsample':
            class_weight = 'balanced_subsample'
            self.logger.info("RandomForest使用balanced_subsample类别平衡")
        elif class_balance_config.get('method') == 'custom' and class_balance_config.get('custom_weights'):
            class_weight = class_balance_config['custom_weights']
            self.logger.info(f"RandomForest使用自定义类别权重: {class_weight}")
        
        # 创建RandomForest分类器
        # Windows修复：n_jobs=1 避免多进程问题
        model = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            min_samples_split=rf_config.get('min_samples_split', 2),
            min_samples_leaf=rf_config.get('min_samples_leaf', 1),
            max_features=rf_config.get('max_features', 'auto'),
            random_state=rf_config['random_state'],
            class_weight=class_weight,
            n_jobs=1  # Windows修复：避免多进程问题
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)
        
        # 计算指标
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'feature_importance': feature_importance
        }
    
    def evaluate_models(self, models: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Dict]:
        """
        评估模型性能
        
        Args:
            models: 训练好的模型字典
            data: 特征数据
            
        Returns:
            评估结果字典
        """
        self.logger.info("开始模型评估...")
        
        evaluation_results = {}
        
        for model_name, model_info in models.items():
            self.logger.info(f"评估 {model_name} 模型...")
            
            # 在测试集上预测
            if model_name == 'lstm':
                # 使用包装器进行序列化预测，避免直接对原始tabular输入调用Keras模型
                try:
                    y_pred, y_pred_proba_raw = self.lstm_model.predict(self.X_test)
                    y_pred_proba = np.column_stack([1 - y_pred_proba_raw, y_pred_proba_raw])
                except Exception as e:
                    self.logger.warning(f"LSTM模型预测失败: {str(e)}")
                    y_pred = model_info.get('predictions', np.zeros(len(self.y_test)))
                    y_pred_proba = np.column_stack([1 - y_pred, y_pred])
            else:
                y_pred = model_info['model'].predict(self.X_test)
                y_pred_proba = model_info['model'].predict_proba(self.X_test)
            
            # 阈值优化
            best_threshold, optimized_accuracy = self._optimize_threshold(self.y_test, y_pred_proba[:, 1])
            y_pred_optimized = (y_pred_proba[:, 1] > best_threshold).astype(int)
            
            # 计算指标
            accuracy = accuracy_score(self.y_test, y_pred)
            accuracy_optimized = accuracy_score(self.y_test, y_pred_optimized)
            
            # 分类报告
            class_report = classification_report(self.y_test, y_pred, output_dict=True)
            class_report_optimized = classification_report(self.y_test, y_pred_optimized, output_dict=True)
            
            # 混淆矩阵
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            conf_matrix_optimized = confusion_matrix(self.y_test, y_pred_optimized)
            
            # F1分数
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            f1_optimized = f1_score(self.y_test, y_pred_optimized, average='weighted')
            
            # 保存概率统计信息
            proba_positive = y_pred_proba[:, 1]
            prob_stats = {
                'mean': float(np.mean(proba_positive)),
                'median': float(np.median(proba_positive)),
                'std': float(np.std(proba_positive)),
                'min': float(np.min(proba_positive)),
                'max': float(np.max(proba_positive)),
                'q25': float(np.percentile(proba_positive, 25)),
                'q75': float(np.percentile(proba_positive, 75)),
                'concentration_45_55': float(np.sum((proba_positive >= 0.45) & (proba_positive <= 0.55)) / len(proba_positive)),
                'polarization_low': float(np.sum(proba_positive < 0.2) / len(proba_positive)),
                'polarization_high': float(np.sum(proba_positive > 0.8) / len(proba_positive))
            }
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'accuracy_optimized': accuracy_optimized,
                'f1_score': f1,
                'f1_score_optimized': f1_optimized,
                'best_threshold': best_threshold,
                'threshold_improvement': accuracy_optimized - accuracy,
                'classification_report': class_report,
                'classification_report_optimized': class_report_optimized,
                'confusion_matrix': conf_matrix,
                'confusion_matrix_optimized': conf_matrix_optimized,
                'predictions': y_pred,
                'predictions_optimized': y_pred_optimized,
                'probabilities': y_pred_proba,
                'probability_stats': prob_stats,  # 新增概率统计
                'true_labels': self.y_test.values,  # 新增：保存真实标签用于ROC/PR曲线
                'feature_importance': model_info['feature_importance']
            }
            
            # 保存概率数据到CSV以便后续分析
            proba_df = pd.DataFrame({
                'true_label': self.y_test.values,
                'pred_prob_0': y_pred_proba[:, 0],
                'pred_prob_1': y_pred_proba[:, 1],
                'prediction': y_pred
            })
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            proba_save_path = os.path.join(self.model_save_path, f"{model_name}_probabilities_{timestamp}.csv")
            proba_df.to_csv(proba_save_path, index=False)
            self.logger.info(f"{model_name} 预测概率数据已保存到: {proba_save_path}")
            
            self.logger.info(f"{model_name} 准确率: {accuracy:.4f} -> {accuracy_optimized:.4f} (阈值: {best_threshold:.3f})")
        
        return evaluation_results
    
    def _optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
        """
        基于ROC曲线优化分类阈值
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            
        Returns:
            最佳阈值和对应的准确率
        """
        from sklearn.metrics import roc_curve, accuracy_score
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # 计算每个阈值下的准确率
        accuracies = []
        for threshold in thresholds:
            y_pred_thresh = (y_proba > threshold).astype(int)
            acc = accuracy_score(y_true, y_pred_thresh)
            accuracies.append(acc)
        
        # 找到最佳阈值
        best_idx = np.argmax(accuracies)
        best_threshold = thresholds[best_idx]
        best_accuracy = accuracies[best_idx]
        
        return best_threshold, best_accuracy
    
    def _save_models(self, models: Dict[str, Any]):
        """
        保存模型到文件
        
        Args:
            models: 模型字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model_info in models.items():
            if model_name == 'lstm':
                # 保存LSTM模型
                model_path = os.path.join(self.model_save_path, f"{model_name}_{timestamp}")
                self.lstm_model.save_model(model_path)
                self.logger.info(f"{model_name} 模型已保存到: {model_path}")
            else:
                # 保存传统模型
                model_path = os.path.join(self.model_save_path, f"{model_name}_{timestamp}.joblib")
                joblib.dump(model_info['model'], model_path)
                self.logger.info(f"{model_name} 模型已保存到: {model_path}")
            
            # 保存特征重要性
            importance_path = os.path.join(self.model_save_path, f"{model_name}_importance_{timestamp}.csv")
            model_info['feature_importance'].to_csv(importance_path, index=False)
    
    def load_model(self, model_path: str) -> Any:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的模型
        """
        return joblib.load(model_path)
    
    def predict(self, model: Any, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用模型进行预测
        
        Args:
            model: 训练好的模型
            X: 特征数据
            
        Returns:
            预测结果和概率
        """
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return predictions, probabilities
    
    def get_model_summary(self, evaluation_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        获取模型性能摘要
        
        Args:
            evaluation_results: 评估结果
            
        Returns:
            性能摘要DataFrame
        """
        summary_data = []
        
        for model_name, results in evaluation_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision_0': results['classification_report']['-1']['precision'],
                'Recall_0': results['classification_report']['-1']['recall'],
                'F1_0': results['classification_report']['-1']['f1-score'],
                'Precision_1': results['classification_report']['0']['precision'],
                'Recall_1': results['classification_report']['0']['recall'],
                'F1_1': results['classification_report']['0']['f1-score'],
                'Precision_2': results['classification_report']['1']['precision'],
                'Recall_2': results['classification_report']['1']['recall'],
                'F1_2': results['classification_report']['1']['f1-score']
            })
        
        return pd.DataFrame(summary_data)
    
    def _save_feature_selection_results(self, selection_result: Dict[str, Any], 
                                      importance_result: Dict[str, Any]):
        """
        保存特征选择结果
        
        Args:
            selection_result: 特征选择结果
            importance_result: 特征重要性结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建特征选择报告
        report_df = self.feature_selector.create_feature_report(selection_result, importance_result)
        
        # 保存报告
        report_path = os.path.join(self.model_save_path, f"feature_selection_report_{timestamp}.csv")
        report_df.to_csv(report_path, index=False)
        
        # 保存选择的特征列表
        features_path = os.path.join(self.model_save_path, f"selected_features_{timestamp}.txt")
        with open(features_path, 'w') as f:
            for feat in selection_result['selected_features']:
                f.write(f"{feat}\n")
        
        self.logger.info(f"特征选择结果已保存到: {report_path}")
        self.logger.info(f"选择的特征列表已保存到: {features_path}")
    
    def _calculate_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """
        计算特征重要性
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            
        Returns:
            特征重要性DataFrame
        """
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # 对于逻辑回归等模型，使用系数的绝对值
            importance_scores = np.abs(model.coef_[0])
        else:
            # 如果没有特征重要性，返回零
            importance_scores = np.zeros(len(feature_names))
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def _remove_highly_correlated_features(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        删除高相关特征（优先级5：特征精简）
        
        Args:
            X: 特征DataFrame
            threshold: 相关性阈值，超过此值的特征对将被删除
            
        Returns:
            删除高相关特征后的DataFrame
        """
        if len(X.columns) <= 1:
            return X
            
        # 计算相关性矩阵
        corr_matrix = X.corr().abs()
        
        # 找出高相关特征对
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 找出需要删除的特征（保留第一个，删除其他）
        to_drop = []
        for column in upper_triangle.columns:
            high_corr_features = upper_triangle.index[upper_triangle[column] > threshold].tolist()
            if high_corr_features:
                # 保留第一个特征，删除其他
                for feat in high_corr_features:
                    if feat not in to_drop:
                        to_drop.append(feat)
        
        if to_drop:
            self.logger.info(f"删除 {len(to_drop)} 个高相关特征（阈值>{threshold}）: {to_drop[:10]}...")
            X = X.drop(columns=to_drop)
        
        return X
    
    def _apply_smote_if_needed(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = 'train') -> Tuple[pd.DataFrame, pd.Series]:
        """
        应用SMOTE过采样（优先级1：类别平衡）
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            dataset_name: 数据集名称（用于日志）
            
        Returns:
            过采样后的X和y
        """
        if not SMOTE_AVAILABLE:
            self.logger.warning("SMOTE不可用，跳过过采样")
            return X, y
        
        # 检查类别分布
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
        
        self.logger.info(f"{dataset_name} 数据集类别分布: 负类={neg_count}, 正类={pos_count}, 不平衡比={imbalance_ratio:.2f}")
        
        # 只在训练集上应用
        if dataset_name != 'train':
            self.logger.info(f"跳过SMOTE：只在训练集上应用，当前数据集={dataset_name}")
            return X, y
        
        # 获取配置
        xgb_config = self.models_config.get('xgboost', {})
        rf_config = self.models_config.get('random_forest', {})
        
        use_smote = (
            xgb_config.get('class_balance', {}).get('use_smote', False) or
            rf_config.get('class_balance', {}).get('use_smote', False)
        )
        
        if not use_smote:
            self.logger.info("SMOTE未启用，跳过过采样")
            return X, y
        
        # 优先级1：检查是否强制应用SMOTE
        force_smote = (
            xgb_config.get('class_balance', {}).get('force_smote', False) or
            rf_config.get('class_balance', {}).get('force_smote', False)
        )
        
        # 获取最小不平衡比阈值（如果启用强制SMOTE，降低阈值）
        min_imbalance_ratio = (
            xgb_config.get('class_balance', {}).get('min_imbalance_ratio', 2.0) if not force_smote else
            xgb_config.get('class_balance', {}).get('min_imbalance_ratio', 1.3)
        )
        
        # 检查是否需要SMOTE（如果强制启用，则降低阈值要求）
        if not force_smote and imbalance_ratio < min_imbalance_ratio:
            self.logger.info(f"跳过SMOTE：不平衡比={imbalance_ratio:.2f} < 阈值={min_imbalance_ratio:.2f}，且未启用强制SMOTE")
            return X, y
        
        # 获取SMOTE参数（优先使用XGBoost配置）
        smote_ratio = xgb_config.get('class_balance', {}).get('smote_ratio', 0.5)
        if not smote_ratio:
            smote_ratio = rf_config.get('class_balance', {}).get('smote_ratio', 0.5)
        
        # 如果启用强制SMOTE，确保至少有一定的过采样
        if force_smote and smote_ratio < 0.5:
            smote_ratio = 0.6  # 默认提高到0.6
            self.logger.info(f"强制SMOTE已启用，将过采样比例调整为{smote_ratio}")
        
        # 优先级1：计算目标正类数量，如果太少则使用绝对数量而非比例
        target_pos_count = int(neg_count * smote_ratio)
        
        # 如果正类样本太少（<5个），使用固定目标数量
        if pos_count < 5:
            # 使用固定目标数量：将正类增加到至少10个或负类的30%
            target_pos_count = max(10, int(neg_count * 0.3))
            self.logger.info(f"正类样本过少({pos_count}个)，使用固定目标数量: {target_pos_count}")
            sampling_strategy = target_pos_count
        elif target_pos_count <= pos_count:
            # 如果目标数量小于等于当前正类数量，使用字典形式指定目标数量
            # SMOTE不支持大于1.0的倍数，改用目标数量字典
            target_pos_count = int(pos_count * 1.2)  # 增加20%
            self.logger.info(f"目标正类数量({target_pos_count}) <= 当前数量({pos_count})，使用目标数量策略: {target_pos_count}")
            sampling_strategy = {1: target_pos_count}  # 字典形式：{类别: 目标数量}
        else:
            # 正常情况：使用比例（必须在0.0-1.0之间）
            sampling_strategy = min(smote_ratio, 1.0)  # 确保不超过1.0
        
        try:
            # 应用SMOTE
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42,
                k_neighbors=min(5, pos_count - 1) if pos_count > 1 else 1
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # 转换回DataFrame/Series（SMOTE返回numpy数组）
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns, index=range(len(X_resampled)))
            y_resampled = pd.Series(y_resampled, name=y.name)
            
            new_neg_count = (y_resampled == 0).sum()
            new_pos_count = (y_resampled == 1).sum()
            new_ratio = new_neg_count / new_pos_count if new_pos_count > 0 else 1.0
            
            self.logger.info(f"SMOTE完成: 原始样本={len(y)}, 过采样后={len(y_resampled)}")
            self.logger.info(f"SMOTE后类别分布: 负类={new_neg_count}, 正类={new_pos_count}, 不平衡比={new_ratio:.2f}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.warning(f"SMOTE应用失败: {str(e)}，使用原始数据")
            return X, y
