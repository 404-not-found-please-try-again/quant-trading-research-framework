"""
可视化模块
创建各种图表和可视化
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Any
import os
from datetime import datetime

class Plotter:
    """可视化类"""
    
    def __init__(self, config: dict):
        """
        初始化可视化器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.viz_config = config['visualization']
        self.logger = logging.getLogger(__name__)
        
        # 设置matplotlib样式
        plt.style.use(self.viz_config.get('style', 'default'))
        
        # 创建保存目录
        self.save_path = self.viz_config.get('save_path', 'results/plots/')
        os.makedirs(self.save_path, exist_ok=True)
        
    def create_plots(self, data: pd.DataFrame, evaluation_results: Dict[str, Dict]):
        """
        创建所有可视化图表
        
        Args:
            data: 特征数据
            evaluation_results: 模型评估结果
        """
        self.logger.info("开始创建可视化图表...")
        
        # 数据概览图
        self._plot_data_overview(data)
        
        # 特征分布图
        self._plot_feature_distributions(data)
        
        # 模型性能对比
        self._plot_model_performance(evaluation_results)
        
        # 混淆矩阵
        self._plot_confusion_matrices(evaluation_results)
        
        # 特征重要性
        self._plot_feature_importance(evaluation_results)
        
        # 预测结果分析
        self._plot_predictions_analysis(evaluation_results)
        
        # 详细概率分布分析（新增）
        self.plot_probability_distribution_detailed(evaluation_results)
        
        # ROC曲线和PR曲线分析（评估类别平衡改善）
        self.plot_roc_pr_curves(evaluation_results)
        
        # 类别平衡评估详细分析
        self.plot_class_balance_evaluation(evaluation_results)
        
        self.logger.info("可视化图表创建完成")
    
    def _plot_data_overview(self, data: pd.DataFrame):
        """创建数据概览图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('数据概览', fontsize=16)
        
        # 1. 股票价格走势
        ax1 = axes[0, 0]
        for symbol in data['symbol'].unique()[:5]:  # 只显示前5只股票
            symbol_data = data[data['symbol'] == symbol]
            ax1.plot(symbol_data['Date'], symbol_data['close'], label=symbol, alpha=0.7)
        ax1.set_title('股票价格走势')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('收盘价')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 成交量分布
        ax2 = axes[0, 1]
        ax2.hist(data['volume'], bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('成交量分布')
        ax2.set_xlabel('成交量')
        ax2.set_ylabel('频次')
        
        # 3. 日收益率分布
        ax3 = axes[1, 0]
        ax3.hist(data['daily_return'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax3.set_title('日收益率分布')
        ax3.set_xlabel('日收益率')
        ax3.set_ylabel('频次')
        
        # 4. 标签分布
        ax4 = axes[1, 1]
        label_counts = data['target'].value_counts()
        ax4.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
        ax4.set_title('标签分布')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'data_overview.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_distributions(self, data: pd.DataFrame):
        """创建特征分布图"""
        # 选择数值特征
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['target', 'target_binary', 'target_return']]
        
        # 选择前12个最重要的特征
        if len(feature_cols) > 12:
            # 计算与目标的相关性
            correlations = data[feature_cols + ['target']].corr()['target'].abs().sort_values(ascending=False)
            top_features = correlations.index[1:13].tolist()  # 排除target本身
        else:
            top_features = feature_cols[:12]
        
        # 创建子图
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('重要特征分布', fontsize=16)
        
        for i, feature in enumerate(top_features):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            # 绘制分布图
            ax.hist(data[feature].dropna(), bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{feature}')
            ax.set_xlabel('值')
            ax.set_ylabel('频次')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_model_performance(self, evaluation_results: Dict[str, Dict]):
        """创建模型性能对比图"""
        # 提取性能指标
        models = list(evaluation_results.keys())
        accuracies = [results['accuracy'] for results in evaluation_results.values()]
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        ax.set_title('模型准确率对比', fontsize=14)
        ax.set_xlabel('模型')
        ax.set_ylabel('准确率')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'model_performance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrices(self, evaluation_results: Dict[str, Dict]):
        """创建混淆矩阵图"""
        n_models = len(evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            conf_matrix = results['confusion_matrix']
            
            # 绘制混淆矩阵
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name} 混淆矩阵')
            axes[i].set_xlabel('预测值')
            axes[i].set_ylabel('真实值')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_importance(self, evaluation_results: Dict[str, Dict]):
        """创建特征重要性图"""
        for model_name, results in evaluation_results.items():
            importance_df = results['feature_importance']
            
            # 选择前20个最重要的特征
            top_features = importance_df.head(20)
            
            # 创建水平柱状图
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('重要性')
            ax.set_title(f'{model_name} 特征重要性 (前20)')
            
            # 反转y轴，使最重要的特征在顶部
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f'{model_name}_feature_importance.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def _plot_predictions_analysis(self, evaluation_results: Dict[str, Dict]):
        """创建预测结果分析图（包含详细的概率分布分析）"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('预测结果分析', fontsize=16)
        
        # 1. 预测概率分布（二分类 - 正类概率，bins=50）
        ax1 = axes[0, 0]
        for model_name, results in evaluation_results.items():
            probabilities = results['probabilities']
            # 二分类：使用正类概率（[:, 1]）
            if probabilities.shape[1] >= 2:
                proba_positive = probabilities[:, 1]
                # 计算统计信息
                mean_prob = np.mean(proba_positive)
                std_prob = np.std(proba_positive)
                median_prob = np.median(proba_positive)
                # 绘制直方图
                ax1.hist(proba_positive, bins=50, alpha=0.6, label=f'{model_name}\n(均值:{mean_prob:.3f}, 中位:{median_prob:.3f})', density=True)
                # 添加均值线
                ax1.axvline(mean_prob, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.set_title('预测概率分布（正类概率，bins=50）', fontsize=12)
        ax1.set_xlabel('预测概率 (P(y=1))')
        ax1.set_ylabel('密度')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(0.5, color='black', linestyle=':', linewidth=1, label='默认阈值(0.5)')
        
        # 2. 预测概率分布箱线图
        ax2 = axes[0, 1]
        proba_data = []
        model_labels = []
        for model_name, results in evaluation_results.items():
            probabilities = results['probabilities']
            if probabilities.shape[1] >= 2:
                proba_positive = probabilities[:, 1]
                proba_data.append(proba_positive)
                model_labels.append(model_name)
        if proba_data:
            bp = ax2.boxplot(proba_data, labels=model_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            ax2.axhline(0.5, color='red', linestyle='--', linewidth=1, label='默认阈值(0.5)')
            ax2.set_title('预测概率分布箱线图')
            ax2.set_ylabel('预测概率 (P(y=1))')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 模型置信度分布
        ax3 = axes[1, 0]
        for model_name, results in evaluation_results.items():
            probabilities = results['probabilities']
            if probabilities.shape[1] >= 2:
                # 二分类：使用正类概率的绝对值距离0.5作为置信度
                proba_positive = probabilities[:, 1]
                confidence = np.abs(proba_positive - 0.5) * 2  # 归一化到0-1
                ax3.hist(confidence, bins=50, alpha=0.6, label=model_name, density=True)
        ax3.set_title('模型置信度分布（距离0.5的绝对距离）')
        ax3.set_xlabel('置信度 (0=不确定, 1=非常确定)')
        ax3.set_ylabel('密度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 预测错误分析
        ax4 = axes[1, 1]
        error_rates = []
        model_names = []
        for model_name, results in evaluation_results.items():
            error_rate = 1 - results['accuracy']
            error_rates.append(error_rate)
            model_names.append(model_name)
        
        bars = ax4.bar(model_names, error_rates, color=['red', 'orange', 'yellow'])
        ax4.set_title('预测错误率')
        ax4.set_xlabel('模型')
        ax4.set_ylabel('错误率')
        
        # 添加数值标签
        for bar, rate in zip(bars, error_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'predictions_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("预测概率分布分析图已保存")
    
    def plot_probability_distribution_detailed(self, evaluation_results: Dict[str, Dict]):
        """
        创建详细的预测概率分布分析图（独立图表）
        用于诊断模型预测行为
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        fig.suptitle('模型预测概率分布详细分析', fontsize=16, fontweight='bold')
        
        colors = {'xgboost': 'blue', 'random_forest': 'green', 'lstm': 'orange'}
        
        for idx, (model_name, results) in enumerate(evaluation_results.items()):
            probabilities = results['probabilities']
            if probabilities.shape[1] >= 2:
                proba_positive = probabilities[:, 1]
                
                # 子图1：概率直方图（bins=50）
                axes[0].hist(proba_positive, bins=50, alpha=0.6, label=model_name, 
                           color=colors.get(model_name, 'gray'), density=True, histtype='step', linewidth=2)
                
                # 子图2：累积分布函数
                sorted_proba = np.sort(proba_positive)
                cumulative = np.arange(1, len(sorted_proba) + 1) / len(sorted_proba)
                axes[1].plot(sorted_proba, cumulative, label=model_name, 
                           color=colors.get(model_name, 'gray'), linewidth=2)
                
                # 子图3：概率统计摘要
                stats = {
                    'mean': np.mean(proba_positive),
                    'median': np.median(proba_positive),
                    'std': np.std(proba_positive),
                    'min': np.min(proba_positive),
                    'max': np.max(proba_positive),
                    'q25': np.percentile(proba_positive, 25),
                    'q75': np.percentile(proba_positive, 75),
                    'concentration_45_55': np.sum((proba_positive >= 0.45) & (proba_positive <= 0.55)) / len(proba_positive),
                    'polarization_low': np.sum(proba_positive < 0.2) / len(proba_positive),
                    'polarization_high': np.sum(proba_positive > 0.8) / len(proba_positive)
                }
                
                # 在第三个子图上显示统计信息
                if idx == 0:
                    axes[2].text(0.1, 0.9 - idx*0.25, 
                               f"{model_name}:\n"
                               f"  均值: {stats['mean']:.3f}, 中位数: {stats['median']:.3f}\n"
                               f"  标准差: {stats['std']:.3f}\n"
                               f"  25%-75%分位: [{stats['q25']:.3f}, {stats['q75']:.3f}]\n"
                               f"  0.45-0.55集中度: {stats['concentration_45_55']:.1%}\n"
                               f"  低概率(<0.2): {stats['polarization_low']:.1%}, 高概率(>0.8): {stats['polarization_high']:.1%}",
                               transform=axes[2].transAxes, fontsize=10,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    axes[2].text(0.1, 0.9 - idx*0.25, 
                               f"{model_name}:\n"
                               f"  均值: {stats['mean']:.3f}, 中位数: {stats['median']:.3f}\n"
                               f"  标准差: {stats['std']:.3f}\n"
                               f"  25%-75%分位: [{stats['q25']:.3f}, {stats['q75']:.3f}]\n"
                               f"  0.45-0.55集中度: {stats['concentration_45_55']:.1%}\n"
                               f"  低概率(<0.2): {stats['polarization_low']:.1%}, 高概率(>0.8): {stats['polarization_high']:.1%}",
                               transform=axes[2].transAxes, fontsize=10,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                
                # 保存统计信息到日志
                self.logger.info(f"{model_name} 概率分布统计:")
                self.logger.info(f"  均值: {stats['mean']:.3f}, 中位数: {stats['median']:.3f}, 标准差: {stats['std']:.3f}")
                self.logger.info(f"  0.45-0.55集中度: {stats['concentration_45_55']:.1%} (过高说明阈值保守)")
                self.logger.info(f"  两极化比例: 低({stats['polarization_low']:.1%}) + 高({stats['polarization_high']:.1%})")
        
        # 设置子图标题和标签
        axes[0].set_title('预测概率分布直方图 (bins=50)', fontsize=12)
        axes[0].set_xlabel('预测概率 P(y=1)')
        axes[0].set_ylabel('密度')
        axes[0].axvline(0.5, color='red', linestyle='--', linewidth=1, label='默认阈值(0.5)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('累积分布函数 (CDF)', fontsize=12)
        axes[1].set_xlabel('预测概率 P(y=1)')
        axes[1].set_ylabel('累积概率')
        axes[1].axvline(0.5, color='red', linestyle='--', linewidth=1, label='默认阈值(0.5)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_title('概率分布统计摘要', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        prob_dist_path = os.path.join(self.save_path, 'probability_distribution_detailed.png')
        plt.savefig(prob_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"详细概率分布分析图已保存到: {prob_dist_path}")
    
    def create_interactive_plots(self, data: pd.DataFrame, evaluation_results: Dict[str, Dict]):
        """创建交互式图表"""
        # 1. 股票价格交互式图表
        fig = go.Figure()
        
        for symbol in data['symbol'].unique()[:3]:  # 只显示前3只股票
            symbol_data = data[data['symbol'] == symbol]
            fig.add_trace(go.Scatter(
                x=symbol_data['Date'],
                y=symbol_data['close'],
                mode='lines',
                name=symbol,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='股票价格走势 (交互式)',
            xaxis_title='日期',
            yaxis_title='收盘价',
            hovermode='x unified'
        )
        
        fig.write_html(os.path.join(self.save_path, 'interactive_price_chart.html'))
        
        # 2. 模型性能对比交互式图表
        models = list(evaluation_results.keys())
        accuracies = [results['accuracy'] for results in evaluation_results.values()]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=accuracies, marker_color=['skyblue', 'lightcoral', 'lightgreen'])
        ])
        
        fig.update_layout(
            title='模型准确率对比 (交互式)',
            xaxis_title='模型',
            yaxis_title='准确率',
            yaxis=dict(range=[0, 1])
        )
        
        fig.write_html(os.path.join(self.save_path, 'interactive_model_performance.html'))
    
    def save_plots_summary(self, evaluation_results: Dict[str, Dict]):
        """保存图表摘要"""
        summary_data = []
        
        for model_name, results in evaluation_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision_Avg': f"{np.mean([results['classification_report'][str(i)]['precision'] for i in [-1, 0, 1]]):.4f}",
                'Recall_Avg': f"{np.mean([results['classification_report'][str(i)]['recall'] for i in [-1, 0, 1]]):.4f}",
                'F1_Avg': f"{np.mean([results['classification_report'][str(i)]['f1-score'] for i in [-1, 0, 1]]):.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.save_path, 'model_performance_summary.csv'), index=False)
        
        self.logger.info(f"图表摘要已保存到: {os.path.join(self.save_path, 'model_performance_summary.csv')}")
    
    def plot_roc_pr_curves(self, evaluation_results: Dict[str, Dict]):
        """
        绘制ROC曲线和PR曲线，用于评估类别平衡改善情况
        
        Args:
            evaluation_results: 评估结果字典，需要包含probabilities和真实标签
        """
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        
        # 需要从评估结果中提取真实标签，这里假设在evaluate_models中已经保存
        # 实际使用时需要传入y_test
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('模型ROC曲线和PR曲线分析（类别平衡评估）', fontsize=16, fontweight='bold')
        
        colors = {'xgboost': 'blue', 'random_forest': 'green', 'lstm': 'orange'}
        
        # 由于evaluation_results中没有直接保存y_test，我们需要从其他地方获取
        # 这里我们假设probabilities的索引对应测试集的顺序
        # 实际上，我们需要从model_trainer中传递y_test
        # 暂时从第一个模型的probabilities推断样本数量
        
        # 注意：这里需要y_test，但evaluation_results中没有
        # 我们需要修改evaluate_models来保存y_test，或者在这里传入
        # 暂时先绘制，如果缺少y_test会报错
        
        for model_name, results in evaluation_results.items():
            probabilities = results['probabilities']
            if probabilities.shape[1] >= 2:
                proba_positive = probabilities[:, 1]
                
                # 尝试从probabilities推断y_test（临时方案）
                # 更好的方案是在evaluate_models中保存y_test
                if 'true_labels' in results:
                    y_test = results['true_labels']
                else:
                    # 如果没有保存，尝试从混淆矩阵推断（不准确，但可以临时使用）
                    self.logger.warning(f"{model_name}: 无法获取真实标签，跳过ROC/PR曲线")
                    continue
                
                # ROC曲线
                fpr, tpr, roc_thresholds = roc_curve(y_test, proba_positive)
                roc_auc = auc(fpr, tpr)
                
                axes[0].plot(fpr, tpr, color=colors.get(model_name, 'gray'), 
                           linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
                
                # PR曲线
                precision, recall, pr_thresholds = precision_recall_curve(y_test, proba_positive)
                ap_score = average_precision_score(y_test, proba_positive)
                
                axes[1].plot(recall, precision, color=colors.get(model_name, 'gray'),
                           linewidth=2, label=f'{model_name} (AP = {ap_score:.3f})')
                
                # 记录关键指标
                self.logger.info(f"{model_name} ROC-AUC: {roc_auc:.4f}, PR-AP: {ap_score:.4f}")
        
        # 设置ROC曲线图
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测 (AUC = 0.500)')
        axes[0].set_xlabel('假阳性率 (False Positive Rate)', fontsize=11)
        axes[0].set_ylabel('真阳性率 (True Positive Rate)', fontsize=11)
        axes[0].set_title('ROC曲线（越大越好）', fontsize=12)
        axes[0].legend(loc='lower right', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])
        
        # 设置PR曲线图
        # 计算随机基线的AP（对于不平衡数据，这是正类比例）
        if 'true_labels' in list(evaluation_results.values())[0]:
            baseline_ap = list(evaluation_results.values())[0]['true_labels'].mean()
            axes[1].axhline(y=baseline_ap, color='k', linestyle='--', linewidth=1, 
                          label=f'随机基线 (AP = {baseline_ap:.3f})')
        axes[1].set_xlabel('召回率 (Recall)', fontsize=11)
        axes[1].set_ylabel('精确率 (Precision)', fontsize=11)
        axes[1].set_title('精确率-召回率曲线（越大越好）', fontsize=12)
        axes[1].legend(loc='lower left', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        roc_pr_path = os.path.join(self.save_path, 'roc_pr_curves.png')
        plt.savefig(roc_pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"ROC/PR曲线已保存到: {roc_pr_path}")
    
    def plot_class_balance_evaluation(self, evaluation_results: Dict[str, Dict]):
        """
        绘制类别平衡评估图（混淆矩阵详细分析 + 类别召回率）
        
        Args:
            evaluation_results: 评估结果字典
        """
        fig, axes = plt.subplots(2, len(evaluation_results), figsize=(5*len(evaluation_results), 10))
        if len(evaluation_results) == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle('类别平衡改善评估', fontsize=16, fontweight='bold')
        
        for idx, (model_name, results) in enumerate(evaluation_results.items()):
            # 上排：混淆矩阵热力图
            ax1 = axes[0, idx] if len(evaluation_results) > 1 else axes[0]
            conf_matrix = results['confusion_matrix']
            im = ax1.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            ax1.figure.colorbar(im, ax=ax1)
            
            # 添加数值标注
            thresh = conf_matrix.max() / 2.
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax1.text(j, i, format(conf_matrix[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if conf_matrix[i, j] > thresh else "black")
            
            ax1.set_title(f'{model_name}\n混淆矩阵', fontsize=12)
            ax1.set_xlabel('预测标签')
            ax1.set_ylabel('真实标签')
            ax1.set_xticks([0, 1])
            ax1.set_yticks([0, 1])
            ax1.set_xticklabels(['0 (下跌)', '1 (上涨)'])
            ax1.set_yticklabels(['0 (下跌)', '1 (上涨)'])
            
            # 下排：类别召回率对比
            ax2 = axes[1, idx] if len(evaluation_results) > 1 else axes[1]
            class_report = results['classification_report']
            
            # 提取各类别的召回率
            recalls = []
            precisions = []
            f1_scores = []
            class_labels = []
            
            for class_label in ['0', '1']:
                if class_label in class_report:
                    recalls.append(class_report[class_label]['recall'])
                    precisions.append(class_report[class_label]['precision'])
                    f1_scores.append(class_report[class_label]['f1-score'])
                    class_labels.append(f'类别{class_label}')
            
            x = np.arange(len(class_labels))
            width = 0.25
            
            ax2.bar(x - width, recalls, width, label='召回率 (Recall)', alpha=0.8)
            ax2.bar(x, precisions, width, label='精确率 (Precision)', alpha=0.8)
            ax2.bar(x + width, f1_scores, width, label='F1分数', alpha=0.8)
            
            ax2.set_title(f'{model_name}\n类别性能指标', fontsize=12)
            ax2.set_xlabel('类别')
            ax2.set_ylabel('分数')
            ax2.set_xticks(x)
            ax2.set_xticklabels(class_labels)
            ax2.legend(loc='upper right', fontsize=8)
            ax2.set_ylim([0, 1])
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 记录关键信息
            if '1' in class_report:
                recall_1 = class_report['1']['recall']
                status = "改善" if recall_1 > 0.1 else "仍需改善"
                self.logger.info(f"{model_name} 类别1召回率: {recall_1:.4f} ({status})")
        
        plt.tight_layout()
        balance_eval_path = os.path.join(self.save_path, 'class_balance_evaluation.png')
        plt.savefig(balance_eval_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"类别平衡评估图已保存到: {balance_eval_path}")

