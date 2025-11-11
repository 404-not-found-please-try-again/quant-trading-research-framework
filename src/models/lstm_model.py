"""
LSTM模型模块
实现LSTM神经网络进行股票预测
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import Adam
# 尝试导入MultiHeadAttention（TensorFlow 2.x+）
try:
    from tensorflow.keras.layers import MultiHeadAttention
    ATTENTION_AVAILABLE = True
except ImportError:
    # 如果MultiHeadAttention不存在，尝试其他导入方式
    try:
        from tensorflow.keras.layers.experimental import MultiHeadAttention
        ATTENTION_AVAILABLE = True
    except ImportError:
        ATTENTION_AVAILABLE = False
        logging.warning("MultiHeadAttention不可用，将使用标准LSTM架构")
# 尝试导入AdamW，如果不存在则使用Adam
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    # 如果AdamW不存在，使用Adam并添加weight_decay（某些旧版本不支持）
    AdamW = None
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import os
from datetime import datetime

class LSTMModel:
    """LSTM模型类"""
    
    def __init__(self, config: dict):
        """
        初始化LSTM模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.lstm_config = config['models']['lstm']
        self.logger = logging.getLogger(__name__)
        
        # 模型参数
        self.sequence_length = self.lstm_config['sequence_length']
        self.hidden_units = self.lstm_config['hidden_units']
        self.dropout_rate = self.lstm_config.get('dropout_rate', 0.2)
        self.learning_rate = self.lstm_config.get('learning_rate', 0.001)
        self.batch_size = self.lstm_config.get('batch_size', 32)
        self.epochs = self.lstm_config.get('epochs', 100)
        
        # 数据标准化器
        self.scaler = MinMaxScaler()
        self.model = None
        
    def prepare_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备LSTM序列数据
        
        Args:
            X: 特征数据
            y: 目标数据
            
        Returns:
            序列化的特征和目标数据
        """
        self.logger.info(f"准备LSTM序列数据，序列长度: {self.sequence_length}")
        
        # 标准化特征数据
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建序列
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
            y_sequences.append(y.iloc[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        self.logger.info(f"序列数据形状: X={X_sequences.shape}, y={y_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def _create_sequences(self, X: pd.DataFrame) -> np.ndarray:
        """
        为预测创建序列数据
        
        Args:
            X: 特征数据
            
        Returns:
            序列化的特征数据
        """
        # 标准化特征数据
        X_scaled = self.scaler.transform(X)
        
        # 创建序列
        X_sequences = []
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
        
        if len(X_sequences) == 0:
            return np.array([])
        
        return np.array(X_sequences)
    
    def build_model(self, input_shape: Tuple[int, int]):
        """
        构建LSTM模型 - 优化版本（支持Attention机制）
        
        Args:
            input_shape: 输入形状 (sequence_length, n_features)
            
        Returns:
            构建好的LSTM模型（Sequential或Functional API）
        """
        self.logger.info("构建LSTM模型...")
        
        # 阶段2优化：如果支持Attention，使用Functional API；否则使用Sequential
        if ATTENTION_AVAILABLE:
            self.logger.info("使用Functional API构建带Attention的LSTM模型")
            from tensorflow.keras import Model, Input
            
            # 输入层
            inputs = Input(shape=input_shape, name='input')
            x = BatchNormalization()(inputs)
            
            # 第一个LSTM层
            x = LSTM(
                units=self.hidden_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate * 0.5,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            )(x)
            # 回退：使用BatchNormalization（LayerNorm导致性能下降）
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
            
            # 第二个LSTM层
            x = LSTM(
                units=self.hidden_units // 2,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate * 0.5,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            )(x)
            # 回退：使用BatchNormalization
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
            
            # 第三个LSTM层 - 返回序列以支持Attention
            lstm_output = LSTM(
                units=self.hidden_units // 4,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate * 0.5,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            )(x)
            # 回退：使用BatchNormalization
            x = BatchNormalization()(lstm_output)
            x = Dropout(self.dropout_rate)(x)
            
            # Multi-Head Attention（query和value/key都使用lstm_output）
            attention_output = MultiHeadAttention(
                num_heads=4,  # 4个注意力头
                key_dim=self.hidden_units // 16,  # 每个头的键维度
                dropout=self.dropout_rate * 0.5,
                name='multi_head_attention'
            )(x, x)  # self-attention: query=value=key=x
            
            # 残差连接（可选，但有助于训练）
            x = tf.keras.layers.Add()([x, attention_output])
            # 回退：使用BatchNormalization
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
            
            # 全局平均池化
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # 全连接层
            x = Dense(units=self.hidden_units // 2, activation='relu')(x)
            # 阶段5优化：全连接层使用BatchNormalization（非序列数据）
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
            
            x = Dense(units=self.hidden_units // 4, activation='relu')(x)
            # 阶段5优化：全连接层使用BatchNormalization
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
            
            # 输出层
            outputs = Dense(units=1, activation='sigmoid', name='output')(x)
            
            model = Model(inputs=inputs, outputs=outputs, name='lstm_with_attention')
        else:
            # 标准Sequential模型（无Attention）
            self.logger.info("使用Sequential API构建标准LSTM模型（Attention不可用）")
            model = Sequential([
                # 输入层 - 添加BatchNormalization
                BatchNormalization(input_shape=input_shape),
                
                # 第一个LSTM层 - 增加单元数
                LSTM(
                    units=self.hidden_units,
                    return_sequences=True,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate * 0.5,
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)
                ),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                
                # 第二个LSTM层
                LSTM(
                    units=self.hidden_units // 2,
                    return_sequences=True,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate * 0.5,
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)
                ),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                
                # 第三个LSTM层
                LSTM(
                    units=self.hidden_units // 4,
                    return_sequences=False,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate * 0.5,
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)
                ),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                
                # 全连接层 - 增加层数
                Dense(units=self.hidden_units // 2, activation='relu'),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                
                Dense(units=self.hidden_units // 4, activation='relu'),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                
                # 输出层
                Dense(units=1, activation='sigmoid')  # 二分类
            ])
        
        # 编译模型 - 使用AdamW优化器（更稳定），如果不可用则使用Adam
        # 注意：loss将在训练时动态设置（如果需要加权loss）
        if AdamW is not None:
            optimizer = AdamW(
                learning_rate=self.learning_rate,
                weight_decay=0.01,  # L2正则化
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
        else:
            # 降级使用Adam
            optimizer = Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
            self.logger.warning("AdamW不可用，使用Adam优化器")
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',  # 将在train方法中替换为加权版本
            metrics=['accuracy']
        )
        
        self.logger.info("LSTM模型构建完成")
        return model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        训练LSTM模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            训练结果
        """
        self.logger.info("开始训练LSTM模型...")
        
        # 准备序列数据
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
        
        # 计算类别权重（用于加权loss）
        neg_count = (y_train_seq == 0).sum()
        pos_count = (y_train_seq == 1).sum()
        pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0
        self.logger.info(f"LSTM类别权重: pos_weight={pos_weight:.2f} (负类:{neg_count}, 正类:{pos_count})")
        
        # 构建模型
        self.model = self.build_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        
        # 如果配置要求使用加权loss，则使用BCEWithLogitsLoss + pos_weight
        use_weighted_loss = self.lstm_config.get('use_weighted_loss', False)
        if use_weighted_loss:
            # 使用BCEWithLogitsLoss并设置pos_weight来处理类别不平衡
            import tensorflow.keras.losses as losses
            weighted_loss = losses.BinaryCrossentropy(from_logits=False)  # 因为输出层是sigmoid
            # 由于Keras的BinaryCrossentropy不支持pos_weight，我们需要使用sample_weight
            # 或者改用自定义loss函数
            # 这里使用sample_weight方式（在fit时传入）
            self.logger.info("使用类别平衡的BinaryCrossentropy loss")
        else:
            weighted_loss = 'binary_crossentropy'
        
        # 重新编译模型以使用加权loss（如果需要）
        if use_weighted_loss:
            if AdamW is not None:
                optimizer = AdamW(
                    learning_rate=self.learning_rate,
                    weight_decay=0.01,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                )
            else:
                optimizer = Adam(
                    learning_rate=self.learning_rate,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                )
            self.model.compile(
                optimizer=optimizer,
                loss=weighted_loss,
                metrics=['accuracy']
            )
        
        # 计算sample weights（用于类别平衡）
        sample_weights = np.ones(len(y_train_seq))
        if use_weighted_loss:
            # 为正类样本设置更高的权重
            sample_weights[y_train_seq == 1] = pos_weight
            self.logger.info(f"使用sample weights进行类别平衡，正类权重: {pos_weight:.2f}")
        
        # 设置回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # 训练模型
        # 如果使用加权loss，传入sample_weight
        fit_kwargs = {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'validation_data': (X_val_seq, y_val_seq),
            'callbacks': callbacks,
            'verbose': 1
        }
        if use_weighted_loss:
            fit_kwargs['sample_weight'] = sample_weights
        
        history = self.model.fit(
            X_train_seq, y_train_seq,
            **fit_kwargs
        )
        
        # 预测
        train_pred = self.model.predict(X_train_seq, verbose=0)
        val_pred = self.model.predict(X_val_seq, verbose=0)
        
        # 转换预测结果为分类
        train_pred_class = (train_pred > 0.5).astype(int).flatten()
        val_pred_class = (val_pred > 0.5).astype(int).flatten()
        
        # 计算指标
        train_accuracy = accuracy_score(y_train_seq, train_pred_class)
        val_accuracy = accuracy_score(y_val_seq, val_pred_class)
        val_f1 = f1_score(y_val_seq, val_pred_class, average='weighted')
        
        self.logger.info(f"LSTM训练完成")
        self.logger.info(f"训练准确率: {train_accuracy:.4f}")
        self.logger.info(f"验证准确率: {val_accuracy:.4f}")
        self.logger.info(f"验证F1分数: {val_f1:.4f}")
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'history': history.history,
            'predictions': val_pred_class,
            'probabilities': val_pred.flatten()
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用LSTM模型进行预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果和概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 使用新的序列创建方法
        X_sequences = self._create_sequences(X)
        
        if len(X_sequences) == 0:
            # 如果没有足够的序列，返回全零预测
            return np.zeros(len(X)), np.zeros(len(X))
        
        # 预测
        predictions = self.model.predict(X_sequences, verbose=0)
        pred_classes = (predictions > 0.5).astype(int).flatten()
        
        # 如果序列长度不足，在前面补零
        if len(pred_classes) < len(X):
            padding = np.zeros(len(X) - len(pred_classes))
            pred_classes = np.concatenate([padding, pred_classes])
            predictions = np.concatenate([padding, predictions.flatten()])
        
        return pred_classes, predictions
    
    def save_model(self, filepath: str):
        """
        保存LSTM模型
        
        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 保存模型
        self.model.save(f"{filepath}_model.h5")
        
        # 保存标准化器
        joblib.dump(self.scaler, f"{filepath}_scaler.joblib")
        
        # 保存配置
        config = {
            'sequence_length': self.sequence_length,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
        joblib.dump(config, f"{filepath}_config.joblib")
        
        self.logger.info(f"LSTM模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载LSTM模型
        
        Args:
            filepath: 模型路径
        """
        # 加载模型
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        
        # 加载标准化器
        self.scaler = joblib.load(f"{filepath}_scaler.joblib")
        
        # 加载配置
        config = joblib.load(f"{filepath}_config.joblib")
        self.sequence_length = config['sequence_length']
        self.hidden_units = config['hidden_units']
        self.dropout_rate = config['dropout_rate']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        
        self.logger.info(f"LSTM模型已从 {filepath} 加载")
    
    def get_feature_importance(self, X: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """
        获取LSTM特征重要性（通过梯度分析）
        
        Args:
            X: 特征数据
            feature_names: 特征名称列表
            
        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 准备序列数据
        X_scaled = self.scaler.transform(X)
        
        # 创建序列
        X_sequences = []
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
        
        X_sequences = np.array(X_sequences)
        
        # 计算梯度
        X_tensor = tf.Variable(X_sequences, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predictions = self.model(X_tensor)
            loss = tf.reduce_mean(predictions)
        
        gradients = tape.gradient(loss, X_tensor)
        
        # 计算特征重要性（梯度的绝对值）
        importance_scores = np.mean(np.abs(gradients.numpy()), axis=(0, 1))
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
