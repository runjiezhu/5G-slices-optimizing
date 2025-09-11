"""
时间序列预测器
基于Transformer的实时动态预测模块
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio
from collections import deque
import threading
import time

from ..models.transformer_model import TransformerPredictor, SliceAwareTransformer
from ..data_processing.data_processor import DataProcessor, UserData
from ..data_processing.feature_extractor import FeatureExtractor


class TimeSeriesPredictor:
    """时间序列预测器"""
    
    def __init__(self, model_config: Dict, data_config: Dict, device: str = 'cpu'):
        """
        初始化预测器
        
        Args:
            model_config: 模型配置
            data_config: 数据配置
            device: 设备 ('cpu' 或 'cuda')
        """
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(device)
        
        # 配置
        self.model_config = model_config
        self.data_config = data_config
        self.prediction_horizon = data_config.get('prediction_horizon', 30)
        self.window_size = data_config.get('window_size', 60)
        self.update_interval = data_config.get('update_interval', 1.0)
        
        # 模型
        self.model = TransformerPredictor(model_config).to(self.device)
        self.slice_aware_model = SliceAwareTransformer(model_config).to(self.device)
        self.is_trained = False
        
        # 数据处理器
        self.data_processor = DataProcessor()
        self.feature_extractor = FeatureExtractor()
        
        # 预测缓存
        self.prediction_cache = {}
        self.cache_expiry_time = 30  # 缓存过期时间（秒）
        
        # 实时数据缓冲区
        self.data_buffer = deque(maxlen=self.window_size * 2)
        self.prediction_history = deque(maxlen=1000)
        
        # 性能监控
        self.prediction_metrics = {
            'total_predictions': 0,
            'avg_latency': 0.0,
            'accuracy_scores': deque(maxlen=100)
        }
        
        self.logger.info("时间序列预测器初始化完成")
    
    def load_model(self, model_path: str, slice_aware: bool = False) -> None:
        """
        加载预训练模型
        
        Args:
            model_path: 模型文件路径
            slice_aware: 是否加载切片感知模型
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if slice_aware:
                self.slice_aware_model.load_state_dict(checkpoint['model_state_dict'])
                self.slice_aware_model.eval()
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
            
            self.is_trained = True
            self.logger.info(f"成功加载模型: {model_path}")
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def predict(self, input_data: torch.Tensor, 
                slice_types: Optional[torch.Tensor] = None,
                use_cache: bool = True) -> Dict[str, torch.Tensor]:
        """
        执行预测
        
        Args:
            input_data: 输入数据 [batch_size, seq_len, feature_dim]
            slice_types: 切片类型（如果使用切片感知模型）
            use_cache: 是否使用缓存
            
        Returns:
            预测结果字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练或加载，无法进行预测")
        
        start_time = time.time()
        
        # 检查缓存
        if use_cache:
            cache_key = self._generate_cache_key(input_data)
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result is not None:
                return cached_result
        
        # 数据预处理
        input_data = input_data.to(self.device)
        
        # 执行预测
        with torch.no_grad():
            if slice_types is not None:
                slice_types = slice_types.to(self.device)
                predictions = self.slice_aware_model(input_data, slice_types)
            else:
                predictions = self.model(input_data)
        
        # 后处理预测结果
        processed_predictions = self._post_process_predictions(predictions)
        
        # 缓存结果
        if use_cache:
            self._cache_prediction(cache_key, processed_predictions)
        
        # 更新性能指标
        prediction_time = time.time() - start_time
        self._update_metrics(prediction_time)
        
        # 存储预测历史
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'predictions': processed_predictions,
            'input_shape': input_data.shape,
            'prediction_time': prediction_time
        })
        
        return processed_predictions
    
    def predict_from_user_data(self, user_data_list: List[UserData],
                              user_id: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        从用户数据直接预测
        
        Args:
            user_data_list: 用户数据列表
            user_id: 指定用户ID（可选）
            
        Returns:
            预测结果
        """
        # 处理用户数据
        if user_id:
            filtered_data = [data for data in user_data_list if data.user_id == user_id]
        else:
            filtered_data = user_data_list
        
        if len(filtered_data) < self.window_size:
            raise ValueError(f"数据不足，需要至少 {self.window_size} 个数据点")
        
        # 添加数据到处理器
        self.data_processor.raw_data_cache = filtered_data[-self.window_size:]
        
        # 处理数据
        processed_data = self.data_processor.process_data()
        
        if len(processed_data.sequences) == 0:
            raise ValueError("数据处理后无有效序列")
        
        # 转换为张量
        input_tensor = torch.tensor(processed_data.sequences[-1:], dtype=torch.float32)
        
        # 如果有切片类型信息，提取切片类型
        slice_types = None
        if len(filtered_data) > 0:
            last_slice_type = filtered_data[-1].slice_type
            slice_mapping = {'eMBB': 0, 'URLLC': 1, 'mMTC': 2}
            slice_id = slice_mapping.get(last_slice_type, 0)
            slice_types = torch.tensor([[slice_id] * self.window_size], dtype=torch.long)
        
        return self.predict(input_tensor, slice_types)
    
    def batch_predict(self, batch_data: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        批量预测
        
        Args:
            batch_data: 批量输入数据
            
        Returns:
            批量预测结果
        """
        results = []
        
        # 合并批量数据
        if batch_data:
            batch_tensor = torch.cat(batch_data, dim=0)
            batch_predictions = self.predict(batch_tensor, use_cache=False)
            
            # 拆分结果
            batch_size = len(batch_data)
            for i in range(batch_size):
                individual_result = {}
                for key, value in batch_predictions.items():
                    individual_result[key] = value[i:i+1]
                results.append(individual_result)
        
        return results
    
    def add_real_time_data(self, user_data: UserData) -> None:
        """
        添加实时数据到缓冲区
        
        Args:
            user_data: 用户数据
        """
        self.data_buffer.append(user_data)
        
        # 如果缓冲区足够，自动触发预测
        if len(self.data_buffer) >= self.window_size:
            try:
                recent_data = list(self.data_buffer)[-self.window_size:]
                self.predict_from_user_data(recent_data)
            except Exception as e:
                self.logger.warning(f"自动预测失败: {e}")
    
    def get_predictions_for_timerange(self, start_time: datetime, 
                                    end_time: datetime) -> List[Dict]:
        """
        获取指定时间范围的预测历史
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            预测历史列表
        """
        filtered_predictions = []
        
        for pred_record in self.prediction_history:
            if start_time <= pred_record['timestamp'] <= end_time:
                filtered_predictions.append(pred_record)
        
        return filtered_predictions
    
    def evaluate_predictions(self, ground_truth: List[Dict]) -> Dict[str, float]:
        """
        评估预测性能
        
        Args:
            ground_truth: 真实值列表
            
        Returns:
            评估指标
        """
        if not self.prediction_history or not ground_truth:
            return {}
        
        metrics = {
            'mae': 0.0,  # 平均绝对误差
            'rmse': 0.0,  # 均方根误差
            'mape': 0.0   # 平均绝对百分比误差
        }
        
        # 简化的评估实现
        recent_predictions = list(self.prediction_history)[-len(ground_truth):]
        
        if len(recent_predictions) == len(ground_truth):
            errors = []
            for pred, truth in zip(recent_predictions, ground_truth):
                # 假设比较带宽预测
                pred_bandwidth = pred['predictions'].get('bandwidth', torch.zeros(1, 30, 1))
                true_bandwidth = truth.get('bandwidth', 0.0)
                
                pred_mean = torch.mean(pred_bandwidth).item()
                error = abs(pred_mean - true_bandwidth)
                errors.append(error)
            
            if errors:
                metrics['mae'] = np.mean(errors)
                metrics['rmse'] = np.sqrt(np.mean([e**2 for e in errors]))
                metrics['mape'] = np.mean([abs(e/t) for e, t in zip(errors, [gt.get('bandwidth', 1.0) for gt in ground_truth]) if t != 0])
        
        return metrics
    
    def _generate_cache_key(self, input_data: torch.Tensor) -> str:
        """生成缓存键"""
        # 使用输入数据的哈希作为缓存键
        data_hash = hash(input_data.detach().cpu().numpy().tobytes())
        return f"pred_{data_hash}_{int(time.time() // self.cache_expiry_time)}"
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """获取缓存的预测结果"""
        if cache_key in self.prediction_cache:
            cached_time, cached_result = self.prediction_cache[cache_key]
            if time.time() - cached_time < self.cache_expiry_time:
                return cached_result
            else:
                del self.prediction_cache[cache_key]
        return None
    
    def _cache_prediction(self, cache_key: str, predictions: Dict[str, torch.Tensor]) -> None:
        """缓存预测结果"""
        self.prediction_cache[cache_key] = (time.time(), predictions)
        
        # 清理过期缓存
        current_time = time.time()
        expired_keys = [
            key for key, (cached_time, _) in self.prediction_cache.items()
            if current_time - cached_time > self.cache_expiry_time
        ]
        for key in expired_keys:
            del self.prediction_cache[key]
    
    def _post_process_predictions(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """后处理预测结果"""
        processed = {}
        
        for key, value in predictions.items():
            if key == 'features':
                # 特征预测可能需要反标准化
                processed[key] = torch.clamp(value, min=0.0)
            elif key == 'bandwidth':
                # 带宽预测确保为正值
                processed[key] = torch.clamp(value, min=0.1)
            elif key == 'slice_type':
                # 切片类型预测应用softmax
                processed[key] = torch.softmax(value, dim=-1)
            else:
                processed[key] = value
        
        return processed
    
    def _update_metrics(self, prediction_time: float) -> None:
        """更新性能指标"""
        self.prediction_metrics['total_predictions'] += 1
        
        # 更新平均延迟
        total_time = (self.prediction_metrics['avg_latency'] * 
                     (self.prediction_metrics['total_predictions'] - 1))
        self.prediction_metrics['avg_latency'] = (
            (total_time + prediction_time) / self.prediction_metrics['total_predictions']
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'prediction_metrics': self.prediction_metrics.copy(),
            'buffer_size': len(self.data_buffer),
            'cache_size': len(self.prediction_cache),
            'prediction_history_size': len(self.prediction_history),
            'model_device': str(self.device),
            'is_trained': self.is_trained
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.prediction_cache.clear()
        self.logger.info("预测缓存已清空")
    
    def reset_metrics(self) -> None:
        """重置性能指标"""
        self.prediction_metrics = {
            'total_predictions': 0,
            'avg_latency': 0.0,
            'accuracy_scores': deque(maxlen=100)
        }
        self.logger.info("性能指标已重置")