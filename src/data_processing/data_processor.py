"""
主要数据处理器
处理5G用户数据包括移动轨迹、速度、行为等信息
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yaml


@dataclass
class UserData:
    """用户数据结构"""
    user_id: str
    timestamp: datetime
    position: Tuple[float, float, float]  # (x, y, z)坐标
    velocity: Tuple[float, float, float]  # (vx, vy, vz)速度
    behavior_features: Dict[str, float]   # 用户行为特征
    network_metrics: Dict[str, float]     # 网络性能指标
    slice_type: str                       # 当前网络切片类型


@dataclass
class ProcessedData:
    """处理后的数据结构"""
    sequences: np.ndarray                 # 时间序列数据
    labels: np.ndarray                    # 标签数据
    metadata: Dict[str, Any]              # 元数据信息
    feature_names: List[str]              # 特征名称


class DataProcessor:
    """5G用户数据处理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化数据处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # 数据配置
        self.window_size = self.config['data']['sampling']['window_size']
        self.prediction_horizon = self.config['data']['sampling']['prediction_horizon']
        self.sampling_freq = self.config['data']['sampling']['frequency']
        
        # 特征维度
        self.trajectory_dim = self.config['data']['features']['trajectory_dim']
        self.velocity_dim = self.config['data']['features']['velocity_dim']
        self.behavior_dim = self.config['data']['features']['behavior_dim']
        
        # 预处理配置
        self.normalize = self.config['data']['preprocessing']['normalize']
        self.remove_outliers = self.config['data']['preprocessing']['remove_outliers']
        self.fill_missing = self.config['data']['preprocessing']['fill_missing']
        
        # 数据缓存
        self.raw_data_cache: List[UserData] = []
        self.processed_cache: Optional[ProcessedData] = None
        
        # 标准化器
        self.scalers = {
            'position': StandardScaler(),
            'velocity': StandardScaler(), 
            'behavior': MinMaxScaler(),
            'network': StandardScaler()
        }
        
        self.logger.info("数据处理器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"配置文件 {config_path} 未找到，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'data': {
                'sampling': {
                    'window_size': 60,
                    'prediction_horizon': 30,
                    'frequency': 1.0
                },
                'features': {
                    'trajectory_dim': 3,
                    'velocity_dim': 3,
                    'behavior_dim': 10
                },
                'preprocessing': {
                    'normalize': True,
                    'remove_outliers': True,
                    'fill_missing': 'interpolation'
                }
            }
        }
    
    def add_user_data(self, user_data: UserData) -> None:
        """
        添加用户数据到缓存
        
        Args:
            user_data: 用户数据对象
        """
        self.raw_data_cache.append(user_data)
        
        # 如果缓存过大，移除旧数据
        max_cache_size = self.window_size * 10
        if len(self.raw_data_cache) > max_cache_size:
            self.raw_data_cache = self.raw_data_cache[-max_cache_size:]
        
        # 清除已处理数据缓存，强制重新处理
        self.processed_cache = None
    
    def batch_add_data(self, user_data_list: List[UserData]) -> None:
        """
        批量添加用户数据
        
        Args:
            user_data_list: 用户数据列表
        """
        for user_data in user_data_list:
            self.add_user_data(user_data)
    
    def process_trajectory_data(self, data_df: pd.DataFrame) -> np.ndarray:
        """
        处理轨迹数据
        
        Args:
            data_df: 包含轨迹数据的DataFrame
            
        Returns:
            处理后的轨迹特征数组
        """
        trajectory_features = []
        
        # 基础位置特征
        positions = data_df[['pos_x', 'pos_y', 'pos_z']].values
        trajectory_features.append(positions)
        
        # 计算移动距离
        distances = []
        for i in range(1, len(positions)):
            dist = euclidean(positions[i], positions[i-1])
            distances.append(dist)
        distances.insert(0, 0.0)  # 第一个点距离为0
        trajectory_features.append(np.array(distances).reshape(-1, 1))
        
        # 计算方向变化
        direction_changes = []
        for i in range(2, len(positions)):
            v1 = positions[i-1] - positions[i-2]
            v2 = positions[i] - positions[i-1]
            
            # 计算角度变化
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_change = np.arccos(cos_angle)
                direction_changes.append(angle_change)
            else:
                direction_changes.append(0.0)
        
        # 前两个点的方向变化设为0
        direction_changes = [0.0, 0.0] + direction_changes
        trajectory_features.append(np.array(direction_changes).reshape(-1, 1))
        
        # 合并所有轨迹特征
        trajectory_matrix = np.concatenate(trajectory_features, axis=1)
        
        return trajectory_matrix
    
    def process_velocity_data(self, data_df: pd.DataFrame) -> np.ndarray:
        """
        处理速度数据
        
        Args:
            data_df: 包含速度数据的DataFrame
            
        Returns:
            处理后的速度特征数组
        """
        velocity_features = []
        
        # 基础速度特征
        velocities = data_df[['vel_x', 'vel_y', 'vel_z']].values
        velocity_features.append(velocities)
        
        # 计算速度大小
        speed = np.linalg.norm(velocities, axis=1).reshape(-1, 1)
        velocity_features.append(speed)
        
        # 计算加速度
        accelerations = []
        for i in range(1, len(velocities)):
            acc = velocities[i] - velocities[i-1]
            accelerations.append(acc)
        accelerations.insert(0, np.array([0.0, 0.0, 0.0]))  # 第一个点加速度为0
        accelerations = np.array(accelerations)
        velocity_features.append(accelerations)
        
        # 计算加速度大小
        acc_magnitude = np.linalg.norm(accelerations, axis=1).reshape(-1, 1)
        velocity_features.append(acc_magnitude)
        
        # 合并所有速度特征
        velocity_matrix = np.concatenate(velocity_features, axis=1)
        
        return velocity_matrix
    
    def process_behavior_data(self, data_df: pd.DataFrame) -> np.ndarray:
        """
        处理用户行为数据
        
        Args:
            data_df: 包含行为数据的DataFrame
            
        Returns:
            处理后的行为特征数组
        """
        behavior_features = []
        
        # 应用使用统计
        app_usage_cols = [col for col in data_df.columns if col.startswith('app_')]
        if app_usage_cols:
            app_usage = data_df[app_usage_cols].values
            behavior_features.append(app_usage)
        
        # 通话和数据使用模式
        if 'call_duration' in data_df.columns:
            behavior_features.append(data_df[['call_duration']].values)
        
        if 'data_usage' in data_df.columns:
            behavior_features.append(data_df[['data_usage']].values)
        
        # 时间特征（小时，星期几等）
        if 'timestamp' in data_df.columns:
            timestamps = pd.to_datetime(data_df['timestamp'])
            hour_of_day = timestamps.dt.hour.values.reshape(-1, 1)
            day_of_week = timestamps.dt.dayofweek.values.reshape(-1, 1)
            behavior_features.extend([hour_of_day, day_of_week])
        
        # 位置类型特征（家、办公室、商业区等）
        if 'location_type' in data_df.columns:
            # 独热编码位置类型
            location_types = pd.get_dummies(data_df['location_type']).values
            behavior_features.append(location_types)
        
        # 如果没有行为特征，创建默认特征
        if not behavior_features:
            default_features = np.zeros((len(data_df), self.behavior_dim))
            behavior_features.append(default_features)
        
        # 合并所有行为特征
        behavior_matrix = np.concatenate(behavior_features, axis=1)
        
        # 确保特征维度正确
        if behavior_matrix.shape[1] < self.behavior_dim:
            # 填充零特征到指定维度
            padding = np.zeros((behavior_matrix.shape[0], 
                              self.behavior_dim - behavior_matrix.shape[1]))
            behavior_matrix = np.concatenate([behavior_matrix, padding], axis=1)
        elif behavior_matrix.shape[1] > self.behavior_dim:
            # 截取到指定维度
            behavior_matrix = behavior_matrix[:, :self.behavior_dim]
        
        return behavior_matrix
    
    def create_sequences(self, features: np.ndarray, 
                        labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列数据
        
        Args:
            features: 特征数据
            labels: 标签数据
            
        Returns:
            序列特征和标签的元组
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(len(features) - self.window_size - self.prediction_horizon + 1):
            # 输入序列
            X_seq = features[i:i+self.window_size]
            X_sequences.append(X_seq)
            
            # 预测目标
            y_seq = labels[i+self.window_size:i+self.window_size+self.prediction_horizon]
            y_sequences.append(y_seq)
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def remove_outliers(self, data: np.ndarray, 
                       threshold: float = 3.0) -> np.ndarray:
        """
        移除异常值
        
        Args:
            data: 输入数据
            threshold: Z-score阈值
            
        Returns:
            清理后的数据
        """
        if not self.remove_outliers:
            return data
        
        z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
        mask = np.all(z_scores < threshold, axis=1)
        
        return data[mask]
    
    def fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        填充缺失值
        
        Args:
            data: 输入DataFrame
            
        Returns:
            填充后的DataFrame
        """
        if self.fill_missing == 'interpolation':
            return data.interpolate(method='linear')
        elif self.fill_missing == 'forward_fill':
            return data.fillna(method='ffill')
        elif self.fill_missing == 'backward_fill':
            return data.fillna(method='bfill')
        elif self.fill_missing == 'mean':
            return data.fillna(data.mean())
        else:
            return data.dropna()
    
    def normalize_features(self, features: Dict[str, np.ndarray], 
                          fit_scalers: bool = True) -> Dict[str, np.ndarray]:
        """
        标准化特征
        
        Args:
            features: 特征字典
            fit_scalers: 是否拟合标准化器
            
        Returns:
            标准化后的特征字典
        """
        if not self.normalize:
            return features
        
        normalized_features = {}
        
        for feature_type, feature_data in features.items():
            if feature_type in self.scalers:
                scaler = self.scalers[feature_type]
                
                if fit_scalers:
                    normalized_data = scaler.fit_transform(feature_data)
                else:
                    normalized_data = scaler.transform(feature_data)
                
                normalized_features[feature_type] = normalized_data
            else:
                normalized_features[feature_type] = feature_data
        
        return normalized_features
    
    def process_data(self, force_reprocess: bool = False) -> ProcessedData:
        """
        处理所有数据
        
        Args:
            force_reprocess: 是否强制重新处理
            
        Returns:
            处理后的数据对象
        """
        if self.processed_cache is not None and not force_reprocess:
            return self.processed_cache
        
        if not self.raw_data_cache:
            raise ValueError("没有可处理的数据，请先添加数据")
        
        self.logger.info(f"开始处理 {len(self.raw_data_cache)} 条用户数据")
        
        # 转换为DataFrame
        data_records = []
        for user_data in self.raw_data_cache:
            record = {
                'user_id': user_data.user_id,
                'timestamp': user_data.timestamp,
                'pos_x': user_data.position[0],
                'pos_y': user_data.position[1], 
                'pos_z': user_data.position[2],
                'vel_x': user_data.velocity[0],
                'vel_y': user_data.velocity[1],
                'vel_z': user_data.velocity[2],
                'slice_type': user_data.slice_type
            }
            
            # 添加行为特征
            for key, value in user_data.behavior_features.items():
                record[f'behavior_{key}'] = value
            
            # 添加网络指标
            for key, value in user_data.network_metrics.items():
                record[f'network_{key}'] = value
            
            data_records.append(record)
        
        data_df = pd.DataFrame(data_records)
        
        # 填充缺失值
        data_df = self.fill_missing_values(data_df)
        
        # 处理各类特征
        trajectory_features = self.process_trajectory_data(data_df)
        velocity_features = self.process_velocity_data(data_df)
        behavior_features = self.process_behavior_data(data_df)
        
        # 创建特征字典
        feature_dict = {
            'position': trajectory_features[:, :self.trajectory_dim],
            'velocity': velocity_features[:, :self.velocity_dim],
            'behavior': behavior_features
        }
        
        # 添加网络指标
        network_cols = [col for col in data_df.columns if col.startswith('network_')]
        if network_cols:
            network_features = data_df[network_cols].values
            feature_dict['network'] = network_features
        
        # 标准化特征
        normalized_features = self.normalize_features(feature_dict, fit_scalers=True)
        
        # 合并所有特征
        all_features = np.concatenate(list(normalized_features.values()), axis=1)
        
        # 移除异常值
        all_features = self.remove_outliers(all_features)
        
        # 创建标签（这里假设标签是网络切片类型和需求预测）
        slice_types = data_df['slice_type'].values
        # 将切片类型转换为数值
        slice_mapping = {'eMBB': 0, 'URLLC': 1, 'mMTC': 2}
        labels = np.array([slice_mapping.get(st, 0) for st in slice_types])
        
        # 确保标签长度与特征一致
        min_length = min(len(all_features), len(labels))
        all_features = all_features[:min_length]
        labels = labels[:min_length]
        
        # 创建时间序列
        sequences, sequence_labels = self.create_sequences(all_features, labels)
        
        # 创建特征名称
        feature_names = []
        for feature_type, features in normalized_features.items():
            feature_names.extend([f"{feature_type}_{i}" 
                                for i in range(features.shape[1])])
        
        # 创建处理后的数据对象
        processed_data = ProcessedData(
            sequences=sequences,
            labels=sequence_labels,
            metadata={
                'total_samples': len(sequences),
                'sequence_length': self.window_size,
                'prediction_horizon': self.prediction_horizon,
                'feature_count': all_features.shape[1],
                'users_count': len(set(data_df['user_id'])),
                'processing_time': datetime.now()
            },
            feature_names=feature_names
        )
        
        # 缓存结果
        self.processed_cache = processed_data
        
        self.logger.info(f"数据处理完成：{processed_data.metadata['total_samples']} 个序列，"
                        f"{processed_data.metadata['feature_count']} 个特征")
        
        return processed_data
    
    def get_latest_features(self, user_id: str, 
                           sequence_length: Optional[int] = None) -> Optional[np.ndarray]:
        """
        获取指定用户的最新特征序列
        
        Args:
            user_id: 用户ID
            sequence_length: 序列长度，默认使用配置中的窗口大小
            
        Returns:
            最新特征序列，如果数据不足则返回None
        """
        if sequence_length is None:
            sequence_length = self.window_size
        
        # 筛选指定用户的数据
        user_data = [data for data in self.raw_data_cache if data.user_id == user_id]
        
        if len(user_data) < sequence_length:
            return None
        
        # 获取最新的数据
        latest_data = user_data[-sequence_length:]
        
        # 创建临时处理器来处理这些数据
        temp_processor = DataProcessor()
        temp_processor.raw_data_cache = latest_data
        
        try:
            processed_data = temp_processor.process_data()
            if len(processed_data.sequences) > 0:
                return processed_data.sequences[-1]  # 返回最后一个序列
        except Exception as e:
            self.logger.error(f"获取用户 {user_id} 最新特征时出错: {e}")
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Returns:
            数据统计字典
        """
        if not self.raw_data_cache:
            return {"message": "没有数据"}
        
        user_ids = set(data.user_id for data in self.raw_data_cache)
        slice_types = [data.slice_type for data in self.raw_data_cache]
        
        stats = {
            "total_records": len(self.raw_data_cache),
            "unique_users": len(user_ids), 
            "time_span": {
                "start": min(data.timestamp for data in self.raw_data_cache),
                "end": max(data.timestamp for data in self.raw_data_cache)
            },
            "slice_type_distribution": pd.Series(slice_types).value_counts().to_dict(),
            "cache_status": {
                "raw_data_count": len(self.raw_data_cache),
                "processed_cache_exists": self.processed_cache is not None
            }
        }
        
        if self.processed_cache:
            stats["processed_data"] = self.processed_cache.metadata
        
        return stats