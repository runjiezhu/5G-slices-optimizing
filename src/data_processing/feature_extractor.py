"""
高级特征提取器
从5G用户数据中提取复杂的时空特征和行为模式
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy.fft import fft, fftfreq
# from tsfresh import extract_features  # 可选依赖
# from tsfresh.feature_extraction import ComprehensiveFCParameters  # 可选依赖
import warnings

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """高级特征提取器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化特征提取器
        
        Args:
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 特征提取配置
        self.extract_spatial_features = self.config.get('extract_spatial_features', True)
        self.extract_temporal_features = self.config.get('extract_temporal_features', True)
        self.extract_behavioral_features = self.config.get('extract_behavioral_features', True)
        self.extract_frequency_features = self.config.get('extract_frequency_features', True)
        
        # 聚类器用于位置分析
        self.location_clusterer = KMeans(n_clusters=5, random_state=42)
        self.location_fitted = False
        
        self.logger.info("特征提取器初始化完成")
    
    def extract_spatial_features(self, positions: np.ndarray, 
                                velocities: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取空间特征
        
        Args:
            positions: 位置数据 [N, 3]
            velocities: 速度数据 [N, 3]
            
        Returns:
            空间特征字典
        """
        features = {}
        
        # 基础统计特征
        features['position_mean'] = np.mean(positions, axis=0)
        features['position_std'] = np.std(positions, axis=0)
        features['position_range'] = np.ptp(positions, axis=0)
        
        # 移动范围和活动区域
        features['movement_range'] = np.sqrt(np.sum(np.var(positions, axis=0)))
        features['activity_radius'] = np.sqrt(np.sum((positions - features['position_mean'])**2, axis=1))
        
        # 轨迹长度和复杂度
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        features['total_distance'] = np.sum(distances)
        features['avg_step_distance'] = np.mean(distances) if len(distances) > 0 else 0
        features['distance_std'] = np.std(distances) if len(distances) > 0 else 0
        
        # 方向特征
        directions = np.diff(positions, axis=0)
        direction_angles = np.arctan2(directions[:, 1], directions[:, 0])
        features['direction_entropy'] = entropy(np.histogram(direction_angles, bins=8)[0] + 1e-10)
        features['direction_consistency'] = np.std(np.diff(direction_angles))
        
        # 速度特征
        speeds = np.sqrt(np.sum(velocities**2, axis=1))
        features['speed_mean'] = np.mean(speeds)
        features['speed_std'] = np.std(speeds)
        features['speed_max'] = np.max(speeds)
        features['speed_min'] = np.min(speeds)
        
        # 加速度特征
        accelerations = np.diff(velocities, axis=0)
        acc_magnitudes = np.sqrt(np.sum(accelerations**2, axis=1))
        features['acceleration_mean'] = np.mean(acc_magnitudes) if len(acc_magnitudes) > 0 else 0
        features['acceleration_std'] = np.std(acc_magnitudes) if len(acc_magnitudes) > 0 else 0
        
        # 停留点检测
        speed_threshold = 0.5  # m/s
        stop_mask = speeds < speed_threshold
        features['stop_ratio'] = np.mean(stop_mask)
        features['stop_duration_mean'] = self._calculate_stop_duration(stop_mask)
        
        # 位置聚类特征
        if not self.location_fitted and len(positions) > 5:
            self.location_clusterer.fit(positions)
            self.location_fitted = True
        
        if self.location_fitted:
            location_labels = self.location_clusterer.predict(positions)
            features['location_diversity'] = len(np.unique(location_labels))
            features['main_location_ratio'] = np.max(np.bincount(location_labels)) / len(location_labels)
        
        return features
    
    def extract_temporal_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        提取时间特征
        
        Args:
            data: 包含时间戳的数据
            
        Returns:
            时间特征字典
        """
        features = {}
        
        if 'timestamp' not in data.columns:
            return features
        
        timestamps = pd.to_datetime(data['timestamp'])
        
        # 基础时间特征
        features['hour_of_day'] = timestamps.dt.hour.values
        features['day_of_week'] = timestamps.dt.dayofweek.values
        features['day_of_month'] = timestamps.dt.day.values
        features['month'] = timestamps.dt.month.values
        features['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(float)
        
        # 时间段特征
        hour = timestamps.dt.hour
        features['is_morning'] = ((hour >= 6) & (hour < 12)).astype(float)
        features['is_afternoon'] = ((hour >= 12) & (hour < 18)).astype(float)
        features['is_evening'] = ((hour >= 18) & (hour < 22)).astype(float)
        features['is_night'] = ((hour >= 22) | (hour < 6)).astype(float)
        
        # 活动规律性
        daily_patterns = self._extract_daily_patterns(timestamps, data)
        features.update(daily_patterns)
        
        # 时间间隔特征
        time_diffs = timestamps.diff().dt.total_seconds().fillna(0)
        features['time_interval_mean'] = np.mean(time_diffs)
        features['time_interval_std'] = np.std(time_diffs)
        features['time_regularity'] = 1.0 / (1.0 + np.std(time_diffs))
        
        return features
    
    def extract_behavioral_features(self, behavior_data: Dict[str, np.ndarray], 
                                  network_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        提取行为特征
        
        Args:
            behavior_data: 行为数据字典
            network_data: 网络数据字典
            
        Returns:
            行为特征字典
        """
        features = {}
        
        # 应用使用模式
        if 'app_usage' in behavior_data:
            app_usage = behavior_data['app_usage']
            features['app_diversity'] = entropy(app_usage + 1e-10)
            features['dominant_app_ratio'] = np.max(app_usage) / np.sum(app_usage)
            features['app_switching_rate'] = self._calculate_switching_rate(app_usage)
        
        # 通信模式
        if 'call_duration' in behavior_data:
            call_durations = behavior_data['call_duration']
            features['call_frequency'] = np.sum(call_durations > 0) / len(call_durations)
            features['avg_call_duration'] = np.mean(call_durations[call_durations > 0])
            features['call_pattern_regularity'] = self._calculate_pattern_regularity(call_durations)
        
        # 数据使用模式
        if 'data_usage' in behavior_data:
            data_usage = behavior_data['data_usage']
            features['data_usage_mean'] = np.mean(data_usage)
            features['data_usage_std'] = np.std(data_usage)
            features['data_usage_peak_ratio'] = np.max(data_usage) / np.mean(data_usage)
            features['data_usage_trend'] = self._calculate_trend(data_usage)
        
        # 网络质量响应
        if 'signal_strength' in network_data:
            signal_strength = network_data['signal_strength']
            features['signal_stability'] = 1.0 / (1.0 + np.std(signal_strength))
            features['signal_quality'] = np.mean(signal_strength)
        
        if 'latency' in network_data:
            latency = network_data['latency']
            features['latency_sensitivity'] = np.std(latency) / np.mean(latency)
            features['avg_latency'] = np.mean(latency)
        
        # 服务质量要求
        if 'throughput' in network_data:
            throughput = network_data['throughput']
            features['throughput_demand'] = np.mean(throughput)
            features['throughput_variability'] = np.std(throughput) / np.mean(throughput)
            features['peak_throughput_ratio'] = np.max(throughput) / np.mean(throughput)
        
        return features
    
    def extract_frequency_features(self, time_series: np.ndarray, 
                                 sampling_rate: float = 1.0) -> Dict[str, np.ndarray]:
        """
        提取频域特征
        
        Args:
            time_series: 时间序列数据
            sampling_rate: 采样率
            
        Returns:
            频域特征字典
        """
        features = {}
        
        if len(time_series) < 4:
            return features
        
        # FFT变换
        fft_values = fft(time_series)
        fft_freqs = fftfreq(len(time_series), 1/sampling_rate)
        fft_magnitude = np.abs(fft_values)
        
        # 功率谱密度
        psd = fft_magnitude ** 2
        
        # 频域统计特征
        features['spectral_centroid'] = np.sum(fft_freqs * fft_magnitude) / np.sum(fft_magnitude)
        features['spectral_bandwidth'] = np.sqrt(np.sum(((fft_freqs - features['spectral_centroid']) ** 2) * fft_magnitude) / np.sum(fft_magnitude))
        features['spectral_rolloff'] = self._calculate_spectral_rolloff(fft_freqs, fft_magnitude)
        
        # 主要频率成分
        dominant_freq_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
        features['dominant_frequency'] = fft_freqs[dominant_freq_idx]
        features['dominant_power'] = fft_magnitude[dominant_freq_idx]
        
        # 频率熵
        psd_norm = psd / np.sum(psd)
        features['spectral_entropy'] = entropy(psd_norm + 1e-10)
        
        return features
    
    def extract_tsfresh_features(self, data: pd.DataFrame, 
                               value_column: str, 
                               id_column: str = 'user_id') -> pd.DataFrame:
        """
        使用tsfresh提取时间序列特征
        
        Args:
            data: 输入数据
            value_column: 值列名
            id_column: ID列名
            
        Returns:
            提取的特征DataFrame
        """
        try:
            # 需要安装tsfresh才能使用
            from tsfresh import extract_features
            from tsfresh.feature_extraction import ComprehensiveFCParameters
            
            # 准备数据格式
            ts_data = data[[id_column, value_column]].copy()
            ts_data['time'] = range(len(ts_data))
            
            # 提取特征
            extraction_settings = ComprehensiveFCParameters()
            features = extract_features(ts_data, 
                                      column_id=id_column,
                                      column_sort='time',
                                      default_fc_parameters=extraction_settings)
            
            return features
        
        except ImportError:
            self.logger.warning("tsfresh未安装，跳过tsfresh特征提取")
            return pd.DataFrame()
        except Exception as e:
            self.logger.warning(f"tsfresh特征提取失败: {e}")
            return pd.DataFrame()
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        提取所有特征
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            包含所有特征的DataFrame
        """
        self.logger.info("开始提取所有特征")
        
        all_features = {}
        
        # 提取空间特征
        if self.extract_spatial_features and all(col in data.columns for col in ['pos_x', 'pos_y', 'pos_z']):
            positions = data[['pos_x', 'pos_y', 'pos_z']].values
            velocities = data[['vel_x', 'vel_y', 'vel_z']].values if all(col in data.columns for col in ['vel_x', 'vel_y', 'vel_z']) else np.zeros_like(positions)
            
            spatial_features = self.extract_spatial_features(positions, velocities)
            all_features.update({f'spatial_{k}': v for k, v in spatial_features.items()})
        
        # 提取时间特征
        if self.extract_temporal_features and 'timestamp' in data.columns:
            temporal_features = self.extract_temporal_features(data)
            all_features.update({f'temporal_{k}': v for k, v in temporal_features.items()})
        
        # 提取行为特征
        if self.extract_behavioral_features:
            behavior_cols = [col for col in data.columns if col.startswith('behavior_')]
            network_cols = [col for col in data.columns if col.startswith('network_')]
            
            if behavior_cols or network_cols:
                behavior_data = {col.replace('behavior_', ''): data[col].values for col in behavior_cols}
                network_data = {col.replace('network_', ''): data[col].values for col in network_cols}
                
                behavioral_features = self.extract_behavioral_features(behavior_data, network_data)
                all_features.update({f'behavioral_{k}': v for k, v in behavioral_features.items()})
        
        # 提取频域特征
        if self.extract_frequency_features:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:5]:  # 限制列数避免过多计算
                if len(data[col].dropna()) >= 4:
                    freq_features = self.extract_frequency_features(data[col].values)
                    all_features.update({f'freq_{col}_{k}': v for k, v in freq_features.items()})
        
        # 转换为DataFrame
        max_length = max(len(v) if isinstance(v, np.ndarray) else 1 for v in all_features.values())
        
        feature_data = {}
        for key, value in all_features.items():
            if isinstance(value, np.ndarray):
                if len(value) == max_length:
                    feature_data[key] = value
                else:
                    # 广播标量值
                    feature_data[key] = np.full(max_length, np.mean(value))
            else:
                feature_data[key] = np.full(max_length, value)
        
        feature_df = pd.DataFrame(feature_data)
        
        self.logger.info(f"特征提取完成，共 {len(feature_df.columns)} 个特征")
        
        return feature_df
    
    def _calculate_stop_duration(self, stop_mask: np.ndarray) -> float:
        """计算平均停留时间"""
        if not np.any(stop_mask):
            return 0.0
        
        stop_durations = []
        current_duration = 0
        
        for is_stop in stop_mask:
            if is_stop:
                current_duration += 1
            else:
                if current_duration > 0:
                    stop_durations.append(current_duration)
                    current_duration = 0
        
        if current_duration > 0:
            stop_durations.append(current_duration)
        
        return np.mean(stop_durations) if stop_durations else 0.0
    
    def _extract_daily_patterns(self, timestamps: pd.Series, 
                              data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """提取每日活动模式"""
        features = {}
        
        # 按小时聚合活动
        hourly_activity = timestamps.dt.hour.value_counts().sort_index()
        peak_hour = hourly_activity.idxmax()
        features['peak_activity_hour'] = np.full(len(timestamps), peak_hour)
        
        # 活动集中度
        activity_entropy = entropy(hourly_activity.values)
        features['activity_concentration'] = np.full(len(timestamps), 1.0 / (1.0 + activity_entropy))
        
        return features
    
    def _calculate_switching_rate(self, app_usage: np.ndarray) -> float:
        """计算应用切换率"""
        if len(app_usage) < 2:
            return 0.0
        
        switches = np.sum(np.diff(app_usage) != 0)
        return switches / (len(app_usage) - 1)
    
    def _calculate_pattern_regularity(self, values: np.ndarray) -> float:
        """计算模式规律性"""
        if len(values) < 2:
            return 0.0
        
        # 使用标准差衡量规律性
        return 1.0 / (1.0 + np.std(values))
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """计算趋势"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]
        return trend
    
    def _calculate_spectral_rolloff(self, freqs: np.ndarray, 
                                  magnitudes: np.ndarray, 
                                  rolloff_percent: float = 0.85) -> float:
        """计算谱滚降"""
        cumsum = np.cumsum(magnitudes)
        rolloff_idx = np.where(cumsum >= rolloff_percent * cumsum[-1])[0]
        
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        else:
            return freqs[-1]