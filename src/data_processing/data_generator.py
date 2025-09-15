"""
5G用户数据生成器
生成模拟的用户移动轨迹、速度、行为数据用于测试和训练
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
from ..data_processing.data_processor import UserData


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    user_type: str  # 'business', 'personal', 'iot'
    mobility_pattern: str  # 'static', 'regular', 'random'
    activity_level: float  # 0-1
    preferred_slice: str  # 'eMBB', 'URLLC', 'mMTC'


class UserDataGenerator:
    """5G用户数据生成器"""
    
    def __init__(self, seed: int = 42):
        """初始化生成器"""
        np.random.seed(seed)
        random.seed(seed)
        
        # 预定义的位置区域
        self.locations = {
            'home': [(0, 0, 0), (5, 5, 0)],
            'office': [(20, 20, 0), (25, 25, 0)], 
            'mall': [(10, 30, 0), (15, 35, 0)],
            'park': [(30, 10, 0), (35, 15, 0)]
        }
        
        # 应用类型和使用模式
        self.app_types = ['social', 'video', 'game', 'work', 'navigation']
        
        # 新增扩展的用户行为类型
        self.behavior_types = {
            # 游戏类行为
            'vr_ar_gaming': {
                'bandwidth_req': 'ultra_high', 
                'latency_req': 'ultra_low', 
                'slice_preference': 'URLLC',
                'data_intensive': True,
                'realtime_critical': True
            },
            'fps_gaming': {
                'bandwidth_req': 'medium_high', 
                'latency_req': 'ultra_low', 
                'slice_preference': 'URLLC',
                'data_intensive': False,
                'realtime_critical': True
            },
            'cloud_gaming': {
                'bandwidth_req': 'high', 
                'latency_req': 'ultra_low', 
                'slice_preference': 'URLLC',
                'data_intensive': True,
                'realtime_critical': True
            },
            'online_gaming': {
                'bandwidth_req': 'medium', 
                'latency_req': 'low', 
                'slice_preference': 'URLLC',
                'data_intensive': False,
                'realtime_critical': True
            },
            
            # 流媒体类行为
            'video_streaming': {
                'bandwidth_req': 'high', 
                'latency_req': 'low', 
                'slice_preference': 'eMBB',
                'data_intensive': True,
                'realtime_critical': False
            },
            'live_streaming': {
                'bandwidth_req': 'very_high', 
                'latency_req': 'low', 
                'slice_preference': 'eMBB',
                'data_intensive': True,
                'realtime_critical': True
            },
            'video_calling': {
                'bandwidth_req': 'medium', 
                'latency_req': 'low', 
                'slice_preference': 'eMBB',
                'data_intensive': False,
                'realtime_critical': True
            },
            
            # 其他应用行为
            'file_download': {
                'bandwidth_req': 'very_high', 
                'latency_req': 'medium', 
                'slice_preference': 'eMBB',
                'data_intensive': True,
                'realtime_critical': False
            },
            'iot_sensor': {
                'bandwidth_req': 'very_low', 
                'latency_req': 'medium', 
                'slice_preference': 'mMTC',
                'data_intensive': False,
                'realtime_critical': False
            },
            'web_browsing': {
                'bandwidth_req': 'low', 
                'latency_req': 'medium', 
                'slice_preference': 'eMBB',
                'data_intensive': False,
                'realtime_critical': False
            }
        }
        
        # 带宽需求映射 (Mbps)
        self.bandwidth_requirements = {
            'very_low': (0.1, 1.0),
            'low': (1.0, 10.0),
            'medium': (10.0, 50.0),
            'medium_high': (30.0, 100.0),
            'high': (50.0, 200.0),
            'very_high': (100.0, 500.0),
            'ultra_high': (200.0, 1000.0)
        }
        
        # 延迟需求映射 (ms)
        self.latency_requirements = {
            'ultra_low': (1, 5),
            'low': (5, 20),
            'medium': (20, 100),
            'high': (100, 500)
        }
        
    def generate_user_profiles(self, num_users: int) -> List[UserProfile]:
        """生成用户画像"""
        profiles = []
        
        for i in range(num_users):
            user_type = np.random.choice(['business', 'personal', 'iot'], 
                                       p=[0.3, 0.6, 0.1])
            
            if user_type == 'business':
                mobility = np.random.choice(['regular', 'random'], p=[0.8, 0.2])
                slice_type = np.random.choice(['eMBB', 'URLLC'], p=[0.6, 0.4])
                activity = np.random.uniform(0.7, 1.0)
            elif user_type == 'personal':
                mobility = np.random.choice(['regular', 'random'], p=[0.6, 0.4])
                slice_type = 'eMBB'
                activity = np.random.uniform(0.4, 0.8)
            else:  # iot
                mobility = 'static'
                slice_type = 'mMTC'
                activity = np.random.uniform(0.1, 0.3)
            
            profile = UserProfile(
                user_id=f"user_{i:04d}",
                user_type=user_type,
                mobility_pattern=mobility,
                activity_level=activity,
                preferred_slice=slice_type
            )
            profiles.append(profile)
        
        return profiles
    
    def generate_trajectory(self, profile: UserProfile, 
                          duration_hours: int = 24,
                          sampling_interval: int = 60) -> List[UserData]:
        """生成用户轨迹数据"""
        data_points = []
        current_time = datetime.now()
        
        # 根据用户类型确定移动模式
        if profile.mobility_pattern == 'static':
            # 静态用户，位置基本不变
            base_pos = self._get_random_position('home')
            
            for i in range(0, duration_hours * 3600, sampling_interval):
                timestamp = current_time + timedelta(seconds=i)
                
                # 添加小量随机噪声
                position = (
                    base_pos[0] + np.random.normal(0, 0.1),
                    base_pos[1] + np.random.normal(0, 0.1), 
                    base_pos[2] + np.random.normal(0, 0.05)
                )
                
                velocity = (0, 0, 0)
                behavior = self._generate_behavior_features(profile, timestamp)
                network = self._generate_network_metrics(profile, position, behavior)
                
                data_points.append(UserData(
                    user_id=profile.user_id,
                    timestamp=timestamp,
                    position=position,
                    velocity=velocity,
                    behavior_features=behavior,
                    network_metrics=network,
                    slice_type=profile.preferred_slice
                ))
        
        elif profile.mobility_pattern == 'regular':
            # 规律移动用户
            locations = ['home', 'office', 'mall', 'home']
            location_times = [0, 8, 18, 22]  # 小时
            
            for i in range(0, duration_hours * 3600, sampling_interval):
                timestamp = current_time + timedelta(seconds=i)
                hour = (i // 3600) % 24
                
                # 确定当前应该在的位置
                current_location = self._get_location_at_time(hour, locations, location_times)
                target_pos = self._get_random_position(current_location)
                
                # 如果在移动时间，生成移动轨迹
                position, velocity = self._generate_movement(i, target_pos, sampling_interval)
                
                behavior = self._generate_behavior_features(profile, timestamp)
                network = self._generate_network_metrics(profile, position, behavior)
                
                data_points.append(UserData(
                    user_id=profile.user_id,
                    timestamp=timestamp,
                    position=position,
                    velocity=velocity,
                    behavior_features=behavior,
                    network_metrics=network,
                    slice_type=profile.preferred_slice
                ))
        
        else:  # random
            # 随机移动用户
            current_pos = self._get_random_position('home')
            
            for i in range(0, duration_hours * 3600, sampling_interval):
                timestamp = current_time + timedelta(seconds=i)
                
                # 随机移动
                move_distance = np.random.exponential(2.0) * profile.activity_level
                move_direction = np.random.uniform(0, 2*np.pi)
                
                new_pos = (
                    current_pos[0] + move_distance * np.cos(move_direction),
                    current_pos[1] + move_distance * np.sin(move_direction),
                    current_pos[2] + np.random.normal(0, 0.1)
                )
                
                velocity = (
                    (new_pos[0] - current_pos[0]) / sampling_interval,
                    (new_pos[1] - current_pos[1]) / sampling_interval,
                    (new_pos[2] - current_pos[2]) / sampling_interval
                )
                
                behavior = self._generate_behavior_features(profile, timestamp)
                network = self._generate_network_metrics(profile, new_pos, behavior)
                
                data_points.append(UserData(
                    user_id=profile.user_id,
                    timestamp=timestamp,
                    position=new_pos,
                    velocity=velocity,
                    behavior_features=behavior,
                    network_metrics=network,
                    slice_type=profile.preferred_slice
                ))
                
                current_pos = new_pos
        
        return data_points
    
    def _get_random_position(self, location_type: str) -> Tuple[float, float, float]:
        """获取指定区域的随机位置"""
        if location_type in self.locations:
            min_pos, max_pos = self.locations[location_type]
            return (
                np.random.uniform(min_pos[0], max_pos[0]),
                np.random.uniform(min_pos[1], max_pos[1]),
                np.random.uniform(min_pos[2], max_pos[2])
            )
        else:
            return (np.random.uniform(-50, 50), 
                   np.random.uniform(-50, 50), 
                   np.random.uniform(0, 10))
    
    def _get_location_at_time(self, hour: int, locations: List[str], 
                            times: List[int]) -> str:
        """根据时间确定位置"""
        for i in range(len(times)):
            if hour < times[i]:
                return locations[i-1] if i > 0 else locations[-1]
        return locations[-1]
    
    def _generate_movement(self, time_seconds: int, target_pos: Tuple[float, float, float], 
                         interval: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """生成移动轨迹"""
        # 简化的移动模拟
        base_pos = (
            10 + 5 * np.sin(time_seconds / 3600),
            10 + 5 * np.cos(time_seconds / 3600),
            0
        )
        
        velocity = (
            5 * np.cos(time_seconds / 3600) / 3600,
            -5 * np.sin(time_seconds / 3600) / 3600,
            0
        )
        
        return base_pos, velocity
    
    def _generate_behavior_features(self, profile: UserProfile, 
                                  timestamp: datetime) -> Dict[str, float]:
        """生成行为特征"""
        hour = timestamp.hour
        
        # 基础应用使用
        app_usage = {}
        for app in self.app_types:
            if profile.user_type == 'business' and app == 'work':
                usage = 0.8 if 9 <= hour <= 17 else 0.1
            elif profile.user_type == 'personal' and app in ['social', 'video']:
                usage = 0.6 if hour >= 18 or hour <= 22 else 0.2
            else:
                usage = np.random.exponential(0.2) * profile.activity_level
            
            app_usage[f'app_{app}'] = min(usage, 1.0)
        
        # 新增用户行为特征生成
        behavior_usage = self._generate_user_behaviors(profile, hour)
        
        # 通话和数据使用
        call_prob = 0.3 * profile.activity_level
        call_duration = np.random.exponential(120) if np.random.random() < call_prob else 0
        
        # 根据行为类型调整数据使用量
        base_data_usage = np.random.lognormal(
            mean=2.0 + profile.activity_level,
            sigma=1.0
        ) * (1.5 if 18 <= hour <= 22 else 1.0)
        
        # 根据高带宽行为调整数据使用量
        data_multiplier = 1.0
        if behavior_usage.get('vr_ar_gaming', 0) > 0.5:
            data_multiplier *= 3.0
        if behavior_usage.get('live_streaming', 0) > 0.5:
            data_multiplier *= 5.0
        if behavior_usage.get('video_streaming', 0) > 0.5:
            data_multiplier *= 2.0
        if behavior_usage.get('file_download', 0) > 0.5:
            data_multiplier *= 4.0
            
        data_usage = base_data_usage * data_multiplier
        
        behavior = {
            **app_usage,
            **behavior_usage,
            'call_duration': call_duration,
            'data_usage': data_usage,
            'location_type': self._get_location_type(timestamp),
            'is_peak_hour': 1.0 if 8 <= hour <= 10 or 17 <= hour <= 19 else 0.0
        }
        
        return behavior
    
    def _generate_user_behaviors(self, profile: UserProfile, hour: int) -> Dict[str, float]:
        """生成用户行为特征"""
        behaviors = {}
        
        # 根据用户类型和时间生成不同行为的使用概率
        for behavior_name, behavior_info in self.behavior_types.items():
            base_prob = self._get_behavior_probability(behavior_name, profile, hour)
            
            # 添加随机性
            actual_usage = base_prob * np.random.uniform(0.5, 1.5) * profile.activity_level
            
            # 添加时间相关性（某些行为在特定时间更常见）
            if behavior_name in ['video_streaming', 'live_streaming']:
                if 19 <= hour <= 23:  # 晚上高峰期
                    actual_usage *= 2.0
                elif 12 <= hour <= 14:  # 午休时间
                    actual_usage *= 1.5
            
            # 游戏类行为时间相关性
            elif behavior_name in ['vr_ar_gaming', 'fps_gaming', 'cloud_gaming', 'online_gaming']:
                if 18 <= hour <= 24 or 0 <= hour <= 2:  # 晚上和深夜游戏高峰期
                    actual_usage *= 2.5
                elif 12 <= hour <= 14:  # 午休时间
                    actual_usage *= 1.3
                elif 9 <= hour <= 17:  # 工作时间减少
                    actual_usage *= 0.2 if profile.user_type == 'business' else 0.5
            
            elif behavior_name == 'video_calling':
                if 9 <= hour <= 11 or 14 <= hour <= 16:  # 会议时间
                    actual_usage *= 1.8
            
            elif behavior_name == 'file_download':
                if 8 <= hour <= 10 or 13 <= hour <= 15:  # 工作时间
                    actual_usage *= 1.5
            
            elif behavior_name == 'iot_sensor':
                # IoT设备通常持续工作，但数据量小
                actual_usage = 0.8 + np.random.normal(0, 0.1)
            
            behaviors[behavior_name] = np.clip(actual_usage, 0.0, 1.0)
        
        return behaviors
    
    def _get_behavior_probability(self, behavior_name: str, profile: UserProfile, hour: int) -> float:
        """获取特定行为的基本使用概率"""
        behavior_info = self.behavior_types[behavior_name]
        
        # 根据用户类型调整基础概率
        if profile.user_type == 'business':
            if behavior_name in ['video_calling', 'file_download', 'web_browsing']:
                base_prob = 0.7
            elif behavior_name in ['vr_ar_gaming', 'fps_gaming', 'cloud_gaming', 'online_gaming']:
                base_prob = 0.1  # 商务用户游戏使用率较低
            elif behavior_name in ['video_streaming', 'live_streaming']:
                base_prob = 0.3
            else:
                base_prob = 0.2
        
        elif profile.user_type == 'personal':
            if behavior_name in ['video_streaming', 'live_streaming']:
                base_prob = 0.8
            elif behavior_name in ['vr_ar_gaming', 'fps_gaming']:
                base_prob = 0.5  # 个人用户VR/AR和FPS游戏使用率较高
            elif behavior_name in ['cloud_gaming', 'online_gaming']:
                base_prob = 0.6  # 个人用户对云游戏和网络游戏使用率高
            elif behavior_name in ['video_calling', 'web_browsing']:
                base_prob = 0.6
            elif behavior_name == 'file_download':
                base_prob = 0.3
            else:
                base_prob = 0.1
        
        else:  # iot
            if behavior_name == 'iot_sensor':
                base_prob = 0.9
            else:
                base_prob = 0.05
        
        # 根据切片偏好调整
        preferred_slice = behavior_info['slice_preference']
        if profile.preferred_slice == preferred_slice:
            base_prob *= 1.5
        
        return min(base_prob, 1.0)
    
    def _generate_network_metrics(self, profile: UserProfile, 
                                position: Tuple[float, float, float],
                                behavior_features: Dict[str, float] = None) -> Dict[str, float]:
        """生成网络性能指标"""
        # 基于位置和用户类型生成网络指标
        distance_from_center = np.sqrt(position[0]**2 + position[1]**2)
        
        # 信号强度（距离越远越弱）
        signal_strength = max(0.1, 1.0 - distance_from_center / 100.0)
        signal_strength += np.random.normal(0, 0.1)
        signal_strength = np.clip(signal_strength, 0.1, 1.0)
        
        # 延迟（距离越远越高）
        base_latency = 10 + distance_from_center * 0.5
        
        # 根据用户行为调整网络指标
        if behavior_features:
            # VR/AR游戏需要极低延迟和高带宽
            if behavior_features.get('vr_ar_gaming', 0) > 0.5:
                base_latency *= 0.05  # 极低延迟需求 (<1ms)
                signal_strength *= 1.3  # 更强信号需求
            
            # FPS游戏需要低延迟
            elif behavior_features.get('fps_gaming', 0) > 0.5:
                base_latency *= 0.1  # 低延迟需求 (<5ms)
                signal_strength *= 1.2
            
            # 云游戏需要低延迟和高带宽
            elif behavior_features.get('cloud_gaming', 0) > 0.5:
                base_latency *= 0.15  # 云游戏对延迟敏感
                signal_strength *= 1.25
            
            # 网络游戏需要稳定的网络
            elif behavior_features.get('online_gaming', 0) > 0.5:
                base_latency *= 0.3
                signal_strength *= 1.1
            
            # 直播流需要高带宽和低延迟
            elif behavior_features.get('live_streaming', 0) > 0.5:
                base_latency *= 0.4
                signal_strength *= 1.4  # 直播对带宽需求最高
            
            # 视频流需要高带宽
            elif behavior_features.get('video_streaming', 0) > 0.5:
                signal_strength *= 1.2
            
            # IoT设备对网络要求较低
            elif behavior_features.get('iot_sensor', 0) > 0.5:
                base_latency *= 3.0  # 可以容忍较高延迟
                signal_strength *= 0.7
        
        # 根据切片类型调整延迟
        if profile.preferred_slice == 'URLLC':
            latency = base_latency * 0.1 + np.random.exponential(2)
        elif profile.preferred_slice == 'eMBB':
            latency = base_latency + np.random.exponential(10)
        else:  # mMTC
            latency = base_latency * 2 + np.random.exponential(50)
        
        # 吞吐量
        base_throughput = signal_strength * 100  # Mbps
        
        # 根据行为调整吞吐量需求
        if behavior_features:
            throughput_multiplier = 1.0
            
            if behavior_features.get('live_streaming', 0) > 0.5:
                throughput_multiplier *= 5.0  # 直播需要最高带宽
            elif behavior_features.get('vr_ar_gaming', 0) > 0.5:
                throughput_multiplier *= 4.0  # VR/AR需要极高带宽
            elif behavior_features.get('cloud_gaming', 0) > 0.5:
                throughput_multiplier *= 3.5  # 云游戏需要高带宽
            elif behavior_features.get('file_download', 0) > 0.5:
                throughput_multiplier *= 3.0  # 文件下载需要高带宽
            elif behavior_features.get('video_streaming', 0) > 0.5:
                throughput_multiplier *= 2.5  # 视频流需要高带宽
            elif behavior_features.get('fps_gaming', 0) > 0.5:
                throughput_multiplier *= 1.8  # FPS游戏中等带宽需求
            elif behavior_features.get('video_calling', 0) > 0.5:
                throughput_multiplier *= 1.5  # 视频通话中等带宽
            elif behavior_features.get('online_gaming', 0) > 0.5:
                throughput_multiplier *= 1.2  # 网络游戏较低带宽需求
            elif behavior_features.get('web_browsing', 0) > 0.5:
                throughput_multiplier *= 0.3  # 网页浏览低带宽需求
            elif behavior_features.get('iot_sensor', 0) > 0.5:
                throughput_multiplier *= 0.05  # IoT设备带宽需求极低
            
            base_throughput *= throughput_multiplier
        
        # 根据切片类型调整吞吐量
        if profile.preferred_slice == 'eMBB':
            throughput = base_throughput * np.random.uniform(0.8, 1.2)
        elif profile.preferred_slice == 'URLLC':
            throughput = base_throughput * 0.3 * np.random.uniform(0.9, 1.1)
        else:  # mMTC
            throughput = base_throughput * 0.1 * np.random.uniform(0.5, 1.5)
        
        # 根据行为调整丢包率和抖动
        packet_loss = np.random.exponential(0.01)
        jitter = np.random.exponential(5.0)
        
        if behavior_features:
            # 游戏类应用对丢包和抖动非常敏感
            if (behavior_features.get('vr_ar_gaming', 0) > 0.5 or 
                behavior_features.get('fps_gaming', 0) > 0.5 or
                behavior_features.get('cloud_gaming', 0) > 0.5):
                packet_loss *= 0.05  # 游戏对丢包极度敏感
                jitter *= 0.05       # 游戏需要极低抖动
            elif behavior_features.get('online_gaming', 0) > 0.5:
                packet_loss *= 0.1   # 网络游戏对丢包敏感
                jitter *= 0.1        # 需要低抖动
            elif behavior_features.get('live_streaming', 0) > 0.5:
                packet_loss *= 0.2   # 直播对丢包较敏感
                jitter *= 0.3        # 直播需要相对稳定的网络
            elif behavior_features.get('video_calling', 0) > 0.5:
                packet_loss *= 0.3   # 视频通话对丢包敏感
                jitter *= 0.4        # 视频通话需要稳定网络
        
        return {
            'signal_strength': np.clip(signal_strength, 0.1, 1.0),
            'latency': max(1.0, latency),
            'throughput': max(0.1, throughput),
            'packet_loss': max(0.0, packet_loss),
            'jitter': max(0.0, jitter)
        }
    
    def _get_location_type(self, timestamp: datetime) -> str:
        """根据时间推断位置类型"""
        hour = timestamp.hour
        
        if 22 <= hour or hour <= 6:
            return 'home'
        elif 9 <= hour <= 17:
            return 'office'
        elif 18 <= hour <= 21:
            return 'commercial'
        else:
            return 'transit'
    
    def generate_dataset(self, num_users: int = 100, 
                        duration_hours: int = 24,
                        sampling_interval: int = 60) -> List[UserData]:
        """生成完整数据集"""
        profiles = self.generate_user_profiles(num_users)
        all_data = []
        
        for profile in profiles:
            user_data = self.generate_trajectory(
                profile, duration_hours, sampling_interval
            )
            all_data.extend(user_data)
        
        return all_data
    
    def test_behavior_features(self, num_users: int = 5, duration_hours: int = 1) -> Dict[str, Any]:
        """测试新增的用户行为特征功能"""
        print("\n📦 测试新增的用户行为特征...")
        
        # 生成测试数据
        test_data = self.generate_dataset(num_users, duration_hours, 300)  # 5分钟间隔
        
        # 统计新增行为特征的使用情况
        behavior_stats = {}
        for behavior_name in self.behavior_types.keys():
            behavior_stats[behavior_name] = []
        
        # 收集所有数据点的行为特征
        for data_point in test_data:
            behavior_features = data_point.behavior_features
            for behavior_name in self.behavior_types.keys():
                if behavior_name in behavior_features:
                    behavior_stats[behavior_name].append(behavior_features[behavior_name])
        
        # 计算统计信息
        test_results = {
            'total_data_points': len(test_data),
            'unique_users': len(set(d.user_id for d in test_data)),
            'behavior_statistics': {},
            'slice_type_distribution': {},
            'network_impact_analysis': {}
        }
        
        print(f"   ✅ 生成测试数据: {len(test_data)} 个数据点")
        print(f"   👥 涉及用户: {len(set(d.user_id for d in test_data))} 个")
        print("\n   🎯 新增行为特征统计:")
        
        for behavior_name, values in behavior_stats.items():
            if values:  # 只处理有数据的行为
                behavior_info = self.behavior_types[behavior_name]
                avg_usage = np.mean(values)
                max_usage = np.max(values)
                usage_rate = sum(1 for v in values if v > 0.1) / len(values)  # 活跃率
                
                test_results['behavior_statistics'][behavior_name] = {
                    'average_usage': avg_usage,
                    'max_usage': max_usage,
                    'usage_rate': usage_rate,
                    'total_samples': len(values),
                    'bandwidth_req': behavior_info['bandwidth_req'],
                    'latency_req': behavior_info['latency_req'],
                    'slice_preference': behavior_info['slice_preference']
                }
                
                print(f"     • {behavior_name}:")
                print(f"       - 平均使用率: {avg_usage:.3f}")
                print(f"       - 最大使用率: {max_usage:.3f}")
                print(f"       - 活跃用户比例: {usage_rate*100:.1f}%")
                print(f"       - 带宽需求: {behavior_info['bandwidth_req']}, 延迟需求: {behavior_info['latency_req']}")
        
        # 分析网络切片分布
        slice_distribution = {}
        for data_point in test_data:
            slice_type = data_point.slice_type
            if slice_type not in slice_distribution:
                slice_distribution[slice_type] = 0
            slice_distribution[slice_type] += 1
        
        print("\n   🔌 网络切片分布:")
        for slice_type, count in slice_distribution.items():
            percentage = (count / len(test_data)) * 100
            test_results['slice_type_distribution'][slice_type] = {
                'count': count,
                'percentage': percentage
            }
            print(f"     • {slice_type}: {count} 个 ({percentage:.1f}%)")
        
        # 分析网络指标影响
        print("\n   📊 网络指标影响分析:")
        
        # 按行为类型分组分析网络指标
        behavior_network_impact = {}
        
        for behavior_name in self.behavior_types.keys():
            # 找到使用该行为的数据点
            relevant_data = []
            for data_point in test_data:
                if data_point.behavior_features.get(behavior_name, 0) > 0.5:
                    relevant_data.append(data_point)
            
            if relevant_data:
                # 计算平均网络指标
                avg_latency = np.mean([d.network_metrics['latency'] for d in relevant_data])
                avg_throughput = np.mean([d.network_metrics['throughput'] for d in relevant_data])
                avg_signal = np.mean([d.network_metrics['signal_strength'] for d in relevant_data])
                
                behavior_network_impact[behavior_name] = {
                    'sample_count': len(relevant_data),
                    'avg_latency': avg_latency,
                    'avg_throughput': avg_throughput,
                    'avg_signal_strength': avg_signal
                }
                
                print(f"     • {behavior_name} (样本数: {len(relevant_data)}):")
                print(f"       - 平均延迟: {avg_latency:.2f} ms")
                print(f"       - 平均吐量: {avg_throughput:.2f} Mbps")
                print(f"       - 平均信号强度: {avg_signal:.3f}")
        
        test_results['network_impact_analysis'] = behavior_network_impact
        
        print("\n   ✅ 用户行为特征测试完成！")
        
        return test_results