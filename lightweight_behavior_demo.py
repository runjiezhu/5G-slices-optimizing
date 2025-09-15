"""
轻量化用户行为特征演示
使用简单的统计模型和规则引擎，展示VR/AR、FPS游戏、云游戏、网络游戏、视频流、直播流等行为特征
避免PyTorch等重型依赖，专注于行为建模的核心逻辑
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class UserBehaviorProfile:
    """用户行为画像"""
    user_id: str
    user_type: str  # 'business', 'personal', 'iot'
    primary_behaviors: List[str]  # 主要行为类型
    activity_level: float  # 活跃度 0-1
    time_preferences: Dict[str, float]  # 时间偏好
    device_type: str  # 设备类型
    network_sensitivity: str  # 网络敏感度 'high', 'medium', 'low'


class LightweightBehaviorEngine:
    """轻量化行为引擎"""
    
    def __init__(self, seed: int = 42):
        """初始化行为引擎"""
        np.random.seed(seed)
        random.seed(seed)
        
        # 扩展的用户行为类型配置
        self.behavior_configs = {
            # 游戏类行为
            'vr_ar_gaming': {
                'chinese_name': 'VR/AR游戏',
                'bandwidth_need': 'ultra_high',  # 200-1000 Mbps
                'latency_need': 'ultra_low',     # <1ms
                'slice_type': 'URLLC',
                'data_intensity': 5.0,
                'realtime_critical': True,
                'typical_duration': 120,  # 分钟
                'peak_hours': [19, 20, 21, 22]
            },
            'fps_gaming': {
                'chinese_name': 'FPS射击游戏',
                'bandwidth_need': 'medium_high',  # 30-100 Mbps
                'latency_need': 'ultra_low',      # <5ms
                'slice_type': 'URLLC',
                'data_intensity': 1.5,
                'realtime_critical': True,
                'typical_duration': 90,
                'peak_hours': [18, 19, 20, 21, 22, 23]
            },
            'cloud_gaming': {
                'chinese_name': '云游戏',
                'bandwidth_need': 'high',         # 50-200 Mbps
                'latency_need': 'ultra_low',      # <10ms
                'slice_type': 'URLLC',
                'data_intensity': 3.5,
                'realtime_critical': True,
                'typical_duration': 150,
                'peak_hours': [17, 18, 19, 20, 21]
            },
            'online_gaming': {
                'chinese_name': '网络游戏',
                'bandwidth_need': 'medium',       # 10-50 Mbps
                'latency_need': 'low',            # <20ms
                'slice_type': 'URLLC',
                'data_intensity': 1.0,
                'realtime_critical': True,
                'typical_duration': 180,
                'peak_hours': [19, 20, 21, 22]
            },
            
            # 流媒体类行为
            'video_streaming': {
                'chinese_name': '视频流媒体',
                'bandwidth_need': 'high',         # 50-200 Mbps
                'latency_need': 'low',            # <20ms
                'slice_type': 'eMBB',
                'data_intensity': 2.5,
                'realtime_critical': False,
                'typical_duration': 120,
                'peak_hours': [19, 20, 21, 22]
            },
            'live_streaming': {
                'chinese_name': '直播流媒体',
                'bandwidth_need': 'very_high',    # 100-500 Mbps
                'latency_need': 'low',            # <10ms
                'slice_type': 'eMBB',
                'data_intensity': 4.0,
                'realtime_critical': True,
                'typical_duration': 180,
                'peak_hours': [19, 20, 21, 22, 23]
            },
            'video_calling': {
                'chinese_name': '视频通话',
                'bandwidth_need': 'medium',       # 10-50 Mbps
                'latency_need': 'low',            # <20ms
                'slice_type': 'eMBB',
                'data_intensity': 1.2,
                'realtime_critical': True,
                'typical_duration': 60,
                'peak_hours': [9, 10, 14, 15, 16]
            },
            
            # 其他应用行为
            'file_download': {
                'chinese_name': '文件下载',
                'bandwidth_need': 'very_high',    # 100-500 Mbps
                'latency_need': 'medium',         # <100ms
                'slice_type': 'eMBB',
                'data_intensity': 3.0,
                'realtime_critical': False,
                'typical_duration': 30,
                'peak_hours': [9, 10, 13, 14, 15]
            },
            'web_browsing': {
                'chinese_name': '网页浏览',
                'bandwidth_need': 'low',          # 1-10 Mbps
                'latency_need': 'medium',         # <100ms
                'slice_type': 'eMBB',
                'data_intensity': 0.3,
                'realtime_critical': False,
                'typical_duration': 45,
                'peak_hours': [8, 9, 13, 14, 15, 16]
            },
            'iot_sensor': {
                'chinese_name': 'IoT传感器',
                'bandwidth_need': 'very_low',     # 0.1-1 Mbps
                'latency_need': 'medium',         # <100ms
                'slice_type': 'mMTC',
                'data_intensity': 0.05,
                'realtime_critical': False,
                'typical_duration': 1440,  # 全天
                'peak_hours': list(range(24))  # 全天
            }
        }
        
        # 带宽需求映射 (Mbps)
        self.bandwidth_ranges = {
            'very_low': (0.1, 1.0),
            'low': (1.0, 10.0),
            'medium': (10.0, 50.0),
            'medium_high': (30.0, 100.0),
            'high': (50.0, 200.0),
            'very_high': (100.0, 500.0),
            'ultra_high': (200.0, 1000.0)
        }
        
        # 延迟需求映射 (ms)
        self.latency_ranges = {
            'ultra_low': (1, 5),
            'low': (5, 20),
            'medium': (20, 100),
            'high': (100, 500)
        }
        
        print("✅ 轻量化行为引擎初始化完成")
        print(f"📊 支持 {len(self.behavior_configs)} 种用户行为类型")
    
    def generate_user_profile(self, user_id: str) -> UserBehaviorProfile:
        """生成用户行为画像"""
        # 随机选择用户类型
        user_type = np.random.choice(['business', 'personal', 'iot'], p=[0.3, 0.6, 0.1])
        
        # 根据用户类型生成行为偏好
        if user_type == 'business':
            primary_behaviors = np.random.choice(
                ['video_calling', 'file_download', 'web_browsing', 'online_gaming'],
                size=np.random.randint(1, 3), replace=False
            ).tolist()
            activity_level = np.random.uniform(0.6, 0.9)
            device_type = np.random.choice(['laptop', 'desktop', 'tablet'])
            network_sensitivity = 'high'
            time_prefs = {'work_hours': 0.8, 'evening': 0.4, 'night': 0.1}
            
        elif user_type == 'personal':
            primary_behaviors = np.random.choice(
                ['vr_ar_gaming', 'fps_gaming', 'cloud_gaming', 'video_streaming', 
                 'live_streaming', 'online_gaming', 'web_browsing'],
                size=np.random.randint(2, 4), replace=False
            ).tolist()
            activity_level = np.random.uniform(0.4, 0.8)
            device_type = np.random.choice(['smartphone', 'gaming_console', 'pc', 'vr_headset'])
            network_sensitivity = 'high' if 'vr_ar_gaming' in primary_behaviors else 'medium'
            time_prefs = {'work_hours': 0.2, 'evening': 0.9, 'night': 0.6}
            
        else:  # iot
            primary_behaviors = ['iot_sensor']
            activity_level = np.random.uniform(0.8, 1.0)  # IoT设备通常持续工作
            device_type = np.random.choice(['sensor', 'smart_device', 'gateway'])
            network_sensitivity = 'low'
            time_prefs = {'work_hours': 1.0, 'evening': 1.0, 'night': 1.0}  # 全天
        
        return UserBehaviorProfile(
            user_id=user_id,
            user_type=user_type,
            primary_behaviors=primary_behaviors,
            activity_level=activity_level,
            time_preferences=time_prefs,
            device_type=device_type,
            network_sensitivity=network_sensitivity
        )
    
    def simulate_behavior_session(self, profile: UserBehaviorProfile, 
                                current_hour: int) -> Dict[str, any]:
        """模拟用户行为会话"""
        session_data = {
            'user_id': profile.user_id,
            'timestamp': datetime.now().replace(hour=current_hour),
            'active_behaviors': {},
            'network_demands': {},
            'predicted_slice': None,
            'session_metrics': {}
        }
        
        # 确定时间段
        if 6 <= current_hour <= 18:
            time_period = 'work_hours'
        elif 18 <= current_hour <= 23:
            time_period = 'evening'
        else:
            time_period = 'night'
        
        base_activity = profile.time_preferences.get(time_period, 0.5)
        
        # 模拟每种行为的活跃度
        total_bandwidth = 0
        min_latency = 1000
        preferred_slices = []
        
        for behavior in profile.primary_behaviors:
            if behavior not in self.behavior_configs:
                continue
                
            config = self.behavior_configs[behavior]
            
            # 计算该行为在当前时间的活跃概率
            hour_boost = 2.0 if current_hour in config['peak_hours'] else 1.0
            activity_prob = base_activity * profile.activity_level * hour_boost * 0.3
            
            # 添加随机性
            if np.random.random() < activity_prob:
                # 行为活跃
                intensity = np.random.uniform(0.3, 1.0)
                session_data['active_behaviors'][behavior] = {
                    'intensity': intensity,
                    'chinese_name': config['chinese_name'],
                    'duration_minutes': config['typical_duration'] * intensity
                }
                
                # 计算网络需求
                bw_range = self.bandwidth_ranges[config['bandwidth_need']]
                lat_range = self.latency_ranges[config['latency_need']]
                
                bandwidth_need = np.random.uniform(bw_range[0], bw_range[1]) * intensity
                latency_need = np.random.uniform(lat_range[0], lat_range[1])
                
                total_bandwidth += bandwidth_need
                min_latency = min(min_latency, latency_need)
                preferred_slices.append(config['slice_type'])
                
                session_data['network_demands'][behavior] = {
                    'bandwidth_mbps': bandwidth_need,
                    'latency_ms': latency_need,
                    'data_intensity': config['data_intensity'] * intensity
                }
        
        # 决定推荐的网络切片
        if preferred_slices:
            # 根据优先级选择切片：URLLC > eMBB > mMTC
            if 'URLLC' in preferred_slices:
                recommended_slice = 'URLLC'
            elif 'eMBB' in preferred_slices:
                recommended_slice = 'eMBB'
            else:
                recommended_slice = 'mMTC'
        else:
            recommended_slice = 'eMBB'  # 默认
        
        session_data['predicted_slice'] = recommended_slice
        session_data['session_metrics'] = {
            'total_bandwidth_demand': total_bandwidth,
            'min_latency_requirement': min_latency if min_latency < 1000 else 100,
            'num_active_behaviors': len(session_data['active_behaviors']),
            'activity_level': base_activity * profile.activity_level
        }
        
        return session_data
    
    def run_behavior_simulation(self, num_users: int = 10, 
                              simulation_hours: int = 24) -> List[Dict]:
        """运行行为模拟"""
        print(f"\n🎯 开始用户行为模拟")
        print(f"   👥 用户数量: {num_users}")
        print(f"   ⏰ 模拟时长: {simulation_hours} 小时")
        
        # 生成用户画像
        user_profiles = []
        for i in range(num_users):
            profile = self.generate_user_profile(f"user_{i:03d}")
            user_profiles.append(profile)
        
        # 运行模拟
        all_sessions = []
        for hour in range(simulation_hours):
            for profile in user_profiles:
                session = self.simulate_behavior_session(profile, hour % 24)
                if session['active_behaviors']:  # 只记录有活动的会话
                    all_sessions.append(session)
        
        print(f"   📊 生成会话数据: {len(all_sessions)} 个")
        return all_sessions
    
    def analyze_simulation_results(self, sessions: List[Dict]) -> Dict:
        """分析模拟结果"""
        print(f"\n📈 行为模拟结果分析")
        print("=" * 40)
        
        if not sessions:
            print("⚠️ 没有会话数据可分析")
            return {}
        
        # 统计行为使用频率
        behavior_stats = {}
        slice_distribution = {}
        hourly_activity = {hour: 0 for hour in range(24)}
        
        total_bandwidth = 0
        total_latency = 0
        latency_samples = 0
        
        for session in sessions:
            # 统计行为频率
            for behavior, data in session['active_behaviors'].items():
                if behavior not in behavior_stats:
                    behavior_stats[behavior] = {
                        'count': 0,
                        'total_intensity': 0,
                        'total_duration': 0,
                        'chinese_name': data['chinese_name']
                    }
                behavior_stats[behavior]['count'] += 1
                behavior_stats[behavior]['total_intensity'] += data['intensity']
                behavior_stats[behavior]['total_duration'] += data['duration_minutes']
            
            # 统计切片分布
            slice_type = session['predicted_slice']
            slice_distribution[slice_type] = slice_distribution.get(slice_type, 0) + 1
            
            # 统计小时活动
            hour = session['timestamp'].hour
            hourly_activity[hour] += 1
            
            # 统计网络指标
            metrics = session['session_metrics']
            total_bandwidth += metrics['total_bandwidth_demand']
            if metrics['min_latency_requirement'] < 1000:
                total_latency += metrics['min_latency_requirement']
                latency_samples += 1
        
        # 计算统计结果
        print("🎮 行为类型统计:")
        for behavior, stats in sorted(behavior_stats.items(), 
                                    key=lambda x: x[1]['count'], reverse=True):
            avg_intensity = stats['total_intensity'] / stats['count']
            avg_duration = stats['total_duration'] / stats['count']
            usage_rate = stats['count'] / len(sessions) * 100
            
            print(f"  • {stats['chinese_name']} ({behavior}):")
            print(f"    - 使用频率: {stats['count']} 次 ({usage_rate:.1f}%)")
            print(f"    - 平均强度: {avg_intensity:.3f}")
            print(f"    - 平均时长: {avg_duration:.1f} 分钟")
        
        print(f"\n🔌 网络切片分布:")
        for slice_type, count in sorted(slice_distribution.items(), 
                                      key=lambda x: x[1], reverse=True):
            percentage = count / len(sessions) * 100
            print(f"  • {slice_type}: {count} 次 ({percentage:.1f}%)")
        
        print(f"\n📊 网络需求统计:")
        avg_bandwidth = total_bandwidth / len(sessions)
        avg_latency = total_latency / latency_samples if latency_samples > 0 else 0
        print(f"  • 平均带宽需求: {avg_bandwidth:.1f} Mbps")
        print(f"  • 平均延迟需求: {avg_latency:.1f} ms")
        print(f"  • 活跃会话总数: {len(sessions)}")
        
        # 找出活动高峰时段
        peak_hours = sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n⏰ 活动高峰时段:")
        for hour, activity in peak_hours:
            print(f"  • {hour:02d}:00 - {activity} 个活跃会话")
        
        return {
            'behavior_stats': behavior_stats,
            'slice_distribution': slice_distribution,
            'hourly_activity': hourly_activity,
            'avg_bandwidth': avg_bandwidth,
            'avg_latency': avg_latency,
            'total_sessions': len(sessions)
        }


def demonstrate_enhanced_behaviors():
    """演示增强的用户行为特征"""
    print("🚀 5G网络切片用户行为特征演示 - 轻量化版本")
    print("=" * 60)
    
    # 初始化行为引擎
    engine = LightweightBehaviorEngine(seed=42)
    
    # 展示支持的行为类型
    print("\n📋 支持的用户行为类型:")
    categories = {
        "🎮 游戏类": ['vr_ar_gaming', 'fps_gaming', 'cloud_gaming', 'online_gaming'],
        "📺 流媒体类": ['video_streaming', 'live_streaming', 'video_calling'],
        "💻 其他应用": ['file_download', 'web_browsing', 'iot_sensor']
    }
    
    for category, behaviors in categories.items():
        print(f"\n{category}:")
        for behavior in behaviors:
            config = engine.behavior_configs[behavior]
            bw_range = engine.bandwidth_ranges[config['bandwidth_need']]
            lat_range = engine.latency_ranges[config['latency_need']]
            
            print(f"  • {config['chinese_name']} ({behavior}):")
            print(f"    带宽: {bw_range[0]:.1f}-{bw_range[1]:.1f} Mbps | "
                  f"延迟: {lat_range[0]}-{lat_range[1]} ms | "
                  f"切片: {config['slice_type']}")
    
    # 运行模拟
    sessions = engine.run_behavior_simulation(num_users=15, simulation_hours=24)
    
    # 分析结果
    analysis = engine.analyze_simulation_results(sessions)
    
    print(f"\n💡 关键洞察:")
    if analysis:
        urllc_ratio = analysis['slice_distribution'].get('URLLC', 0) / analysis['total_sessions'] * 100
        embb_ratio = analysis['slice_distribution'].get('eMBB', 0) / analysis['total_sessions'] * 100
        
        print(f"  • 游戏类应用推动 {urllc_ratio:.1f}% 的会话需要 URLLC 切片")
        print(f"  • 流媒体应用推动 {embb_ratio:.1f}% 的会话需要 eMBB 切片")
        print(f"  • 平均每个会话的带宽需求为 {analysis['avg_bandwidth']:.1f} Mbps")
        
        if analysis['avg_latency'] < 10:
            print(f"  • 大量低延迟应用使平均延迟需求降至 {analysis['avg_latency']:.1f} ms")
    
    print(f"\n✅ 轻量化用户行为特征演示完成！")
    
    return sessions, analysis


def export_simulation_data(sessions: List[Dict], filename: str = "behavior_simulation.json"):
    """导出模拟数据"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # 转换datetime对象为字符串以便JSON序列化
            export_data = []
            for session in sessions:
                session_copy = session.copy()
                session_copy['timestamp'] = session['timestamp'].isoformat()
                export_data.append(session_copy)
            
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"📁 模拟数据已导出到 {filename}")
    except Exception as e:
        print(f"⚠️ 导出失败: {e}")


if __name__ == "__main__":
    try:
        # 运行演示
        sessions, analysis = demonstrate_enhanced_behaviors()
        
        # 导出数据
        export_simulation_data(sessions)
        
        print(f"\n🎉 演示成功完成！")
        print(f"📊 共生成 {len(sessions)} 个行为会话数据")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()