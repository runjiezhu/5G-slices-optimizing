"""
5G网络切片优化系统独立演示
完全独立运行，不依赖复杂的深度学习框架
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class SliceType(Enum):
    """网络切片类型"""
    eMBB = "增强型移动宽带"
    URLLC = "超可靠低时延通信"
    mMTC = "大规模机器类型通信"


@dataclass
class UserData:
    """用户数据结构"""
    user_id: str
    timestamp: datetime
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    behavior: Dict[str, float]
    network_metrics: Dict[str, float]
    slice_type: str


class SimpleDataGenerator:
    """简化的数据生成器"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_user_data(self, num_users: int = 10, hours: int = 1) -> List[UserData]:
        """生成用户数据"""
        data_list = []
        start_time = datetime.now()
        
        for user_idx in range(num_users):
            user_id = f"user_{user_idx:03d}"
            
            # 确定用户类型和偏好
            user_type = random.choice(['business', 'personal', 'iot'])
            if user_type == 'business':
                preferred_slice = SliceType.URLLC.name
                activity_level = np.random.uniform(0.7, 1.0)
            elif user_type == 'personal':
                preferred_slice = SliceType.eMBB.name
                activity_level = np.random.uniform(0.4, 0.8)
            else:
                preferred_slice = SliceType.mMTC.name
                activity_level = np.random.uniform(0.1, 0.3)
            
            # 生成该用户的时间序列数据
            num_points = hours * 60  # 每分钟一个数据点
            
            # 初始位置
            base_x = np.random.uniform(-50, 50)
            base_y = np.random.uniform(-50, 50)
            
            for minute in range(num_points):
                timestamp = start_time + timedelta(minutes=minute)
                
                # 位置（简单的随机游走）
                noise_x = np.random.normal(0, 1.0)
                noise_y = np.random.normal(0, 1.0) 
                position = (
                    base_x + noise_x,
                    base_y + noise_y,
                    np.random.uniform(0, 2)  # 高度
                )                
                # 速度
                velocity = (
                    noise_x / 60,  # m/s
                    noise_y / 60,
                    np.random.normal(0, 0.1)
                )
                
                # 行为特征
                hour = timestamp.hour
                behavior = {
                    'data_usage': np.random.lognormal(2, 1) * activity_level,
                    'call_duration': np.random.exponential(30) if np.random.random() < 0.2 else 0,
                    'app_video': 0.8 if hour >= 19 and hour <= 22 else 0.2,
                    'app_social': 0.6 if hour >= 8 and hour <= 23 else 0.1,
                    'app_work': 0.9 if hour >= 9 and hour <= 17 and user_type == 'business' else 0.1,
                    'is_peak_hour': 1.0 if (8 <= hour <= 10) or (17 <= hour <= 19) else 0.0
                }
                
                # 网络指标
                distance_from_center = np.sqrt(position[0]**2 + position[1]**2)
                signal_strength = max(0.1, 1.0 - distance_from_center / 100.0)
                
                network_metrics = {
                    'signal_strength': signal_strength + np.random.normal(0, 0.1),
                    'latency': max(1, 10 + distance_from_center * 0.5 + np.random.exponential(5)),
                    'throughput': max(0.1, signal_strength * 100 * np.random.uniform(0.8, 1.2)),
                    'packet_loss': np.random.exponential(0.01),
                    'jitter': np.random.exponential(2.0)
                }
                
                data_point = UserData(
                    user_id=user_id,
                    timestamp=timestamp,
                    position=position,
                    velocity=velocity,
                    behavior=behavior,
                    network_metrics=network_metrics,
                    slice_type=preferred_slice
                )
                
                data_list.append(data_point)
        
        return data_list


def softmax(x):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def simulate_ai_prediction(features: np.ndarray) -> Dict:
    """模拟AI预测"""
    num_users = features.shape[0]
    
    # 模拟Transformer预测
    slice_logits = np.random.randn(num_users, 3)  # 3种切片类型
    slice_probs = np.array([softmax(logits) for logits in slice_logits])
    
    # 带宽需求预测
    bandwidth_pred = np.random.uniform(5, 200, num_users)
    
    # 置信度
    confidence = np.random.uniform(0.6, 0.95, num_users)
    
    return {
        'slice_probabilities': slice_probs,
        'bandwidth_demand': bandwidth_pred,
        'confidence': confidence
    }


def optimize_network_slices(predictions: Dict, user_data: List[UserData]) -> List[Dict]:
    """网络切片优化"""
    allocations = []
    
    # 总资源
    total_bandwidth = 1000.0  # Mbps
    remaining_bandwidth = total_bandwidth
    
    slice_names = ['eMBB', 'URLLC', 'mMTC']
    slice_priorities = [1, 3, 2]
    
    # 按优先级排序用户
    user_priorities = []
    for i, user in enumerate(user_data):
        slice_idx = np.argmax(predictions['slice_probabilities'][i])
        priority = slice_priorities[slice_idx]
        user_priorities.append((priority, i, user))
    
    user_priorities.sort(key=lambda x: x[0], reverse=True)
    
    # 分配资源
    for priority, i, user in user_priorities:
        if remaining_bandwidth <= 0:
            break
        
        slice_idx = np.argmax(predictions['slice_probabilities'][i])
        slice_type = slice_names[slice_idx]
        
        # 预测带宽需求
        demanded_bandwidth = predictions['bandwidth_demand'][i]
        
        # 分配策略
        if slice_type == 'URLLC':
            # 高优先级，保证最小带宽
            allocated = min(max(demanded_bandwidth, 10), remaining_bandwidth)
        elif slice_type == 'eMBB':
            # 中优先级，根据需求和可用性
            allocated = min(demanded_bandwidth, remaining_bandwidth * 0.3)
        else:  # mMTC
            # 低优先级，分配较少带宽
            allocated = min(demanded_bandwidth, remaining_bandwidth * 0.1, 20)
        
        remaining_bandwidth -= allocated
        
        # 满意度计算
        satisfaction = min(1.0, allocated / max(demanded_bandwidth, 1))
        
        allocation = {
            'user_id': user.user_id,
            'slice_type': slice_type,
            'allocated_bandwidth': allocated,
            'demanded_bandwidth': demanded_bandwidth,
            'satisfaction': satisfaction,
            'confidence': predictions['confidence'][i],
            'priority': priority
        }
        
        allocations.append(allocation)
    
    return allocations


def run_standalone_demo():
    """运行独立演示"""
    print("🌐 5G动态网络切片优化系统 - 独立演示")
    print("=" * 70)
    
    # 1. 数据生成
    print("📊 1. 生成5G用户数据...")
    generator = SimpleDataGenerator(seed=42)
    user_data = generator.generate_user_data(num_users=20, hours=2)
    
    print(f"   ✅ 生成了 {len(user_data)} 条数据记录")
    print(f"   👥 涵盖 {len(set(d.user_id for d in user_data))} 个用户")
    print(f"   ⏰ 时间跨度: 2小时")
    
    # 2. 数据分析
    print("\n📈 2. 数据统计分析...")
    
    # 转换为DataFrame便于分析
    records = []
    for data in user_data:
        record = {
            'user_id': data.user_id,
            'timestamp': data.timestamp,
            'pos_x': data.position[0],
            'pos_y': data.position[1],
            'slice_type': data.slice_type,
            'data_usage': data.behavior['data_usage'],
            'signal_strength': data.network_metrics['signal_strength'],
            'latency': data.network_metrics['latency'],
            'throughput': data.network_metrics['throughput']
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # 切片类型分布
    slice_dist = df['slice_type'].value_counts()
    print("   🔧 切片类型分布:")
    for slice_type, count in slice_dist.items():
        print(f"     • {slice_type}: {count} ({count/len(df)*100:.1f}%)")
    
    # 基础统计
    print(f"   📊 平均数据使用: {df['data_usage'].mean():.2f} MB")
    print(f"   📶 平均信号强度: {df['signal_strength'].mean():.3f}")
    print(f"   ⏱️  平均延迟: {df['latency'].mean():.2f} ms")
    print(f"   🚀 平均吞吐量: {df['throughput'].mean():.2f} Mbps")
    
    # 3. 用户行为分析
    print("\n🎯 3. 用户行为模式分析...")
    
    user_stats = []
    for user_id in df['user_id'].unique()[:8]:  # 分析前8个用户
        user_df = df[df['user_id'] == user_id]
        
        # 移动模式
        positions = user_df[['pos_x', 'pos_y']].values
        if len(positions) > 1:
            distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            total_distance = np.sum(distances)
            max_distance = np.max(distances) if len(distances) > 0 else 0
        else:
            total_distance = 0
            max_distance = 0
        
        # 行为特征
        avg_usage = user_df['data_usage'].mean()
        preferred_slice = user_df['slice_type'].iloc[0]
        
        user_stats.append({
            'user_id': user_id,
            'total_movement': total_distance,
            'max_step': max_distance,
            'avg_data_usage': avg_usage,
            'preferred_slice': preferred_slice
        })
        
        print(f"   📱 {user_id}: 移动距离 {total_distance:.1f}m, "
              f"数据使用 {avg_usage:.1f}MB, 偏好切片 {preferred_slice}")
    
    # 4. AI预测模拟
    print("\n🧠 4. AI预测算法模拟...")
    
    # 提取特征
    latest_data = []
    unique_users = df['user_id'].unique()
    
    for user_id in unique_users:
        user_df = df[df['user_id'] == user_id].tail(10)  # 最近10个数据点
        
        if len(user_df) >= 5:
            features = [
                user_df['pos_x'].mean(),
                user_df['pos_y'].mean(),
                user_df['data_usage'].mean(),
                user_df['signal_strength'].mean(),
                user_df['latency'].mean(),
                user_df['throughput'].mean(),
                user_df['data_usage'].std(),
                len(user_df)
            ]
            latest_data.append({
                'user_id': user_id,
                'features': features,
                'actual_slice': user_df['slice_type'].iloc[-1]
            })
    
    features_matrix = np.array([d['features'] for d in latest_data])
    print(f"   ✅ 提取了 {len(latest_data)} 个用户的特征向量")
    
    # 执行预测
    predictions = simulate_ai_prediction(features_matrix)
    print(f"   🎯 预测完成，平均置信度: {np.mean(predictions['confidence']):.3f}")
    
    # 5. 网络切片优化
    print("\n⚡ 5. 网络切片优化分配...")
    
    # 获取每个用户的最新数据
    user_latest = []
    for d in latest_data:
        user_data_point = [ud for ud in user_data if ud.user_id == d['user_id']][-1]
        user_latest.append(user_data_point)
    
    allocations = optimize_network_slices(predictions, user_latest)
    
    print(f"   ✅ 完成 {len(allocations)} 个用户的切片分配")
    
    # 统计结果
    total_allocated = sum(a['allocated_bandwidth'] for a in allocations)
    print(f"   📊 总分配带宽: {total_allocated:.1f} Mbps / 1000 Mbps")
    print(f"   📈 带宽利用率: {total_allocated/1000*100:.1f}%")
    
    # 6. 结果分析
    print("\n📋 6. 优化结果分析...")
    
    # 按切片类型统计
    slice_stats = {}
    for alloc in allocations:
        slice_type = alloc['slice_type']
        if slice_type not in slice_stats:
            slice_stats[slice_type] = {'count': 0, 'bandwidth': 0, 'satisfaction': []}
        
        slice_stats[slice_type]['count'] += 1
        slice_stats[slice_type]['bandwidth'] += alloc['allocated_bandwidth']
        slice_stats[slice_type]['satisfaction'].append(alloc['satisfaction'])
    
    print("   🎯 按切片类型统计:")
    for slice_type, stats in slice_stats.items():
        avg_satisfaction = np.mean(stats['satisfaction'])
        print(f"     • {slice_type}:")
        print(f"       - 用户数: {stats['count']}")
        print(f"       - 总带宽: {stats['bandwidth']:.1f} Mbps")
        print(f"       - 平均满意度: {avg_satisfaction:.3f}")
    
    # 显示详细分配结果
    print("\n   📊 详细分配结果(前10个用户):")
    for i, alloc in enumerate(allocations[:10]):
        print(f"     {i+1:2d}. {alloc['user_id']} | "
              f"{alloc['slice_type']:6s} | "
              f"分配: {alloc['allocated_bandwidth']:6.1f} Mbps | "
              f"需求: {alloc['demanded_bandwidth']:6.1f} Mbps | "
              f"满意度: {alloc['satisfaction']:.3f}")
    
    # 7. 性能指标
    print("\n⚡ 7. 系统性能指标...")
    
    # 总体满意度
    avg_satisfaction = np.mean([a['satisfaction'] for a in allocations])
    print(f"   📈 平均用户满意度: {avg_satisfaction*100:.1f}%")
    
    # 高优先级用户服务质量
    high_priority = [a for a in allocations if a['priority'] >= 3]
    if high_priority:
        hp_satisfaction = np.mean([a['satisfaction'] for a in high_priority])
        print(f"   🔥 高优先级用户满意度: {hp_satisfaction*100:.1f}%")
    
    # 资源利用效率
    bandwidth_efficiency = total_allocated / 1000.0
    print(f"   💪 带宽利用效率: {bandwidth_efficiency*100:.1f}%")
    
    # 预测准确性（简化评估）
    correct_predictions = 0
    for i, alloc in enumerate(allocations):
        actual_slice = latest_data[i]['actual_slice']
        predicted_slice = alloc['slice_type']
        if actual_slice == predicted_slice:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(allocations) if allocations else 0
    print(f"   🎯 切片预测准确率: {accuracy*100:.1f}%")
    
    # 8. 导出结果
    print("\n💾 8. 导出演示结果...")
    
    export_data = {
        'demo_info': {
            'timestamp': datetime.now().isoformat(),
            'total_users': len(unique_users),
            'total_data_points': len(user_data),
            'duration_hours': 2
        },
        'performance_metrics': {
            'bandwidth_utilization': bandwidth_efficiency,
            'average_satisfaction': avg_satisfaction,
            'prediction_accuracy': accuracy,
            'total_allocated_bandwidth': total_allocated
        },
        'slice_statistics': slice_stats,
        'sample_allocations': allocations[:10]  # 保存前10个结果
    }
    
    # 保存为JSON文件
    filename = f"standalone_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"   ✅ 结果已导出到: {filename}")
    
    # 9. 总结
    print("\n" + "=" * 70)
    print("🎉 独立演示完成！")
    print("\n✨ 演示特色:")
    print("   🎯 完全独立运行，无需复杂依赖")
    print("   📊 真实的5G用户数据模拟")
    print("   🧠 AI预测算法效果展示")
    print("   ⚡ 智能网络切片优化")
    print("   📈 详细的性能分析报告")
    
    print(f"\n📊 核心指标:")
    print(f"   • 处理用户数: {len(unique_users)}")
    print(f"   • 带宽利用率: {bandwidth_efficiency*100:.1f}%")
    print(f"   • 用户满意度: {avg_satisfaction*100:.1f}%") 
    print(f"   • 预测准确率: {accuracy*100:.1f}%")
    
    print("\n🚀 完整版功能:")
    print("   • 安装PyTorch后可运行完整的Transformer模型")
    print("   • 支持模型训练和实时预测")
    print("   • 提供Web可视化仪表板")
    
    return export_data


if __name__ == "__main__":
    try:
        result = run_standalone_demo()
        print(f"\n✅ 演示成功完成！")
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()