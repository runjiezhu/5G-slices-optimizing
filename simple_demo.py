"""
5G网络切片优化系统简化演示
不依赖PyTorch，展示核心数据处理和分析功能
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import json

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 直接导入需要的模块，避免PyTorch依赖
try:
    from src.data_processing.data_generator import UserDataGenerator
    from src.utils.config_manager import ConfigManager
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("正在使用备用导入方式...")
    
    # 备用：直接在这里定义简化版本
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'data_processing'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'utils'))
    
    try:
        from data_generator import UserDataGenerator
        from config_manager import ConfigManager
    except ImportError:
        print("❌ 无法导入必要模块，将使用内置简化版本")
        UserDataGenerator = None
        ConfigManager = None


def softmax(x, axis=-1):
    """简单的softmax实现"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def simulate_transformer_prediction(input_data: np.ndarray) -> Dict:
    """模拟Transformer预测结果"""
    batch_size = input_data.shape[0]
    
    # 模拟预测结果
    predictions = {
        'slice_type': softmax(np.random.randn(batch_size, 3), axis=1),  # eMBB, URLLC, mMTC
        'bandwidth': np.random.uniform(10, 100, (batch_size, 30, 1)),  # 30个时间步的带宽预测
        'confidence': np.random.uniform(0.6, 0.95, batch_size)
    }
    
    return predictions


def simple_slice_optimization(predictions: Dict, user_demands: List[Dict]) -> List[Dict]:
    """简化的切片优化算法"""
    allocations = []
    
    # 总可用带宽
    total_bandwidth = 1000.0  # Mbps
    remaining_bandwidth = total_bandwidth
    
    # 切片类型映射
    slice_names = ['eMBB', 'URLLC', 'mMTC']
    slice_priorities = [1, 3, 2]  # 优先级
    
    for i, demand in enumerate(user_demands):
        if i >= len(predictions['slice_type']):
            continue
            
        # 预测的切片类型
        predicted_slice_idx = np.argmax(predictions['slice_type'][i])
        slice_type = slice_names[predicted_slice_idx]
        
        # 预测的带宽需求
        predicted_bandwidth = np.mean(predictions['bandwidth'][i])
        
        # 分配带宽（简化算法）
        allocated_bandwidth = min(predicted_bandwidth, remaining_bandwidth * 0.1)
        remaining_bandwidth -= allocated_bandwidth
        
        allocation = {
            'user_id': demand['user_id'],
            'slice_type': slice_type,
            'allocated_bandwidth': allocated_bandwidth,
            'predicted_bandwidth': predicted_bandwidth,
            'confidence': predictions['confidence'][i],
            'priority': slice_priorities[predicted_slice_idx]
        }
        
        allocations.append(allocation)
    
    return allocations


def run_simple_demo():
    """运行简化演示"""
    print("🌐 5G动态网络切片优化系统 - 简化演示")
    print("=" * 60)
    
    # 初始化配置
    config_manager = ConfigManager()
    print("✅ 配置管理器初始化完成")
    
    # 生成模拟用户数据
    print("\n📊 1. 生成模拟5G用户数据...")
    data_generator = UserDataGenerator(seed=42)
    
    user_data = data_generator.generate_dataset(
        num_users=15,
        duration_hours=1,  # 1小时数据
        sampling_interval=60  # 60秒间隔
    )
    
    print(f"   ✅ 生成了 {len(user_data)} 条数据记录")
    print(f"   📈 涵盖 {len(set(data.user_id for data in user_data))} 个用户")
    
    # 数据统计分析
    print("\n📈 2. 数据统计分析...")
    
    # 转换为DataFrame进行分析
    data_records = []
    for data in user_data:
        record = {
            'user_id': data.user_id,
            'timestamp': data.timestamp,
            'pos_x': data.position[0],
            'pos_y': data.position[1],
            'pos_z': data.position[2],
            'vel_x': data.velocity[0],
            'vel_y': data.velocity[1],
            'vel_z': data.velocity[2],
            'slice_type': data.slice_type,
            'data_usage': data.behavior_features.get('data_usage', 0),
            'call_duration': data.behavior_features.get('call_duration', 0),
            'signal_strength': data.network_metrics.get('signal_strength', 0),
            'latency': data.network_metrics.get('latency', 0),
            'throughput': data.network_metrics.get('throughput', 0)
        }
        data_records.append(record)
    
    df = pd.DataFrame(data_records)
    
    # 基础统计
    print(f"   • 数据时间跨度: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"   • 切片类型分布:")
    slice_counts = df['slice_type'].value_counts()
    for slice_type, count in slice_counts.items():
        print(f"     - {slice_type}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"   • 平均数据使用量: {df['data_usage'].mean():.2f} MB")
    print(f"   • 平均信号强度: {df['signal_strength'].mean():.3f}")
    print(f"   • 平均延迟: {df['latency'].mean():.2f} ms")
    print(f"   • 平均吞吐量: {df['throughput'].mean():.2f} Mbps")
    
    # 移动模式分析
    print("\n🚶 3. 用户移动模式分析...")
    for user_id in df['user_id'].unique()[:5]:  # 分析前5个用户
        user_df = df[df['user_id'] == user_id]
        
        # 计算移动距离
        distances = []
        for i in range(1, len(user_df)):
            prev_pos = (user_df.iloc[i-1]['pos_x'], user_df.iloc[i-1]['pos_y'])
            curr_pos = (user_df.iloc[i]['pos_x'], user_df.iloc[i]['pos_y'])
            dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            distances.append(dist)
        
        total_distance = sum(distances)
        avg_speed = np.sqrt(user_df['vel_x']**2 + user_df['vel_y']**2).mean()
        
        print(f"   • {user_id}: 总移动距离 {total_distance:.1f}m, 平均速度 {avg_speed:.2f}m/s")
    
    # 模拟预测过程
    print("\n🧠 4. 模拟AI预测过程...")
    
    # 准备特征数据（简化）
    features = []
    user_demands = []
    
    for user_id in df['user_id'].unique():
        user_df = df[df['user_id'] == user_id].tail(10)  # 取最近10个数据点
        
        if len(user_df) < 5:
            continue
        
        # 简化特征提取
        feature_vector = [
            user_df['pos_x'].mean(),
            user_df['pos_y'].mean(),
            user_df['vel_x'].mean(),
            user_df['vel_y'].mean(),
            user_df['data_usage'].mean(),
            user_df['signal_strength'].mean(),
            user_df['latency'].mean(),
            user_df['throughput'].mean()
        ]
        
        features.append(feature_vector)
        
        # 用户需求
        user_demands.append({
            'user_id': user_id,
            'bandwidth_req': user_df['throughput'].mean(),
            'latency_req': user_df['latency'].mean(),
            'slice_preference': user_df['slice_type'].iloc[-1]
        })
    
    features_array = np.array(features)
    print(f"   ✅ 提取了 {len(features)} 个用户的特征向量")
    
    # 模拟Transformer预测
    print("\n🔮 5. 执行预测...")
    predictions = simulate_transformer_prediction(features_array)
    
    print(f"   ✅ 完成 {len(predictions['slice_type'])} 个用户的预测")
    print(f"   📊 平均预测置信度: {np.mean(predictions['confidence']):.3f}")
    
    # 切片优化
    print("\n🔧 6. 网络切片优化...")
    allocations = simple_slice_optimization(predictions, user_demands)
    
    print(f"   ✅ 完成 {len(allocations)} 个切片分配")
    
    # 优化结果分析
    total_allocated = sum(alloc['allocated_bandwidth'] for alloc in allocations)
    print(f"   📈 总分配带宽: {total_allocated:.1f} Mbps")
    
    # 按切片类型统计
    slice_stats = {}
    for alloc in allocations:
        slice_type = alloc['slice_type']
        if slice_type not in slice_stats:
            slice_stats[slice_type] = {'count': 0, 'bandwidth': 0}
        slice_stats[slice_type]['count'] += 1
        slice_stats[slice_type]['bandwidth'] += alloc['allocated_bandwidth']
    
    print(f"   🎯 切片分配统计:")
    for slice_type, stats in slice_stats.items():
        print(f"     - {slice_type}: {stats['count']} 用户, {stats['bandwidth']:.1f} Mbps")
    
    # 显示详细结果
    print("\n📋 7. 预测结果详情 (前5个用户):")
    for i, alloc in enumerate(allocations[:5]):
        print(f"   {i+1}. 用户: {alloc['user_id']}")
        print(f"      切片类型: {alloc['slice_type']}")
        print(f"      分配带宽: {alloc['allocated_bandwidth']:.1f} Mbps")
        print(f"      预测带宽: {alloc['predicted_bandwidth']:.1f} Mbps")
        print(f"      置信度: {alloc['confidence']:.3f}")
        print(f"      优先级: {alloc['priority']}")
        print()
    
    # 系统性能指标
    print("\n⚡ 8. 系统性能指标:")
    
    # 带宽利用率
    bandwidth_utilization = total_allocated / 1000.0
    print(f"   • 带宽利用率: {bandwidth_utilization*100:.1f}%")
    
    # 用户满意度（模拟）
    satisfaction_scores = []
    for alloc in allocations:
        ratio = alloc['allocated_bandwidth'] / max(alloc['predicted_bandwidth'], 1.0)
        satisfaction = min(1.0, ratio)
        satisfaction_scores.append(satisfaction)
    
    avg_satisfaction = np.mean(satisfaction_scores)
    print(f"   • 平均用户满意度: {avg_satisfaction*100:.1f}%")
    
    # 延迟满足度
    high_priority_count = sum(1 for alloc in allocations if alloc['priority'] >= 3)
    print(f"   • 高优先级用户: {high_priority_count} 个")
    
    # 导出结果
    print("\n💾 9. 导出结果...")
    
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_users': len(allocations),
            'total_bandwidth_allocated': total_allocated,
            'bandwidth_utilization': bandwidth_utilization,
            'average_satisfaction': avg_satisfaction,
            'slice_distribution': slice_stats
        },
        'allocations': allocations[:10]  # 只导出前10个结果
    }
    
    export_file = f"simple_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(export_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"   ✅ 结果已导出到: {export_file}")
    
    # 总结
    print("\n" + "=" * 60)
    print("🎉 简化演示完成！")
    print("\n💡 演示内容:")
    print("   ✅ 5G用户数据生成和分析")
    print("   ✅ 用户移动模式识别")
    print("   ✅ AI预测算法模拟")
    print("   ✅ 网络切片智能分配")
    print("   ✅ 系统性能评估")
    
    print("\n🚀 完整功能请安装PyTorch后运行:")
    print("   • python demo.py - 完整演示")
    print("   • python main.py - 交互模式")
    print("   • python train.py - 模型训练")


if __name__ == "__main__":
    try:
        run_simple_demo()
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()