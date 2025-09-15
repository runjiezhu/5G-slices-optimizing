"""
测试增强的用户行为特征
包含VR/AR游戏、FPS游戏、云游戏、网络游戏、视频流、直播流等行为
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.data_generator import UserDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def test_enhanced_user_behaviors():
    """测试增强的用户行为特征"""
    print("🚀 测试增强的用户行为特征系统")
    print("=" * 60)
    
    # 初始化数据生成器
    generator = UserDataGenerator(seed=42)
    
    # 展示新增的行为类型
    print("\n📋 支持的用户行为类型:")
    behavior_categories = {
        "🎮 游戏类": ['vr_ar_gaming', 'fps_gaming', 'cloud_gaming', 'online_gaming'],
        "📺 流媒体类": ['video_streaming', 'live_streaming', 'video_calling'],
        "💻 其他应用": ['file_download', 'web_browsing', 'iot_sensor']
    }
    
    for category, behaviors in behavior_categories.items():
        print(f"\n{category}:")
        for behavior in behaviors:
            if behavior in generator.behavior_types:
                info = generator.behavior_types[behavior]
                print(f"  • {behavior}: 带宽需求={info['bandwidth_req']}, 延迟需求={info['latency_req']}, 首选切片={info['slice_preference']}")
    
    # 展示带宽和延迟需求映射
    print(f"\n📊 带宽需求级别 (Mbps):")
    for level, (min_bw, max_bw) in generator.bandwidth_requirements.items():
        print(f"  • {level}: {min_bw} - {max_bw} Mbps")
    
    print(f"\n⏱️ 延迟需求级别 (ms):")
    for level, (min_lat, max_lat) in generator.latency_requirements.items():
        print(f"  • {level}: {min_lat} - {max_lat} ms")
    
    # 生成测试数据
    print(f"\n🔬 生成测试数据...")
    test_results = generator.test_behavior_features(num_users=10, duration_hours=2)
    
    return test_results


def analyze_behavior_performance(test_results):
    """分析行为特征的性能影响"""
    print("\n📈 用户行为对网络性能影响分析")
    print("=" * 50)
    
    network_impact = test_results.get('network_impact_analysis', {})
    
    if not network_impact:
        print("⚠️ 没有足够的网络影响分析数据")
        return
    
    # 按延迟要求排序
    behaviors_by_latency = []
    for behavior_name, impact_data in network_impact.items():
        if impact_data['sample_count'] > 0:
            behaviors_by_latency.append((
                behavior_name, 
                impact_data['avg_latency'],
                impact_data['avg_throughput'],
                impact_data['sample_count']
            ))
    
    behaviors_by_latency.sort(key=lambda x: x[1])  # 按平均延迟排序
    
    print("\n🏆 按延迟性能排名 (从低到高):")
    for i, (behavior, latency, throughput, samples) in enumerate(behaviors_by_latency, 1):
        print(f"  {i}. {behavior}:")
        print(f"     延迟: {latency:.2f} ms | 吞吐量: {throughput:.1f} Mbps | 样本数: {samples}")
    
    # 分析带宽使用情况
    behaviors_by_bandwidth = sorted(behaviors_by_latency, key=lambda x: x[2], reverse=True)
    
    print("\n🌐 按带宽需求排名 (从高到低):")
    for i, (behavior, latency, throughput, samples) in enumerate(behaviors_by_bandwidth, 1):
        print(f"  {i}. {behavior}:")
        print(f"     吞吐量: {throughput:.1f} Mbps | 延迟: {latency:.2f} ms | 样本数: {samples}")


def demonstrate_behavior_scenarios():
    """演示不同行为场景"""
    print("\n🎯 用户行为场景演示")
    print("=" * 40)
    
    generator = UserDataGenerator(seed=42)
    
    # 定义典型用户场景
    scenarios = [
        {
            'name': '🎮 VR游戏玩家',
            'user_type': 'personal',
            'primary_behavior': 'vr_ar_gaming',
            'description': '需要极低延迟和超高带宽的VR/AR游戏体验'
        },
        {
            'name': '📺 主播用户',
            'user_type': 'personal', 
            'primary_behavior': 'live_streaming',
            'description': '进行直播需要极高带宽和稳定的网络连接'
        },
        {
            'name': '☁️ 云游戏用户',
            'user_type': 'personal',
            'primary_behavior': 'cloud_gaming', 
            'description': '云游戏需要低延迟和高带宽的稳定连接'
        },
        {
            'name': '💼 商务用户',
            'user_type': 'business',
            'primary_behavior': 'video_calling',
            'description': '视频会议需要稳定的中等带宽和低延迟'
        },
        {
            'name': '🏭 IoT设备',
            'user_type': 'iot',
            'primary_behavior': 'iot_sensor',
            'description': '传感器数据传输，低带宽需求但需要可靠连接'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print(f"  场景描述: {scenario['description']}")
        
        # 模拟该场景的网络需求
        behavior_info = generator.behavior_types[scenario['primary_behavior']]
        bandwidth_range = generator.bandwidth_requirements[behavior_info['bandwidth_req']]
        latency_range = generator.latency_requirements[behavior_info['latency_req']]
        
        print(f"  网络需求:")
        print(f"    • 带宽: {bandwidth_range[0]:.1f} - {bandwidth_range[1]:.1f} Mbps ({behavior_info['bandwidth_req']})")
        print(f"    • 延迟: {latency_range[0]} - {latency_range[1]} ms ({behavior_info['latency_req']})")
        print(f"    • 推荐切片: {behavior_info['slice_preference']}")
        print(f"    • 实时关键: {'是' if behavior_info.get('realtime_critical', False) else '否'}")
        print(f"    • 数据密集: {'是' if behavior_info.get('data_intensive', False) else '否'}")


def create_behavior_comparison_chart(test_results):
    """创建行为特征对比图表"""
    try:
        import matplotlib.pyplot as plt
        
        behavior_stats = test_results.get('behavior_statistics', {})
        if not behavior_stats:
            print("⚠️ 没有足够的行为统计数据用于绘图")
            return
        
        # 准备数据
        behaviors = []
        usage_rates = []
        avg_usages = []
        
        for behavior_name, stats in behavior_stats.items():
            if stats['usage_rate'] > 0:  # 只显示有使用的行为
                behaviors.append(behavior_name)
                usage_rates.append(stats['usage_rate'] * 100)
                avg_usages.append(stats['average_usage'])
        
        if not behaviors:
            print("⚠️ 没有活跃的行为数据用于绘图")
            return
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 用户活跃率柱状图
        bars1 = ax1.bar(range(len(behaviors)), usage_rates, color='skyblue')
        ax1.set_xlabel('用户行为类型')
        ax1.set_ylabel('活跃用户比例 (%)')
        ax1.set_title('各行为类型的用户活跃率')
        ax1.set_xticks(range(len(behaviors)))
        ax1.set_xticklabels(behaviors, rotation=45, ha='right')
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars1, usage_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 平均使用强度柱状图
        bars2 = ax2.bar(range(len(behaviors)), avg_usages, color='lightcoral')
        ax2.set_xlabel('用户行为类型')
        ax2.set_ylabel('平均使用强度')
        ax2.set_title('各行为类型的平均使用强度')
        ax2.set_xticks(range(len(behaviors)))
        ax2.set_xticklabels(behaviors, rotation=45, ha='right')
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars2, avg_usages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('behavior_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 行为分析图表已保存为 'behavior_analysis.png'")
        plt.show()
        
    except ImportError:
        print("⚠️ matplotlib 未安装，跳过图表生成")
    except Exception as e:
        print(f"⚠️ 图表生成失败: {e}")


def main():
    """主函数"""
    print("🎯 5G网络切片用户行为特征增强测试")
    print("=" * 60)
    
    try:
        # 测试增强的用户行为
        test_results = test_enhanced_user_behaviors()
        
        # 分析行为性能影响
        analyze_behavior_performance(test_results)
        
        # 演示行为场景
        demonstrate_behavior_scenarios()
        
        # 创建对比图表
        create_behavior_comparison_chart(test_results)
        
        print("\n🎉 用户行为特征增强测试完成！")
        print("\n📋 新增特征总结:")
        print("  ✅ 添加了云游戏(cloud_gaming)和网络游戏(online_gaming)")
        print("  ✅ 优化了VR/AR游戏的网络需求配置")
        print("  ✅ 完善了各行为类型的带宽和延迟映射")
        print("  ✅ 增强了网络性能指标的动态调整")
        print("  ✅ 改进了时间相关的行为模式")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()