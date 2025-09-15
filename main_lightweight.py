"""
5G网络切片优化系统 - 轻量化版本主程序
使用轻量级行为建模，避免重型深度学习依赖
"""

import sys
import os
from datetime import datetime
import argparse

# 导入轻量化模块
from lightweight_behavior_demo import LightweightBehaviorEngine, demonstrate_enhanced_behaviors


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='5G网络切片优化系统 - 轻量化版本')
    parser.add_argument('mode', nargs='?', default='interactive', 
                       choices=['demo', 'interactive', 'behavior', 'simulate'],
                       help='运行模式: demo(演示), interactive(交互), behavior(行为测试), simulate(模拟)')
    parser.add_argument('--users', type=int, default=15, help='模拟用户数量')
    parser.add_argument('--hours', type=int, default=24, help='模拟时长(小时)')
    
    args = parser.parse_args()
    
    print("🚀 5G网络切片优化系统 - 轻量化版本")
    print("=" * 50)
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 运行模式: {args.mode}")
    
    try:
        if args.mode == 'demo':
            run_demo_mode()
        elif args.mode == 'behavior':
            run_behavior_test()
        elif args.mode == 'simulate':
            run_simulation_mode(args.users, args.hours)
        else:
            run_interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 程序运行错误: {e}")
        import traceback
        traceback.print_exc()


def run_demo_mode():
    """演示模式"""
    print("\n🎬 进入演示模式...")
    sessions, analysis = demonstrate_enhanced_behaviors()
    
    print("\n📊 演示总结:")
    if analysis:
        print(f"  • 总会话数: {analysis['total_sessions']}")
        print(f"  • 平均带宽需求: {analysis['avg_bandwidth']:.1f} Mbps")
        print(f"  • 平均延迟需求: {analysis['avg_latency']:.1f} ms")
        
        # 显示最受欢迎的行为
        behavior_stats = analysis.get('behavior_stats', {})
        if behavior_stats:
            top_behavior = max(behavior_stats.items(), key=lambda x: x[1]['count'])
            print(f"  • 最活跃行为: {top_behavior[1]['chinese_name']} ({top_behavior[1]['count']} 次)")


def run_behavior_test():
    """行为测试模式"""
    print("\n🧪 进入行为特征测试模式...")
    
    engine = LightweightBehaviorEngine(seed=42)
    
    # 展示行为配置
    print("\n📋 行为类型详细配置:")
    for behavior_name, config in engine.behavior_configs.items():
        bw_range = engine.bandwidth_ranges[config['bandwidth_need']]
        lat_range = engine.latency_ranges[config['latency_need']]
        
        print(f"\n🔸 {config['chinese_name']} ({behavior_name}):")
        print(f"   带宽需求: {bw_range[0]:.1f} - {bw_range[1]:.1f} Mbps ({config['bandwidth_need']})")
        print(f"   延迟需求: {lat_range[0]} - {lat_range[1]} ms ({config['latency_need']})")
        print(f"   推荐切片: {config['slice_type']}")
        print(f"   数据强度: {config['data_intensity']}")
        print(f"   实时关键: {'是' if config['realtime_critical'] else '否'}")
        print(f"   典型时长: {config['typical_duration']} 分钟")
        print(f"   高峰时段: {config['peak_hours']}")


def run_simulation_mode(num_users: int, num_hours: int):
    """模拟模式"""
    print(f"\n🎯 进入模拟模式...")
    print(f"   👥 模拟用户: {num_users} 个")
    print(f"   ⏰ 模拟时长: {num_hours} 小时")
    
    engine = LightweightBehaviorEngine(seed=42)
    
    # 运行模拟
    sessions = engine.run_behavior_simulation(num_users, num_hours)
    analysis = engine.analyze_simulation_results(sessions)
    
    # 详细分析
    print(f"\n📈 详细分析结果:")
    if analysis:
        slice_dist = analysis['slice_distribution']
        total = analysis['total_sessions']
        
        print(f"\n🔌 网络切片需求分析:")
        for slice_type in ['URLLC', 'eMBB', 'mMTC']:
            count = slice_dist.get(slice_type, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  • {slice_type}: {count} 个会话 ({percentage:.1f}%)")
        
        print(f"\n⚡ 网络资源规划建议:")
        urllc_ratio = slice_dist.get('URLLC', 0) / total if total > 0 else 0
        embb_ratio = slice_dist.get('eMBB', 0) / total if total > 0 else 0
        
        if urllc_ratio > 0.3:
            print(f"  🎮 高游戏用户场景：建议增加URLLC切片资源配置")
        if embb_ratio > 0.6:
            print(f"  📺 高流媒体场景：建议优化eMBB切片带宽分配")
        if analysis['avg_bandwidth'] > 100:
            print(f"  🚀 高带宽需求：建议升级基站带宽容量")


def run_interactive_mode():
    """交互模式"""
    print("\n💬 进入交互模式...")
    print("可用命令:")
    print("  demo     - 运行完整演示")
    print("  behavior - 查看行为配置")
    print("  simulate - 自定义模拟")
    print("  exit     - 退出程序")
    
    while True:
        try:
            command = input("\n> 请输入命令: ").strip().lower()
            
            if command == 'exit':
                print("👋 再见！")
                break
            elif command == 'demo':
                run_demo_mode()
            elif command == 'behavior':
                run_behavior_test()
            elif command == 'simulate':
                try:
                    users = int(input("请输入用户数量 (默认15): ") or "15")
                    hours = int(input("请输入模拟小时数 (默认24): ") or "24")
                    run_simulation_mode(users, hours)
                except ValueError:
                    print("⚠️ 输入格式错误，使用默认值")
                    run_simulation_mode(15, 24)
            elif command == 'help' or command == '?':
                print("可用命令: demo, behavior, simulate, exit")
            elif command == '':
                continue
            else:
                print(f"❌ 未知命令: {command}")
                print("输入 'help' 查看可用命令")
                
        except KeyboardInterrupt:
            print("\n👋 用户中断，程序退出")
            break
        except Exception as e:
            print(f"❌ 命令执行错误: {e}")


if __name__ == "__main__":
    main()