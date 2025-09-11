"""
5G动态网络切片优化系统主程序
"""

import argparse
import sys
import time
from datetime import datetime
import logging

from src.prediction_engine.realtime_predictor import RealTimePredictionEngine
from src.data_processing.data_generator import UserDataGenerator
from src.utils.config_manager import ConfigManager
from src.utils.logger import Logger


def run_demo():
    """运行演示"""
    print("🚀 5G动态网络切片优化系统演示")
    
    # 初始化系统
    config_manager = ConfigManager()
    Logger.setup_project_logging(config_manager.get_config())
    
    logger = logging.getLogger("MainDemo")
    logger.info("系统启动中...")
    
    # 创建实时预测引擎
    engine = RealTimePredictionEngine()
    
    # 创建数据生成器
    data_generator = UserDataGenerator(seed=42)
    
    # 生成模拟数据
    logger.info("生成模拟5G用户数据...")
    user_data = data_generator.generate_dataset(
        num_users=20,
        duration_hours=2,  # 2小时数据
        sampling_interval=30  # 30秒间隔
    )
    
    # 启动预测引擎
    logger.info("启动实时预测引擎...")
    engine.start()
    
    try:
        # 模拟实时数据流
        logger.info("开始实时数据流模拟...")
        for i, data in enumerate(user_data):
            engine.add_user_data(data)
            
            # 每100个数据点显示一次状态
            if i % 100 == 0:
                status = engine.get_system_status()
                print(f"📊 处理进度: {i}/{len(user_data)}")
                print(f"   活跃用户: {status.active_users}")
                print(f"   总预测数: {status.total_predictions}")
                print(f"   平均延迟: {status.average_latency:.3f}s")
                print()
            
            # 模拟实时间隔
            time.sleep(0.01)
        
        # 等待处理完成
        time.sleep(5)
        
        # 显示最终结果
        final_status = engine.get_system_status()
        recent_predictions = engine.get_latest_predictions(limit=5)
        
        print("🎯 最终结果:")
        print(f"   总预测数: {final_status.total_predictions}")
        print(f"   活跃用户: {final_status.active_users}")
        print(f"   平均延迟: {final_status.average_latency:.3f}s")
        print(f"   最近预测: {len(recent_predictions)} 条")
        
        # 显示部分预测结果
        if recent_predictions:
            print("\n📈 最近预测结果:")
            for pred in recent_predictions[:3]:
                print(f"   用户: {pred.user_id}")
                print(f"   时间: {pred.timestamp}")
                print(f"   置信度: {pred.confidence_score:.3f}")
                print(f"   切片分配: {len(pred.slice_allocations)} 个")
                print()
        
        # 导出结果
        export_file = f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        engine.export_prediction_results(export_file)
        print(f"📄 结果已导出到: {export_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
    finally:
        # 停止引擎
        engine.stop()
        logger.info("系统已停止")
        print("✅ 演示完成")


def run_training():
    """运行模型训练"""
    print("🔧 开始模型训练...")
    
    from train import main as train_main
    train_main()
    
    print("✅ 训练完成")


def run_interactive():
    """运行交互模式"""
    print("🎮 5G网络切片优化系统 - 交互模式")
    print("可用命令:")
    print("  demo  - 运行系统演示")
    print("  train - 开始模型训练")
    print("  exit  - 退出程序")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == "demo":
                run_demo()
            elif command == "train":
                run_training()
            elif command == "exit":
                print("👋 再见!")
                break
            elif command == "help":
                print("可用命令: demo, train, exit")
            else:
                print("❌ 未知命令，输入 'help' 查看帮助")
                
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="5G动态网络切片优化系统",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        choices=["demo", "train", "interactive"],
        default="interactive",
        help="运行模式: demo(演示), train(训练), interactive(交互)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌐 5G动态网络切片优化系统")
    print("   基于Transformer架构的实时预测和优化")
    print("=" * 60)
    
    if args.command == "demo":
        run_demo()
    elif args.command == "train":
        run_training()
    else:
        run_interactive()


if __name__ == "__main__":
    main()