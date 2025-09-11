"""
5G网络切片优化系统快速演示脚本
"""

import sys
import os
import time
from datetime import datetime

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing.data_generator import UserDataGenerator
from src.data_processing.data_processor import DataProcessor
from src.prediction_engine.realtime_predictor import RealTimePredictionEngine
from src.utils.config_manager import ConfigManager
from src.utils.logger import Logger


def run_quick_demo():
    """运行快速演示"""
    print("🌐 5G动态网络切片优化系统 - 快速演示")
    print("=" * 50)
    
    # 初始化配置和日志
    config_manager = ConfigManager()
    Logger.setup_project_logging(config_manager.get_config())
    
    print("📊 1. 生成模拟用户数据...")
    # 生成用户数据
    data_generator = UserDataGenerator(seed=42)
    user_data = data_generator.generate_dataset(
        num_users=10,
        duration_hours=1,  # 1小时数据
        sampling_interval=30  # 30秒间隔
    )
    
    print(f"   ✅ 生成了 {len(user_data)} 条数据记录")
    
    print("\n🧠 2. 初始化预测引擎...")
    # 创建预测引擎
    engine = RealTimePredictionEngine()
    
    print("   ✅ 预测引擎初始化完成")
    
    print("\n🚀 3. 启动实时预测...")
    # 启动引擎
    engine.start()
    
    print("   ✅ 预测引擎已启动")
    
    try:
        print("\n📡 4. 处理用户数据流...")
        # 添加数据并进行预测
        processed_count = 0
        
        for i, data in enumerate(user_data[:100]):  # 处理前100条数据
            engine.add_user_data(data)
            processed_count += 1
            
            # 每20条数据显示一次进度
            if (i + 1) % 20 == 0:
                status = engine.get_system_status()
                print(f"   📈 进度: {processed_count}/100, "
                      f"活跃用户: {status.active_users}, "
                      f"预测数: {status.total_predictions}")
            
            # 短暂延迟模拟实时流
            time.sleep(0.05)
        
        # 等待处理完成
        print("\n⏳ 5. 等待预测完成...")
        time.sleep(3)
        
        # 获取结果
        print("\n📊 6. 展示预测结果...")
        final_status = engine.get_system_status()
        recent_predictions = engine.get_latest_predictions(limit=5)
        
        print(f"   📈 最终统计:")
        print(f"      • 总预测数: {final_status.total_predictions}")
        print(f"      • 活跃用户: {final_status.active_users}")
        print(f"      • 平均延迟: {final_status.average_latency:.3f}s")
        print(f"      • 错误数量: {final_status.error_count}")
        
        if recent_predictions:
            print(f"\n   🎯 近期预测结果 (最新5条):")
            for i, pred in enumerate(recent_predictions, 1):
                slice_types = [alloc.slice_type.name for alloc in pred.slice_allocations]
                total_bandwidth = sum(alloc.allocated_bandwidth for alloc in pred.slice_allocations)
                
                print(f"      {i}. 用户: {pred.user_id}")
                print(f"         时间: {pred.timestamp.strftime('%H:%M:%S')}")
                print(f"         置信度: {pred.confidence_score:.3f}")
                print(f"         切片类型: {', '.join(slice_types) if slice_types else 'None'}")
                print(f"         总带宽: {total_bandwidth:.1f} Mbps")
                print(f"         处理时间: {pred.processing_time*1000:.1f} ms")
                print()
        
        # 导出结果
        print("💾 7. 导出预测结果...")
        export_file = f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        engine.export_prediction_results(export_file)
        print(f"   ✅ 结果已导出到: {export_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断演示")
    
    finally:
        # 停止引擎
        print("\n🛑 8. 停止预测引擎...")
        engine.stop()
        print("   ✅ 引擎已停止")
    
    print("\n" + "=" * 50)
    print("🎉 演示完成！")
    print("\n💡 接下来您可以:")
    print("   • 运行 'python main.py' 进入交互模式")
    print("   • 运行 'python run_dashboard.py' 启动可视化仪表板")
    print("   • 运行 'python train.py' 训练自定义模型")


if __name__ == "__main__":
    run_quick_demo()