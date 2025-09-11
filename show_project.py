"""
项目结构查看器
展示5G网络切片优化项目的完整结构
"""

import os
from pathlib import Path


def print_tree(directory, prefix="", max_depth=3, current_depth=0):
    """打印目录树结构"""
    if current_depth > max_depth:
        return
    
    directory = Path(directory)
    items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    
    for i, item in enumerate(items):
        # 跳过一些不重要的文件和目录
        if item.name in {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', 'node_modules'}:
            continue
        
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir() and current_depth < max_depth:
            extension = "    " if is_last else "│   "
            print_tree(item, prefix + extension, max_depth, current_depth + 1)


def show_project_info():
    """显示项目信息"""
    print("🌐 5G动态网络切片优化系统")
    print("=" * 60)
    print("📁 项目结构:")
    print()
    
    # 打印项目树
    print_tree(".", max_depth=3)
    
    print("\n" + "=" * 60)
    print("📊 项目统计:")
    
    # 统计文件
    python_files = list(Path(".").rglob("*.py"))
    yaml_files = list(Path(".").rglob("*.yaml"))
    md_files = list(Path(".").rglob("*.md"))
    txt_files = list(Path(".").rglob("*.txt"))
    
    print(f"   • Python文件: {len(python_files)}")
    print(f"   • 配置文件: {len(yaml_files)}")
    print(f"   • 文档文件: {len(md_files)}")
    print(f"   • 其他文件: {len(txt_files)}")
    
    # 计算代码行数
    total_lines = 0
    code_lines = 0
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                total_lines += len(lines)
                code_lines += len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        except:
            pass
    
    print(f"   • 总代码行数: {total_lines:,}")
    print(f"   • 有效代码行数: {code_lines:,}")
    
    print("\n" + "=" * 60)
    print("🚀 快速开始:")
    print("   1. 安装依赖: pip install -r requirements.txt")
    print("   2. 快速演示: python demo.py")
    print("   3. 交互模式: python main.py")
    print("   4. 可视化仪表板: python run_dashboard.py")
    print("   5. 模型训练: python train.py")
    
    print("\n" + "=" * 60)
    print("📋 核心模块:")
    
    modules = [
        ("数据处理", "src/data_processing/", "用户轨迹、行为数据处理和特征工程"),
        ("Transformer模型", "src/models/", "基于注意力机制的预测模型"),
        ("预测引擎", "src/prediction_engine/", "实时预测和系统集成"),
        ("可视化", "src/visualization/", "Streamlit仪表板和监控界面"),
        ("工具模块", "src/utils/", "配置管理和日志系统")
    ]
    
    for name, path, desc in modules:
        print(f"   • {name:12} ({path:25}) - {desc}")
    
    print("\n" + "=" * 60)
    print("🎯 系统特性:")
    print("   • 🧠 基于Transformer的时间序列预测")
    print("   • 📱 实时用户行为分析和轨迹跟踪")
    print("   • 🔧 智能网络切片分配和资源优化")
    print("   • 📊 多任务学习：切片分类、带宽预测、特征预测")
    print("   • ⚡ 毫秒级预测响应和实时监控")
    print("   • 🎮 交互式可视化仪表板")
    
    print("\n✨ 项目就绪！开始您的5G网络切片优化之旅吧！")


if __name__ == "__main__":
    show_project_info()