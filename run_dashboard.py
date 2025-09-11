"""
启动Streamlit仪表板
"""

import subprocess
import sys
import os

def main():
    """启动仪表板"""
    dashboard_path = os.path.join("src", "visualization", "dashboard.py")
    
    if not os.path.exists(dashboard_path):
        print("❌ 仪表板文件不存在")
        return
    
    print("🚀 启动5G网络切片优化系统仪表板...")
    print("📱 仪表板将在浏览器中打开: http://localhost:8501")
    print("🛑 按 Ctrl+C 停止服务")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n✅ 仪表板已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()