"""
Streamlit可视化仪表板
实时监控5G网络切片优化系统的性能和预测结果
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from ..prediction_engine.realtime_predictor import RealTimePredictionEngine, PredictionResult
from ..data_processing.data_generator import UserDataGenerator
from ..models.slice_optimizer import SliceType


class Dashboard:
    """5G网络切片优化系统仪表板"""
    
    def __init__(self):
        """初始化仪表板"""
        self.logger = logging.getLogger(__name__)
        
        # 初始化会话状态
        if 'engine' not in st.session_state:
            st.session_state.engine = None
        if 'data_generator' not in st.session_state:
            st.session_state.data_generator = UserDataGenerator()
        if 'is_running' not in st.session_state:
            st.session_state.is_running = False
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
    
    def run(self):
        """运行仪表板"""
        st.set_page_config(
            page_title="5G网络切片优化系统",
            page_icon="🌐",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 页面标题
        st.title("🌐 5G动态网络切片优化系统")
        st.markdown("基于Transformer架构的实时预测和优化监控")
        
        # 侧边栏控制
        self._render_sidebar()
        
        # 主要内容区域
        if st.session_state.engine is None:
            self._render_welcome()
        else:
            self._render_main_dashboard()
    
    def _render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.header("🔧 系统控制")
        
        # 系统启动/停止
        if st.sidebar.button("🚀 启动系统" if not st.session_state.is_running else "⏹️ 停止系统"):
            if not st.session_state.is_running:
                self._start_system()
            else:
                self._stop_system()
        
        st.sidebar.divider()
        
        # 数据生成控制
        st.sidebar.subheader("📊 数据生成")
        
        num_users = st.sidebar.slider("用户数量", 5, 50, 20, 5)
        duration = st.sidebar.slider("持续时间(小时)", 1, 24, 2, 1)
        
        if st.sidebar.button("🎲 生成模拟数据"):
            self._generate_simulation_data(num_users, duration)
        
        st.sidebar.divider()
        
        # 系统配置
        st.sidebar.subheader("⚙️ 系统配置")
        
        refresh_rate = st.sidebar.selectbox(
            "刷新率(秒)",
            [1, 2, 5, 10],
            index=1
        )
        
        confidence_threshold = st.sidebar.slider(
            "置信度阈值",
            0.0, 1.0, 0.8, 0.1
        )
        
        # 显示选项
        st.sidebar.subheader("📈 显示选项")
        
        show_real_time = st.sidebar.checkbox("实时更新", True)
        show_predictions = st.sidebar.checkbox("预测结果", True)
        show_slice_info = st.sidebar.checkbox("切片信息", True)
        show_performance = st.sidebar.checkbox("性能监控", True)
        
        # 将配置存储到会话状态
        st.session_state.refresh_rate = refresh_rate
        st.session_state.confidence_threshold = confidence_threshold
        st.session_state.show_real_time = show_real_time
        st.session_state.show_predictions = show_predictions
        st.session_state.show_slice_info = show_slice_info
        st.session_state.show_performance = show_performance
    
    def _render_welcome(self):
        """渲染欢迎页面"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ## 🎯 欢迎使用5G网络切片优化系统
            
            ### 系统特色：
            - 🧠 **Transformer架构**: 先进的时间序列预测
            - 📱 **实时处理**: 毫秒级预测响应
            - 🔧 **智能切片**: 自适应资源分配
            - 📊 **数据驱动**: 基于用户行为的优化
            
            ### 开始使用：
            1. 点击侧边栏的 "🚀 启动系统" 按钮
            2. 生成模拟数据进行测试
            3. 观察实时预测结果和性能指标
            
            ### 网络切片类型：
            - **eMBB**: 增强型移动宽带 (高带宽)
            - **URLLC**: 超可靠低时延通信 (低延迟)
            - **mMTC**: 大规模机器类型通信 (大连接)
            """)
            
            # 系统架构图
            st.markdown("### 🏗️ 系统架构")
            
            # 创建简化的架构图
            fig = go.Figure()
            
            # 添加架构组件
            components = [
                {"name": "用户数据", "x": 1, "y": 3, "color": "#FF6B6B"},
                {"name": "数据处理", "x": 2, "y": 3, "color": "#4ECDC4"},
                {"name": "Transformer\n预测器", "x": 3, "y": 3, "color": "#45B7D1"},
                {"name": "切片优化器", "x": 4, "y": 3, "color": "#96CEB4"},
                {"name": "实时监控", "x": 3, "y": 2, "color": "#FFEAA7"}
            ]
            
            for comp in components:
                fig.add_trace(go.Scatter(
                    x=[comp["x"]],
                    y=[comp["y"]],
                    mode='markers+text',
                    marker=dict(size=60, color=comp["color"]),
                    text=comp["name"],
                    textposition="middle center",
                    showlegend=False
                ))
            
            # 添加连接线
            connections = [
                (1, 3, 2, 3), (2, 3, 3, 3), (3, 3, 4, 3), (3, 3, 3, 2)
            ]
            
            for x1, y1, x2, y2 in connections:
                fig.add_trace(go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            fig.update_layout(
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_main_dashboard(self):
        """渲染主仪表板"""
        # 状态指示器
        self._render_status_indicators()
        
        # 实时数据
        if st.session_state.show_real_time:
            self._render_real_time_metrics()
        
        # 预测结果
        if st.session_state.show_predictions:
            self._render_prediction_results()
        
        # 切片信息
        if st.session_state.show_slice_info:
            self._render_slice_information()
        
        # 性能监控
        if st.session_state.show_performance:
            self._render_performance_monitoring()
        
        # 自动刷新
        if st.session_state.show_real_time:
            time.sleep(st.session_state.refresh_rate)
            st.rerun()
    
    def _render_status_indicators(self):
        """渲染状态指示器"""
        st.subheader("📊 系统状态")
        
        if st.session_state.engine:
            status = st.session_state.engine.get_system_status()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "系统状态",
                    "🟢 运行中" if status.is_running else "🔴 已停止",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "活跃用户",
                    status.active_users,
                    delta=None
                )
            
            with col3:
                st.metric(
                    "总预测数",
                    status.total_predictions,
                    delta=None
                )
            
            with col4:
                st.metric(
                    "平均延迟",
                    f"{status.average_latency:.3f}s",
                    delta=None
                )
    
    def _render_real_time_metrics(self):
        """渲染实时指标"""
        st.subheader("📈 实时指标")
        
        if st.session_state.engine:
            # 获取最新预测结果
            recent_predictions = st.session_state.engine.get_latest_predictions(limit=20)
            
            if recent_predictions:
                # 准备数据
                df_predictions = []
                for pred in recent_predictions:
                    df_predictions.append({
                        'timestamp': pred.timestamp,
                        'user_id': pred.user_id,
                        'confidence': pred.confidence_score,
                        'processing_time': pred.processing_time,
                        'slice_count': len(pred.slice_allocations)
                    })
                
                df = pd.DataFrame(df_predictions)
                
                # 创建实时图表
                col1, col2 = st.columns(2)
                
                with col1:
                    # 置信度趋势
                    fig_confidence = px.line(
                        df, x='timestamp', y='confidence',
                        title='预测置信度趋势',
                        labels={'confidence': '置信度', 'timestamp': '时间'}
                    )
                    fig_confidence.update_layout(height=300)
                    st.plotly_chart(fig_confidence, use_container_width=True)
                
                with col2:
                    # 处理时间分布
                    fig_time = px.histogram(
                        df, x='processing_time',
                        title='预测处理时间分布',
                        labels={'processing_time': '处理时间(秒)', 'count': '频次'}
                    )
                    fig_time.update_layout(height=300)
                    st.plotly_chart(fig_time, use_container_width=True)
    
    def _render_prediction_results(self):
        """渲染预测结果"""
        st.subheader("🎯 预测结果")
        
        if st.session_state.engine:
            recent_predictions = st.session_state.engine.get_latest_predictions(limit=10)
            
            if recent_predictions:
                # 创建结果表格
                results_data = []
                for pred in recent_predictions:
                    slice_types = [alloc.slice_type.name for alloc in pred.slice_allocations]
                    total_bandwidth = sum(alloc.allocated_bandwidth for alloc in pred.slice_allocations)
                    
                    results_data.append({
                        '时间': pred.timestamp.strftime('%H:%M:%S'),
                        '用户ID': pred.user_id,
                        '置信度': f"{pred.confidence_score:.3f}",
                        '切片类型': ', '.join(slice_types) if slice_types else 'None',
                        '总带宽(Mbps)': f"{total_bandwidth:.1f}",
                        '处理时间(ms)': f"{pred.processing_time*1000:.1f}"
                    })
                
                df_results = pd.DataFrame(results_data)
                st.dataframe(df_results, use_container_width=True)
                
                # 切片类型分布饼图
                if results_data:
                    slice_counts = {}
                    for pred in recent_predictions:
                        for alloc in pred.slice_allocations:
                            slice_type = alloc.slice_type.name
                            slice_counts[slice_type] = slice_counts.get(slice_type, 0) + 1
                    
                    if slice_counts:
                        fig_pie = px.pie(
                            values=list(slice_counts.values()),
                            names=list(slice_counts.keys()),
                            title='切片类型分布'
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
    
    def _render_slice_information(self):
        """渲染切片信息"""
        st.subheader("🔧 网络切片信息")
        
        # 切片类型说明
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### eMBB (增强型移动宽带)
            - 🎯 **目标**: 高带宽应用
            - 📱 **应用**: 视频流、文件下载
            - 🚀 **带宽**: 10-1000 Mbps
            - ⏱️ **延迟**: ≤20ms
            - 🎚️ **优先级**: 1 (低)
            """)
        
        with col2:
            st.markdown("""
            ### URLLC (超可靠低时延通信)
            - 🎯 **目标**: 超低延迟
            - 🚗 **应用**: 自动驾驶、工业控制
            - 🚀 **带宽**: 1-50 Mbps
            - ⏱️ **延迟**: ≤1ms
            - 🎚️ **优先级**: 3 (高)
            """)
        
        with col3:
            st.markdown("""
            ### mMTC (大规模机器类型通信)
            - 🎯 **目标**: 大连接数
            - 🌐 **应用**: IoT设备、传感器
            - 🚀 **带宽**: 0.1-10 Mbps
            - ⏱️ **延迟**: ≤100ms
            - 🎚️ **优先级**: 2 (中)
            """)
        
        # 网络资源状态
        if st.session_state.engine:
            resources = st.session_state.engine.network_resources
            
            st.markdown("### 📊 网络资源状态")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总带宽", f"{resources.total_bandwidth:.0f} Mbps")
            
            with col2:
                st.metric("可用带宽", f"{resources.available_bandwidth:.0f} Mbps")
            
            with col3:
                st.metric("CPU使用率", f"{resources.cpu_usage*100:.1f}%")
            
            with col4:
                st.metric("内存使用率", f"{resources.memory_usage*100:.1f}%")
    
    def _render_performance_monitoring(self):
        """渲染性能监控"""
        st.subheader("⚡ 性能监控")
        
        if st.session_state.engine:
            # 获取性能统计
            perf_stats = st.session_state.engine.predictor.get_performance_stats()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 预测性能")
                metrics = perf_stats['prediction_metrics']
                
                st.metric("总预测次数", metrics['total_predictions'])
                st.metric("平均延迟", f"{metrics['avg_latency']*1000:.1f} ms")
                st.metric("缓冲区大小", perf_stats['buffer_size'])
                st.metric("缓存大小", perf_stats['cache_size'])
            
            with col2:
                st.markdown("### 🎯 系统状态")
                st.metric("预测历史", perf_stats['prediction_history_size'])
                st.metric("运行设备", perf_stats['model_device'])
                st.metric(
                    "模型状态", 
                    "✅ 已训练" if perf_stats['is_trained'] else "❌ 未训练"
                )
                
                # 内存使用情况（模拟）
                import psutil
                memory_percent = psutil.virtual_memory().percent
                st.metric("系统内存", f"{memory_percent:.1f}%")
    
    def _start_system(self):
        """启动系统"""
        try:
            st.session_state.engine = RealTimePredictionEngine()
            st.session_state.engine.start()
            st.session_state.is_running = True
            st.success("✅ 系统启动成功！")
            st.rerun()
        except Exception as e:
            st.error(f"❌ 系统启动失败: {e}")
    
    def _stop_system(self):
        """停止系统"""
        try:
            if st.session_state.engine:
                st.session_state.engine.stop()
                st.session_state.engine = None
            st.session_state.is_running = False
            st.success("✅ 系统已停止！")
            st.rerun()
        except Exception as e:
            st.error(f"❌ 系统停止失败: {e}")
    
    def _generate_simulation_data(self, num_users: int, duration: int):
        """生成模拟数据"""
        if not st.session_state.engine:
            st.error("❌ 请先启动系统！")
            return
        
        try:
            with st.spinner("正在生成模拟数据..."):
                # 生成数据
                user_data = st.session_state.data_generator.generate_dataset(
                    num_users=num_users,
                    duration_hours=duration,
                    sampling_interval=60
                )
                
                # 添加到引擎
                st.session_state.engine.batch_add_data(user_data)
                
                st.success(f"✅ 已生成 {len(user_data)} 条数据记录！")
                
                # 等待一段时间让系统处理
                time.sleep(2)
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ 数据生成失败: {e}")


def main():
    """主函数"""
    dashboard = Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main()