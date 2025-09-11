"""
实时预测引擎
集成所有模块，实现端到端的实时预测系统
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
import json

from .time_series_predictor import TimeSeriesPredictor
from ..models.slice_optimizer import NetworkSliceOptimizer, NetworkResources, SliceAllocation
from ..data_processing.data_processor import DataProcessor, UserData
from ..utils.config_manager import ConfigManager


@dataclass
class PredictionResult:
    """预测结果"""
    timestamp: datetime
    user_id: str
    predictions: Dict[str, Any]
    slice_allocations: List[SliceAllocation]
    confidence_score: float
    processing_time: float


@dataclass
class SystemStatus:
    """系统状态"""
    is_running: bool
    total_predictions: int
    active_users: int
    average_latency: float
    last_update: datetime
    error_count: int


class RealTimePredictionEngine:
    """实时预测引擎"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化实时预测引擎
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # 预测配置
        self.update_interval = self.config['prediction']['update_interval']
        self.confidence_threshold = self.config['prediction']['confidence_threshold']
        self.max_prediction_time = self.config['prediction']['max_prediction_time']
        
        # 初始化组件
        self.predictor = TimeSeriesPredictor(
            self.config['model']['transformer'],
            self.config['data']
        )
        
        self.slice_optimizer = NetworkSliceOptimizer(
            self.config['network_slicing']
        )
        
        self.data_processor = DataProcessor(config_path)
        
        # 运行状态
        self.is_running = False
        self.prediction_thread = None
        self.event_loop = None
        
        # 数据存储
        self.user_data_streams = {}  # 用户数据流
        self.prediction_results = []  # 预测结果历史
        self.system_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_processing_time': 0.0,
            'active_users': set()
        }
        
        # 回调函数
        self.prediction_callbacks = []
        self.error_callbacks = []
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 网络资源状态（模拟）
        self.network_resources = NetworkResources(
            total_bandwidth=1000.0,  # 1Gbps
            available_bandwidth=800.0,  # 800Mbps 可用
            cpu_usage=0.3,
            memory_usage=0.4,
            latency=10.0
        )
        
        self.logger.info("实时预测引擎初始化完成")
    
    def start(self) -> None:
        """启动实时预测引擎"""
        if self.is_running:
            self.logger.warning("预测引擎已在运行")
            return
        
        self.is_running = True
        
        # 启动异步事件循环
        self.prediction_thread = threading.Thread(target=self._run_prediction_loop)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
        
        self.logger.info("实时预测引擎已启动")
    
    def stop(self) -> None:
        """停止实时预测引擎"""
        self.is_running = False
        
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        
        self.logger.info("实时预测引擎已停止")
    
    def add_user_data(self, user_data: UserData) -> None:
        """
        添加用户数据
        
        Args:
            user_data: 用户数据
        """
        user_id = user_data.user_id
        
        # 初始化用户数据流
        if user_id not in self.user_data_streams:
            self.user_data_streams[user_id] = []
        
        # 添加数据到流
        self.user_data_streams[user_id].append(user_data)
        
        # 限制数据流大小
        max_stream_size = self.config['data']['sampling']['window_size'] * 2
        if len(self.user_data_streams[user_id]) > max_stream_size:
            self.user_data_streams[user_id] = self.user_data_streams[user_id][-max_stream_size:]
        
        # 更新活跃用户
        self.system_metrics['active_users'].add(user_id)
        
        # 触发实时数据处理
        self.predictor.add_real_time_data(user_data)
    
    def batch_add_data(self, user_data_list: List[UserData]) -> None:
        """
        批量添加用户数据
        
        Args:
            user_data_list: 用户数据列表
        """
        for user_data in user_data_list:
            self.add_user_data(user_data)
    
    def predict_for_user(self, user_id: str, 
                        use_cache: bool = True) -> Optional[PredictionResult]:
        """
        为特定用户进行预测
        
        Args:
            user_id: 用户ID
            use_cache: 是否使用缓存
            
        Returns:
            预测结果
        """
        if user_id not in self.user_data_streams:
            self.logger.warning(f"用户 {user_id} 没有数据流")
            return None
        
        user_data_stream = self.user_data_streams[user_id]
        window_size = self.config['data']['sampling']['window_size']
        
        if len(user_data_stream) < window_size:
            self.logger.warning(f"用户 {user_id} 数据不足，需要至少 {window_size} 个数据点")
            return None
        
        start_time = time.time()
        
        try:
            # 执行预测
            predictions = self.predictor.predict_from_user_data(
                user_data_stream, user_id
            )
            
            # 计算置信度
            confidence_score = self._calculate_confidence(predictions)
            
            # 网络切片优化
            slice_allocations = []
            if confidence_score >= self.confidence_threshold:
                # 构造用户需求
                user_demands = self._extract_user_demands(user_data_stream[-1])
                slice_allocations = self.slice_optimizer.optimize_slices(
                    predictions, self.network_resources, [user_demands]
                )
            
            processing_time = time.time() - start_time
            
            # 创建预测结果
            result = PredictionResult(
                timestamp=datetime.now(),
                user_id=user_id,
                predictions=self._tensor_to_dict(predictions),
                slice_allocations=slice_allocations,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
            # 存储结果
            self.prediction_results.append(result)
            
            # 限制结果历史大小
            max_results = 1000
            if len(self.prediction_results) > max_results:
                self.prediction_results = self.prediction_results[-max_results:]
            
            # 更新系统指标
            self._update_system_metrics(processing_time, success=True)
            
            # 触发回调
            self._trigger_prediction_callbacks(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"用户 {user_id} 预测失败: {e}")
            self._update_system_metrics(time.time() - start_time, success=False)
            self._trigger_error_callbacks(user_id, str(e))
            return None
    
    def predict_for_all_users(self) -> List[PredictionResult]:
        """为所有活跃用户进行预测"""
        results = []
        
        # 使用线程池并行预测
        futures = []
        for user_id in list(self.user_data_streams.keys()):
            future = self.executor.submit(self.predict_for_user, user_id, True)
            futures.append((user_id, future))
        
        # 收集结果
        for user_id, future in futures:
            try:
                result = future.result(timeout=self.max_prediction_time)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"用户 {user_id} 并行预测失败: {e}")
        
        return results
    
    def get_latest_predictions(self, user_id: Optional[str] = None, 
                             limit: int = 10) -> List[PredictionResult]:
        """
        获取最新预测结果
        
        Args:
            user_id: 用户ID（可选）
            limit: 结果数量限制
            
        Returns:
            预测结果列表
        """
        if user_id:
            filtered_results = [
                result for result in self.prediction_results 
                if result.user_id == user_id
            ]
        else:
            filtered_results = self.prediction_results
        
        # 按时间倒序排列
        sorted_results = sorted(
            filtered_results, 
            key=lambda x: x.timestamp, 
            reverse=True
        )
        
        return sorted_results[:limit]
    
    def get_system_status(self) -> SystemStatus:
        """获取系统状态"""
        return SystemStatus(
            is_running=self.is_running,
            total_predictions=self.system_metrics['total_predictions'],
            active_users=len(self.system_metrics['active_users']),
            average_latency=self.system_metrics['average_processing_time'],
            last_update=datetime.now(),
            error_count=self.system_metrics['failed_predictions']
        )
    
    def register_prediction_callback(self, callback: Callable[[PredictionResult], None]) -> None:
        """注册预测结果回调函数"""
        self.prediction_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable[[str, str], None]) -> None:
        """注册错误回调函数"""
        self.error_callbacks.append(callback)
    
    def update_network_resources(self, resources: NetworkResources) -> None:
        """更新网络资源状态"""
        self.network_resources = resources
        self.logger.info("网络资源状态已更新")
    
    def _run_prediction_loop(self) -> None:
        """运行预测循环"""
        self.logger.info("启动预测循环")
        
        while self.is_running:
            try:
                loop_start_time = time.time()
                
                # 为所有用户进行预测
                if self.user_data_streams:
                    results = self.predict_for_all_users()
                    self.logger.debug(f"完成 {len(results)} 个预测")
                
                # 计算循环时间
                loop_time = time.time() - loop_start_time
                
                # 等待到下一个更新间隔
                sleep_time = max(0, self.update_interval - loop_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"预测循环出错: {e}")
                time.sleep(self.update_interval)
        
        self.logger.info("预测循环已停止")
    
    def _calculate_confidence(self, predictions: Dict) -> float:
        """计算预测置信度"""
        # 简化的置信度计算
        confidence_scores = []
        
        # 基于切片类型预测的置信度
        if 'slice_type' in predictions:
            slice_probs = predictions['slice_type']
            max_prob = float(slice_probs.max())
            confidence_scores.append(max_prob)
        
        # 基于带宽预测的稳定性
        if 'bandwidth' in predictions:
            bandwidth_pred = predictions['bandwidth']
            std_ratio = float(bandwidth_pred.std() / (bandwidth_pred.mean() + 1e-6))
            stability_score = 1.0 / (1.0 + std_ratio)
            confidence_scores.append(stability_score)
        
        # 综合置信度
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        else:
            return 0.5  # 默认置信度
    
    def _extract_user_demands(self, user_data: UserData) -> Dict:
        """从用户数据提取需求"""
        # 根据用户行为和网络指标推断需求
        behavior = user_data.behavior_features
        network = user_data.network_metrics
        
        # 基于应用使用推断带宽需求
        video_usage = behavior.get('app_video', 0.0)
        game_usage = behavior.get('app_game', 0.0)
        work_usage = behavior.get('app_work', 0.0)
        
        bandwidth_req = 10.0  # 默认带宽需求
        if video_usage > 0.5:
            bandwidth_req = 50.0
        elif game_usage > 0.3:
            bandwidth_req = 30.0
        elif work_usage > 0.4:
            bandwidth_req = 20.0
        
        # 基于应用类型推断延迟需求
        latency_req = 50.0  # 默认延迟需求
        if game_usage > 0.3 or work_usage > 0.4:
            latency_req = 10.0
        
        return {
            'user_id': user_data.user_id,
            'bandwidth_req': bandwidth_req,
            'latency_req': latency_req,
            'reliability_req': 0.95,
            'slice_preference': user_data.slice_type
        }
    
    def _tensor_to_dict(self, tensor_dict: Dict) -> Dict:
        """将张量字典转换为普通字典"""
        result = {}
        for key, value in tensor_dict.items():
            if hasattr(value, 'detach'):
                result[key] = value.detach().cpu().numpy().tolist()
            else:
                result[key] = value
        return result
    
    def _update_system_metrics(self, processing_time: float, success: bool) -> None:
        """更新系统指标"""
        self.system_metrics['total_predictions'] += 1
        
        if success:
            self.system_metrics['successful_predictions'] += 1
        else:
            self.system_metrics['failed_predictions'] += 1
        
        # 更新平均处理时间
        total_time = (self.system_metrics['average_processing_time'] * 
                     (self.system_metrics['total_predictions'] - 1))
        self.system_metrics['average_processing_time'] = (
            (total_time + processing_time) / self.system_metrics['total_predictions']
        )
    
    def _trigger_prediction_callbacks(self, result: PredictionResult) -> None:
        """触发预测回调"""
        for callback in self.prediction_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"预测回调执行失败: {e}")
    
    def _trigger_error_callbacks(self, user_id: str, error_msg: str) -> None:
        """触发错误回调"""
        for callback in self.error_callbacks:
            try:
                callback(user_id, error_msg)
            except Exception as e:
                self.logger.error(f"错误回调执行失败: {e}")
    
    def export_prediction_results(self, filepath: str, 
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None) -> None:
        """
        导出预测结果
        
        Args:
            filepath: 导出文件路径
            start_time: 开始时间
            end_time: 结束时间
        """
        # 筛选结果
        results_to_export = self.prediction_results
        
        if start_time:
            results_to_export = [r for r in results_to_export if r.timestamp >= start_time]
        
        if end_time:
            results_to_export = [r for r in results_to_export if r.timestamp <= end_time]
        
        # 转换为JSON可序列化格式
        export_data = []
        for result in results_to_export:
            export_data.append({
                'timestamp': result.timestamp.isoformat(),
                'user_id': result.user_id,
                'predictions': result.predictions,
                'slice_allocations': [
                    {
                        'slice_id': alloc.slice_id,
                        'slice_type': alloc.slice_type.name,
                        'allocated_bandwidth': alloc.allocated_bandwidth,
                        'satisfaction_score': alloc.satisfaction_score
                    }
                    for alloc in result.slice_allocations
                ],
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time
            })
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"已导出 {len(export_data)} 条预测结果到 {filepath}")