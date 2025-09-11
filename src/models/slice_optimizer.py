"""
网络切片优化器
实现基于预测结果的动态资源分配和切片优化
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum


class SliceType(Enum):
    """网络切片类型"""
    eMBB = 0    # 增强型移动宽带
    URLLC = 1   # 超可靠低时延通信  
    mMTC = 2    # 大规模机器类型通信


@dataclass
class SliceRequirements:
    """切片需求"""
    slice_type: SliceType
    min_bandwidth: float    # Mbps
    max_bandwidth: float    # Mbps
    latency_req: float      # ms
    reliability_req: float  # 0-1
    priority: int          # 优先级 (1-3)


@dataclass 
class NetworkResources:
    """网络资源状态"""
    total_bandwidth: float     # 总带宽 Mbps
    available_bandwidth: float # 可用带宽 Mbps
    cpu_usage: float          # CPU使用率 0-1
    memory_usage: float       # 内存使用率 0-1
    latency: float           # 当前延迟 ms


@dataclass
class SliceAllocation:
    """切片分配结果"""
    slice_id: str
    slice_type: SliceType
    allocated_bandwidth: float
    allocated_priority: int
    expected_latency: float
    satisfaction_score: float


class NetworkSliceOptimizer(nn.Module):
    """网络切片优化器"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # 配置参数
        self.config = config
        self.optimization_method = config.get('optimization_algorithm', 'reinforcement_learning')
        self.reward_weights = config.get('reward_weights', {
            'bandwidth_utilization': 0.4,
            'latency_satisfaction': 0.3, 
            'energy_efficiency': 0.2,
            'user_satisfaction': 0.1
        })
        
        # 切片需求定义
        self.slice_requirements = {
            SliceType.eMBB: SliceRequirements(
                slice_type=SliceType.eMBB,
                min_bandwidth=10.0,
                max_bandwidth=1000.0,
                latency_req=20.0,
                reliability_req=0.95,
                priority=1
            ),
            SliceType.URLLC: SliceRequirements(
                slice_type=SliceType.URLLC,
                min_bandwidth=1.0,
                max_bandwidth=50.0,
                latency_req=1.0,
                reliability_req=0.99999,
                priority=3
            ),
            SliceType.mMTC: SliceRequirements(
                slice_type=SliceType.mMTC,  
                min_bandwidth=0.1,
                max_bandwidth=10.0,
                latency_req=100.0,
                reliability_req=0.9,
                priority=2
            )
        }
        
        # 神经网络组件（用于强化学习优化）
        self.state_dim = 32  # 网络状态维度
        self.action_dim = 16  # 动作空间维度
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.logger.info("网络切片优化器初始化完成")
    
    def optimize_slices(self, predictions: Dict[str, torch.Tensor],
                       current_resources: NetworkResources,
                       user_demands: List[Dict]) -> List[SliceAllocation]:
        """
        优化网络切片分配
        
        Args:
            predictions: Transformer预测结果
            current_resources: 当前网络资源状态
            user_demands: 用户需求列表
            
        Returns:
            切片分配结果列表
        """
        self.logger.info("开始网络切片优化")
        
        # 解析预测结果
        predicted_bandwidth = predictions.get('bandwidth', torch.zeros(1, 30, 1))
        predicted_slice_types = predictions.get('slice_type', torch.zeros(1, 3))
        
        # 统计各类切片需求
        slice_demands = self._aggregate_slice_demands(user_demands, predicted_slice_types)
        
        # 根据优化算法选择策略
        if self.optimization_method == 'reinforcement_learning':
            allocations = self._rl_optimize(slice_demands, current_resources, predicted_bandwidth)
        elif self.optimization_method == 'greedy':
            allocations = self._greedy_optimize(slice_demands, current_resources)
        else:
            allocations = self._proportional_optimize(slice_demands, current_resources)
        
        self.logger.info(f"完成切片优化，分配了 {len(allocations)} 个切片")
        return allocations
    
    def _aggregate_slice_demands(self, user_demands: List[Dict], 
                               predicted_types: torch.Tensor) -> Dict[SliceType, Dict]:
        """聚合切片需求"""
        slice_stats = {
            SliceType.eMBB: {'count': 0, 'total_bandwidth': 0.0, 'users': []},
            SliceType.URLLC: {'count': 0, 'total_bandwidth': 0.0, 'users': []},
            SliceType.mMTC: {'count': 0, 'total_bandwidth': 0.0, 'users': []}
        }
        
        # 从预测结果获取切片类型分布
        slice_probs = torch.softmax(predicted_types, dim=-1).squeeze().cpu().numpy()
        
        for demand in user_demands:
            # 根据用户需求和预测确定切片类型
            user_slice_type = self._determine_slice_type(demand, slice_probs)
            
            slice_stats[user_slice_type]['count'] += 1
            slice_stats[user_slice_type]['total_bandwidth'] += demand.get('bandwidth_req', 10.0)
            slice_stats[user_slice_type]['users'].append(demand)
        
        return slice_stats
    
    def _determine_slice_type(self, user_demand: Dict, slice_probs: np.ndarray) -> SliceType:
        """确定用户的切片类型"""
        # 基于用户需求特征和预测概率确定切片类型
        latency_req = user_demand.get('latency_req', 50.0)
        bandwidth_req = user_demand.get('bandwidth_req', 10.0)
        reliability_req = user_demand.get('reliability_req', 0.95)
        
        # 规则判断
        if latency_req <= 5.0 and reliability_req >= 0.999:
            return SliceType.URLLC
        elif bandwidth_req >= 100.0:
            return SliceType.eMBB  
        elif bandwidth_req <= 5.0:
            return SliceType.mMTC
        else:
            # 使用预测概率
            predicted_type = np.argmax(slice_probs)
            return SliceType(predicted_type)
    
    def _rl_optimize(self, slice_demands: Dict[SliceType, Dict],
                    resources: NetworkResources,
                    predicted_bandwidth: torch.Tensor) -> List[SliceAllocation]:
        """基於強化學習的優化"""
        # 构建状态向量
        state = self._build_state_vector(slice_demands, resources, predicted_bandwidth)
        
        # 前向传播获取动作和价值
        with torch.no_grad():
            encoded_state = self.state_encoder(state)
            action_probs = self.policy_net(encoded_state)
            state_value = self.value_net(encoded_state)
        
        # 基于动作概率进行资源分配
        allocations = self._allocate_resources_from_actions(
            action_probs, slice_demands, resources
        )
        
        return allocations
    
    def _greedy_optimize(self, slice_demands: Dict[SliceType, Dict],
                        resources: NetworkResources) -> List[SliceAllocation]:
        """贪心优化算法"""
        allocations = []
        remaining_bandwidth = resources.available_bandwidth
        
        # 按优先级排序切片类型
        sorted_slices = sorted(
            slice_demands.items(),
            key=lambda x: self.slice_requirements[x[0]].priority,
            reverse=True
        )
        
        for slice_type, demand_info in sorted_slices:
            if demand_info['count'] == 0:
                continue
                
            requirements = self.slice_requirements[slice_type]
            
            # 计算需要的带宽
            needed_bandwidth = max(
                demand_info['total_bandwidth'],
                requirements.min_bandwidth * demand_info['count']
            )
            
            # 分配带宽
            allocated_bandwidth = min(needed_bandwidth, remaining_bandwidth)
            
            if allocated_bandwidth >= requirements.min_bandwidth:
                allocation = SliceAllocation(
                    slice_id=f"slice_{slice_type.name}_{len(allocations)}",
                    slice_type=slice_type,
                    allocated_bandwidth=allocated_bandwidth,
                    allocated_priority=requirements.priority,
                    expected_latency=requirements.latency_req,
                    satisfaction_score=min(1.0, allocated_bandwidth / needed_bandwidth)
                )
                
                allocations.append(allocation)
                remaining_bandwidth -= allocated_bandwidth
        
        return allocations
    
    def _proportional_optimize(self, slice_demands: Dict[SliceType, Dict],
                             resources: NetworkResources) -> List[SliceAllocation]:
        """比例分配优化"""
        allocations = []
        total_demand = sum(info['total_bandwidth'] for info in slice_demands.values())
        
        if total_demand == 0:
            return allocations
        
        for slice_type, demand_info in slice_demands.items():
            if demand_info['count'] == 0:
                continue
            
            requirements = self.slice_requirements[slice_type]
            
            # 按比例分配带宽
            proportion = demand_info['total_bandwidth'] / total_demand
            allocated_bandwidth = resources.available_bandwidth * proportion
            
            # 确保满足最小要求
            allocated_bandwidth = max(
                allocated_bandwidth,
                requirements.min_bandwidth * demand_info['count']
            )
            
            allocation = SliceAllocation(
                slice_id=f"slice_{slice_type.name}_{len(allocations)}",
                slice_type=slice_type,
                allocated_bandwidth=allocated_bandwidth,
                allocated_priority=requirements.priority,
                expected_latency=requirements.latency_req,
                satisfaction_score=min(1.0, allocated_bandwidth / demand_info['total_bandwidth'])
            )
            
            allocations.append(allocation)
        
        return allocations
    
    def _build_state_vector(self, slice_demands: Dict[SliceType, Dict],
                           resources: NetworkResources,
                           predicted_bandwidth: torch.Tensor) -> torch.Tensor:
        """构建状态向量"""
        state_features = []
        
        # 资源状态特征
        state_features.extend([
            resources.available_bandwidth / resources.total_bandwidth,
            resources.cpu_usage,
            resources.memory_usage,
            resources.latency / 100.0  # 归一化
        ])
        
        # 切片需求特征
        for slice_type in SliceType:
            demand_info = slice_demands.get(slice_type, {'count': 0, 'total_bandwidth': 0.0})
            state_features.extend([
                demand_info['count'] / 100.0,  # 归一化用户数
                demand_info['total_bandwidth'] / 1000.0  # 归一化带宽需求
            ])
        
        # 预测带宽特征（使用前几个时间步）
        pred_bandwidth_features = predicted_bandwidth.squeeze().cpu().numpy()[:10]
        pred_bandwidth_features = pred_bandwidth_features / 1000.0  # 归一化
        state_features.extend(pred_bandwidth_features.tolist())
        
        # 填充到固定维度
        while len(state_features) < self.state_dim:
            state_features.append(0.0)
        
        return torch.tensor(state_features[:self.state_dim], dtype=torch.float32).unsqueeze(0)
    
    def _allocate_resources_from_actions(self, action_probs: torch.Tensor,
                                       slice_demands: Dict[SliceType, Dict],
                                       resources: NetworkResources) -> List[SliceAllocation]:
        """根据动作概率分配资源"""
        allocations = []
        actions = action_probs.squeeze().cpu().numpy()
        
        # 将动作向量转换为资源分配决策
        total_weight = 0.0
        slice_weights = {}
        
        for i, slice_type in enumerate(SliceType):
            if slice_demands[slice_type]['count'] > 0:
                # 每个切片类型使用多个动作维度
                start_idx = i * 4
                weight = np.sum(actions[start_idx:start_idx+4])
                slice_weights[slice_type] = weight
                total_weight += weight
        
        # 按权重分配带宽
        if total_weight > 0:
            for slice_type, weight in slice_weights.items():
                if slice_demands[slice_type]['count'] == 0:
                    continue
                
                proportion = weight / total_weight
                allocated_bandwidth = resources.available_bandwidth * proportion
                requirements = self.slice_requirements[slice_type]
                
                # 确保满足最小要求
                min_required = requirements.min_bandwidth * slice_demands[slice_type]['count']
                allocated_bandwidth = max(allocated_bandwidth, min_required)
                
                allocation = SliceAllocation(
                    slice_id=f"slice_{slice_type.name}_{len(allocations)}",
                    slice_type=slice_type,
                    allocated_bandwidth=allocated_bandwidth,
                    allocated_priority=requirements.priority,
                    expected_latency=requirements.latency_req,
                    satisfaction_score=min(1.0, allocated_bandwidth / slice_demands[slice_type]['total_bandwidth'])
                )
                
                allocations.append(allocation)
        
        return allocations
    
    def calculate_reward(self, allocations: List[SliceAllocation],
                        resources: NetworkResources) -> float:
        """计算优化奖励"""
        if not allocations:
            return 0.0
        
        # 带宽利用率奖励
        total_allocated = sum(alloc.allocated_bandwidth for alloc in allocations)
        bandwidth_utilization = total_allocated / resources.total_bandwidth
        bandwidth_reward = min(1.0, bandwidth_utilization) * self.reward_weights['bandwidth_utilization']
        
        # 延迟满足度奖励
        latency_satisfaction = np.mean([
            1.0 if alloc.expected_latency <= self.slice_requirements[alloc.slice_type].latency_req else 0.5
            for alloc in allocations
        ])
        latency_reward = latency_satisfaction * self.reward_weights['latency_satisfaction']
        
        # 用户满意度奖励
        user_satisfaction = np.mean([alloc.satisfaction_score for alloc in allocations])
        user_reward = user_satisfaction * self.reward_weights['user_satisfaction']
        
        # 能效奖励（简化）
        energy_efficiency = 1.0 - (resources.cpu_usage + resources.memory_usage) / 2.0
        energy_reward = energy_efficiency * self.reward_weights['energy_efficiency']
        
        total_reward = bandwidth_reward + latency_reward + user_reward + energy_reward
        
        return total_reward
    
    def get_optimization_stats(self, allocations: List[SliceAllocation]) -> Dict:
        """获取优化统计信息"""
        stats = {
            'total_slices': len(allocations),
            'total_bandwidth_allocated': sum(alloc.allocated_bandwidth for alloc in allocations),
            'average_satisfaction': np.mean([alloc.satisfaction_score for alloc in allocations]) if allocations else 0.0,
            'slice_distribution': {}
        }
        
        for slice_type in SliceType:
            slice_allocs = [alloc for alloc in allocations if alloc.slice_type == slice_type]
            stats['slice_distribution'][slice_type.name] = {
                'count': len(slice_allocs),
                'total_bandwidth': sum(alloc.allocated_bandwidth for alloc in slice_allocs),
                'avg_satisfaction': np.mean([alloc.satisfaction_score for alloc in slice_allocs]) if slice_allocs else 0.0
            }
        
        return stats