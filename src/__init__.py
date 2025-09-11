"""
5G动态网络切片优化项目
基于Transformer架构的实时预测和优化系统
"""

__version__ = "1.0.0"
__author__ = "5G Slicing Team"
__description__ = "基于Transformer架构的5G动态网络切片实时优化系统"

# 导入核心模块
from .data_processing import DataProcessor, FeatureExtractor
from .models import TransformerPredictor, NetworkSliceOptimizer
from .prediction_engine import RealTimePredictionEngine
from .visualization import Dashboard
from .utils import ConfigManager, Logger

__all__ = [
    "DataProcessor",
    "FeatureExtractor", 
    "TransformerPredictor",
    "NetworkSliceOptimizer",
    "RealTimePredictionEngine",
    "Dashboard",
    "ConfigManager",
    "Logger"
]