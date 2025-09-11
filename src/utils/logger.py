"""
日志管理器
统一的日志记录和管理
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    """日志管理器"""
    
    def __init__(self, name: str = "5G-Slicing", 
                 log_file: Optional[str] = None,
                 log_level: str = "INFO"):
        """
        初始化日志管理器
        
        Args:
            name: 日志记录器名称
            log_file: 日志文件路径
            log_level: 日志级别
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers(log_file)
    
    def _setup_handlers(self, log_file: Optional[str]) -> None:
        """设置日志处理器"""
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用回转文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """获取日志记录器"""
        return self.logger
    
    @staticmethod
    def setup_project_logging(config: dict) -> None:
        """设置项目全局日志"""
        log_config = config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('file', 'logs/5g_slicing.log')
        
        # 创建主日志记录器
        main_logger = Logger("5G-Slicing", log_file, log_level)
        
        # 为各个模块设置日志
        modules = [
            'data_processing', 'models', 'prediction_engine',
            'visualization', 'utils'
        ]
        
        for module in modules:
            module_logger = logging.getLogger(module)
            module_logger.setLevel(getattr(logging, log_level.upper()))
            
            # 如果没有处理器，继承主日志记录器的处理器
            if not module_logger.handlers:
                for handler in main_logger.get_logger().handlers:
                    module_logger.addHandler(handler)