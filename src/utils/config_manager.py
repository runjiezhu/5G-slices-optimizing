"""
配置管理器
处理项目配置文件的加载、验证和管理
"""

import yaml
import os
from typing import Dict, Any, Optional
import logging
from pathlib import Path


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = {}
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.load_config()
        self.validate_config()
    
    def load_config(self) -> None:
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                self.logger.info(f"配置文件加载成功: {self.config_path}")
            else:
                self.logger.warning(f"配置文件不存在: {self.config_path}, 使用默认配置")
                self.config = self._get_default_config()
                self.save_config()
        except Exception as e:
            self.logger.error(f"配置文件加载失败: {e}")
            self.config = self._get_default_config()
    
    def save_config(self) -> None:
        """保存配置文件"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            self.logger.info(f"配置文件保存成功: {self.config_path}")
        except Exception as e:
            self.logger.error(f"配置文件保存失败: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置项
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        # 导航到父级字典
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
        self.logger.info(f"配置项已更新: {key} = {value}")
    
    def update(self, new_config: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            new_config: 新配置字典
        """
        self._deep_update(self.config, new_config)
        self.logger.info("配置已更新")
    
    def validate_config(self) -> bool:
        """验证配置完整性"""
        required_sections = [
            'model', 'data', 'network_slicing', 
            'prediction', 'logging'
        ]
        
        for section in required_sections:
            if section not in self.config:
                self.logger.warning(f"缺少配置节: {section}")
                return False
        
        # 验证模型配置
        model_config = self.config.get('model', {})
        transformer_config = model_config.get('transformer', {})
        
        required_model_keys = ['d_model', 'nhead', 'num_layers']
        for key in required_model_keys:
            if key not in transformer_config:
                self.logger.warning(f"缺少模型配置: model.transformer.{key}")
                return False
        
        return True
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'project': {
                'name': '5G Dynamic Network Slicing Optimization',
                'version': '1.0.0'
            },
            'model': {
                'transformer': {
                    'd_model': 512,
                    'nhead': 8,
                    'num_layers': 6,
                    'dropout': 0.1,
                    'max_seq_length': 100
                },
                'training': {
                    'batch_size': 32,
                    'learning_rate': 0.0001,
                    'epochs': 100
                }
            },
            'data': {
                'features': {
                    'trajectory_dim': 3,
                    'velocity_dim': 3,
                    'behavior_dim': 10
                },
                'sampling': {
                    'frequency': 1.0,
                    'window_size': 60,
                    'prediction_horizon': 30
                }
            },
            'network_slicing': {
                'optimization': {
                    'algorithm': 'reinforcement_learning'
                }
            },
            'prediction': {
                'update_interval': 1.0,
                'confidence_threshold': 0.8
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value