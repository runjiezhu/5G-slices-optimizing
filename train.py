"""
模型训练脚本
实现Transformer模型的训练、验证和保存功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
import os
from datetime import datetime
import json
from tqdm import tqdm

from src.models.transformer_model import TransformerPredictor, SliceAwareTransformer
from src.data_processing.data_processor import DataProcessor
from src.data_processing.data_generator import UserDataGenerator
from src.utils.config_manager import ConfigManager
from src.utils.logger import Logger


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # 训练配置
        self.model_config = self.config['model']['transformer']
        self.training_config = self.config['model']['training']
        self.data_config = self.config['data']
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
        # 模型初始化
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # 训练状态
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # 数据生成器和处理器
        self.data_generator = UserDataGenerator()
        self.data_processor = DataProcessor(config_path)
        
        self.logger.info("模型训练器初始化完成")
    
    def prepare_data(self, num_users: int = 100, 
                    duration_hours: int = 24,
                    train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        准备训练数据
        
        Args:
            num_users: 用户数量
            duration_hours: 数据持续时间（小时）
            train_split: 训练集比例
            
        Returns:
            训练数据加载器和验证数据加载器
        """
        self.logger.info(f"开始生成训练数据: {num_users} 用户, {duration_hours} 小时")
        
        # 生成用户数据
        user_data_list = self.data_generator.generate_dataset(
            num_users=num_users,
            duration_hours=duration_hours,
            sampling_interval=60
        )
        
        # 添加到数据处理器
        self.data_processor.batch_add_data(user_data_list)
        
        # 处理数据
        processed_data = self.data_processor.process_data()
        
        # 转换为张量
        X = torch.tensor(processed_data.sequences, dtype=torch.float32)
        y = torch.tensor(processed_data.labels, dtype=torch.long)
        
        self.logger.info(f"数据形状: X={X.shape}, y={y.shape}")
        
        # 创建数据集
        dataset = TensorDataset(X, y)
        
        # 划分训练集和验证集
        train_size = int(len(dataset) * train_split)
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        batch_size = self.training_config['batch_size']
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.logger.info(f"数据准备完成: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def create_model(self, slice_aware: bool = False) -> nn.Module:
        """
        创建模型
        
        Args:
            slice_aware: 是否使用切片感知模型
            
        Returns:
            模型实例
        """
        # 更新模型配置中的输入输出维度
        feature_count = (self.data_config['features']['trajectory_dim'] + 
                        self.data_config['features']['velocity_dim'] + 
                        self.data_config['features']['behavior_dim'])
        
        model_config = self.model_config.copy()
        model_config['input_dim'] = feature_count
        model_config['output_dim'] = 32  # 预测特征维度
        model_config['prediction_horizon'] = self.data_config['sampling']['prediction_horizon']
        
        if slice_aware:
            self.model = SliceAwareTransformer(model_config)
        else:
            self.model = TransformerPredictor(model_config)
        
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型创建完成:")
        self.logger.info(f"  总参数: {total_params:,}")
        self.logger.info(f"  可训练参数: {trainable_params:,}")
        
        return self.model
    
    def setup_training(self) -> None:
        """设置训练组件"""
        if self.model is None:
            raise ValueError("请先创建模型")
        
        # 优化器
        learning_rate = self.training_config['learning_rate']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 损失函数（多任务损失）
        self.criterion = MultiTaskLoss()
        
        self.logger.info("训练组件设置完成")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        训练单个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均损失和准确率
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # 计算损失
            loss = self.criterion(outputs, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 计算准确率（仅针对切片分类）
            if 'slice_type' in outputs:
                slice_pred = torch.argmax(outputs['slice_type'], dim=1)
                accuracy = (slice_pred == target).float().mean()
                total_accuracy += accuracy.item()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        验证单个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            平均损失和准确率
        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                outputs = self.model(data)
                
                # 计算损失
                loss = self.criterion(outputs, target)
                total_loss += loss.item()
                
                # 计算准确率
                if 'slice_type' in outputs:
                    slice_pred = torch.argmax(outputs['slice_type'], dim=1)
                    accuracy = (slice_pred == target).float().mean()
                    total_accuracy += accuracy.item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/(len(pbar.finished_tasks)+1):.4f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: Optional[int] = None,
              save_path: str = "models/transformer_5g.pth") -> Dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_path: 模型保存路径
            
        Returns:
            训练历史
        """
        if num_epochs is None:
            num_epochs = self.training_config['epochs']
        
        self.logger.info(f"开始训练，共 {num_epochs} 轮")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.training_config.get('early_stopping_patience', 10)
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['val_accuracy'].append(val_acc)
            
            # 日志记录
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(save_path, epoch, val_loss)
                self.logger.info(f"保存最佳模型: {save_path}")
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        self.logger.info("训练完成")
        
        return self.training_history
    
    def save_model(self, save_path: str, epoch: int, val_loss: float) -> None:
        """
        保存模型
        
        Args:
            save_path: 保存路径
            epoch: 当前轮数
            val_loss: 验证损失
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, save_path)
    
    def load_model(self, load_path: str) -> None:
        """
        加载模型
        
        Args:
            load_path: 模型路径
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 重新创建模型
        self.model_config = checkpoint['model_config']
        self.create_model()
        
        # 加载状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        self.logger.info(f"模型加载成功: {load_path}")
    
    def plot_training_history(self, save_path: str = "training_history.png") -> None:
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 损失曲线
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.training_history['train_accuracy'], label='Train Acc')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练历史图表已保存: {save_path}")


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self):
        super().__init__()
        self.slice_criterion = nn.CrossEntropyLoss()
        self.bandwidth_criterion = nn.MSELoss()
        self.feature_criterion = nn.MSELoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> torch.Tensor:
        """
        计算多任务损失
        
        Args:
            outputs: 模型输出字典
            targets: 目标标签
            
        Returns:
            总损失
        """
        total_loss = 0.0
        loss_count = 0
        
        # 切片类型分类损失
        if 'slice_type' in outputs:
            slice_loss = self.slice_criterion(outputs['slice_type'], targets)
            total_loss += slice_loss
            loss_count += 1
        
        # 带宽预测损失（如果有带宽标签）
        if 'bandwidth' in outputs:
            # 这里简化处理，实际应该有相应的带宽标签
            bandwidth_target = torch.randn_like(outputs['bandwidth'])
            bandwidth_loss = self.bandwidth_criterion(outputs['bandwidth'], bandwidth_target)
            total_loss += bandwidth_loss * 0.5  # 降低权重
            loss_count += 1
        
        # 特征预测损失
        if 'features' in outputs:
            # 简化处理，实际应该有相应的特征标签
            feature_target = torch.randn_like(outputs['features'])
            feature_loss = self.feature_criterion(outputs['features'], feature_target)
            total_loss += feature_loss * 0.3  # 降低权重
            loss_count += 1
        
        return total_loss / max(loss_count, 1)


def main():
    """主训练函数"""
    # 设置日志
    logger_manager = Logger("ModelTrainer", "logs/training.log", "INFO")
    logger = logger_manager.get_logger()
    
    # 创建训练器
    trainer = ModelTrainer()
    
    # 准备数据
    train_loader, val_loader = trainer.prepare_data(
        num_users=50,
        duration_hours=12
    )
    
    # 创建和设置模型
    trainer.create_model(slice_aware=False)
    trainer.setup_training()
    
    # 训练模型
    history = trainer.train(train_loader, val_loader, save_path="models/transformer_5g.pth")
    
    # 绘制训练历史
    trainer.plot_training_history("training_history.png")
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()