"""
精确度提升工具包
包含数据质量优化、高级训练策略、模型集成等核心功能
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import logging


class DataQualityOptimizer:
    """数据质量优化器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = RobustScaler()  # 对异常值更鲁棒
        self.logger = logging.getLogger(__name__)
    
    def detect_and_clean_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """检测和清洗异常值"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data
        
        # 异常值检测
        outlier_mask = self.outlier_detector.fit_predict(data[numeric_cols])
        outlier_ratio = np.sum(outlier_mask == -1) / len(data)
        
        self.logger.info(f"检测到异常值比例: {outlier_ratio:.2%}")
        
        # 清洗异常值（用中位数替换）
        cleaned_data = data.copy()
        for col in numeric_cols:
            col_median = data[col].median()
            outlier_indices = np.where(outlier_mask == -1)[0]
            for idx in outlier_indices:
                if not pd.isna(data.iloc[idx][col]):
                    cleaned_data.iloc[idx, cleaned_data.columns.get_loc(col)] = col_median
        
        return cleaned_data
    
    def advanced_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """高级缺失值插补"""
        from sklearn.impute import KNNImputer
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return data
        
        # KNN插补
        imputer = KNNImputer(n_neighbors=5)
        imputed_data = data.copy()
        imputed_data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        
        return imputed_data
    
    def add_data_quality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加数据质量特征"""
        enhanced_data = data.copy()
        
        # 缺失值比例特征
        enhanced_data['missing_ratio'] = data.isnull().sum(axis=1) / len(data.columns)
        
        # 数据变化率特征
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(data) > 1:
                change_rate = np.abs(data[col].diff().fillna(0))
                enhanced_data[f'{col}_change_rate'] = change_rate
        
        return enhanced_data


class AdvancedLoss:
    """高级损失函数集合"""
    
    @staticmethod
    def focal_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                   alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """焦点损失，适用于不平衡分类"""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    @staticmethod
    def smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                       beta: float = 1.0) -> torch.Tensor:
        """平滑L1损失，对异常值更鲁棒"""
        diff = torch.abs(predictions - targets)
        loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return loss.mean()
    
    @staticmethod
    def uncertainty_weighted_loss(losses: List[torch.Tensor], 
                                  log_vars: torch.Tensor) -> torch.Tensor:
        """不确定性加权多任务损失"""
        weighted_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-log_vars[i])
            weighted_loss += precision * loss + log_vars[i]
        return weighted_loss


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []
        
        # 回归任务损失
        if 'bandwidth' in predictions:
            bandwidth_loss = AdvancedLoss.smooth_l1_loss(
                predictions['bandwidth'], targets['bandwidth'])
            losses.append(bandwidth_loss)
        
        if 'latency' in predictions:
            latency_loss = AdvancedLoss.smooth_l1_loss(
                predictions['latency'], targets['latency'])
            losses.append(latency_loss)
        
        # 分类任务损失
        if 'slice_type' in predictions:
            slice_loss = AdvancedLoss.focal_loss(
                predictions['slice_type'], targets['slice_type'])
            losses.append(slice_loss)
        
        # 不确定性加权
        return AdvancedLoss.uncertainty_weighted_loss(losses, self.log_vars)


class EnhancedTrainer:
    """增强训练器"""
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # 优化器配置
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=config.get('restart_period', 10),
            T_mult=2,
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # 早停配置
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001)
        )
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader, loss_fn) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(data)
                    loss = loss_fn(predictions, targets)
                
                self.scaler.scale(loss).backward()
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(data)
                loss = loss_fn(predictions, targets)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        self.scheduler.step()
        return total_loss / num_batches
    
    def validate(self, val_loader, loss_fn) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                predictions = self.model(data)
                loss = loss_fn(predictions, targets)
                
                total_loss += loss.item()
                predictions_list.append(predictions)
                targets_list.append(targets)
        
        avg_loss = total_loss / len(val_loader)
        
        # 计算精确度指标
        metrics = self._calculate_metrics(predictions_list, targets_list)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _calculate_metrics(self, predictions_list: List[Dict], 
                          targets_list: List[Dict]) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}
        
        # 合并所有批次的预测和目标
        all_predictions = {}
        all_targets = {}
        
        for pred_batch, target_batch in zip(predictions_list, targets_list):
            for key in pred_batch:
                if key not in all_predictions:
                    all_predictions[key] = []
                    all_targets[key] = []
                all_predictions[key].append(pred_batch[key].cpu())
                all_targets[key].append(target_batch[key].cpu())
        
        # 计算各任务指标
        for key in all_predictions:
            pred_tensor = torch.cat(all_predictions[key], dim=0)
            target_tensor = torch.cat(all_targets[key], dim=0)
            
            if key in ['bandwidth', 'latency']:  # 回归任务
                mae = F.l1_loss(pred_tensor, target_tensor).item()
                mse = F.mse_loss(pred_tensor, target_tensor).item()
                rmse = np.sqrt(mse)
                
                metrics[f'{key}_mae'] = mae
                metrics[f'{key}_rmse'] = rmse
                
            elif key == 'slice_type':  # 分类任务
                pred_classes = pred_tensor.argmax(dim=1)
                accuracy = (pred_classes == target_tensor).float().mean().item()
                metrics[f'{key}_accuracy'] = accuracy
        
        return metrics


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


class DataAugmentation:
    """数据增强"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.noise_level = self.config.get('noise_level', 0.01)
        self.time_jitter_ratio = self.config.get('time_jitter_ratio', 0.05)
        self.feature_dropout_prob = self.config.get('feature_dropout_prob', 0.1)
    
    def augment(self, data: torch.Tensor) -> torch.Tensor:
        """数据增强"""
        augmented_data = data.clone()
        
        # 高斯噪声
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(augmented_data) * self.noise_level
            augmented_data += noise
        
        # 特征dropout
        if torch.rand(1) < 0.2:
            dropout_mask = torch.rand(augmented_data.shape[-1]) > self.feature_dropout_prob
            augmented_data = augmented_data * dropout_mask.unsqueeze(0).unsqueeze(0)
        
        # 时间抖动
        if torch.rand(1) < 0.3:
            seq_len = augmented_data.shape[1]
            jitter_amount = int(seq_len * self.time_jitter_ratio)
            if jitter_amount > 0:
                start_idx = torch.randint(0, jitter_amount + 1, (1,)).item()
                end_idx = start_idx + seq_len - jitter_amount
                augmented_data = augmented_data[:, start_idx:end_idx, :]
        
        return augmented_data


class ModelEnsemble:
    """模型集成"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """集成预测"""
        ensemble_predictions = {}
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                predictions = model(x)
                
                for key, value in predictions.items():
                    if key not in ensemble_predictions:
                        ensemble_predictions[key] = []
                    ensemble_predictions[key].append(value * self.weights[i])
        
        # 加权平均
        final_predictions = {}
        for key, pred_list in ensemble_predictions.items():
            final_predictions[key] = sum(pred_list)
        
        return final_predictions


class AccuracyAnalyzer:
    """精确度分析器"""
    
    @staticmethod
    def calculate_comprehensive_metrics(predictions: Dict[str, torch.Tensor], 
                                      targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """计算综合精确度指标"""
        metrics = {}
        
        for key in predictions:
            pred = predictions[key]
            target = targets[key]
            
            if key in ['bandwidth', 'latency']:  # 回归任务
                # MAE
                mae = F.l1_loss(pred, target).item()
                metrics[f'{key}_mae'] = mae
                
                # RMSE  
                rmse = torch.sqrt(F.mse_loss(pred, target)).item()
                metrics[f'{key}_rmse'] = rmse
                
                # MAPE
                mape = torch.mean(torch.abs((pred - target) / (target + 1e-8))).item() * 100
                metrics[f'{key}_mape'] = mape
                
                # R²分数
                ss_res = torch.sum((target - pred) ** 2)
                ss_tot = torch.sum((target - torch.mean(target)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                metrics[f'{key}_r2'] = r2.item()
                
            elif key == 'slice_type':  # 分类任务
                pred_classes = pred.argmax(dim=1)
                accuracy = (pred_classes == target).float().mean().item()
                metrics[f'{key}_accuracy'] = accuracy
                
                # F1分数
                from sklearn.metrics import f1_score
                f1 = f1_score(target.cpu().numpy(), pred_classes.cpu().numpy(), average='weighted')
                metrics[f'{key}_f1'] = f1
        
        return metrics
    
    @staticmethod
    def generate_accuracy_report(metrics: Dict[str, float]) -> str:
        """生成精确度报告"""
        report = "\n📊 模型精确度评估报告\n"
        report += "=" * 50 + "\n"
        
        # 回归任务指标
        if any('bandwidth' in k for k in metrics):
            report += "\n🎯 带宽预测精确度:\n"
            if 'bandwidth_mae' in metrics:
                report += f"  • MAE: {metrics['bandwidth_mae']:.4f}\n"
            if 'bandwidth_rmse' in metrics:
                report += f"  • RMSE: {metrics['bandwidth_rmse']:.4f}\n"
            if 'bandwidth_mape' in metrics:
                report += f"  • MAPE: {metrics['bandwidth_mape']:.2f}%\n"
            if 'bandwidth_r2' in metrics:
                report += f"  • R²: {metrics['bandwidth_r2']:.4f}\n"
        
        if any('latency' in k for k in metrics):
            report += "\n⏱️ 延迟预测精确度:\n"
            if 'latency_mae' in metrics:
                report += f"  • MAE: {metrics['latency_mae']:.4f}\n"
            if 'latency_rmse' in metrics:
                report += f"  • RMSE: {metrics['latency_rmse']:.4f}\n"
            if 'latency_mape' in metrics:
                report += f"  • MAPE: {metrics['latency_mape']:.2f}%\n"
            if 'latency_r2' in metrics:
                report += f"  • R²: {metrics['latency_r2']:.4f}\n"
        
        # 分类任务指标
        if any('slice_type' in k for k in metrics):
            report += "\n🔀 切片类型分类精确度:\n"
            if 'slice_type_accuracy' in metrics:
                report += f"  • 准确率: {metrics['slice_type_accuracy']:.4f}\n"
            if 'slice_type_f1' in metrics:
                report += f"  • F1分数: {metrics['slice_type_f1']:.4f}\n"
        
        report += "\n" + "=" * 50 + "\n"
        return report


def create_accuracy_improvement_config() -> Dict:
    """创建精确度提升配置"""
    return {
        'data_quality': {
            'outlier_detection': True,
            'advanced_imputation': True,
            'noise_filtering': True
        },
        'feature_engineering': {
            'lag_features': [1, 3, 6, 12],
            'rolling_windows': [3, 6, 12, 24],
            'frequency_features': True,
            'interaction_features': True
        },
        'model_optimization': {
            'multi_task_learning': True,
            'attention_heads': 16,
            'hidden_dim': 256,
            'dropout_rate': 0.1
        },
        'training_strategy': {
            'learning_rate': 1e-3,
            'batch_size': 64,
            'max_epochs': 100,
            'early_stopping_patience': 10,
            'gradient_clipping': 1.0,
            'use_amp': False
        },
        'data_augmentation': {
            'noise_level': 0.01,
            'time_jitter_ratio': 0.05,
            'feature_dropout_prob': 0.1
        },
        'ensemble': {
            'num_models': 3,
            'diversity_penalty': 0.1
        }
    }