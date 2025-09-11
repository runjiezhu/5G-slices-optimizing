"""
Transformer模型实现
用于5G网络切片的时间序列预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math
import logging


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 残差连接
        residual = query
        
        # 线性变换
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 输出投影
        output = self.W_o(context)
        
        # 残差连接和层归一化
        return self.layer_norm(output + residual)


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.layer_norm(x + residual)


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(x, x, x, mask)
        x = self.feed_forward(x)
        return x


class TransformerPredictor(nn.Module):
    """基于Transformer的5G网络切片预测器"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # 模型配置
        self.d_model = config.get('d_model', 512)
        self.n_heads = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.d_ff = config.get('d_ff', 2048)
        self.dropout = config.get('dropout', 0.1)
        self.max_seq_length = config.get('max_seq_length', 100)
        
        # 输入输出维度
        self.input_dim = config.get('input_dim', 64)  # 特征维度
        self.output_dim = config.get('output_dim', 32)  # 预测维度
        self.prediction_horizon = config.get('prediction_horizon', 30)
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        
        # Transformer编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff // 2, self.output_dim * self.prediction_horizon)
        )
        
        # 网络切片分类头
        self.slice_classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff // 4, 3)  # eMBB, URLLC, mMTC
        )
        
        # 带宽需求预测头
        self.bandwidth_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff // 4, self.prediction_horizon)
        )
        
        self.logger.info(f"Transformer预测器初始化完成: {self.d_model}维, {self.num_layers}层")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            mask: 注意力掩码
        
        Returns:
            预测结果字典
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入嵌入和位置编码
        x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Transformer编码器
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # 使用最后一个时间步的表示进行预测
        last_hidden = x[:, -1, :]  # [batch_size, d_model]
        
        # 多任务预测
        outputs = {}
        
        # 1. 通用特征预测
        feature_pred = self.prediction_head(last_hidden)
        feature_pred = feature_pred.view(batch_size, self.prediction_horizon, self.output_dim)
        outputs['features'] = feature_pred
        
        # 2. 网络切片类型预测
        slice_logits = self.slice_classifier(last_hidden)
        outputs['slice_type'] = slice_logits
        
        # 3. 带宽需求预测
        bandwidth_pred = self.bandwidth_predictor(last_hidden)
        outputs['bandwidth'] = bandwidth_pred.unsqueeze(-1)  # [batch_size, horizon, 1]
        
        return outputs
    
    def predict_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """单步预测"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def generate_attention_mask(self, sequence_length: int, 
                              device: torch.device) -> torch.Tensor:
        """生成注意力掩码"""
        mask = torch.tril(torch.ones(sequence_length, sequence_length))
        return mask.to(device)


class SliceAwareTransformer(TransformerPredictor):
    """切片感知的Transformer模型"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 切片类型嵌入
        self.slice_embedding = nn.Embedding(3, self.d_model // 4)
        
        # 调整输入嵌入层以考虑切片信息
        self.input_embedding = nn.Linear(
            self.input_dim + self.d_model // 4, 
            self.d_model
        )
    
    def forward(self, x: torch.Tensor, slice_types: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播（包含切片类型信息）
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            slice_types: 切片类型 [batch_size, seq_len]
            mask: 注意力掩码
        
        Returns:
            预测结果字典
        """
        batch_size, seq_len, _ = x.shape
        
        # 切片类型嵌入
        slice_emb = self.slice_embedding(slice_types)  # [batch_size, seq_len, d_model//4]
        
        # 拼接输入特征和切片嵌入
        x_concat = torch.cat([x, slice_emb], dim=-1)
        
        # 后续处理与基础Transformer相同
        x = self.input_embedding(x_concat)
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        last_hidden = x[:, -1, :]
        
        outputs = {}
        feature_pred = self.prediction_head(last_hidden)
        feature_pred = feature_pred.view(batch_size, self.prediction_horizon, self.output_dim)
        outputs['features'] = feature_pred
        
        slice_logits = self.slice_classifier(last_hidden)
        outputs['slice_type'] = slice_logits
        
        bandwidth_pred = self.bandwidth_predictor(last_hidden)
        outputs['bandwidth'] = bandwidth_pred.unsqueeze(-1)
        
        return outputs


def create_transformer_model(config: Dict, slice_aware: bool = False) -> nn.Module:
    """
    创建Transformer模型
    
    Args:
        config: 模型配置
        slice_aware: 是否使用切片感知模型
    
    Returns:
        Transformer模型实例
    """
    if slice_aware:
        return SliceAwareTransformer(config)
    else:
        return TransformerPredictor(config)