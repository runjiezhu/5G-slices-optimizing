"""
数据预处理模块
处理原始数据，进行清洗、标准化、特征选择等操作
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import logging


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化预处理器"""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 预处理配置
        self.scaling_method = self.config.get('scaling_method', 'standard')
        self.imputation_method = self.config.get('imputation_method', 'mean')
        self.feature_selection = self.config.get('feature_selection', False)
        self.dimensionality_reduction = self.config.get('dimensionality_reduction', False)
        
        # 预处理器存储
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.pca = None
        
        # 拟合状态
        self.is_fitted = False
        
        self.logger.info("数据预处理器初始化完成")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DataPreprocessor':
        """
        拟合预处理器
        
        Args:
            X: 特征数据
            y: 标签数据（可选）
        
        Returns:
            自身实例
        """
        self.logger.info("开始拟合预处理器")
        
        # 1. 缺失值处理
        self._fit_imputer(X)
        X_imputed = self.imputer.transform(X)
        
        # 2. 特征缩放
        self._fit_scaler(X_imputed)
        X_scaled = self.scaler.transform(X_imputed)
        
        # 3. 特征选择
        if self.feature_selection and y is not None:
            self._fit_feature_selector(X_scaled, y)
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # 4. 降维
        if self.dimensionality_reduction:
            self._fit_pca(X_selected)
        
        self.is_fitted = True
        self.logger.info("预处理器拟合完成")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        应用预处理
        
        Args:
            X: 输入特征数据
        
        Returns:
            预处理后的数据
        """
        if not self.is_fitted:
            raise ValueError("预处理器尚未拟合，请先调用 fit() 方法")
        
        # 缺失值处理
        X_processed = self.imputer.transform(X)
        
        # 特征缩放
        X_processed = self.scaler.transform(X_processed)
        
        # 特征选择
        if self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)
        
        # 降维
        if self.pca is not None:
            X_processed = self.pca.transform(X_processed)
        
        return X_processed
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """拟合并转换数据"""
        return self.fit(X, y).transform(X)
    
    def _fit_imputer(self, X: np.ndarray) -> None:
        """拟合缺失值填充器"""
        if self.imputation_method == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        elif self.imputation_method == 'median':
            self.imputer = SimpleImputer(strategy='median')
        elif self.imputation_method == 'most_frequent':
            self.imputer = SimpleImputer(strategy='most_frequent')
        elif self.imputation_method == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            self.imputer = SimpleImputer(strategy='mean')
        
        self.imputer.fit(X)
    
    def _fit_scaler(self, X: np.ndarray) -> None:
        """拟合特征缩放器"""
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.scaler.fit(X)
    
    def _fit_feature_selector(self, X: np.ndarray, y: np.ndarray) -> None:
        """拟合特征选择器"""
        k_features = min(50, X.shape[1])  # 最多选择50个特征
        self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
        self.feature_selector.fit(X, y)
    
    def _fit_pca(self, X: np.ndarray) -> None:
        """拟合PCA降维"""
        n_components = min(20, X.shape[1], X.shape[0])  # 最多20个主成分
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性"""
        if self.feature_selector is not None:
            return self.feature_selector.scores_
        return None
    
    def get_selected_features(self, feature_names: List[str]) -> List[str]:
        """获取选中的特征名称"""
        if self.feature_selector is not None:
            mask = self.feature_selector.get_support()
            return [name for name, selected in zip(feature_names, mask) if selected]
        return feature_names
    
    def get_pca_explained_variance(self) -> Optional[np.ndarray]:
        """获取PCA解释的方差比例"""
        if self.pca is not None:
            return self.pca.explained_variance_ratio_
        return None
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆变换（尽可能恢复原始尺度）"""
        X_inv = X.copy()
        
        # PCA逆变换
        if self.pca is not None:
            X_inv = self.pca.inverse_transform(X_inv)
        
        # 特征选择无法完全逆变换，保持当前状态
        
        # 缩放逆变换
        if hasattr(self.scaler, 'inverse_transform'):
            X_inv = self.scaler.inverse_transform(X_inv)
        
        return X_inv
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """获取预处理信息"""
        info = {
            'is_fitted': self.is_fitted,
            'scaling_method': self.scaling_method,
            'imputation_method': self.imputation_method,
            'feature_selection_enabled': self.feature_selection,
            'dimensionality_reduction_enabled': self.dimensionality_reduction
        }
        
        if self.is_fitted:
            if self.feature_selector is not None:
                info['selected_features_count'] = self.feature_selector.k
                info['total_features'] = len(self.feature_selector.scores_)
            
            if self.pca is not None:
                info['pca_components'] = self.pca.n_components_
                info['pca_explained_variance_ratio'] = self.pca.explained_variance_ratio_.tolist()
                info['pca_total_explained_variance'] = np.sum(self.pca.explained_variance_ratio_)
        
        return info