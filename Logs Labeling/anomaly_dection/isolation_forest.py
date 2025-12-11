"""Isolation Forest 異常偵測模組

利用樹狀結構進行幾何隔離，異常點在空間中分佈稀疏，
容易被隨機超平面隔離（路徑較短）。
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class IsolationForestConfig:
    """Isolation Forest 超參數設定"""
    n_estimators: int = 100
    contamination: float | str = "auto"
    max_samples: str | int = "auto"
    random_state: int = 42
    n_jobs: int = -1


class IsolationForestDetector:
    """Isolation Forest 異常偵測器
    
    透過隨機切割空間，計算樣本隔離所需的平均路徑長度。
    路徑越短代表越容易被隔離，異常分數越高。
    """
    
    def __init__(self, config: Optional[IsolationForestConfig] = None):
        self.config = config or IsolationForestConfig()
        self.model: Optional[IsolationForest] = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """訓練 Isolation Forest 模型
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
        """
        # * 建立並訓練 Isolation Forest，使用隨機子採樣建立多棵 iTrees
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        self.model.fit(X)
        self._is_fitted = True
        return self
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """計算異常分數
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
            
        Returns:
            異常分數陣列，值越高代表越異常
        """
        if not self._is_fitted:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()")
        
        # * 取得 decision_function 分數並轉換：原始分數越負代表越異常，乘以 -1 使越高越異常
        raw_scores = self.model.decision_function(X)
        return -raw_scores
    
    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """預測異常標籤
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
            
        Returns:
            標籤陣列 (0: 正常, 1: 異常)
        """
        if not self._is_fitted:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()")
        
        # * sklearn 輸出 1=正常, -1=異常，轉換為 0=正常, 1=異常
        predictions = self.model.predict(X)
        return (predictions == -1).astype(int)
    
    def fit_predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """訓練並預測
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
            
        Returns:
            包含 'scores' 和 'labels' 的字典
        """
        self.fit(X)
        return {
            "scores": self.predict_scores(X),
            "labels": self.predict_labels(X)
        }
