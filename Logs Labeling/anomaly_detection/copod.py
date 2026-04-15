"""COPOD 異常偵測模組

基於 Copula 函數的異常偵測，評估樣本在多變數分佈中的尾端機率。
不依賴距離計算，計算效率極高。
"""

import numpy as np
from pyod.models.copod import COPOD
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class COPODConfig:
    """COPOD 超參數設定"""
    contamination: float = 0.05
    n_jobs: int = -1


class COPODDetector:
    """COPOD 異常偵測器
    
    透過經驗累積分佈函數 (ECDF) 評估樣本在各維度的極端程度，
    結合機率估算總體異常分數。
    """
    
    def __init__(self, config: Optional[COPODConfig] = None):
        self.config = config or COPODConfig()
        self.model: Optional[COPOD] = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> "COPODDetector":
        """訓練 COPOD 模型
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
        """
        # * 建立 COPOD 模型，計算各維度的 ECDF 並擬合 Copula
        self.model = COPOD(
            contamination=self.config.contamination,
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
        
        # * 計算基於 Copula 的異常分數（結合左尾與右尾機率）
        return self.model.decision_function(X)
    
    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """預測異常標籤
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
            
        Returns:
            標籤陣列 (0: 正常, 1: 異常)
        """
        if not self._is_fitted:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()")
        
        return self.model.predict(X)
    
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
