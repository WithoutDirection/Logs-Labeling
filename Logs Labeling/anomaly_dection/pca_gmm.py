"""PCA + GMM 異常偵測模組

結合 PCA 降維與高斯混合模型進行機率密度估計，
解決高維空間中 GMM 難以收斂的問題。
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PCAGMMConfig:
    """PCA + GMM 超參數設定"""
    pca_explained_var: float = 0.95
    gmm_n_components_range: Tuple[int, int] = (2, 10)
    gmm_covariance_type: str = "full"
    gmm_random_state: int = 42
    gmm_max_iter: int = 200
    use_bic: bool = True


class PCAGMMDetector:
    """PCA + GMM 異常偵測器
    
    先以 PCA 降維保留主要變異，再用 GMM 擬合資料分佈，
    透過 Log-Likelihood 評估異常程度。
    """
    
    def __init__(self, config: Optional[PCAGMMConfig] = None):
        self.config = config or PCAGMMConfig()
        self.pca: Optional[PCA] = None
        self.gmm: Optional[GaussianMixture] = None
        self._is_fitted = False
        self._n_components: Optional[int] = None
    
    def _select_gmm_components(self, X_reduced: np.ndarray) -> int:
        """使用 BIC 選擇最佳 GMM 元件數
        
        Args:
            X_reduced: 降維後的資料矩陣
            
        Returns:
            最佳元件數
        """
        min_comp, max_comp = self.config.gmm_n_components_range
        best_bic = np.inf
        best_n = min_comp
        
        # * 遍歷可能的元件數，選擇 BIC 最低者
        for n in range(min_comp, max_comp + 1):
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=self.config.gmm_covariance_type,
                random_state=self.config.gmm_random_state,
                max_iter=self.config.gmm_max_iter
            )
            gmm.fit(X_reduced)
            bic = gmm.bic(X_reduced)
            
            if bic < best_bic:
                best_bic = bic
                best_n = n
        
        return best_n
    
    def fit(self, X: np.ndarray) -> "PCAGMMDetector":
        """訓練 PCA + GMM 模型
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
        """
        # * 步驟一：PCA 降維，保留指定比例的變異量
        self.pca = PCA(n_components=self.config.pca_explained_var, svd_solver='full')
        X_reduced = self.pca.fit_transform(X)
        
        # * 步驟二：使用 BIC 選擇 GMM 最佳元件數（或使用固定值）
        if self.config.use_bic:
            self._n_components = self._select_gmm_components(X_reduced)
        else:
            self._n_components = self.config.gmm_n_components_range[0]
        
        # * 步驟三：訓練最終 GMM 模型
        self.gmm = GaussianMixture(
            n_components=self._n_components,
            covariance_type=self.config.gmm_covariance_type,
            random_state=self.config.gmm_random_state,
            max_iter=self.config.gmm_max_iter
        )
        self.gmm.fit(X_reduced)
        
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
        
        X_reduced = self.pca.transform(X)
        
        # * 計算負對數似然：Log-Likelihood 越低（負數絕對值越大）代表越異常
        log_likelihood = self.gmm.score_samples(X_reduced)
        return -log_likelihood
    
    def predict_labels(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """預測異常標籤
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
            threshold: 異常閾值，若為 None 則使用 mean + 2*std
            
        Returns:
            標籤陣列 (0: 正常, 1: 異常)
        """
        scores = self.predict_scores(X)
        
        if threshold is None:
            threshold = np.mean(scores) + 2 * np.std(scores)
        
        return (scores > threshold).astype(int)
    
    def fit_predict(self, X: np.ndarray, threshold: Optional[float] = None) -> Dict[str, np.ndarray]:
        """訓練並預測
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
            threshold: 異常閾值
            
        Returns:
            包含 'scores' 和 'labels' 的字典
        """
        self.fit(X)
        return {
            "scores": self.predict_scores(X),
            "labels": self.predict_labels(X, threshold)
        }
    
    @property
    def n_pca_components(self) -> Optional[int]:
        """PCA 降維後的維度數"""
        return self.pca.n_components_ if self.pca else None
    
    @property
    def n_gmm_components(self) -> Optional[int]:
        """GMM 使用的高斯分佈數"""
        return self._n_components
