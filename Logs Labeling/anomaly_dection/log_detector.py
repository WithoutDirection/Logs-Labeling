"""Log Detector 整合模組

整合多種異常偵測模型，提供分數正規化、閾值決策與 Ensemble 機制。
"""

import numpy as np
import os
import sys
from typing import Dict, List, Optional, Literal, Any
from dataclasses import dataclass, field
from enum import Enum
from datasets import Dataset, load_from_disk

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from anomaly_dection.isolation_forest import IsolationForestDetector, IsolationForestConfig
from anomaly_dection.copod import COPODDetector, COPODConfig
from anomaly_dection.autoencoder import AutoEncoderDetector, AutoEncoderConfig
from anomaly_dection.pca_gmm import PCAGMMDetector, PCAGMMConfig


class ScalerType(str, Enum):
    """分數正規化方法"""
    MINMAX = "minmax"
    RANK = "rank"
    ZSCORE = "zscore"


class ThresholdMethod(str, Enum):
    """閾值決策策略"""
    PERCENTILE = "percentile"
    STD = "std"
    TOP_N = "top_n"


@dataclass
class LogDetectorConfig:
    """Log Detector 設定"""
    models: List[str] = field(default_factory=lambda: ["isolation_forest", "copod", "autoencoder", "pca_gmm"])
    score_scaler: ScalerType = ScalerType.MINMAX
    threshold_method: ThresholdMethod = ThresholdMethod.PERCENTILE
    threshold_params: Dict[str, Any] = field(default_factory=lambda: {"percentile": 95})
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "isolation_forest": 0.25,
        "copod": 0.25,
        "autoencoder": 0.25,
        "pca_gmm": 0.25
    })
    output_dir: str = os.path.join(config.DATA_DIR, "Detection_Results")
    
    # 各模型設定
    if_config: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    copod_config: COPODConfig = field(default_factory=COPODConfig)
    ae_config: AutoEncoderConfig = field(default_factory=AutoEncoderConfig)
    pca_gmm_config: PCAGMMConfig = field(default_factory=PCAGMMConfig)


class LogDetector:
    """日誌異常偵測整合器
    
    整合 Isolation Forest、COPOD、AutoEncoder、PCA+GMM 四種模型，
    提供統一的異常偵測介面與結果整合。
    """
    
    MODEL_REGISTRY = {
        "isolation_forest": (IsolationForestDetector, "if_config"),
        "copod": (COPODDetector, "copod_config"),
        "autoencoder": (AutoEncoderDetector, "ae_config"),
        "pca_gmm": (PCAGMMDetector, "pca_gmm_config")
    }
    
    def __init__(self, config: Optional[LogDetectorConfig] = None):
        self.config = config or LogDetectorConfig()
        self.detectors: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, np.ndarray]] = {}
        self._is_fitted = False
        
        # * 根據設定初始化所需的偵測器
        for model_name in self.config.models:
            if model_name in self.MODEL_REGISTRY:
                detector_cls, config_attr = self.MODEL_REGISTRY[model_name]
                model_config = getattr(self.config, config_attr)
                self.detectors[model_name] = detector_cls(model_config)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """正規化分數至 [0, 1] 區間
        
        Args:
            scores: 原始異常分數
            
        Returns:
            正規化後的分數
        """
        if self.config.score_scaler == ScalerType.MINMAX:
            # * Min-Max 正規化
            min_val, max_val = scores.min(), scores.max()
            if max_val - min_val == 0:
                return np.zeros_like(scores)
            return (scores - min_val) / (max_val - min_val)
        
        elif self.config.score_scaler == ScalerType.RANK:
            # * 排名正規化：轉為百分位數
            ranks = np.argsort(np.argsort(scores))
            return ranks / (len(scores) - 1) if len(scores) > 1 else np.zeros_like(scores)
        
        elif self.config.score_scaler == ScalerType.ZSCORE:
            # * Z-Score 正規化後映射到 [0, 1]
            mean, std = scores.mean(), scores.std()
            if std == 0:
                return np.zeros_like(scores)
            z_scores = (scores - mean) / std
            return 1 / (1 + np.exp(-z_scores))  # Sigmoid 映射
        
        return scores
    
    def _apply_threshold(self, scores: np.ndarray) -> np.ndarray:
        """根據閾值策略決定異常標籤
        
        Args:
            scores: 正規化後的分數
            
        Returns:
            標籤陣列 (0: 正常, 1: 異常)
        """
        params = self.config.threshold_params
        
        if self.config.threshold_method == ThresholdMethod.PERCENTILE:
            # * 百分位數閾值：取高於指定百分位的樣本為異常
            percentile = params.get("percentile", 95)
            threshold = np.percentile(scores, percentile)
            return (scores >= threshold).astype(int)
        
        elif self.config.threshold_method == ThresholdMethod.STD:
            # * 標準差閾值：mean + n*std
            n_std = params.get("n_std", 2)
            threshold = scores.mean() + n_std * scores.std()
            return (scores >= threshold).astype(int)
        
        elif self.config.threshold_method == ThresholdMethod.TOP_N:
            # * 固定數量：取分數最高的 N 個
            top_n = params.get("n", 100)
            threshold = np.sort(scores)[-top_n] if len(scores) >= top_n else scores.min()
            return (scores >= threshold).astype(int)
        
        return np.zeros(len(scores), dtype=int)
    
    def fit(self, X: np.ndarray) -> "LogDetector":
        """訓練所有偵測器
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
        """
        for model_name, detector in self.detectors.items():
            print(f"訓練 {model_name}...")
            detector.fit(X)
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """執行異常偵測並整合結果
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
            
        Returns:
            包含各模型結果與 Ensemble 結果的字典
        """
        if not self._is_fitted:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()")
        
        n_samples = X.shape[0]
        all_normalized_scores = {}
        
        # * 步驟一：收集各模型的原始分數並正規化
        for model_name, detector in self.detectors.items():
            raw_scores = detector.predict_scores(X)
            normalized_scores = self._normalize_scores(raw_scores)
            
            self.results[model_name] = {
                "raw_scores": raw_scores,
                "normalized_scores": normalized_scores,
                "labels": self._apply_threshold(normalized_scores)
            }
            all_normalized_scores[model_name] = normalized_scores
        
        # * 步驟二：計算加權 Ensemble 分數
        ensemble_scores = np.zeros(n_samples)
        total_weight = 0.0
        
        for model_name, scores in all_normalized_scores.items():
            weight = self.config.ensemble_weights.get(model_name, 0.0)
            ensemble_scores += weight * scores
            total_weight += weight
        
        if total_weight > 0:
            ensemble_scores /= total_weight
        
        # * 步驟三：對 Ensemble 分數應用閾值
        ensemble_labels = self._apply_threshold(ensemble_scores)
        
        self.results["ensemble"] = {
            "scores": ensemble_scores,
            "labels": ensemble_labels
        }
        
        return self.results
    
    def fit_predict(self, X: np.ndarray) -> Dict[str, Any]:
        """訓練並預測"""
        self.fit(X)
        return self.predict(X)
    
    def detect_from_dataset(self, dataset_path: str) -> Dataset:
        """從 Hugging Face Dataset 載入並偵測
        
        Args:
            dataset_path: Dataset 資料夾路徑
            
        Returns:
            包含偵測結果的 Dataset
        """
        dataset = load_from_disk(dataset_path)
        
        # * 擷取 log_vector 欄位轉為 numpy 陣列
        X = np.array(dataset["log_vector"])
        
        results = self.fit_predict(X)
        
        # * 將結果加入 Dataset
        new_columns = {}
        for model_name, model_results in results.items():
            if model_name == "ensemble":
                new_columns["ensemble_score"] = model_results["scores"].tolist()
                new_columns["ensemble_label"] = model_results["labels"].tolist()
            else:
                new_columns[f"{model_name}_raw_score"] = model_results["raw_scores"].tolist()
                new_columns[f"{model_name}_score"] = model_results["normalized_scores"].tolist()
                new_columns[f"{model_name}_label"] = model_results["labels"].tolist()
        
        for col_name, col_data in new_columns.items():
            dataset = dataset.add_column(col_name, col_data)
        
        return dataset
    
    def save_results(self, dataset: Dataset, output_name: str) -> str:
        """儲存偵測結果
        
        Args:
            dataset: 包含結果的 Dataset
            output_name: 輸出名稱
            
        Returns:
            儲存路徑
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, f"{output_name}_detection")
        dataset.save_to_disk(output_path)
        return output_path


def detect_anomalies(
    dataset_path: str,
    models: List[str] = None,
    score_scaler: str = "minmax",
    threshold_method: str = "percentile",
    threshold_params: Dict[str, Any] = None,
    ensemble_weights: Dict[str, float] = None,
    save_results: bool = True
) -> Dataset:
    """便捷函數：執行完整異常偵測流程
    
    Args:
        dataset_path: Log Vector Dataset 路徑
        models: 要使用的模型列表
        score_scaler: 分數正規化方法
        threshold_method: 閾值決策策略
        threshold_params: 閾值參數
        ensemble_weights: Ensemble 權重
        save_results: 是否儲存結果
        
    Returns:
        包含偵測結果的 Dataset
    """
    config = LogDetectorConfig(
        models=models or ["isolation_forest", "copod", "autoencoder", "pca_gmm"],
        score_scaler=ScalerType(score_scaler),
        threshold_method=ThresholdMethod(threshold_method),
        threshold_params=threshold_params or {"percentile": 95},
        ensemble_weights=ensemble_weights or {}
    )
    
    detector = LogDetector(config)
    result_dataset = detector.detect_from_dataset(dataset_path)
    
    if save_results:
        output_name = os.path.basename(dataset_path).replace("_logvectors", "")
        detector.save_results(result_dataset, output_name)
    
    return result_dataset


def visualize_anomaly_distribution(results: Dict[str, Any], title: str = "Anomaly Score Distribution"):
    """視覺化各模型的異常分數分佈"""
    import matplotlib.pyplot as plt
    
    model_names = [k for k in results.keys() if k != "ensemble"]
    n_models = len(model_names) + 1  # +1 for ensemble
    
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        scores = results[model_name]["normalized_scores"]
        labels = results[model_name]["labels"]
        
        # * 繪製正常與異常分數直方圖
        ax.hist(scores[labels == 0], bins=50, alpha=0.7, label="Normal", color="steelblue")
        ax.hist(scores[labels == 1], bins=50, alpha=0.7, label="Anomaly", color="crimson")
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("Score")
        ax.legend()
    
    # Ensemble
    ax = axes[len(model_names)]
    scores = results["ensemble"]["scores"]
    labels = results["ensemble"]["labels"]
    ax.hist(scores[labels == 0], bins=50, alpha=0.7, label="Normal", color="steelblue")
    ax.hist(scores[labels == 1], bins=50, alpha=0.7, label="Anomaly", color="crimson")
    ax.set_title("Ensemble")
    ax.set_xlabel("Score")
    ax.legend()
    
    for j in range(len(model_names) + 1, len(axes)):
        axes[j].axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(config.DATA_DIR, "anomaly_distribution.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    from datasets import load_from_disk
    
    # * 測試資料路徑
    TEST_DATASET = os.path.join(config.LOG_VECTORS_DIR, "0223d96d-40c2-44dc-b006-887cc322b025_logvectors")
    
    print("=" * 60)
    print("異常偵測模組測試")
    print("=" * 60)
    
    # * 載入測試資料
    print("\n[1] 載入 Log Vector Dataset...")
    dataset = load_from_disk(TEST_DATASET)
    X = np.array(dataset["log_vector"])
    print(f"    資料形狀: {X.shape}")
    
    # * 測試各獨立偵測器
    print("\n[2] 測試 Isolation Forest...")
    if_detector = IsolationForestDetector()
    if_results = if_detector.fit_predict(X)
    print(f"    異常數量: {if_results['labels'].sum()}")
    
    print("\n[3] 測試 COPOD...")
    copod_detector = COPODDetector()
    copod_results = copod_detector.fit_predict(X)
    print(f"    異常數量: {copod_results['labels'].sum()}")
    
    print("\n[4] 測試 AutoEncoder...")
    ae_detector = AutoEncoderDetector(AutoEncoderConfig(epochs=10))
    ae_results = ae_detector.fit_predict(X)
    print(f"    異常數量: {ae_results['labels'].sum()}")
    
    print("\n[5] 測試 PCA + GMM...")
    pca_gmm_detector = PCAGMMDetector()
    pca_gmm_results = pca_gmm_detector.fit_predict(X)
    print(f"    異常數量: {pca_gmm_results['labels'].sum()}")
    print(f"    PCA 維度: {pca_gmm_detector.n_pca_components}, GMM 元件: {pca_gmm_detector.n_gmm_components}")
    
    # * 測試整合偵測器
    print("\n[6] 測試 LogDetector 整合模組...")
    detector = LogDetector()
    results = detector.fit_predict(X)
    
    print("\n    各模型異常統計:")
    for model_name, model_results in results.items():
        labels = model_results.get("labels", model_results.get("labels"))
        print(f"    - {model_name}: {labels.sum()} 異常")
    
    # * 視覺化
    print("\n[7] 生成視覺化...")
    visualize_anomaly_distribution(results, "Anomaly Detection Results")
    
    print("\n" + "=" * 60)
    print("測試完成！")
    print("=" * 60)