"""Log Detector 整合模組

整合多種異常偵測模型，提供分數正規化、閾值決策與 Ensemble 機制。
"""

import numpy as np
import os
import sys
from typing import Dict, List, Optional, Literal, Any
from dataclasses import dataclass, field
from enum import Enum
from datasets import Dataset, load_from_disk, concatenate_datasets
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from anomaly_dection.isolation_forest import IsolationForestDetector, IsolationForestConfig
from anomaly_dection.copod import COPODDetector, COPODConfig
from anomaly_dection.autoencoder import AutoEncoderDetector, AutoEncoderConfig
from anomaly_dection.pca_gmm import PCAGMMDetector, PCAGMMConfig

SEED = getattr(config, "SEED", 42)
RESULT_FIG_DIR = os.path.join("result", "unsupervised_anomaly_dection")


def _set_global_seed(seed: int):
    """Set seeds for reproducibility across numpy/torch."""
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


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
        _set_global_seed(SEED)
        self.config = config or LogDetectorConfig()
        self.detectors: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, np.ndarray]] = {}
        self._is_fitted = False
        
        # * 根據設定初始化所需的偵測器
        for model_name in self.config.models:
            if model_name in self.MODEL_REGISTRY:
                detector_cls, config_attr = self.MODEL_REGISTRY[model_name]
                model_config = getattr(self.config, config_attr)
                if hasattr(model_config, "random_state"):
                    model_config.random_state = SEED
                if hasattr(model_config, "gmm_random_state"):
                    model_config.gmm_random_state = SEED
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


if __name__ == "__main__":
    from visualization.aggregator import ResultAggregator
    from visualization.trend_analysis import plot_trend_analysis, plot_anomaly_count_trend
    from visualization.distribution_plot import (
        plot_score_histogram, plot_distribution_evolution, plot_comparison_violin
    )

    _set_global_seed(SEED)
    print("=" * 60)
    print("多規模異常偵測實驗")
    print("=" * 60)
    
    # * 載入所有可用的 Log Vector Dataset
    print("\n[1] 掃描 Log Vector Dataset...")
    logvector_dirs = sorted([
        os.path.join(config.LOG_VECTORS_DIR, d)
        for d in os.listdir(config.LOG_VECTORS_DIR)
        if d.endswith("_logvectors") and os.path.isdir(os.path.join(config.LOG_VECTORS_DIR, d))
    ])

    if not logvector_dirs:
        raise RuntimeError(f"在 {config.LOG_VECTORS_DIR} 下找不到任何 *_logvectors 資料夾")

    total_datasets = len(logvector_dirs)
    print(f"    找到 {total_datasets} 個資料集")
    
    # * 定義實驗規模：1, 5, 10, 15, 20...
    EXPERIMENT_SIZES = [1, 5, 10, 15, 20, 25, total_datasets]
    EXPERIMENT_SIZES = [s for s in EXPERIMENT_SIZES if s <= total_datasets]
    EXPERIMENT_SIZES = sorted(set(EXPERIMENT_SIZES))
    print(f"    實驗規模: {EXPERIMENT_SIZES}")
    
    # * 結果聚合器
    aggregator = ResultAggregator(scaler_type="minmax")
    
    # * 執行多規模實驗
    for n_datasets in EXPERIMENT_SIZES:
        print(f"\n{'='*40}")
        print(f"[實驗] 使用 {n_datasets} 個資料集")
        print("="*40)
        
        # * 載入並合併指定數量的資料集
        selected_dirs = logvector_dirs[:n_datasets]
        datasets_list = [load_from_disk(path) for path in selected_dirs]
        dataset = concatenate_datasets(datasets_list)
        X = np.array(dataset["log_vector"])
        print(f"    樣本數: {X.shape[0]}, 維度: {X.shape[1]}")
        
        # * 執行異常偵測
        detector = LogDetector()
        results = detector.fit_predict(X)
        
        # * 記錄結果
        aggregator.add_experiment(results, dataset_size=n_datasets)
        
        # * 輸出摘要
        print("    各模型異常數:")
        for model_name, model_results in results.items():
            n_anomalies = model_results["labels"].sum()
            print(f"      - {model_name}: {n_anomalies}")
    
    # * 生成視覺化
    print("\n" + "="*60)
    print("生成視覺化報告")
    print("="*60)
    
    print("\n[1] 效能趨勢分析...")
    path = plot_trend_analysis(aggregator, RESULT_FIG_DIR)
    print(f"    儲存至: {path}")
    
    print("\n[2] 異常數量趨勢...")
    path = plot_anomaly_count_trend(aggregator, RESULT_FIG_DIR)
    print(f"    儲存至: {path}")
    
    print("\n[3] 分佈演變圖...")
    paths = plot_distribution_evolution(aggregator, RESULT_FIG_DIR)
    for p in paths:
        print(f"    儲存至: {p}")
    
    print("\n[4] 小提琴對比圖...")
    path = plot_comparison_violin(aggregator, RESULT_FIG_DIR)
    print(f"    儲存至: {path}")
    
    # * 最大規模的直方圖
    print("\n[5] 最大規模分數分佈...")
    largest_results = {
        r.model_name: {
            "normalized_scores": r.normalized_scores,
            "labels": r.labels,
            "scores": r.normalized_scores
        }
        for r in aggregator.get_by_size(max(EXPERIMENT_SIZES))
    }
    path = plot_score_histogram(largest_results, RESULT_FIG_DIR, 
                                f"Score Distribution ({max(EXPERIMENT_SIZES)} Datasets)")
    print(f"    儲存至: {path}")
    
    print("\n" + "="*60)
    print("實驗完成！")
    print("="*60)