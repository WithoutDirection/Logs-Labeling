"""Log Detector 整合模組

整合多種異常偵測模型，提供分數正規化、閾值決策與 Ensemble 機制。
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datasets import Dataset, concatenate_datasets
import matplotlib.pyplot as plt

# * 調整匯入路徑，確保能載入同專案上層的模組
CURRENT_DIR = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(CURRENT_DIR).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.path import (
    get_current_dir, get_parent_dir, join_path, get_basename,
    ensure_dir, get_dirs
)
from utils.dataset import load_dataset, save_dataset

import config
from anomaly_dection.isolation_forest import IsolationForestDetector, IsolationForestConfig
from anomaly_dection.copod import COPODDetector, COPODConfig
from anomaly_dection.autoencoder import AutoEncoderDetector, AutoEncoderConfig
from anomaly_dection.pca_gmm import PCAGMMDetector, PCAGMMConfig

SEED = getattr(config, "SEED", 42)
RESULT_FIG_DIR = join_path("result", "unsupervised_anomaly_dection")


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
    models: List[str] = field(default_factory=lambda: getattr(config, "DETECTION_MODELS", ["isolation_forest", "copod", "autoencoder", "pca_gmm"]))
    score_scaler: ScalerType = field(default_factory=lambda: ScalerType(getattr(config, "SCORE_SCALER", "minmax")))
    threshold_method: ThresholdMethod = field(default_factory=lambda: ThresholdMethod(getattr(config, "THRESHOLDING_METHOD", "percentile")))
    threshold_params: Dict[str, Any] = field(default_factory=lambda: getattr(config, "THRESHOLDING_PARAMS", {"percentile": 95}))
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: getattr(config, "ENSEMBLE_WEIGHTS", {
        "isolation_forest": 0.25,
        "copod": 0.25,
        "autoencoder": 0.25,
        "pca_gmm": 0.25
    }))
    output_dir: str = field(default_factory=lambda: getattr(config, "DETECTION_RESULTS_DIR", join_path(config.DATA_DIR, "Detection_Results")))
    embedding_column: str = "embedding"  # 嵌入向量欄位名稱（支援 embedding 或 log_vector）
    
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
        dataset = load_dataset(dataset_path)
        
        # * 擷取嵌入向量欄位轉為 numpy 陣列（支援 embedding 或 log_vector）
        embedding_col = self.config.embedding_column
        if embedding_col not in dataset.column_names:
            # 嘗試其他可能的欄位名稱
            for alt_col in ["embedding", "log_vector", "embeddings", "vector"]:
                if alt_col in dataset.column_names:
                    embedding_col = alt_col
                    break
            else:
                raise ValueError(f"找不到嵌入向量欄位，可用欄位: {dataset.column_names}")
        
        X = np.array(dataset[embedding_col])
        
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
        ensure_dir(self.config.output_dir)
        output_path = join_path(self.config.output_dir, f"{output_name}_detection")
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
        output_name = get_basename(dataset_path).replace("_logvectors", "")
        detector.save_results(result_dataset, output_name)
    
    return result_dataset


def run_detection_pipeline(
    input_dir: str = None,
    output_dir: str = None,
    models: List[str] = None,
    verbose: bool = True
) -> Dict[str, Dataset]:
    """執行完整的異常偵測流程
    
    遍歷 input_dir 中的所有 embedding 資料夾，對每個資料集執行異常偵測。
    
    Args:
        input_dir: 輸入資料目錄（預設使用 config.LOG_VECTORS_DIR）
        output_dir: 輸出結果目錄（預設使用 config.DETECTION_RESULTS_DIR）
        models: 要使用的模型列表
        verbose: 是否顯示進度訊息
        
    Returns:
        Dict[str, Dataset]: 資料集名稱到偵測結果的映射
    """
    input_dir = input_dir or getattr(config, "LOG_VECTORS_DIR", join_path(config.DATA_DIR, "Embeddings"))
    output_dir = output_dir or getattr(config, "DETECTION_RESULTS_DIR", join_path(config.DATA_DIR, "Detection_Results"))
    
    # 確保輸出目錄存在
    ensure_dir(output_dir)
    
    # 取得所有 embedding 資料夾
    embedding_dirs = get_dirs(input_dir)
    if not embedding_dirs:
        print(f"警告：在 {input_dir} 中找不到任何資料夾")
        return {}
    
    if verbose:
        print(f"找到 {len(embedding_dirs)} 個資料集")
        print(f"輸入目錄: {input_dir}")
        print(f"輸出目錄: {output_dir}")
        print(f"使用模型: {models or getattr(config, 'DETECTION_MODELS', ['isolation_forest', 'copod', 'autoencoder', 'pca_gmm'])}")
        print("=" * 60)
    
    # 建立偵測器設定
    detector_config = LogDetectorConfig(
        models=models or getattr(config, "DETECTION_MODELS", ["isolation_forest", "copod", "autoencoder", "pca_gmm"]),
        output_dir=output_dir
    )
    
    results = {}
    
    for i, embed_dir in enumerate(embedding_dirs, 1):
        dataset_path = join_path(input_dir, embed_dir)
        dataset_name = embed_dir.replace("_embeddings", "").replace("_logvectors", "")
        
        if verbose:
            print(f"\n[{i}/{len(embedding_dirs)}] 處理資料集: {dataset_name}")
        
        try:
            # 為每個資料集建立新的偵測器
            detector = LogDetector(detector_config)
            
            # 執行偵測
            result_dataset = detector.detect_from_dataset(dataset_path)
            
            # 儲存結果
            output_path = join_path(output_dir, f"{dataset_name}_detection")
            result_dataset.save_to_disk(output_path)
            
            results[dataset_name] = result_dataset
            
            if verbose:
                n_samples = len(result_dataset)
                n_anomalies = sum(result_dataset["ensemble_label"])
                anomaly_ratio = n_anomalies / n_samples * 100 if n_samples > 0 else 0
                print(f"   - 樣本數: {n_samples}")
                print(f"   - 異常數: {n_anomalies} ({anomaly_ratio:.2f}%)")
                print(f"   - 已儲存至: {output_path}")
                
        except Exception as e:
            print(f"   - 處理失敗: {e}")
            continue
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"完成！共處理 {len(results)}/{len(embedding_dirs)} 個資料集")
    
    return results


def generate_detection_summary(results: Dict[str, Dataset], output_dir: str = None) -> None:
    """生成偵測結果摘要與視覺化
    
    Args:
        results: run_detection_pipeline 的返回結果
        output_dir: 輸出目錄
    """
    output_dir = output_dir or getattr(config, "DETECTION_RESULTS_DIR", RESULT_FIG_DIR)
    ensure_dir(output_dir)
    
    summary_data = []
    
    for dataset_name, dataset in results.items():
        n_samples = len(dataset)
        
        row = {"dataset": dataset_name, "n_samples": n_samples}
        
        # 統計各模型的異常數量
        for col in dataset.column_names:
            if col.endswith("_label"):
                model_name = col.replace("_label", "")
                n_anomalies = sum(dataset[col])
                row[f"{model_name}_anomalies"] = n_anomalies
                row[f"{model_name}_ratio"] = n_anomalies / n_samples * 100 if n_samples > 0 else 0
        
        summary_data.append(row)
    
    # 輸出摘要統計
    print("\n偵測結果摘要：")
    print("-" * 80)
    for row in summary_data:
        print(f"資料集: {row['dataset']}")
        print(f"  樣本數: {row['n_samples']}")
        for key, value in row.items():
            if key.endswith("_ratio"):
                model = key.replace("_ratio", "")
                anomalies = row.get(f"{model}_anomalies", 0)
                print(f"  {model}: {anomalies} 異常 ({value:.2f}%)")
        print()


if __name__ == "__main__":
    
    
    # 執行偵測流程
    results = run_detection_pipeline(
        input_dir=config.LOG_VECTORS_DIR,
        output_dir=config.DETECTION_RESULTS_DIR,
        models=getattr(config, "DETECTION_MODELS", None),
        verbose=True
    )
    
    # 生成摘要
    if results:
        generate_detection_summary(results, output_dir=config.DETECTION_RESULTS_DIR)