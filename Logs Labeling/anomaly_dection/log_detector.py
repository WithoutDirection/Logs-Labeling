"""Log Detector 整合模組

整合多種異常偵測模型，提供分數正規化、閾值決策與 Ensemble 機制。
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datasets import Dataset
import pandas as pd
from scipy import stats

# * 調整匯入路徑，確保能載入同專案上層的模組
CURRENT_DIR = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(CURRENT_DIR).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.path import join_path, ensure_dir, get_dirs
from utils.dataset import load_dataset

import config
from anomaly_dection.isolation_forest import IsolationForestDetector, IsolationForestConfig
from anomaly_dection.copod import COPODDetector, COPODConfig
from anomaly_dection.autoencoder import AutoEncoderDetector, AutoEncoderConfig
from anomaly_dection.pca_gmm import PCAGMMDetector, PCAGMMConfig

SEED = config.SEED


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
    MAD = "mad"  # Median Absolute Deviation


@dataclass
class LogDetectorConfig:
    """Log Detector 設定 - 直接從 config.py 讀取參數"""
    # 各模型設定（保留物件形式以便模型初始化）
    if_config: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    copod_config: COPODConfig = field(default_factory=COPODConfig)
    ae_config: AutoEncoderConfig = field(default_factory=AutoEncoderConfig)
    pca_gmm_config: PCAGMMConfig = field(default_factory=PCAGMMConfig)
    
    # 從 config.py 讀取的參數（使用 property 以便動態取值）
    @property
    def models(self) -> List[str]:
        return config.DETECTION_MODELS
    
    @property
    def score_scaler(self) -> ScalerType:
        return ScalerType(config.SCORE_SCALER)
    
    @property
    def threshold_method(self) -> ThresholdMethod:
        return ThresholdMethod(config.THRESHOLDING_METHOD)
    
    @property
    def threshold_params(self) -> Dict[str, Any]:
        return config.THRESHOLDING_PARAMS
    
    @property
    def ensemble_weights(self) -> Dict[str, float]:
        return config.ENSEMBLE_WEIGHTS
    
    @property
    def output_dir(self) -> str:
        return config.DETECTION_RESULTS_DIR
    
    @property
    def mad_multiplier(self) -> float:
        return config.MAD_THRESHOLD_MULTIPLIER
    
    @property
    def mad_use_modified(self) -> bool:
        return config.MAD_USE_MODIFIED
    
    @property
    def enable_smoothing(self) -> bool:
        return config.ENABLE_TIME_SERIES_SMOOTHING
    
    @property
    def smoothing_window(self) -> int:
        return config.SMOOTHING_WINDOW_SIZE
    
    @property
    def smoothing_method(self) -> str:
        return config.SMOOTHING_METHOD
    
    @property
    def enable_correlation(self) -> bool:
        return config.ENABLE_CORRELATION_ANALYSIS
    
    @property
    def correlation_method(self) -> str:
        return config.CORRELATION_METHOD


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
    
    def _compute_mad_threshold(self, scores: np.ndarray) -> float:
        """使用 MAD (Median Absolute Deviation) 計算自適應閾值
        
        Args:
            scores: 分數陣列
            
        Returns:
            閾值
        """
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        
        if self.config.mad_use_modified:
            # 修正版 MAD：更接近標準差
            mad = mad * 1.4826
        
        threshold = median + self.config.mad_multiplier * mad
        return threshold
    
    def _smooth_scores(self, scores: np.ndarray) -> np.ndarray:
        """對分數進行時間序列平滑化
        
        Args:
            scores: 原始分數
            
        Returns:
            平滑化後的分數
        """
        if not self.config.enable_smoothing or len(scores) < self.config.smoothing_window:
            return scores
        
        window_size = self.config.smoothing_window
        smoothed = np.zeros_like(scores)
        
        if self.config.smoothing_method == "mean":
            # 簡單移動平均
            for i in range(len(scores)):
                start = max(0, i - window_size // 2)
                end = min(len(scores), i + window_size // 2 + 1)
                smoothed[i] = np.mean(scores[start:end])
                
        elif self.config.smoothing_method == "median":
            # 移動中位數
            for i in range(len(scores)):
                start = max(0, i - window_size // 2)
                end = min(len(scores), i + window_size // 2 + 1)
                smoothed[i] = np.median(scores[start:end])
                
        elif self.config.smoothing_method == "gaussian":
            # 高斯加權平滑
            from scipy.ndimage import gaussian_filter1d
            sigma = window_size / 3.0  # 標準差為視窗大小的 1/3
            smoothed = gaussian_filter1d(scores, sigma=sigma, mode='nearest')
        
        return smoothed
    
    def _compute_correlation_matrix(self, all_scores: Dict[str, np.ndarray]) -> pd.DataFrame:
        """計算模型分數之間的相關性
        
        Args:
            all_scores: 各模型的分數字典
            
        Returns:
            相關性矩陣 DataFrame
        """
        if not self.config.enable_correlation or len(all_scores) < 2:
            return None
        
        # 將分數轉換為 DataFrame
        scores_df = pd.DataFrame(all_scores)
        
        # 計算相關性
        if self.config.correlation_method == "pearson":
            corr_matrix = scores_df.corr(method='pearson')
        elif self.config.correlation_method == "spearman":
            corr_matrix = scores_df.corr(method='spearman')
        elif self.config.correlation_method == "kendall":
            corr_matrix = scores_df.corr(method='kendall')
        else:
            corr_matrix = scores_df.corr(method='pearson')
        
        return corr_matrix
    
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
        
        elif self.config.threshold_method == ThresholdMethod.MAD:
            # * MAD (Median Absolute Deviation) 自適應閾值
            threshold = self._compute_mad_threshold(scores)
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
        
        # * 步驟二：計算模型相關性
        if self.config.enable_correlation:
            corr_matrix = self._compute_correlation_matrix(all_normalized_scores)
            if corr_matrix is not None:
                self.results["correlation_matrix"] = corr_matrix
                print("\n模型分數相關性矩陣:")
                print(corr_matrix.to_string())
        
        # * 步驟三：計算加權 Ensemble 分數
        ensemble_scores = np.zeros(n_samples)
        total_weight = 0.0
        
        for model_name, scores in all_normalized_scores.items():
            weight = self.config.ensemble_weights.get(model_name, 0.0)
            ensemble_scores += weight * scores
            total_weight += weight
        
        if total_weight > 0:
            ensemble_scores /= total_weight
        
        # * 步驟四：時間序列平滑化
        if self.config.enable_smoothing:
            smoothed_scores = self._smooth_scores(ensemble_scores)
            self.results["ensemble_raw"] = {
                "scores": ensemble_scores,
                "labels": self._apply_threshold(ensemble_scores)
            }
            ensemble_scores = smoothed_scores
        
        # * 步驟五：對 Ensemble 分數應用閾值
        ensemble_labels = self._apply_threshold(ensemble_scores)
        
        self.results["ensemble"] = {
            "scores": ensemble_scores,
            "labels": ensemble_labels
        }
        
        return self.results
    
    def batch_detect_from_datasets(self, dataset_paths: List[str], dataset_names: List[str]) -> Dict[str, Dataset]:
        """批次載入多個資料集，合併訓練，然後分別預測
        
        Args:
            dataset_paths: 資料集路徑列表
            dataset_names: 資料集名稱列表
            
        Returns:
            各資料集名稱到偵測結果 Dataset 的映射
        """
        if len(dataset_paths) != len(dataset_names):
            raise ValueError("資料集路徑與名稱列表長度不一致")
        
        print(f"\n開始批次載入 {len(dataset_paths)} 個資料集...")
        
        # 步驟 1: 載入所有資料集並收集 embeddings
        all_embeddings = []
        all_datasets = []
        dataset_indices = []  # 記錄每個資料集在合併陣列中的起始和結束索引
        current_idx = 0
        
        for i, (path, name) in enumerate(zip(dataset_paths, dataset_names)):
            print(f"  [{i+1}/{len(dataset_paths)}] 載入資料集: {name}")
            dataset = load_dataset(path)
            
            # 擷取嵌入向量欄位
            embedding_col = None
            for col_name in ["embedding", "log_vector", "embeddings", "vector"]:
                if col_name in dataset.column_names:
                    embedding_col = col_name
                    break
            
            if embedding_col is None:
                raise ValueError(f"資料集 {name} 找不到嵌入向量欄位，可用欄位: {dataset.column_names}")
            
            embeddings = np.array(dataset[embedding_col])
            all_embeddings.append(embeddings)
            all_datasets.append(dataset)
            
            # 記錄索引範圍
            dataset_indices.append((current_idx, current_idx + len(embeddings)))
            current_idx += len(embeddings)
            
            print(f"     樣本數: {len(embeddings)}")
        
        # 步驟 2: 合併所有 embeddings 並訓練
        print(f"\n合併所有資料集，總樣本數: {current_idx}")
        X_combined = np.vstack(all_embeddings)
        
        print("開始訓練模型（在合併的資料集上）...")
        self.fit(X_combined)
        
        # 步驟 3: 對合併的資料集進行預測
        print("\n開始預測（在合併的資料集上）...")
        combined_results = self.predict(X_combined)
        
        # 步驟 4: 將結果分割回各個資料集
        print("\n將結果分割回各個資料集...")
        results_dict = {}
        
        for i, (name, dataset, (start_idx, end_idx)) in enumerate(zip(dataset_names, all_datasets, dataset_indices)):
            print(f"  [{i+1}/{len(dataset_names)}] 處理資料集: {name}")
            
            # 從合併結果中擷取該資料集的部分
            dataset_results = self._split_results_by_index(combined_results, start_idx, end_idx)
            
            # 將結果加入 Dataset
            result_dataset = self._add_results_to_dataset(dataset, dataset_results)
            results_dict[name] = result_dataset
            
            # 統計資訊
            n_samples = len(result_dataset)
            n_anomalies = sum(result_dataset["ensemble_label"])
            anomaly_ratio = n_anomalies / n_samples * 100 if n_samples > 0 else 0
            print(f"     樣本數: {n_samples}, 異常數: {n_anomalies} ({anomaly_ratio:.2f}%)")
        
        return results_dict
    
    def _split_results_by_index(self, combined_results: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """從合併的結果中擷取指定索引範圍的結果
        
        Args:
            combined_results: 合併的預測結果
            start_idx: 起始索引
            end_idx: 結束索引
            
        Returns:
            該索引範圍的結果字典
        """
        split_results = {}
        
        for model_name, model_results in combined_results.items():
            if model_name == "correlation_matrix":
                # 相關性矩陣是全域的，不需要分割
                continue
            
            split_results[model_name] = {}
            for key, value in model_results.items():
                if isinstance(value, np.ndarray):
                    split_results[model_name][key] = value[start_idx:end_idx]
                else:
                    split_results[model_name][key] = value
        
        return split_results
    
    def _add_results_to_dataset(self, dataset: Dataset, results: Dict[str, Any]) -> Dataset:
        """將偵測結果加入 Dataset
        
        Args:
            dataset: 原始 Dataset
            results: 偵測結果
            
        Returns:
            包含結果的 Dataset
        """
        new_columns = {}
        
        for model_name, model_results in results.items():
            if model_name == "ensemble":
                new_columns["ensemble_score"] = model_results["scores"].tolist()
                new_columns["ensemble_label"] = model_results["labels"].tolist()
            elif model_name == "ensemble_raw":
                new_columns["ensemble_raw_score"] = model_results["scores"].tolist()
                new_columns["ensemble_raw_label"] = model_results["labels"].tolist()
            else:
                if "raw_scores" in model_results:
                    new_columns[f"{model_name}_raw_score"] = model_results["raw_scores"].tolist()
                new_columns[f"{model_name}_score"] = model_results["normalized_scores"].tolist()
                new_columns[f"{model_name}_label"] = model_results["labels"].tolist()
        
        for col_name, col_data in new_columns.items():
            dataset = dataset.add_column(col_name, col_data)
        
        return dataset


def run_detection_pipeline(
    input_dir: str = None,
    output_dir: str = None,
    models: List[str] = None,
    verbose: bool = True
) -> Dict[str, Dataset]:
    """執行完整的異常偵測流程（批次模式）
    
    一次載入所有資料集、合併訓練、整體視覺化，最後逐一儲存結果。
    
    Args:
        input_dir: 輸入資料目錄（預設使用 config.LOG_VECTORS_DIR）
        output_dir: 輸出結果目錄（預設使用 config.DETECTION_RESULTS_DIR）
        models: 要使用的模型列表
        verbose: 是否顯示進度訊息
        
    Returns:
        Dict[str, Dataset]: 資料集名稱到偵測結果的映射
    """
    input_dir = input_dir or config.LOG_VECTORS_DIR
    output_dir = output_dir or config.DETECTION_RESULTS_DIR
    
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
        print(f"使用模型: {models or config.DETECTION_MODELS}")
        print(f"處理模式: 批次模式（合併訓練）")
        print("=" * 60)
    
    # 準備資料集路徑和名稱列表
    dataset_paths = []
    dataset_names = []
    
    for embed_dir in embedding_dirs:
        dataset_path = join_path(input_dir, embed_dir)
        dataset_name = embed_dir.replace("_embeddings", "").replace("_logvectors", "")
        dataset_paths.append(dataset_path)
        dataset_names.append(dataset_name)
    
    # 建立偵測器設定並執行批次處理
    detector_config = LogDetectorConfig()
    detector = LogDetector(detector_config)
    
    try:
        # 批次載入所有資料集、合併訓練、分別預測
        results = detector.batch_detect_from_datasets(dataset_paths, dataset_names)
        
        # 儲存所有結果
        if verbose:
            print("\n儲存所有結果...")
        
        for i, (dataset_name, result_dataset) in enumerate(results.items(), 1):
            output_path = join_path(output_dir, f"{dataset_name}_detection")
            result_dataset.save_to_disk(output_path)
            
            if verbose:
                print(f"  [{i}/{len(results)}] 已儲存: {dataset_name} -> {output_path}")
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"完成！共處理 {len(results)}/{len(embedding_dirs)} 個資料集")
            print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"批次處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    # 使用便捷 API 執行完整流程
    from anomaly_dection import run_detection
    result = run_detection(verbose=True)
    print(f"\n完成！處理了 {result['n_datasets']} 個資料集")