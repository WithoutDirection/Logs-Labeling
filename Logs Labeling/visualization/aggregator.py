"""結果聚合器模組

負責標準化與聚合多規模實驗結果，使其具備可比較性。
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from sklearn.preprocessing import MinMaxScaler, RobustScaler


@dataclass
class AggregatedResult:
    """聚合後的實驗結果"""
    model_name: str
    dataset_size: int
    normalized_scores: np.ndarray
    labels: np.ndarray
    raw_scores: Optional[np.ndarray] = None
    
    @property
    def n_samples(self) -> int:
        return len(self.labels)
    
    @property
    def n_anomalies(self) -> int:
        return int(self.labels.sum())
    
    @property
    def anomaly_ratio(self) -> float:
        return self.n_anomalies / self.n_samples if self.n_samples > 0 else 0.0
    
    @property
    def score_gap(self) -> float:
        """正常與異常分數間距"""
        if self.n_anomalies == 0 or self.n_anomalies == self.n_samples:
            return 0.0
        normal_mean = self.normalized_scores[self.labels == 0].mean()
        anomaly_mean = self.normalized_scores[self.labels == 1].mean()
        return anomaly_mean - normal_mean


class ResultAggregator:
    """多規模實驗結果聚合器"""
    
    def __init__(self, scaler_type: str = "minmax"):
        self.scaler_type = scaler_type
        self.results: List[AggregatedResult] = []
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """將分數正規化至 [0, 1]"""
        if self.scaler_type == "robust":
            # * Robust Scaler 處理極端值
            scaler = RobustScaler()
            scaled = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
            return np.clip(scaled, 0, 1)
        else:
            # * Min-Max 正規化
            min_val, max_val = scores.min(), scores.max()
            if max_val - min_val == 0:
                return np.zeros_like(scores)
            return (scores - min_val) / (max_val - min_val)
    
    def add_experiment(
        self, 
        results: Dict[str, Any], 
        dataset_size: int,
        normalize: bool = True
    ) -> None:
        """加入一次實驗結果
        
        Args:
            results: LogDetector.fit_predict() 的輸出
            dataset_size: 使用的資料集數量
            normalize: 是否重新正規化分數
        """
        for model_name, model_results in results.items():
            if model_name == "ensemble":
                scores = model_results["scores"]
                raw_scores = None
            else:
                scores = model_results["normalized_scores"]
                raw_scores = model_results.get("raw_scores")
            
            # * 重新正規化確保跨實驗可比性
            normalized = self._normalize(scores) if normalize else scores
            
            self.results.append(AggregatedResult(
                model_name=model_name,
                dataset_size=dataset_size,
                normalized_scores=normalized,
                labels=model_results["labels"],
                raw_scores=raw_scores
            ))
    
    def get_by_model(self, model_name: str) -> List[AggregatedResult]:
        """取得特定模型的所有結果"""
        return [r for r in self.results if r.model_name == model_name]
    
    def get_by_size(self, dataset_size: int) -> List[AggregatedResult]:
        """取得特定規模的所有結果"""
        return [r for r in self.results if r.dataset_size == dataset_size]
    
    def get_model_names(self) -> List[str]:
        """取得所有模型名稱"""
        return list(set(r.model_name for r in self.results))
    
    def get_dataset_sizes(self) -> List[int]:
        """取得所有資料集規模（已排序）"""
        return sorted(set(r.dataset_size for r in self.results))
    
    def to_dataframe(self):
        """轉換為 pandas DataFrame（長格式）"""
        import pandas as pd
        
        rows = []
        for r in self.results:
            for i, (score, label) in enumerate(zip(r.normalized_scores, r.labels)):
                rows.append({
                    "model": r.model_name,
                    "dataset_size": r.dataset_size,
                    "score": score,
                    "label": "Anomaly" if label == 1 else "Normal"
                })
        return pd.DataFrame(rows)
    
    def get_metrics_summary(self) -> Dict[str, List[Dict[str, Any]]]:
        """取得各模型各規模的指標摘要"""
        summary = {}
        for model_name in self.get_model_names():
            model_metrics = []
            for r in self.get_by_model(model_name):
                model_metrics.append({
                    "dataset_size": r.dataset_size,
                    "n_samples": r.n_samples,
                    "n_anomalies": r.n_anomalies,
                    "anomaly_ratio": r.anomaly_ratio,
                    "score_gap": r.score_gap,
                    "score_mean": r.normalized_scores.mean(),
                    "score_std": r.normalized_scores.std()
                })
            summary[model_name] = sorted(model_metrics, key=lambda x: x["dataset_size"])
        return summary
