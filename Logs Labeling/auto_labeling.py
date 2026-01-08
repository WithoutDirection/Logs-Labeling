"""
AutoLabeling：自動標註模組

根據序列分群結果與 MITRE ATT&CK 外部知識進行自動標註。

# * 核心功能：
# * 1. 載入 HMM 分群結果與異常偵測分數
# * 2. 計算各 Cluster 的 Centroid 向量
# * 3. 將 MITRE 嵌入轉換至相同概念空間
# * 4. 計算 Cluster Centroid 與 MITRE 向量的相似度
# * 5. 根據相似度與異常分數產生最終標註
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow.feather as feather

# * 調整匯入路徑
CURRENT_DIR = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(CURRENT_DIR).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from utils.path import ensure_dir, get_dirs


@dataclass
class LabelingConfig:
    """自動標註配置"""
    
    # 閾值設定
    similarity_threshold: float = getattr(config, 'LABELING_SIMILARITY_THRESHOLD', 0.3)
    confidence_threshold: float = getattr(config, 'LABELING_CONFIDENCE_THRESHOLD', 0.2)
    
    # 權重設定
    anomaly_weight: float = getattr(config, 'LABELING_ANOMALY_WEIGHT', 0.3)
    similarity_weight: float = getattr(config, 'LABELING_SIMILARITY_WEIGHT', 0.7)
    top_k_techniques: int = getattr(config, 'LABELING_TOP_K', 3)
    
    # 路徑設定
    concept_vectors_dir: str = config.CONCEPT_VECTORS_DIR
    cluster_results_dir: str = config.CLUSTER_RESULTS_DIR
    detection_results_dir: str = config.DETECTION_RESULTS_DIR
    mitre_embeddings_dir: str = getattr(config, 'MITRE_EXTERNAL_KNOWLEDGE_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_ATTACK"))
    input_logs_dir: str = config.INPUT_LOGS_DIR
    labeling_results_dir: str = getattr(config, 'LABELING_RESULTS_DIR', os.path.join(config.RESULT_DIR, "Labeling_Results"))
    nmf_model_path: str = config.NMF_MODEL_PATH


class AutoLabeler:
    """
    自動標註器
    
    # * 流程：
    # * 1. load_cluster_results() - 載入 HMM 分群結果
    # * 2. load_anomaly_scores() - 載入異常偵測分數
    # * 3. load_mitre_embeddings() - 載入 MITRE 嵌入
    # * 4. compute_cluster_centroids() - 計算各 Cluster 的 Centroid
    # * 5. label_dataset() - 對單一資料集進行標註
    # * 6. batch_label_all() - 批次標註所有資料集
    """
    
    def __init__(self, config: Optional[LabelingConfig] = None):
        self.config = config or LabelingConfig()
        
        # 資料儲存
        self.concept_vectors: Dict[str, np.ndarray] = {}
        self.cluster_labels: Dict[str, np.ndarray] = {}
        self.anomaly_scores: Dict[str, np.ndarray] = {}
        
        # MITRE 資料
        self.mitre_embeddings: Optional[np.ndarray] = None
        self.mitre_technique_ids: Optional[List[str]] = None
        self.mitre_technique_names: Optional[List[str]] = None
        self.mitre_concept_vectors: Optional[np.ndarray] = None
        
        # NMF 模型
        self.nmf_model = None
        self._nmf_scaler = None
        
        # 結果
        self.labeling_results: Dict[str, pd.DataFrame] = {}
    
    # ======================== 資料載入 ========================
    
    def _load_from_subdirs(
        self,
        base_dir: str,
        loader_fn,
        dataset_ids: Optional[List[str]] = None,
        id_suffix: str = "",
        desc: str = "資料",
    ) -> Dict[str, np.ndarray]:
        """通用的子目錄載入器"""
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"找不到目錄: {base_dir}")
        
        result = {}
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            
            dataset_id = subdir.replace(id_suffix, "") if id_suffix else subdir
            if dataset_ids is not None and dataset_id not in dataset_ids:
                continue
            
            try:
                data = loader_fn(subdir_path)
                if data is not None:
                    result[dataset_id] = data
            except Exception as e:
                print(f"[Warning] 載入失敗 {subdir_path}: {e}")
        
        print(f"已載入 {len(result)} 個資料集的{desc}")
        return result
    
    def load_nmf_model(self, model_path: Optional[str] = None) -> None:
        """載入 NMF 模型"""
        model_path = model_path or self.config.nmf_model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到 NMF 模型: {model_path}")
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            self.nmf_model = data.get('model') or data.get('nmf_model')
            self._nmf_scaler = data.get('scaler')
        else:
            self.nmf_model = data
        
        print(f"[Info] 已載入 NMF 模型: {model_path}")
    
    def load_concept_vectors(
        self,
        dataset_ids: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """載入概念向量"""
        print("\n載入概念向量...")
        
        def loader(subdir_path):
            arrow_path = os.path.join(subdir_path, "data-00000-of-00001.arrow")
            if not os.path.exists(arrow_path):
                return None
            table = feather.read_table(arrow_path)
            if "concept_vector" in table.column_names:
                return np.array(table["concept_vector"].to_pylist())
            return table.to_pandas().values
        
        self.concept_vectors = self._load_from_subdirs(
            self.config.concept_vectors_dir, loader, dataset_ids, "_concepts", "概念向量"
        )
        return self.concept_vectors
    
    def load_cluster_labels(
        self,
        dataset_ids: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """載入 HMM 分群標籤"""
        print("\n載入分群標籤...")
        
        def loader(subdir_path):
            labels_path = os.path.join(subdir_path, "labels.npy")
            return np.load(labels_path) if os.path.exists(labels_path) else None
        
        self.cluster_labels = self._load_from_subdirs(
            self.config.cluster_results_dir, loader, dataset_ids, "", "分群標籤"
        )
        return self.cluster_labels
    
    def load_anomaly_scores(
        self,
        dataset_ids: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """載入異常偵測分數"""
        detection_dir = self.config.detection_results_dir
        
        if not os.path.exists(detection_dir):
            print(f"[Warning] 找不到異常偵測結果目錄，將使用預設分數")
            return self.anomaly_scores
        
        print("\n載入異常偵測分數...")
        
        # 嘗試載入整合結果
        scores = _load_scores_dict(detection_dir)
        if scores:
            self.anomaly_scores = scores
            print(f"已載入整合異常分數: {len(scores)} 個資料集")
            return self.anomaly_scores
        
        # 載入各資料集的個別結果
        def loader(subdir_path):
            for fname in ["ensemble_scores.npy", "scores.npy", "anomaly_scores.npy"]:
                path = os.path.join(subdir_path, fname)
                if os.path.exists(path):
                    return np.load(path)
            return None
        
        self.anomaly_scores = self._load_from_subdirs(
            detection_dir, loader, dataset_ids, "_detection", "異常分數"
        )
        return self.anomaly_scores
    
    def load_mitre_embeddings(
        self,
        embeddings_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """載入 MITRE ATT&CK 嵌入向量"""
        embeddings_dir = embeddings_dir or self.config.mitre_embeddings_dir
        print(f"\n載入 MITRE 嵌入向量: {embeddings_dir}")
        
        # 嘗試 NumPy 格式
        for fname in ["embeddings.npy", "mitre_embeddings.npy", "technique_embeddings.npy"]:
            path = os.path.join(embeddings_dir, fname)
            if os.path.exists(path):
                self.mitre_embeddings = np.load(path)
                self._load_mitre_metadata(embeddings_dir)
                break
        else:
            # 嘗試 Arrow 格式
            arrow_path = os.path.join(embeddings_dir, "data-00000-of-00001.arrow")
            if not os.path.exists(arrow_path):
                raise FileNotFoundError(f"找不到 MITRE 嵌入檔案於: {embeddings_dir}")
            
            df = feather.read_table(arrow_path).to_pandas()
            embed_col = next((c for c in ["embedding", "embeddings", "vector", "concept_vector"] if c in df.columns), None)
            if embed_col:
                self.mitre_embeddings = np.array(df[embed_col].tolist())
                self.mitre_technique_ids = (df.get("technique_id") or df.get("id", pd.Series(range(len(df))))).tolist()
                self.mitre_technique_names = (df.get("technique") or df.get("name", self.mitre_technique_ids)).tolist()
        
        print(f"已載入 {len(self.mitre_embeddings)} 個 MITRE 技術嵌入")
        return self.mitre_embeddings, self.mitre_technique_ids, self.mitre_technique_names
    
    def _load_mitre_metadata(self, embeddings_dir: str) -> None:
        """載入 MITRE metadata"""
        metadata_path = os.path.join(embeddings_dir, "metadata.csv")
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            self.mitre_technique_ids = (metadata.get("technique_id") or metadata.get("id", pd.Series(range(len(metadata))))).tolist()
            self.mitre_technique_names = (metadata.get("technique") or metadata.get("name", self.mitre_technique_ids)).tolist()
        else:
            n = len(self.mitre_embeddings)
            self.mitre_technique_ids = [f"T{i:04d}" for i in range(n)]
            self.mitre_technique_names = self.mitre_technique_ids
    
    def transform_mitre_to_concepts(self) -> np.ndarray:
        """將 MITRE 嵌入轉換至概念空間"""
        if self.mitre_embeddings is None:
            raise ValueError("請先載入 MITRE 嵌入")
        if self.nmf_model is None:
            self.load_nmf_model()
        
        print("\n將 MITRE 嵌入轉換至概念空間...")
        embeddings = self.mitre_embeddings
        if self._nmf_scaler is not None:
            embeddings = self._nmf_scaler.transform(embeddings)
        
        self.mitre_concept_vectors = self.nmf_model.transform(np.maximum(embeddings, 0))
        print(f"MITRE 概念向量形狀: {self.mitre_concept_vectors.shape}")
        return self.mitre_concept_vectors
    
    # ======================== 核心標註邏輯 ========================
    
    def compute_cluster_centroids(
        self,
        dataset_id: str,
        concept_vectors: np.ndarray,
        cluster_labels: np.ndarray,
        anomaly_scores: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """
        計算各 Cluster 的 Centroid 向量與平均異常分數
        
        Args:
            dataset_id: 資料集 ID
            concept_vectors: 概念向量矩陣 (n_samples, n_concepts)
            cluster_labels: 分群標籤 (n_samples,)
            anomaly_scores: 異常分數 (n_samples,)，可選
            
        Returns:
            Tuple of:
                - centroids: Dict[cluster_id -> centroid_vector]
                - avg_anomaly_scores: Dict[cluster_id -> avg_score]
        """
        unique_clusters = np.unique(cluster_labels)
        centroids = {}
        avg_anomaly_scores = {}
        
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_vectors = concept_vectors[mask]
            
            # 計算 Centroid（加權平均，權重為異常分數）
            if anomaly_scores is not None and len(anomaly_scores) == len(cluster_labels):
                cluster_scores = anomaly_scores[mask]
                # 異常分數作為權重（越高越重要）
                weights = cluster_scores / (cluster_scores.sum() + 1e-8)
                centroid = np.average(cluster_vectors, axis=0, weights=weights)
                avg_score = float(np.mean(cluster_scores))
            else:
                centroid = np.mean(cluster_vectors, axis=0)
                avg_score = 0.5  # 預設中等異常分數
            
            centroids[int(cluster_id)] = centroid
            avg_anomaly_scores[int(cluster_id)] = avg_score
        
        return centroids, avg_anomaly_scores
    
    def match_cluster_to_technique(
        self,
        centroid: np.ndarray,
        avg_anomaly_score: float,
    ) -> Dict[str, Any]:
        """
        將單一 Cluster Centroid 與 MITRE 技術進行比對
        
        Args:
            centroid: Cluster 的 Centroid 向量
            avg_anomaly_score: Cluster 的平均異常分數
            
        Returns:
            Dict with matching results
        """
        if self.mitre_concept_vectors is None:
            raise ValueError("請先執行 transform_mitre_to_concepts()")
        
        # 計算與所有 MITRE 技術的相似度
        centroid_2d = centroid.reshape(1, -1)
        similarities = cosine_similarity(centroid_2d, self.mitre_concept_vectors)[0]
        
        # 取 Top-K
        top_k_indices = np.argsort(similarities)[-self.config.top_k_techniques:][::-1]
        
        # 最高相似度
        best_idx = top_k_indices[0]
        best_similarity = float(similarities[best_idx])
        best_technique_id = self.mitre_technique_ids[best_idx]
        best_technique_name = self.mitre_technique_names[best_idx]
        
        # 計算最終信心度
        # confidence = anomaly_weight * anomaly_score + similarity_weight * similarity
        confidence = (
            self.config.anomaly_weight * avg_anomaly_score +
            self.config.similarity_weight * best_similarity
        )
        
        # 決定最終標籤
        final_score = best_similarity * confidence
        if final_score < self.config.confidence_threshold:
            predicted_technique = "Benign"
        else:
            predicted_technique = best_technique_id
        
        return {
            "predicted_technique": predicted_technique,
            "technique_name": best_technique_name if predicted_technique != "Benign" else "Benign",
            "similarity_score": best_similarity,
            "anomaly_score": avg_anomaly_score,
            "confidence": confidence,
            "final_score": final_score,
            "top_k_techniques": [
                {
                    "technique_id": self.mitre_technique_ids[idx],
                    "technique_name": self.mitre_technique_names[idx],
                    "similarity": float(similarities[idx]),
                }
                for idx in top_k_indices
            ],
        }
    
    def label_dataset(
        self,
        dataset_id: str,
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        對單一資料集進行標註
        
        Args:
            dataset_id: 資料集 ID
            output_dir: 輸出目錄
            
        Returns:
            標註結果 DataFrame
        """
        output_dir = output_dir or self.config.labeling_results_dir
        ensure_dir(output_dir)
        
        print(f"\n[Labeling] {dataset_id}")
        
        # 檢查資料是否存在
        if dataset_id not in self.concept_vectors:
            raise ValueError(f"找不到資料集 {dataset_id} 的概念向量")
        if dataset_id not in self.cluster_labels:
            raise ValueError(f"找不到資料集 {dataset_id} 的分群標籤")
        
        concept_vectors = self.concept_vectors[dataset_id]
        cluster_labels = self.cluster_labels[dataset_id]
        anomaly_scores = self.anomaly_scores.get(dataset_id)
        
        n_samples = len(concept_vectors)
        print(f"    樣本數: {n_samples}")
        print(f"    群集數: {len(np.unique(cluster_labels))}")
        
        # 計算 Cluster Centroids
        centroids, avg_anomaly_scores = self.compute_cluster_centroids(
            dataset_id, concept_vectors, cluster_labels, anomaly_scores
        )
        
        # 對每個 Cluster 進行標註
        cluster_results = {}
        for cluster_id, centroid in centroids.items():
            result = self.match_cluster_to_technique(
                centroid, avg_anomaly_scores[cluster_id]
            )
            cluster_results[cluster_id] = result
        
        # 將標註結果映射回每個樣本
        sample_results = []
        for i in range(n_samples):
            cluster_id = int(cluster_labels[i])
            result = cluster_results[cluster_id]
            
            sample_result = {
                "log_index": i,
                "cluster_id": cluster_id,
                "predicted_technique": result["predicted_technique"],
                "technique_name": result["technique_name"],
                "similarity_score": result["similarity_score"],
                "anomaly_score": anomaly_scores[i] if anomaly_scores is not None else result["anomaly_score"],
                "confidence": result["confidence"],
            }
            sample_results.append(sample_result)
        
        # 建立結果 DataFrame
        result_df = pd.DataFrame(sample_results)
        
        # 載入原始日誌並合併
        original_log_path = os.path.join(self.config.input_logs_dir, f"{dataset_id}.csv")
        if os.path.exists(original_log_path):
            try:
                original_df = pd.read_csv(original_log_path)
                
                # 確保長度一致
                if len(original_df) == n_samples:
                    # 合併原始欄位
                    for col in original_df.columns:
                        result_df[col] = original_df[col].values
                else:
                    print(f"    [Warning] 原始日誌長度 ({len(original_df)}) 與樣本數 ({n_samples}) 不一致")
            except Exception as e:
                print(f"    [Warning] 載入原始日誌失敗: {e}")
        
        # 儲存結果
        output_path = os.path.join(output_dir, f"{dataset_id}_Labeled.csv")
        result_df.to_csv(output_path, index=False)
        print(f"    [完成] 已儲存至: {output_path}")
        
        # 統計
        technique_counts = result_df["predicted_technique"].value_counts()
        print(f"    標註分布:")
        for tech, count in technique_counts.items():
            print(f"        {tech}: {count} ({count/n_samples*100:.1f}%)")
        
        self.labeling_results[dataset_id] = result_df
        return result_df
    
    def batch_label_all(
        self,
        dataset_ids: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        批次標註所有資料集
        
        Args:
            dataset_ids: 指定的資料集 ID 列表（None 表示全部）
            output_dir: 輸出目錄
            
        Returns:
            Dict[dataset_id -> result_df]
        """
        output_dir = output_dir or self.config.labeling_results_dir
        
        # 確定要處理的資料集
        if dataset_ids is None:
            dataset_ids = list(set(self.concept_vectors.keys()) & set(self.cluster_labels.keys()))
        
        print("=" * 60)
        print(f"批次自動標註 - 共 {len(dataset_ids)} 個資料集")
        print("=" * 60)
        
        results = {}
        for idx, dataset_id in enumerate(dataset_ids, 1):
            print(f"\n=== [{idx}/{len(dataset_ids)}] ===")
            try:
                result_df = self.label_dataset(dataset_id, output_dir)
                results[dataset_id] = result_df
            except Exception as e:
                print(f"    [Error] 標註失敗: {e}")
                continue
        
        # 生成摘要
        print("\n" + "=" * 60)
        print("標註摘要")
        print("=" * 60)
        print(f"成功標註: {len(results)}/{len(dataset_ids)} 個資料集")
        
        if results:
            # 統計所有標註結果
            all_techniques = pd.concat([df["predicted_technique"] for df in results.values()])
            technique_counts = all_techniques.value_counts()
            print(f"\n整體標註分布:")
            for tech, count in technique_counts.head(10).items():
                print(f"    {tech}: {count}")
        
        return results


# ======================== 便捷函式 ========================

def run_auto_labeling(
    dataset_ids: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    mitre_embeddings_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    執行自動標註流程
    
    Args:
        dataset_ids: 指定的資料集 ID 列表（None 表示全部）
        output_dir: 輸出目錄
        mitre_embeddings_dir: MITRE 嵌入目錄
        
    Returns:
        Dict[dataset_id -> result_df]
    """
    labeler = AutoLabeler()
    
    # 載入所有必要資料
    labeler.load_nmf_model()
    labeler.load_concept_vectors(dataset_ids)
    labeler.load_cluster_labels(dataset_ids)
    labeler.load_anomaly_scores(dataset_ids)
    labeler.load_mitre_embeddings(mitre_embeddings_dir)
    labeler.transform_mitre_to_concepts()
    
    # 執行標註
    return labeler.batch_label_all(dataset_ids, output_dir)


def _load_scores_dict(detection_dir: str) -> Optional[Dict[str, np.ndarray]]:
    """嘗試從目錄載入整合的異常分數字典"""
    ensemble_path = os.path.join(detection_dir, "ensemble_scores.npy")
    if not os.path.exists(ensemble_path):
        return None
    try:
        data = np.load(ensemble_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        return data if isinstance(data, dict) else None
    except Exception as e:
        print(f"[Warning] 載入整合分數失敗: {e}")
        return None


def load_anomaly_weights(
    detection_results_dir: str = config.DETECTION_RESULTS_DIR,
) -> Dict[str, np.ndarray]:
    """
    載入異常偵測分數作為後續標註的權重（供 Pipeline.py STAGE_II 使用）
    """
    if not os.path.exists(detection_results_dir):
        print(f"[Warning] 找不到異常偵測結果目錄: {detection_results_dir}")
        return {}
    
    # 嘗試載入整合結果
    scores = _load_scores_dict(detection_results_dir)
    if scores:
        print(f"[Info] 已載入 {len(scores)} 個資料集的異常分數權重")
        return scores
    
    # 遍歷各資料集目錄
    weights = {}
    for subdir in os.listdir(detection_results_dir):
        subdir_path = os.path.join(detection_results_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        dataset_id = subdir.replace("_detection", "").replace("_embeddings", "")
        for fname in ["ensemble_scores.npy", "scores.npy", "anomaly_scores.npy"]:
            path = os.path.join(subdir_path, fname)
            if os.path.exists(path):
                try:
                    weights[dataset_id] = np.load(path)
                    break
                except Exception as e:
                    print(f"[Warning] 載入失敗 {path}: {e}")
    
    print(f"[Info] 已載入 {len(weights)} 個資料集的異常分數權重")
    return weights


# ======================== 主程式 ========================

if __name__ == "__main__":
    print("=" * 60)
    print("自動標註 - MITRE ATT&CK 技術比對")
    print("=" * 60)
    
    results = run_auto_labeling()
    print("\n[完成] 自動標註已完成。")
