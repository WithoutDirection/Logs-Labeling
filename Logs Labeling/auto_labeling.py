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
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow.feather as feather
import pyarrow
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# * 調整匯入路徑
CURRENT_DIR = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(CURRENT_DIR).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from utils.path import ensure_dir, get_dirs


def ensure_mitre_raw_embeddings(
    embeddings_dir: str,
    *,
    mitre_csv: Optional[str] = None,
    bert_model: Optional[str] = None,
    force_rebuild: bool = False,
) -> str:
    """Ensure MITRE raw embedding dataset exists on disk.

    This integrates the external_sources builder into the auto-labeling flow.
    """
    # Heuristic: HuggingFace datasets saved via save_to_disk() include state.json
    if (
        not force_rebuild
        and os.path.exists(embeddings_dir)
        and os.path.exists(os.path.join(embeddings_dir, "state.json"))
    ):
        return embeddings_dir

    from external_sources.build_mitre_raw_embeddings import build_mitre_raw_embeddings

    return build_mitre_raw_embeddings(
        mitre_csv=mitre_csv,
        out_dir=embeddings_dir,
        bert_model=bert_model,
        force_rebuild=force_rebuild,
    )


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
    use_raw_embeddings: bool = getattr(config, 'LABELING_USE_RAW_EMBEDDINGS', False)
    
    # TF-IDF Hybrid Scoring
    use_tfidf: bool = getattr(config, 'LABELING_USE_TFIDF', False)
    weight_embedding: float = getattr(config, 'LABELING_WEIGHT_EMBEDDING', 0.7)
    weight_tfidf: float = getattr(config, 'LABELING_WEIGHT_TFIDF', 0.3)
    
    # 路徑設定
    concept_vectors_dir: str = config.CONCEPT_VECTORS_DIR
    cluster_results_dir: str = config.CLUSTER_RESULTS_DIR
    detection_results_dir: str = config.DETECTION_RESULTS_DIR
    mitre_embeddings_dir: str = getattr(config, 'MITRE_EXTERNAL_KNOWLEDGE_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_ATTACK"))
    mitre_tfidf_dir: str = getattr(config, 'MITRE_TFIDF_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_TFIDF"))
    input_logs_dir: str = config.INPUT_LOGS_DIR
    log_vectors_dir: str = config.LOG_VECTORS_DIR  # Added this field
    intermediate_data_dir: str = config.INTERMEDIATE_DATA_DIR
    labeling_results_dir: str = getattr(config, 'LABELING_RESULTS_DIR', os.path.join(config.RESULT_DIR, "Labeling_Results"))
    nmf_model_path: str = config.NMF_MODEL_PATH
    ground_truth_path: str = os.path.join(config.DATA_DIR, "groundtruth", "abilities.csv")


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
        self.mitre_tfidf_matrix: Optional[scipy.sparse.csr_matrix] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        
        # NMF 模型
        self.nmf_model = None
        self._nmf_scaler = None
        
        # 結果
        self.labeling_results: Dict[str, pd.DataFrame] = {}

        # Ground Truth
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> Dict[str, Dict[str, str]]:
        """載入 Ground Truth"""
        if not os.path.exists(self.config.ground_truth_path):
            print(f"[Warning] Ground Truth 不存在: {self.config.ground_truth_path}")
            return {}
        try:
            df = pd.read_csv(self.config.ground_truth_path)
            # filename -> {tid, t_name}
            gt = {}
            for _, row in df.iterrows():
                fname = str(row['filename']).strip()
                gt[fname] = {
                    "tid": row['tid'],
                    "t_name": row['t_name']
                }
            return gt
        except Exception as e:
            print(f"[Error] 載入 Ground Truth 失敗: {e}")
            return {}
    
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
    
    def load_log_vectors_for_dataset(self, dataset_id: str) -> Optional[np.ndarray]:
        """載入原始日誌嵌入向量"""
        base_dir = config.LOG_VECTORS_DIR 
        
        # 尋找對應目錄
        target_dir = None
        candidates = [
            os.path.join(base_dir, f"{dataset_id}_raw_events_embeddings"),
        ]
        
        for cand in candidates:
            if os.path.exists(cand):
                target_dir = cand
                break
        
        if not target_dir and os.path.exists(base_dir):
            # 模糊比對
            for sub in os.listdir(base_dir):
                if sub.startswith(dataset_id) and "embeddings" in sub:
                     target_dir = os.path.join(base_dir, sub)
                     break
        
        if not target_dir:
            print(f"[Warning] 找不到嵌入目錄 for {dataset_id}")
            return None
            
        # 載入向量
        try:
            arrow_path = os.path.join(target_dir, "data-00000-of-00001.arrow")
            if os.path.exists(arrow_path):
                try:
                    table = feather.read_table(arrow_path)
                    for col in ["embeddings", "embedding", "vectors", "vector"]:
                        if col in table.column_names:
                            return np.array(table[col].to_pylist())
                except Exception:
                    # Fallback to datasets library
                    try:
                        from datasets import load_from_disk
                        ds = load_from_disk(target_dir)
                        if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
                             first_key = next(iter(ds.keys()))
                             ds = ds[first_key]
                        
                        for col in ["embeddings", "embedding", "vectors", "vector"]:
                            if col in ds.column_names:
                                return np.array(ds[col])
                    except ImportError:
                        print("[Error] 需要 'datasets' 套件來讀取此格式，但未安裝。")
                        return None
                    except Exception as e_ds:
                         print(f"[Error] load_from_disk 失敗: {e_ds}")
            
        except Exception as e:
            print(f"[Error] 載入嵌入向量失敗 {target_dir}: {e}")
            
        return None

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

            # Many of our external-source builders save HuggingFace datasets via `save_to_disk()`.
            # Those Arrow shards are NOT Feather; load them via `datasets.load_from_disk()`.
            try:
                df = feather.read_table(arrow_path).to_pandas()
                embed_col = next((c for c in ["embedding", "embeddings", "vector", "concept_vector"] if c in df.columns), None)
                if embed_col:
                    self.mitre_embeddings = np.array(df[embed_col].tolist())
                    self.mitre_technique_ids = (df.get("technique_id") or df.get("id", pd.Series(range(len(df))))).tolist()
                    self.mitre_technique_names = (df.get("technique") or df.get("name", self.mitre_technique_ids)).tolist()
            except (pyarrow.lib.ArrowInvalid, OSError):
                from datasets import load_from_disk

                ds = load_from_disk(embeddings_dir)
                # load_from_disk() can return Dataset or DatasetDict
                if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
                    first_key = next(iter(ds.keys()))
                    ds = ds[first_key]

                embed_col = next(
                    (c for c in ["embedding", "embeddings", "vector", "concept_vector"] if c in ds.column_names),
                    None,
                )
                if embed_col is None:
                    raise ValueError(
                        f"MITRE dataset missing embedding column. Found columns: {ds.column_names}"
                    )

                self.mitre_embeddings = np.array(ds[embed_col])
                if "technique_id" in ds.column_names:
                    self.mitre_technique_ids = list(ds["technique_id"])
                elif "id" in ds.column_names:
                    self.mitre_technique_ids = list(ds["id"])
                else:
                    self.mitre_technique_ids = [str(i) for i in range(len(ds))]

                if "technique" in ds.column_names:
                    self.mitre_technique_names = list(ds["technique"])
                elif "name" in ds.column_names:
                    self.mitre_technique_names = list(ds["name"])
                else:
                    self.mitre_technique_names = self.mitre_technique_ids
        
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
            
    def load_mitre_tfidf(self) -> None:
        """載入 MITRE TF-IDF 向量與模型"""
        if not self.config.use_tfidf:
            return

        tfidf_dir = self.config.mitre_tfidf_dir
        print(f"\n載入 MITRE TF-IDF 資料: {tfidf_dir}")
        
        vec_path = os.path.join(tfidf_dir, "tfidf_vectorizer.pkl")
        mat_path = os.path.join(tfidf_dir, "mitre_tfidf_matrix.pkl")
        
        if not os.path.exists(vec_path) or not os.path.exists(mat_path):
            print(f"[Warning] TF-IDF 檔案遺失，將跳過 TF-IDF 載入 ({tfidf_dir})")
            return
            
        with open(vec_path, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)
            
        with open(mat_path, "rb") as f:
            self.mitre_tfidf_matrix = pickle.load(f)
            
        print(f"已載入 TF-IDF Vectorizer 與 Matrix {self.mitre_tfidf_matrix.shape}")
    
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
    
    def load_log_tfidf(self, dataset_id: str) -> Optional[scipy.sparse.csr_matrix]:
        """載入預計算的日誌 TF-IDF 向量"""
        base_dir = self.config.log_vectors_dir 
        
        # 尋找候選路徑
        candidates = [
            os.path.join(base_dir, f"{dataset_id}_embeddings", "tfidf.npz"),
            os.path.join(base_dir, f"{dataset_id}_raw_events_embeddings", "tfidf.npz"),
            os.path.join(base_dir, dataset_id, "tfidf.npz")
        ]
        
        # 嘗試去除後綴
        cleaned_id = dataset_id
        for suffix in ["_embeddings", "_concepts", "_vectors"]:
            if cleaned_id.endswith(suffix):
                cleaned_id = cleaned_id[:-len(suffix)]
        
        candidates.append(os.path.join(base_dir, f"{cleaned_id}_embeddings", "tfidf.npz"))
        candidates.append(os.path.join(base_dir, f"{cleaned_id}_raw_events_embeddings", "tfidf.npz"))

        for p in candidates:
            if os.path.exists(p):
                try:
                    return scipy.sparse.load_npz(p)
                except Exception as e:
                    print(f"[Warning] 載入預計算 TF-IDF 失敗 {p}: {e}")
        return None

    def load_log_texts(self, dataset_id: str) -> List[str]:
        """載入資料集的原始日誌文字（用於 TF-IDF）"""
        # 嘗試去除常規後綴以匹配原始日誌檔名
        cleaned_id = dataset_id
        for suffix in ["_embeddings", "_concepts", "_vectors"]:
            if cleaned_id.endswith(suffix):
                cleaned_id = cleaned_id[:-len(suffix)]
                
        # 嘗試從 Intermediate_data 尋找檔案
        candidates = [
            f"{dataset_id}.csv",
            f"{cleaned_id}.csv",  # 嘗試去除後綴的 ID
            f"{dataset_id}_raw_events.csv",
            f"{cleaned_id}_raw_events.csv",
            f"syslogs_{dataset_id}_audit_log.csv",
            f"syslogs_{cleaned_id}_audit_log.csv"
        ]
        
        df = None
        # 1. 嘗試 Intermediate Data
        for cand in candidates:
            p = os.path.join(self.config.intermediate_data_dir, cand)
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    break
                except:
                    pass
                    
        # 2. 回退到 Input Logs
        if df is None:
            input_path = os.path.join(self.config.input_logs_dir, f"{dataset_id}.csv")
            if os.path.exists(input_path):
                try:
                    df = pd.read_csv(input_path)
                except:
                    pass

        if df is None:
             print(f"[Warning] 無法載入日誌文字: {dataset_id}")
             return []

        # 提取文字
        if "ConcatenatedLog" in df.columns:
            return df["ConcatenatedLog"].fillna("").astype(str).tolist()
        elif "Template" in df.columns and "Parameters" in df.columns:
            return (df["Template"].fillna("") + " " + df["Parameters"].fillna("")).astype(str).tolist()
        elif "Content" in df.columns:
            return df["Content"].fillna("").astype(str).tolist()
        elif "Event" in df.columns:
            return df["Event"].fillna("").astype(str).tolist()
        else:
             # 合併所有欄位
             return df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()

    def compute_tfidf_centroids(
        self,
        log_input: Union[List[str], scipy.sparse.csr_matrix],
        cluster_labels: np.ndarray,
        anomaly_scores: Optional[np.ndarray] = None
    ) -> Dict[int, scipy.sparse.csr_matrix]:
        """計算各 Cluster 的 TF-IDF Centroid"""
        if not self.tfidf_vectorizer:
            return {}
            
        unique_clusters = np.unique(cluster_labels)
        centroids = {}
        
        try:
            # 轉換所有日誌
            if isinstance(log_input, list):
                tfidf_matrix = self.tfidf_vectorizer.transform(log_input)
            else:
                tfidf_matrix = log_input
            
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                cluster_vectors = tfidf_matrix[mask]
                
                # 計算 Centroid
                if anomaly_scores is not None and len(anomaly_scores) == len(cluster_labels):
                    scores = anomaly_scores[mask]
                    weights = scores / (scores.sum() + 1e-8)
                    # 加權平均: weights (n,) @ vectors (n, d) -> (1, d)
                    centroid = scipy.sparse.csr_matrix(weights.reshape(1, -1) @ cluster_vectors)
                else:
                    centroid = cluster_vectors.mean(axis=0)
                    centroid = scipy.sparse.csr_matrix(centroid)
                    
                centroids[int(cluster_id)] = centroid
        except Exception as e:
            print(f"[Error] TF-IDF Centroid 計算失敗: {e}")
            
        return centroids

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
        tfidf_centroid: Optional[scipy.sparse.csr_matrix] = None
    ) -> Dict[str, Any]:
        """
        將單一 Cluster Centroid 與 MITRE 技術進行比對
        
        Args:
            centroid: Cluster 的 Centroid 向量
            avg_anomaly_score: Cluster 的平均異常分數
            
        Returns:
            Dict with matching results
        """
        # Determine which vectors to compare against based on config
        if self.config.use_raw_embeddings:
            if self.mitre_embeddings is None:
                raise ValueError("請先載入 MITRE 嵌入 (mitre_embeddings)")
            target_vectors = self.mitre_embeddings
        else:
            if self.mitre_concept_vectors is None:
                raise ValueError("請先執行 transform_mitre_to_concepts()")
            target_vectors = self.mitre_concept_vectors
        
        # 1. 計算嵌入向量相似度
        centroid_2d = centroid.reshape(1, -1)
        if centroid_2d.shape[1] != target_vectors.shape[1]:
             raise ValueError(f"維度不匹配: Centroid {centroid_2d.shape[1]} vs Target {target_vectors.shape[1]}")

        emb_similarities = cosine_similarity(centroid_2d, target_vectors)[0]
        
        # 2. 計算 TF-IDF 相似度 (若啟用)
        tfidf_similarities = None
        if self.config.use_tfidf and tfidf_centroid is not None and self.mitre_tfidf_matrix is not None:
            try:
                tfidf_similarities = cosine_similarity(tfidf_centroid, self.mitre_tfidf_matrix)[0]
            except Exception as e:
                print(f"[Warning] TF-IDF 相似度計算失敗: {e}")
        
        # 3. 融合分數
        if tfidf_similarities is not None:
            w_emb = self.config.weight_embedding
            w_tfidf = self.config.weight_tfidf
            total = w_emb + w_tfidf
            if total > 0:
                final_similarities = (w_emb * emb_similarities + w_tfidf * tfidf_similarities) / total
            else:
                final_similarities = emb_similarities
        else:
            final_similarities = emb_similarities

        # 使用融合後的相似度進行排名
        similarities = final_similarities
        
        # Collect all techniques that pass the score of top one * 0.95
        passing_indices = np.where(similarities >= (similarities.max() * 0.8))[0]
        passing_indices = passing_indices[np.argsort(similarities[passing_indices])[::-1]]

        # Also keep a Top-K view for quick inspection.
        top_k_indices = np.argsort(similarities)[-self.config.top_k_techniques:][::-1]
        
        # 最高相似度
        best_idx = int(np.argmax(similarities))
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
        if best_similarity < self.config.similarity_threshold or final_score < self.config.confidence_threshold:
            predicted_technique = "Benign"
        else:
            predicted_technique = best_technique_id
        
        return {
            "predicted_technique": predicted_technique,
            "technique_name": best_technique_name,  # Always show actual name
            "similarity_score": best_similarity,
            "emb_similarity": float(emb_similarities[best_idx]),
            "tfidf_similarity": float(tfidf_similarities[best_idx]) if tfidf_similarities is not None else 0.0,
            "anomaly_score": avg_anomaly_score,
            "confidence": confidence,
            "final_score": final_score,
            "passing_techniques": [
                {
                    "technique_id": self.mitre_technique_ids[idx],
                    "technique_name": self.mitre_technique_names[idx],
                    "similarity": float(similarities[idx]),
                    "emb_sim": float(emb_similarities[idx]),
                    "tfidf_sim": float(tfidf_similarities[idx]) if tfidf_similarities is not None else 0.0,
                }
                for idx in passing_indices
            ],
            "top_k_techniques": [
                {
                    "technique_id": self.mitre_technique_ids[idx],
                    "technique_name": self.mitre_technique_names[idx],
                    "similarity": float(similarities[idx]),
                    "emb_sim": float(emb_similarities[idx]),
                    "tfidf_sim": float(tfidf_similarities[idx]) if tfidf_similarities is not None else 0.0,
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
        
        # 準備向量資料 (Concept vs Raw)
        if self.config.use_raw_embeddings:
            print(f"    [Mode] 使用原始嵌入向量 (Raw Embeddings)")
            vectors = self.load_log_vectors_for_dataset(dataset_id)
            if vectors is None:
                raise ValueError(f"無法載入資料集 {dataset_id} 的原始嵌入")
        else:
            if dataset_id not in self.concept_vectors:
                raise ValueError(f"找不到資料集 {dataset_id} 的概念向量")
            vectors = self.concept_vectors[dataset_id]

        if dataset_id not in self.cluster_labels:
            raise ValueError(f"找不到資料集 {dataset_id} 的分群標籤")
        
        cluster_labels = self.cluster_labels[dataset_id]
        anomaly_scores = self.anomaly_scores.get(dataset_id)
        
        n_samples = len(vectors)
        print(f"    樣本數: {n_samples}")
        print(f"    群集數: {len(np.unique(cluster_labels))}")
        
        # 計算 Cluster Centroids
        centroids, avg_anomaly_scores = self.compute_cluster_centroids(
            dataset_id, vectors, cluster_labels, anomaly_scores
        )
        
        # 計算 TF-IDF Centroids
        tfidf_centroids = {}
        if self.config.use_tfidf:
            # 優先嘗試載入預計算的 TF-IDF 向量
            precomputed_tfidf = self.load_log_tfidf(dataset_id)
            if precomputed_tfidf is not None:
                print(f"    [Info] 使用預計算的 TF-IDF 向量 (shape={precomputed_tfidf.shape})...")
                if precomputed_tfidf.shape[0] == len(cluster_labels):
                    tfidf_centroids = self.compute_tfidf_centroids(
                        precomputed_tfidf, cluster_labels, anomaly_scores
                    )
                else:
                     print(f"    [Warning] 預計算 TF-IDF 數量 ({precomputed_tfidf.shape[0]}) 與標籤數量 ({len(cluster_labels)}) 不一致，嘗試回退至原始文本...")
                     precomputed_tfidf = None # Fallback

            if not tfidf_centroids and precomputed_tfidf is None:
                # Fallback to loading text
                log_texts = self.load_log_texts(dataset_id)
                if log_texts:
                    if len(log_texts) == len(cluster_labels):
                        print(f"    [Info] 計算 {len(log_texts)} 筆日誌的 TF-IDF Centroids...")
                        tfidf_centroids = self.compute_tfidf_centroids(
                            log_texts, cluster_labels, anomaly_scores
                        )
                    else:
                        print(f"    [Warning] 日誌文本數量 ({len(log_texts)}) 與標籤數量 ({len(cluster_labels)}) 不一致，略過 TF-IDF")
        
        # 對每個 Cluster 進行標註
        cluster_results = {}
        for cluster_id, centroid in centroids.items():
            result = self.match_cluster_to_technique(
                centroid, 
                avg_anomaly_scores[cluster_id],
                tfidf_centroid=tfidf_centroids.get(cluster_id)
            )
            cluster_results[cluster_id] = result
        
        # 尋找 Ground Truth
        gt_tid = None
        gt_name = None
        
        # 嘗試匹配 dataset_id
        candidates = [dataset_id]
        
        # 去除常見後綴
        cleaned_id = dataset_id
        for suffix in ["_embeddings", "_concepts", "_vectors", "_raw_events"]:
             if cleaned_id.endswith(suffix):
                 cleaned_id = cleaned_id[:-len(suffix)]
        candidates.append(cleaned_id)
        
        
        for cand in candidates:
             if cand in self.ground_truth:
                 gt_tid = self.ground_truth[cand]["tid"]
                 gt_name = self.ground_truth[cand]["t_name"]
                 break

        # 將標註結果映射回每個樣本
        sample_results = []
        for i in range(n_samples):
            cluster_id = int(cluster_labels[i])
            result = cluster_results[cluster_id]

            top_k = result.get("top_k_techniques") or []
            top_k_str = "; ".join(
                f"{c['technique_id']} {c['technique_name']} ({c['similarity']:.4f} [E:{c.get('emb_sim',0):.2f} T:{c.get('tfidf_sim',0):.2f}])"
                for c in top_k
            )
            
            sample_result = {
                "log_index": i,
                "cluster_id": cluster_id,
                "ground_truth_tid": gt_tid,
                "ground_truth_name": gt_name,
                "predicted_technique": result["predicted_technique"],
                "technique_name": result["technique_name"],
                "similarity_score": result["similarity_score"],
                "emb_similarity": result.get("emb_similarity", 0.0),
                "tfidf_similarity": result.get("tfidf_similarity", 0.0),
                "anomaly_score": anomaly_scores[i] if anomaly_scores is not None else result["anomaly_score"],
                "confidence": result["confidence"],
                "passing_techniques": top_k_str,  # Replaced with Top-K instead of Passing
                "passing_techniques_count": len(top_k),
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
    
    def process_single_dataset(
        self,
        dataset_id: str,
        concept_vectors: np.ndarray,
        cluster_labels: np.ndarray,
        output_dir: Optional[str] = None,
        nmf_extractor=None,
    ) -> Optional[Dict[str, Any]]:
        """
        對單一 Dataset 執行自動標註（Per-Dataset API）
        
        此方法與 ConceptExtractor.process_single_dataset() 和 
        SequenceClustering.process_single_dataset() 保持一致的 API 風格，
        方便在 Pipeline 中統一調用。
        
        核心邏輯：
        1. 如果有 NMF extractor，將 MITRE 嵌入投影至相同概念空間
        2. 計算 Cluster Centroid 與投影後 MITRE 向量的餘弦相似度
        3. 根據相似度為每個 Cluster 標註最匹配的 MITRE 技術
        
        Args:
            dataset_id: Dataset 識別碼
            concept_vectors: 概念向量矩陣 (N, n_concepts)
            cluster_labels: HMM 分群標籤
            output_dir: 輸出目錄
            nmf_extractor: ConceptExtractor 物件（用於投影 MITRE 嵌入）
            
        Returns:
            標註結果字典，包含 'labels' 和 'output_path'
        """
        output_dir = output_dir or self.config.labeling_results_dir
        
        if self.mitre_embeddings is None:
            print(f"    [Warning] MITRE 嵌入未載入，跳過標註")
            return None
        
        try:
            # 計算每個 Cluster 的 Centroid
            unique_clusters = np.unique(cluster_labels)
            cluster_centroids = {}
            
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                cluster_vectors = concept_vectors[mask]
                centroid = np.mean(cluster_vectors, axis=0)
                cluster_centroids[cluster_id] = centroid
            
            centroid_matrix = np.array([cluster_centroids[c] for c in unique_clusters])
            
            # 判斷是否需要 NMF 投影
            mitre_dim = self.mitre_embeddings.shape[1]
            concept_dim = centroid_matrix.shape[1]
            
            if nmf_extractor is not None and hasattr(nmf_extractor, '_is_fitted') and nmf_extractor._is_fitted:
                # 使用 extractor 投影 MITRE 嵌入
                nmf_input_dim = nmf_extractor.model.components_.shape[1]
                
                if mitre_dim == nmf_input_dim:
                    print(f"    [NMF 投影] 將 MITRE 嵌入投影至概念空間...")
                    mitre_projected = nmf_extractor.transform_local(self.mitre_embeddings)
                    similarities = cosine_similarity(centroid_matrix, mitre_projected)
                else:
                    print(f"    [Warning] NMF 輸入維度 ({nmf_input_dim}) 與 MITRE 維度 ({mitre_dim}) 不符")
                    return self._generate_placeholder_result(dataset_id, cluster_labels, output_dir)
            elif mitre_dim != concept_dim:
                # 無 NMF 模型且維度不符
                print(f"    [Info] 維度不符 (MITRE={mitre_dim}, Concept={concept_dim})，無 NMF 模型，使用簡化標註")
                return self._generate_placeholder_result(dataset_id, cluster_labels, output_dir)
            else:
                # 維度匹配，直接計算相似度
                similarities = cosine_similarity(centroid_matrix, self.mitre_embeddings)
            
            # 為每個 cluster 找最佳匹配
            cluster_to_technique = {}
            for i, cluster_id in enumerate(unique_clusters):
                best_idx = np.argmax(similarities[i])
                best_sim = similarities[i, best_idx]
                technique_id = self.mitre_technique_ids[best_idx] if self.mitre_technique_ids else f"T{best_idx}"
                technique_name = self.mitre_technique_names[best_idx] if self.mitre_technique_names else "Unknown"
                cluster_to_technique[cluster_id] = {
                    "technique_id": technique_id,
                    "technique_name": technique_name,
                    "similarity": float(best_sim),
                }
            
            # 顯示 Top-3 匹配技術
            print(f"    [Top-3 匹配技術]")
            for cid in list(unique_clusters)[:3]:
                tech = cluster_to_technique[cid]
                name_display = tech['technique_name'][:30] + "..." if len(tech['technique_name']) > 30 else tech['technique_name']
                print(f"      Cluster {cid}: {tech['technique_id'][:30]}... ({name_display}) sim={tech['similarity']:.3f}")
            
            # 生成每筆日誌的標註
            labeling_results = []
            for log_idx in range(len(cluster_labels)):
                cluster_id = cluster_labels[log_idx]
                tech_info = cluster_to_technique[cluster_id]
                labeling_results.append({
                    "log_index": log_idx,
                    "cluster_id": int(cluster_id),
                    "technique_id": tech_info["technique_id"],
                    "technique_name": tech_info["technique_name"],
                    "confidence": tech_info["similarity"],
                })
            
            # 儲存結果
            ensure_dir(output_dir)
            output_path = os.path.join(output_dir, f"{dataset_id}_labels.csv")
            df = pd.DataFrame(labeling_results)
            df.to_csv(output_path, index=False)
            print(f"    標註結果已存至 {output_path}")
            
            return {"labels": labeling_results, "output_path": output_path}
            
        except Exception as e:
            print(f"    [Error] 標註失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_placeholder_result(
        self,
        dataset_id: str,
        cluster_labels: np.ndarray,
        output_dir: str,
    ) -> Dict[str, Any]:
        """生成佔位符標註結果（維度不符時使用）"""
        labels = [
            {
                "log_index": i,
                "cluster_id": int(cluster_labels[i]),
                "technique_id": "TBD",
                "technique_name": "待人工標註",
                "confidence": 0.0,
            }
            for i in range(len(cluster_labels))
        ]
        
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, f"{dataset_id}_labels.csv")
        df = pd.DataFrame(labels)
        df.to_csv(output_path, index=False)
        print(f"    [Placeholder] 標註結果已存至 {output_path}")
        
        return {"labels": labels, "output_path": output_path}
    
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
            if self.config.use_raw_embeddings:
                # 原始嵌入模式：只要有 Cluster Labels 就嘗試標註 (Embeddings 會在 label_dataset 中動態載入)
                dataset_ids = list(self.cluster_labels.keys())
            else:
                # 概念向量模式：需要同時有 Concept Vectors 和 Cluster Labels
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
    
    if not labeler.config.use_raw_embeddings:
        try:
            labeler.load_concept_vectors(dataset_ids)
        except Exception as e:
            print(f"[Warning] 概念向量載入失敗: {e}. 若使用 Raw Embeddings 可忽略。")

    labeler.load_cluster_labels(dataset_ids)
    labeler.load_anomaly_scores(dataset_ids)

    resolved_mitre_dir = mitre_embeddings_dir or labeler.config.mitre_embeddings_dir
    ensure_mitre_raw_embeddings(
        resolved_mitre_dir,
        mitre_csv=getattr(config, "MITRE_TECHNIQUES_CSV", None),
        bert_model=getattr(config, "BERT_MODEL_NAME", None),
        force_rebuild=False,
    )
    labeler.load_mitre_embeddings(resolved_mitre_dir)
    labeler.load_mitre_tfidf()
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
    
    支援以下格式：
    1. HuggingFace Datasets (Arrow 格式) - 優先
    2. NumPy .npy 檔案 - 備選
    """
    from datasets import load_from_disk
    
    if not os.path.exists(detection_results_dir):
        print(f"[Warning] 找不到異常偵測結果目錄: {detection_results_dir}")
        return {}
    
    # 嘗試載入整合結果 (舊格式)
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
        
        # 方法 1：嘗試載入 Arrow 格式 (HuggingFace Datasets)
        state_json = os.path.join(subdir_path, "state.json")
        if os.path.exists(state_json):
            try:
                ds = load_from_disk(subdir_path)
                # 優先使用 ensemble 分數，否則找任何 score 欄位
                score_col = None
                for col in ["ensemble", "ensemble_score", "ensemble_raw"]:
                    if col in ds.column_names:
                        score_col = col
                        break
                if score_col is None:
                    for col in ds.column_names:
                        if "score" in col.lower() or "anomaly" in col.lower():
                            score_col = col
                            break
                
                if score_col:
                    weights[dataset_id] = np.array(ds[score_col])
                    continue
            except Exception as e:
                print(f"[Warning] 載入 Arrow 格式失敗 {subdir_path}: {e}")
        
        # 方法 2：嘗試載入 .npy 格式
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
