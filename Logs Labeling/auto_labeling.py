"""
AutoLabeling：自動標註模組
# * 根據序列分群結果與 MITRE ATT&CK 外部知識進行自動標註
# * 支援 TF-IDF 混合評分以增強外部資料引用的有效性
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pyarrow.feather as feather
import pyarrow

CURRENT_DIR = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(CURRENT_DIR).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from utils.path import ensure_dir


@dataclass
class LabelingConfig:
    top_k_techniques: int = getattr(config, 'LABELING_TOP_K', 3)
    similarity_threshold: float = getattr(config, 'LABELING_SIMILARITY_THRESHOLD', 0.3)
    confidence_threshold: float = getattr(config, 'LABELING_CONFIDENCE_THRESHOLD', 0.2)
    anomaly_weight: float = getattr(config, 'LABELING_ANOMALY_WEIGHT', 0.3)
    similarity_weight: float = getattr(config, 'LABELING_SIMILARITY_WEIGHT', 0.7)
    
    # * TF-IDF 混合評分設定
    use_tfidf: bool = getattr(config, 'LABELING_USE_TFIDF', True)
    weight_embedding: float = getattr(config, 'LABELING_WEIGHT_EMBEDDING', 0.7)
    weight_tfidf: float = getattr(config, 'LABELING_WEIGHT_TFIDF', 0.3)
    mitre_tfidf_dir: str = getattr(config, 'MITRE_TFIDF_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_TFIDF"))
    
    mitre_embeddings_dir: str = getattr(config, 'MITRE_EXTERNAL_KNOWLEDGE_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_ATTACK"))
    input_logs_dir: str = config.INPUT_LOGS_DIR
    log_vectors_dir: str = config.LOG_VECTORS_DIR
    intermediate_data_dir: str = config.INTERMEDIATE_DATA_DIR
    labeling_results_dir: str = getattr(config, 'LABELING_RESULTS_DIR', os.path.join(config.RESULT_DIR, "Labeling_Results"))
    detection_results_dir: str = getattr(config, 'DETECTION_RESULTS_DIR', os.path.join(config.DATA_DIR, "Detection_Results"))


class AutoLabeler:
    # * 自動標註器：結合嵌入向量與 TF-IDF 混合評分
    
    def __init__(self, labeling_config: Optional[LabelingConfig] = None):
        self.config = labeling_config or LabelingConfig()
        
        self.mitre_embeddings: Optional[np.ndarray] = None
        self.mitre_technique_ids: Optional[List[str]] = None
        self.mitre_technique_names: Optional[List[str]] = None
        self.mitre_concept_vectors: Optional[np.ndarray] = None
        
        # * TF-IDF 資料
        self.mitre_tfidf_matrix: Optional[scipy.sparse.csr_matrix] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        
        self.nmf_model = None
        self._nmf_scaler = None
        self.labeling_results: Dict[str, pd.DataFrame] = {}

    def load_mitre_embeddings(self, embeddings_dir: Optional[str] = None) -> None:
        # * 載入 MITRE ATT&CK 嵌入向量
        embeddings_dir = embeddings_dir or self.config.mitre_embeddings_dir
        print(f"\n載入 MITRE 嵌入向量: {embeddings_dir}")
        
        for fname in ["embeddings.npy", "mitre_embeddings.npy", "technique_embeddings.npy"]:
            path = os.path.join(embeddings_dir, fname)
            if os.path.exists(path):
                self.mitre_embeddings = np.load(path)
                self._load_mitre_metadata(embeddings_dir)
                print(f"已載入 {len(self.mitre_embeddings)} 個 MITRE 技術嵌入")
                return
        
        arrow_path = os.path.join(embeddings_dir, "data-00000-of-00001.arrow")
        if not os.path.exists(arrow_path):
            raise FileNotFoundError(f"找不到 MITRE 嵌入檔案於: {embeddings_dir}")

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
            if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
                ds = ds[next(iter(ds.keys()))]

            embed_col = next((c for c in ["embedding", "embeddings", "vector", "concept_vector"] if c in ds.column_names), None)
            if embed_col is None:
                raise ValueError(f"MITRE dataset missing embedding column. Found: {ds.column_names}")

            self.mitre_embeddings = np.array(ds[embed_col])
            
            # * 正確處理 HuggingFace Dataset 的欄位存取
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

    def _load_mitre_metadata(self, embeddings_dir: str) -> None:
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
        # * 載入 MITRE TF-IDF 向量與模型（用於混合評分）
        if not self.config.use_tfidf:
            return

        tfidf_dir = self.config.mitre_tfidf_dir
        print(f"\n載入 MITRE TF-IDF 資料: {tfidf_dir}")
        
        vec_path = os.path.join(tfidf_dir, "tfidf_vectorizer.pkl")
        mat_path = os.path.join(tfidf_dir, "mitre_tfidf_matrix.pkl")
        
        if not os.path.exists(vec_path) or not os.path.exists(mat_path):
            print(f"[Warning] TF-IDF 檔案遺失，跳過 TF-IDF 載入")
            self.config.use_tfidf = False
            return
            
        with open(vec_path, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open(mat_path, "rb") as f:
            self.mitre_tfidf_matrix = pickle.load(f)
            
        print(f"已載入 TF-IDF Vectorizer 與 Matrix {self.mitre_tfidf_matrix.shape}")

    def _compute_tfidf_similarity(self, log_texts: List[str], cluster_labels: np.ndarray) -> Optional[np.ndarray]:
        # * 計算日誌文本與 MITRE 技術的 TF-IDF 相似度
        if not self.config.use_tfidf or self.tfidf_vectorizer is None or self.mitre_tfidf_matrix is None:
            return None
        
        try:
            unique_clusters = np.unique(cluster_labels)
            cluster_texts = []
            for cid in unique_clusters:
                mask = cluster_labels == cid
                cluster_text = " ".join([log_texts[i] for i in range(len(log_texts)) if mask[i]])
                cluster_texts.append(cluster_text)
            
            cluster_tfidf = self.tfidf_vectorizer.transform(cluster_texts)
            tfidf_similarities = cosine_similarity(cluster_tfidf, self.mitre_tfidf_matrix)
            return tfidf_similarities
        except Exception as e:
            print(f"    [Warning] TF-IDF 相似度計算失敗: {e}")
            return None

    def _load_log_texts(self, dataset_id: str) -> Optional[List[str]]:
        # * 載入原始日誌文本（用於 TF-IDF 計算）
        candidates = [
            os.path.join(self.config.intermediate_data_dir, f"{dataset_id}.csv"),
            os.path.join(self.config.input_logs_dir, f"{dataset_id}.csv"),
        ]
        
        for path in candidates:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    for col in ["ConcatenatedLog", "Template", "Content", "Event"]:
                        if col in df.columns:
                            return df[col].fillna("").astype(str).tolist()
                    return df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
                except Exception:
                    continue
        return None

    def process_single_dataset(
        self,
        dataset_id: str,
        concept_vectors: np.ndarray,
        cluster_labels: np.ndarray,
        output_dir: Optional[str] = None,
        nmf_extractor=None,
    ) -> Optional[Dict[str, Any]]:
        # * 對單一 Dataset 執行自動標註（Per-Dataset API）
        # * 支援嵌入向量 + TF-IDF 混合評分
        output_dir = output_dir or self.config.labeling_results_dir
        
        if self.mitre_embeddings is None:
            print(f"    [Warning] MITRE 嵌入未載入，跳過標註")
            return None
        
        try:
            unique_clusters = np.unique(cluster_labels)
            cluster_centroids = {}
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                cluster_centroids[cluster_id] = np.mean(concept_vectors[mask], axis=0)
            
            centroid_matrix = np.array([cluster_centroids[c] for c in unique_clusters])
            
            # * 計算嵌入向量相似度
            mitre_dim = self.mitre_embeddings.shape[1]
            concept_dim = centroid_matrix.shape[1]
            
            if nmf_extractor is not None and hasattr(nmf_extractor, '_is_fitted') and nmf_extractor._is_fitted:
                nmf_input_dim = nmf_extractor.model.components_.shape[1]
                if mitre_dim == nmf_input_dim:
                    print(f"    [NMF 投影] 將 MITRE 嵌入投影至概念空間...")
                    mitre_projected = nmf_extractor.transform_local(self.mitre_embeddings)
                    embedding_similarities = cosine_similarity(centroid_matrix, mitre_projected)
                else:
                    print(f"    [Warning] NMF 維度不符，使用簡化標註")
                    return self._generate_placeholder_result(dataset_id, cluster_labels, output_dir)
            elif mitre_dim != concept_dim:
                print(f"    [Info] 維度不符，使用簡化標註")
                return self._generate_placeholder_result(dataset_id, cluster_labels, output_dir)
            else:
                embedding_similarities = cosine_similarity(centroid_matrix, self.mitre_embeddings)
            
            # * 計算 TF-IDF 相似度並混合評分
            final_similarities = embedding_similarities
            log_texts = self._load_log_texts(dataset_id)
            
            if self.config.use_tfidf and log_texts is not None:
                tfidf_similarities = self._compute_tfidf_similarity(log_texts, cluster_labels)
                if tfidf_similarities is not None:
                    print(f"    [混合評分] Embedding={self.config.weight_embedding:.1f}, TF-IDF={self.config.weight_tfidf:.1f}")
                    final_similarities = (
                        self.config.weight_embedding * embedding_similarities +
                        self.config.weight_tfidf * tfidf_similarities
                    )
            
            # * 為每個 cluster 找最佳匹配
            cluster_to_technique = {}
            for i, cluster_id in enumerate(unique_clusters):
                best_idx = np.argmax(final_similarities[i])
                best_sim = final_similarities[i, best_idx]
                cluster_to_technique[cluster_id] = {
                    "technique_id": self.mitre_technique_ids[best_idx] if self.mitre_technique_ids else f"T{best_idx}",
                    "technique_name": self.mitre_technique_names[best_idx] if self.mitre_technique_names else "Unknown",
                    "similarity": float(best_sim),
                }
            
            print(f"    [Top-3 匹配技術]")
            for cid in list(unique_clusters)[:3]:
                tech = cluster_to_technique[cid]
                name_display = tech['technique_name'][:30] + "..." if len(tech['technique_name']) > 30 else tech['technique_name']
                print(f"      Cluster {cid}: {tech['technique_id'][:30]}... ({name_display}) sim={tech['similarity']:.3f}")
            
            # * 生成標註結果
            labeling_results = [
                {
                    "log_index": log_idx,
                    "cluster_id": int(cluster_labels[log_idx]),
                    "technique_id": cluster_to_technique[cluster_labels[log_idx]]["technique_id"],
                    "technique_name": cluster_to_technique[cluster_labels[log_idx]]["technique_name"],
                    "confidence": cluster_to_technique[cluster_labels[log_idx]]["similarity"],
                }
                for log_idx in range(len(cluster_labels))
            ]
            
            ensure_dir(output_dir)
            output_path = os.path.join(output_dir, f"{dataset_id}_labels.csv")
            pd.DataFrame(labeling_results).to_csv(output_path, index=False)
            print(f"    標註結果已存至 {output_path}")
            
            return {"labels": labeling_results, "output_path": output_path}
            
        except Exception as e:
            print(f"    [Error] 標註失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_placeholder_result(self, dataset_id: str, cluster_labels: np.ndarray, output_dir: str) -> Dict[str, Any]:
        labels = [
            {"log_index": i, "cluster_id": int(cluster_labels[i]), "technique_id": "TBD", "technique_name": "待人工標註", "confidence": 0.0}
            for i in range(len(cluster_labels))
        ]
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, f"{dataset_id}_labels.csv")
        pd.DataFrame(labels).to_csv(output_path, index=False)
        print(f"    [Placeholder] 標註結果已存至 {output_path}")
        return {"labels": labels, "output_path": output_path}


def load_anomaly_weights(detection_results_dir: str = config.DETECTION_RESULTS_DIR) -> Dict[str, np.ndarray]:
    # * 載入異常偵測分數作為後續標註的權重
    from datasets import load_from_disk
    
    if not os.path.exists(detection_results_dir):
        print(f"[Warning] 找不到異常偵測結果目錄: {detection_results_dir}")
        return {}
    
    ensemble_path = os.path.join(detection_results_dir, "ensemble_scores.npy")
    if os.path.exists(ensemble_path):
        try:
            data = np.load(ensemble_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item()
            if isinstance(data, dict):
                print(f"[Info] 已載入 {len(data)} 個資料集的異常分數權重")
                return data
        except Exception as e:
            print(f"[Warning] 載入整合分數失敗: {e}")
    
    weights = {}
    for subdir in os.listdir(detection_results_dir):
        subdir_path = os.path.join(detection_results_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        dataset_id = subdir.replace("_detection", "").replace("_embeddings", "")
        
        state_json = os.path.join(subdir_path, "state.json")
        if os.path.exists(state_json):
            try:
                ds = load_from_disk(subdir_path)
                score_col = next((c for c in ["ensemble", "ensemble_score", "ensemble_raw"] if c in ds.column_names), None)
                if score_col is None:
                    score_col = next((c for c in ds.column_names if "score" in c.lower() or "anomaly" in c.lower()), None)
                if score_col:
                    weights[dataset_id] = np.array(ds[score_col])
                    continue
            except Exception:
                pass
        
        for fname in ["ensemble_scores.npy", "scores.npy", "anomaly_scores.npy"]:
            path = os.path.join(subdir_path, fname)
            if os.path.exists(path):
                try:
                    weights[dataset_id] = np.load(path)
                    break
                except Exception:
                    pass
    
    print(f"[Info] 已載入 {len(weights)} 個資料集的異常分數權重")
    return weights


if __name__ == "__main__":
    print("=" * 60)
    print("自動標註 - MITRE ATT&CK 技術比對")
    print("=" * 60)
    
    labeler = AutoLabeler()
    labeler.load_mitre_embeddings()
    labeler.load_mitre_tfidf()
    print("\n[完成] 自動標註模組初始化完成。")
