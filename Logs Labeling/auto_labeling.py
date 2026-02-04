"""
AutoLabeling：自動標註模組
# * 根據序列分群結果與 MITRE ATT&CK 外部知識進行自動標註
# * 支援 TF-IDF 混合評分以增強外部資料引用的有效性
# * 結合 Anomaly Score (Stage II) 計算最終 Threat Confidence

# * 公式：
# * 1. Similarity Score = (w_emb × Embedding_Sim) + (w_tfidf × TF-IDF_Sim) + DualBoost
# * 2. Threat Confidence = (α × Similarity_Score) + (β × Anomaly_Score)
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
    
    # * Threat Confidence 權重：confidence = α × similarity + β × anomaly
    # * α (similarity_weight) + β (anomaly_weight) 應等於 1.0
    similarity_weight: float = getattr(config, 'LABELING_SIMILARITY_WEIGHT', 0.7)  # α
    anomaly_weight: float = getattr(config, 'LABELING_ANOMALY_WEIGHT', 0.3)        # β
    
    # * Similarity Score 混合評分設定
    # * similarity = w_emb × embedding_sim + w_tfidf × tfidf_sim + dual_boost
    use_tfidf: bool = getattr(config, 'LABELING_USE_TFIDF', True)
    weight_embedding: float = getattr(config, 'LABELING_WEIGHT_EMBEDDING', 0.6)   # w_emb
    weight_tfidf: float = getattr(config, 'LABELING_WEIGHT_TFIDF', 0.3)           # w_tfidf
    
    # * 雙高加分 (Embedding + TF-IDF 同時高時的額外加分)
    enable_dual_boost: bool = getattr(config, 'LABELING_ENABLE_DUAL_BOOST', True)
    dual_boost_threshold: float = getattr(config, 'LABELING_DUAL_BOOST_THRESHOLD', 0.5)
    dual_boost_weight: float = getattr(config, 'LABELING_DUAL_BOOST_WEIGHT', 0.1)
    
    mitre_tfidf_dir: str = getattr(config, 'MITRE_TFIDF_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_TFIDF"))
    
    mitre_embeddings_dir: str = getattr(config, 'MITRE_EXTERNAL_KNOWLEDGE_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_ATTACK"))
    input_logs_dir: str = config.INPUT_LOGS_DIR
    log_vectors_dir: str = config.LOG_VECTORS_DIR
    intermediate_data_dir: str = config.INTERMEDIATE_DATA_DIR
    labeling_results_dir: str = getattr(config, 'LABELING_RESULTS_DIR', os.path.join(config.RESULT_DIR, "Labeling_Results"))
    detection_results_dir: str = getattr(config, 'DETECTION_RESULTS_DIR', os.path.join(config.DATA_DIR, "Detection_Results"))


class AutoLabeler:
    """
    自動標註器：結合嵌入向量、TF-IDF 混合評分與異常分數
    
    Threat Confidence = α × Similarity_Score + β × Anomaly_Score
    其中：
    - Similarity_Score = w_emb × Embedding_Sim + w_tfidf × TF-IDF_Sim + DualBoost
    - Anomaly_Score = Stage II 異常偵測結果（0~1，代表惡意可能性）
    """
    
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

        # * Ground Truth Data
        self.ground_truth: Dict[str, Dict[str, str]] = {}
        self._load_ground_truth()

    def _load_ground_truth(self) -> None:
        gt_path = os.path.join(PROJECT_ROOT, "data", "groundtruth", "abilities.csv")
        if os.path.exists(gt_path):
            try:
                df = pd.read_csv(gt_path)
                # filename, tid, t_name
                if 'filename' in df.columns:
                    df['filename'] = df['filename'].astype(str)
                    for _, row in df.iterrows():
                        key = row['filename'].strip()
                        self.ground_truth[key] = {
                            "tid": str(row['tid']),
                            "t_name": str(row['t_name'])
                        }
                    print(f"[Info] 已載入 Ground Truth: {len(self.ground_truth)} 筆")
            except Exception as e:
                print(f"[Warning] 載入 Ground Truth 失敗: {e}")
        else:
            print(f"[Warning] 找不到 Ground Truth 檔案: {gt_path}")

    def load_log_vectors(self, input_path: str) -> Optional[np.ndarray]:
        # * 載入原始 BERT 嵌入向量（用於計算 cluster centroids）
        # * 支援 Arrow、npy、HuggingFace Dataset 格式
        
        # 嘗試 Arrow 格式
        arrow_path = os.path.join(input_path, "data-00000-of-00001.arrow")
        if os.path.exists(arrow_path):
            try:
                df = feather.read_table(arrow_path).to_pandas()
                for col in ["embedding", "embeddings", "vector", "log_vector"]:
                    if col in df.columns:
                        return np.array(df[col].tolist())
            except Exception:
                pass
        
        # 嘗試 npy 格式
        for fname in ["embeddings.npy", "log_vectors.npy", "vectors.npy"]:
            npy_path = os.path.join(input_path, fname)
            if os.path.exists(npy_path):
                try:
                    return np.load(npy_path)
                except Exception:
                    pass
        
        # 嘗試 HuggingFace Dataset
        try:
            from datasets import load_from_disk
            ds = load_from_disk(input_path)
            if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
                ds = ds[next(iter(ds.keys()))]
            for col in ["embedding", "embeddings", "vector", "log_vector"]:
                if col in ds.column_names:
                    return np.array(ds[col])
        except Exception:
            pass
        
        return None

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

    def _compute_sequence_tfidf(self, log_texts: List[str], cluster_labels: np.ndarray) -> tuple:
        """
        計算 HMM Sequence (Cluster) 的 TF-IDF 向量
        
        Returns:
            (cluster_tfidf_matrix, tfidf_similarities, unique_clusters)
            - cluster_tfidf_matrix: 各 cluster 聚合的 TF-IDF 向量
            - tfidf_similarities: cluster 與 MITRE techniques 的 TF-IDF 相似度
            - unique_clusters: cluster IDs 順序
        """
        if not self.config.use_tfidf or self.tfidf_vectorizer is None or self.mitre_tfidf_matrix is None:
            return None, None, None
        
        try:
            unique_clusters = np.unique(cluster_labels)
            cluster_texts = []
            
            # 聚合每個 cluster 內所有 log 的文本
            for cid in unique_clusters:
                mask = cluster_labels == cid
                cluster_text = " ".join([log_texts[i] for i in range(len(log_texts)) if mask[i]])
                cluster_texts.append(cluster_text)
            
            # 使用 Reference Vectorizer 轉換 Sequence 文本
            cluster_tfidf = self.tfidf_vectorizer.transform(cluster_texts)
            
            # 計算與 MITRE Technique 指紋的相似度
            tfidf_similarities = cosine_similarity(cluster_tfidf, self.mitre_tfidf_matrix)
            
            return cluster_tfidf, tfidf_similarities, unique_clusters
        except Exception as e:
            print(f"    [Warning] Sequence TF-IDF 計算失敗: {e}")
            return None, None, None
    
    def _compute_hybrid_score(
        self,
        embedding_sim: np.ndarray,
        tfidf_sim: np.ndarray
    ) -> np.ndarray:
        """
        計算混合評分：Embedding + TF-IDF + 雙高加分
        
        邏輯：
        1. 基礎分數 = w_emb * embedding_sim + w_tfidf * tfidf_sim
        2. 若 embedding_sim 與 tfidf_sim 同時高於閾值，額外加分
        
        Args:
            embedding_sim: [n_clusters, n_techniques] embedding 相似度
            tfidf_sim: [n_clusters, n_techniques] TF-IDF 相似度
        
        Returns:
            hybrid_scores: [n_clusters, n_techniques] 最終混合分數
        """
        w_emb = self.config.weight_embedding
        w_tfidf = self.config.weight_tfidf
        
        # 基礎混合分數
        base_score = w_emb * embedding_sim + w_tfidf * tfidf_sim
        
        # 雙高加分
        if self.config.enable_dual_boost:
            threshold = self.config.dual_boost_threshold
            boost_weight = self.config.dual_boost_weight
            
            # 識別雙高情況：embedding 和 TF-IDF 同時高於閾值
            dual_high_mask = (embedding_sim >= threshold) & (tfidf_sim >= threshold)
            
            # 加分 = boost_weight * min(embedding_sim, tfidf_sim)，僅對雙高位置
            boost_score = np.where(
                dual_high_mask,
                boost_weight * np.minimum(embedding_sim, tfidf_sim),
                0.0
            )
            
            final_score = base_score + boost_score
        else:
            final_score = base_score
        
        return final_score

    def _compute_threat_confidence(
        self,
        similarity_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        cluster_labels: np.ndarray,
        unique_clusters: np.ndarray
    ) -> np.ndarray:
        """
        計算最終 Threat Confidence：結合 Similarity Score 與 Anomaly Score
        
        公式：Threat Confidence = α × Similarity_Score + β × Anomaly_Score
        
        Args:
            similarity_scores: [n_clusters, n_techniques] 每個 cluster 對各 technique 的相似度
            anomaly_scores: [n_logs] 每筆 log 的異常分數 (0~1)
            cluster_labels: [n_logs] 每筆 log 所屬的 cluster ID
            unique_clusters: [n_clusters] cluster IDs 順序（對應 similarity_scores 的行）
        
        Returns:
            threat_confidence: [n_clusters, n_techniques] 最終威脅信心度
        """
        alpha = self.config.similarity_weight  # α
        beta = self.config.anomaly_weight      # β
        
        # 計算每個 cluster 的平均異常分數
        cluster_anomaly = np.zeros(len(unique_clusters))
        for i, cid in enumerate(unique_clusters):
            mask = cluster_labels == cid
            if np.sum(mask) > 0:
                cluster_anomaly[i] = np.mean(anomaly_scores[mask])
        
        # 擴展為 [n_clusters, n_techniques] 維度
        n_techniques = similarity_scores.shape[1]
        anomaly_matrix = np.tile(cluster_anomaly.reshape(-1, 1), (1, n_techniques))
        
        # 計算 Threat Confidence
        threat_confidence = alpha * similarity_scores + beta * anomaly_matrix
        
        return threat_confidence

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
        log_vectors: Optional[np.ndarray] = None,
        log_vectors_path: Optional[str] = None,
        anomaly_scores: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        對單一 Dataset 執行自動標註（Per-Dataset API）
        
        評分機制：
        1. Similarity Score = Embedding × w_emb + TF-IDF × w_tfidf + DualBoost
        2. Threat Confidence = α × Similarity_Score + β × Anomaly_Score
        
        Args:
            dataset_id: 資料集 ID
            concept_vectors: NMF 概念向量 [n_logs, n_concepts]
            cluster_labels: HMM 分群標籤 [n_logs]
            output_dir: 輸出目錄
            nmf_extractor: NMF 提取器（用於投影 MITRE 嵌入）
            log_vectors: 原始 BERT 嵌入 [n_logs, 768]
            log_vectors_path: log_vectors 路徑
            anomaly_scores: Stage II 異常分數 [n_logs]，範圍 0~1
        
        Returns:
            標註結果 dict 或 None
        """
        output_dir = output_dir or self.config.labeling_results_dir
        top_k = self.config.top_k_techniques
        
        if self.mitre_embeddings is None:
            print(f"    [Warning] MITRE 嵌入未載入，跳過標註")
            return None
        
        try:
            # * 載入原始資料集
            original_df = self._load_original_dataset(dataset_id)
            
            # * 若未提供 log_vectors，嘗試從 log_vectors_path 載入
            if log_vectors is None and log_vectors_path is not None:
                log_vectors = self.load_log_vectors(log_vectors_path)
            
            # * 若未提供 anomaly_scores，嘗試載入
            if anomaly_scores is None:
                anomaly_scores = self._load_anomaly_scores(dataset_id)
            
            unique_clusters = np.unique(cluster_labels)
            mitre_dim = self.mitre_embeddings.shape[1]
            
            # * 決定使用哪種向量計算 cluster centroids
            # * 優先使用 log_vectors (768維)，確保與 MITRE embeddings 維度匹配
            if log_vectors is not None and log_vectors.shape[1] == mitre_dim:
                print(f"    [使用原始嵌入] log_vectors {log_vectors.shape} 與 MITRE {self.mitre_embeddings.shape}")
                vectors_for_centroid = log_vectors
            elif nmf_extractor is not None and hasattr(nmf_extractor, '_is_fitted') and nmf_extractor._is_fitted:
                # 嘗試將 MITRE 嵌入投影至 NMF 概念空間
                nmf_input_dim = nmf_extractor.model.components_.shape[1]
                if mitre_dim == nmf_input_dim:
                    print(f"    [NMF 投影] 將 MITRE 嵌入投影至概念空間...")
                    mitre_projected = nmf_extractor.transform_local(self.mitre_embeddings)
                    # 使用概念向量計算 centroids，與投影後的 MITRE 比較
                    cluster_centroids = {}
                    for cluster_id in unique_clusters:
                        mask = cluster_labels == cluster_id
                        cluster_centroids[cluster_id] = np.mean(concept_vectors[mask], axis=0)
                    centroid_matrix = np.array([cluster_centroids[c] for c in unique_clusters])
                    embedding_similarities = cosine_similarity(centroid_matrix, mitre_projected)
                    # 跳過後續 centroid 計算，直接進入 TF-IDF 混合
                    vectors_for_centroid = None
                else:
                    print(f"    [Warning] NMF 維度不符，使用簡化標註")
                    return self._generate_placeholder_result(dataset_id, cluster_labels, output_dir, original_df, top_k)
            else:
                concept_dim = concept_vectors.shape[1]
                if mitre_dim != concept_dim:
                    print(f"    [Info] 維度不符 (MITRE={mitre_dim}, concept={concept_dim})，使用簡化標註")
                    return self._generate_placeholder_result(dataset_id, cluster_labels, output_dir, original_df, top_k)
                vectors_for_centroid = concept_vectors
            
            # * 計算 Cluster Centroids（如果尚未計算）
            if vectors_for_centroid is not None:
                cluster_centroids = {}
                for cluster_id in unique_clusters:
                    mask = cluster_labels == cluster_id
                    cluster_centroids[cluster_id] = np.mean(vectors_for_centroid[mask], axis=0)
                centroid_matrix = np.array([cluster_centroids[c] for c in unique_clusters])
                embedding_similarities = cosine_similarity(centroid_matrix, self.mitre_embeddings)
            
            # * 計算 Sequence TF-IDF 並混合評分 → Similarity Score
            similarity_scores = embedding_similarities
            tfidf_similarities = None
            log_texts = self._load_log_texts(dataset_id)
            
            if self.config.use_tfidf and log_texts is not None:
                cluster_tfidf, tfidf_similarities, _ = self._compute_sequence_tfidf(log_texts, cluster_labels)
                if tfidf_similarities is not None:
                    boost_status = "On" if self.config.enable_dual_boost else "Off"
                    print(f"    [Similarity] Emb={self.config.weight_embedding:.1f}, TF-IDF={self.config.weight_tfidf:.1f}, Boost={boost_status}")
                    similarity_scores = self._compute_hybrid_score(embedding_similarities, tfidf_similarities)
            
            # * 計算 Threat Confidence：結合 Similarity Score 與 Anomaly Score
            if anomaly_scores is not None:
                alpha = self.config.similarity_weight
                beta = self.config.anomaly_weight
                print(f"    [Threat Confidence] α(Sim)={alpha:.1f}, β(Anomaly)={beta:.1f}")
                final_scores = self._compute_threat_confidence(
                    similarity_scores, anomaly_scores, cluster_labels, unique_clusters
                )
            else:
                print(f"    [Warning] 無 Anomaly Score，僅使用 Similarity Score")
                final_scores = similarity_scores
            
            # * 為每個 cluster 找 Top-K 匹配
            cluster_to_techniques = {}
            for i, cluster_id in enumerate(unique_clusters):
                top_k_indices = np.argsort(final_scores[i])[-top_k:][::-1]
                
                techniques_list = []
                for idx in top_k_indices:
                    technique_id = self.mitre_technique_ids[idx] if self.mitre_technique_ids else f"T{idx}"
                    technique_name = self.mitre_technique_names[idx] if self.mitre_technique_names else "Unknown"
                    threat_conf = float(final_scores[i, idx])
                    sim_score = float(similarity_scores[i, idx])
                    
                    techniques_list.append({
                        "technique_name": technique_name,
                        "threat_confidence": threat_conf,
                        "similarity_score": sim_score,
                    })
                
                cluster_to_techniques[cluster_id] = techniques_list
            
            # * 顯示 Top-3 Clusters 的匹配結果
            print(f"    [Top-3 Cluster 匹配結果]")
            for cid in list(unique_clusters)[:3]:
                tech = cluster_to_techniques[cid][0]
                name_display = tech['technique_name'][:30] + "..." if len(tech['technique_name']) > 30 else tech['technique_name']
                print(f"      Cluster {cid}: {name_display} (threat_conf={tech['threat_confidence']:.3f})")
            
            # * 計算每筆 log 的 anomaly_score（用於輸出）
            log_anomaly_scores = anomaly_scores if anomaly_scores is not None else np.zeros(len(cluster_labels))

            # * 生成 Top-5 技術摘要（Embedding-only / TF-IDF-only / Hybrid）
            def _technique_name_by_index(idx: int) -> str:
                if self.mitre_technique_names and idx < len(self.mitre_technique_names):
                    return self.mitre_technique_names[idx]
                if self.mitre_technique_ids and idx < len(self.mitre_technique_ids):
                    return self.mitre_technique_ids[idx]
                return f"T{idx}"

            def _summarize_top5(scores: np.ndarray, mode: str) -> List[Dict[str, Any]]:
                top1_indices = np.argmax(scores, axis=1)
                cluster_top1 = {
                    cluster_id: _technique_name_by_index(top1_indices[i])
                    for i, cluster_id in enumerate(unique_clusters)
                }
                counts: Dict[str, int] = {}
                for log_idx in range(len(cluster_labels)):
                    tech = cluster_top1[cluster_labels[log_idx]]
                    counts[tech] = counts.get(tech, 0) + 1
                total = len(cluster_labels)
                rows = []
                for tech, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    rows.append({
                        "dataset_id": dataset_id,
                        "mode": mode,
                        "technique": tech,
                        "count": int(count),
                        "percent": round((count / total) * 100.0, 2),
                    })
                return rows

            summary_rows: List[Dict[str, Any]] = []
            embedding_summary = _summarize_top5(embedding_similarities, "embedding_only") if embedding_similarities is not None else []
            tfidf_summary = _summarize_top5(tfidf_similarities, "tfidf_only") if tfidf_similarities is not None else []
            hybrid_summary = _summarize_top5(similarity_scores, "hybrid_similarity") if similarity_scores is not None else []

            def _format_summary(rows: List[Dict[str, Any]]) -> str:
                return ", ".join([f"{r['technique']}: {r['count']} ({r['percent']}%)" for r in rows])

            def _expand_top5(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, str]:
                expanded: Dict[str, str] = {}
                for i in range(5):
                    key = f"{prefix}_top{i+1}"
                    if i < len(rows):
                        r = rows[i]
                        expanded[key] = f"{r['technique']}: {r['count']} ({r['percent']}%)"
                    else:
                        expanded[key] = ""
                return expanded

            summary_row = {
                "dataset_id": dataset_id,
                "groundtruth": f"{gt['tid']} | {gt['t_name']}",
            }
            summary_row.update(_expand_top5(embedding_summary, "embedding"))
            summary_row.update(_expand_top5(tfidf_summary, "tfidf"))
            summary_row.update(_expand_top5(hybrid_summary, "hybrid"))
            summary_rows.append(summary_row)
            
            # * 建立結果 DataFrame（包含原始資料 + Top-K 預測）
            
            # Get Ground Truth
            clean_id = dataset_id.replace("_raw_events", "").replace("_detection", "")
            gt = self.ground_truth.get(clean_id, {"tid": "Unknown", "t_name": "Unknown"})
            
            result_data = []
            for log_idx in range(len(cluster_labels)):
                cluster_id = cluster_labels[log_idx]
                techniques = cluster_to_techniques[cluster_id]
                
                row = {
                    "original_idx": log_idx,
                    "anomaly_score": float(log_anomaly_scores[log_idx]),
                }
                
                # Attach Ground Truth
                row["groundtruth_tid"] = gt["tid"]
                row["groundtruth_t_name"] = gt["t_name"]
                
                # 加入原始資料欄位
                if original_df is not None and log_idx < len(original_df):
                    for col in original_df.columns:
                        row[col] = original_df.iloc[log_idx][col]
                
                # 加入 Top-K 技術預測
                for k_idx, tech in enumerate(techniques, 1):
                    row[f"predicted_technique_{k_idx}_name"] = tech["technique_name"]
                    row[f"predicted_technique_{k_idx}_threat_confidence"] = tech["threat_confidence"]
                    row[f"predicted_technique_{k_idx}_similarity"] = tech["similarity_score"]
                
                result_data.append(row)
            
            result_df = pd.DataFrame(result_data)
            
            # * 儲存結果
            ensure_dir(output_dir)
            output_path = os.path.join(output_dir, f"{dataset_id}_Labeled.csv")
            result_df.to_csv(output_path, index=False)
            print(f"    標註結果已存至 {output_path}")

            # * 儲存摘要（Top-5 技術分布）
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_path = os.path.join(output_dir, f"{dataset_id}_Summary.csv")
                summary_df.to_csv(summary_path, index=False)
                print(f"    摘要已存至 {summary_path}")
                print(f"    [embedding] {summary_df.iloc[0].get('embedding_top1', '')}")
                if summary_df.iloc[0].get('tfidf_top1', ''):
                    print(f"    [tfidf] {summary_df.iloc[0].get('tfidf_top1', '')}")
                print(f"    [hybrid] {summary_df.iloc[0].get('hybrid_top1', '')}")

                # Append to aggregate summary CSV (one row per dataset)
                aggregate_path = os.path.join(output_dir, "Summary_All.csv")
                write_header = not os.path.exists(aggregate_path)
                summary_df.to_csv(aggregate_path, mode="a", index=False, header=write_header)
            
            # * 統計 Top-1 標註分布
            technique_counts = result_df["predicted_technique_1_name"].value_counts()
            print(f"    Top-1 標註分布:")
            for tech, count in list(technique_counts.items())[:5]:
                print(f"        {tech}: {count} ({count/len(result_df)*100:.1f}%)")
            
            return {
                "labels": result_data,
                "output_path": output_path,
                "result_df": result_df,
            }
            
        except Exception as e:
            print(f"    [Error] 標註失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_anomaly_scores(self, dataset_id: str) -> Optional[np.ndarray]:
        """
        載入 Stage II 異常偵測分數
        
        搜尋路徑：
        1. data/Detection_Results/{dataset_id}_detection/ (HuggingFace Dataset)
        2. data/Detection_Results/{dataset_id}_detection/ensemble_scores.npy
        
        Returns:
            anomaly_scores: [n_logs] 異常分數 (0~1)，或 None
        """
        detection_dir = self.config.detection_results_dir
        dataset_detection_dir = os.path.join(detection_dir, f"{dataset_id}_detection")
        
        if not os.path.exists(dataset_detection_dir):
            print(f"    [Warning] 找不到異常偵測結果: {dataset_detection_dir}")
            return None
        
        # 嘗試載入 HuggingFace Dataset
        state_json = os.path.join(dataset_detection_dir, "state.json")
        if os.path.exists(state_json):
            try:
                from datasets import load_from_disk
                ds = load_from_disk(dataset_detection_dir)
                
                # 搜尋可能的分數欄位
                score_cols = ["ensemble", "ensemble_score", "ensemble_raw", "anomaly_score"]
                for col in score_cols:
                    if col in ds.column_names:
                        scores = np.array(ds[col])
                        print(f"    [載入 Anomaly] {col}: mean={scores.mean():.3f}, max={scores.max():.3f}")
                        return scores
                
                # 備用：搜尋任何包含 score 的欄位
                for col in ds.column_names:
                    if "score" in col.lower() or "anomaly" in col.lower():
                        scores = np.array(ds[col])
                        print(f"    [載入 Anomaly] {col}: mean={scores.mean():.3f}, max={scores.max():.3f}")
                        return scores
            except Exception as e:
                print(f"    [Warning] 載入 HuggingFace Dataset 失敗: {e}")
        
        # 嘗試載入 npy 檔案
        for fname in ["ensemble_scores.npy", "scores.npy", "anomaly_scores.npy"]:
            npy_path = os.path.join(dataset_detection_dir, fname)
            if os.path.exists(npy_path):
                try:
                    scores = np.load(npy_path)
                    print(f"    [載入 Anomaly] {fname}: mean={scores.mean():.3f}, max={scores.max():.3f}")
                    return scores
                except Exception as e:
                    print(f"    [Warning] 載入 {fname} 失敗: {e}")
        
        print(f"    [Warning] 找不到有效的異常分數資料")
        return None

    def _load_original_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        # * 載入原始資料集（用於合併輸出）
        candidates = [
            os.path.join(self.config.input_logs_dir, f"{dataset_id}.csv"),
            os.path.join(self.config.intermediate_data_dir, f"{dataset_id}.csv"),
        ]
        
        for path in candidates:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    print(f"    [載入原始資料] {path} ({len(df)} 筆, {len(df.columns)} 欄)")
                    return df
                except Exception as e:
                    print(f"    [Warning] 載入原始資料失敗: {e}")
        
        print(f"    [Warning] 找不到原始資料集: {dataset_id}")
        return None

    def _generate_placeholder_result(
        self,
        dataset_id: str,
        cluster_labels: np.ndarray,
        output_dir: str,
        original_df: Optional[pd.DataFrame] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        # * 生成佔位符標註結果（維度不符時使用）
        
        # Get Ground Truth
        clean_id = dataset_id.replace("_raw_events", "").replace("_detection", "")
        gt = self.ground_truth.get(clean_id, {"tid": "Unknown", "t_name": "Unknown"})
        
        result_data = []
        for i in range(len(cluster_labels)):
            row = {"original_idx": i}
            
            # Attach Ground Truth
            row["groundtruth_tid"] = gt["tid"]
            row["groundtruth_t_name"] = gt["t_name"]
            
            if original_df is not None and i < len(original_df):
                for col in original_df.columns:
                    row[col] = original_df.iloc[i][col]
            
            for k in range(1, top_k + 1):
                row[f"predicted_technique_{k}_name"] = "TBD"
                row[f"predicted_technique_{k}_confidence"] = 0.0
            
            result_data.append(row)
        
        result_df = pd.DataFrame(result_data)
        
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, f"{dataset_id}_Labeled.csv")
        result_df.to_csv(output_path, index=False)
        print(f"    [Placeholder] 標註結果已存至 {output_path}")
        
        return {
            "labels": result_data,
            "output_path": output_path,
            "result_df": result_df,
        }


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
