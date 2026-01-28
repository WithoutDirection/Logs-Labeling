"""
AutoLabeling：自動標註模組

根據序列分群結果與 MITRE ATT&CK 外部知識進行自動標註。

# * 核心功能（供 Pipeline.py STAGE_IV 使用）：
# * 1. 載入 MITRE 嵌入向量
# * 2. 計算 Cluster Centroid 與 MITRE 向量的相似度
# * 3. 根據相似度為每個 Cluster 標註最匹配的 MITRE 技術 (Top-K)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow.feather as feather
import pyarrow

# * 調整匯入路徑
CURRENT_DIR = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(CURRENT_DIR).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from utils.path import ensure_dir


@dataclass
class LabelingConfig:
    """自動標註配置"""
    
    # Top-K 設定
    top_k_techniques: int = getattr(config, 'LABELING_TOP_K', 3)
    
    # 閾值設定
    similarity_threshold: float = getattr(config, 'LABELING_SIMILARITY_THRESHOLD', 0.3)
    confidence_threshold: float = getattr(config, 'LABELING_CONFIDENCE_THRESHOLD', 0.2)
    
    # 權重設定
    anomaly_weight: float = getattr(config, 'LABELING_ANOMALY_WEIGHT', 0.3)
    similarity_weight: float = getattr(config, 'LABELING_SIMILARITY_WEIGHT', 0.7)
    
    # 路徑設定
    mitre_embeddings_dir: str = getattr(config, 'MITRE_EXTERNAL_KNOWLEDGE_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_ATTACK"))
    input_logs_dir: str = config.INPUT_LOGS_DIR
    labeling_results_dir: str = getattr(config, 'LABELING_RESULTS_DIR', os.path.join(config.RESULT_DIR, "Labeling_Results"))
    detection_results_dir: str = getattr(config, 'DETECTION_RESULTS_DIR', os.path.join(config.DATA_DIR, "Detection_Results"))


class AutoLabeler:
    """
    自動標註器
    
    # * 流程（供 Pipeline.py 使用）：
    # * 1. load_mitre_embeddings() - 載入 MITRE 嵌入
    # * 2. process_single_dataset() - 對單一資料集進行標註
    """
    
    def __init__(self, labeling_config: Optional[LabelingConfig] = None):
        self.config = labeling_config or LabelingConfig()
        
        # MITRE 資料
        self.mitre_embeddings: Optional[np.ndarray] = None
        self.mitre_technique_ids: Optional[List[str]] = None
        self.mitre_technique_names: Optional[List[str]] = None
    
    def load_mitre_embeddings(
        self,
        embeddings_dir: Optional[str] = None,
    ) -> None:
        """載入 MITRE ATT&CK 嵌入向量"""
        embeddings_dir = embeddings_dir or self.config.mitre_embeddings_dir
        print(f"\n載入 MITRE 嵌入向量: {embeddings_dir}")
        
        # 嘗試 NumPy 格式
        for fname in ["embeddings.npy", "mitre_embeddings.npy", "technique_embeddings.npy"]:
            path = os.path.join(embeddings_dir, fname)
            if os.path.exists(path):
                self.mitre_embeddings = np.load(path)
                self._load_mitre_metadata(embeddings_dir)
                print(f"已載入 {len(self.mitre_embeddings)} 個 MITRE 技術嵌入")
                return
        
        # 嘗試 Arrow 格式
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
                first_key = next(iter(ds.keys()))
                ds = ds[first_key]

            embed_col = next(
                (c for c in ["embedding", "embeddings", "vector", "concept_vector"] if c in ds.column_names),
                None,
            )
            if embed_col is None:
                raise ValueError(f"MITRE dataset missing embedding column. Found columns: {ds.column_names}")

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
    
    def _load_anomaly_scores(self, dataset_id: str, expected_length: int) -> np.ndarray:
        """
        載入異常分數用於信心度計算
        
        Args:
            dataset_id: 資料集識別碼
            expected_length: 預期的樣本數量
            
        Returns:
            異常分數陣列 (N,)，若載入失敗則返回預設值 0.5
        """
        default_score = 0.5
        
        # 嘗試從 Detection_Results 載入
        detection_dir = self.config.detection_results_dir
        
        # 嘗試多種可能的檔案路徑
        possible_paths = [
            os.path.join(detection_dir, f"{dataset_id}_detection.csv"),
            os.path.join(detection_dir, f"{dataset_id}_anomaly.csv"),
            os.path.join(detection_dir, f"{dataset_id}.csv"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    # 嘗試多種欄位名稱
                    score_col = None
                    for col_name in ["ensemble_score", "anomaly_score", "score", "ensemble_anomaly_score"]:
                        if col_name in df.columns:
                            score_col = col_name
                            break
                    
                    if score_col and len(df) == expected_length:
                        scores = df[score_col].values.astype(float)
                        # 正規化到 0-1 範圍
                        if scores.max() > 1.0 or scores.min() < 0.0:
                            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                        print(f"    [異常分數] 已載入 {path} (mean={scores.mean():.3f})")
                        return scores
                except Exception as e:
                    print(f"    [Warning] 載入異常分數失敗 ({path}): {e}")
        
        # 若無法載入，使用預設值
        print(f"    [異常分數] 使用預設值 {default_score}")
        return np.full(expected_length, default_score)
    
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
        
        核心邏輯：
        1. 如果有 NMF extractor，將 MITRE 嵌入投影至相同概念空間
        2. 計算 Cluster Centroid 與投影後 MITRE 向量的餘弦相似度
        3. 根據相似度為每個 Cluster 標註最匹配的 MITRE 技術 (Top-K)
        
        輸出格式：
        - original_idx: 原始資料列索引
        - 原始資料集的所有欄位
        - predicted_technique_1_name, predicted_technique_1_confidence, ...
        - predicted_technique_k_name, predicted_technique_k_confidence, ...
        
        Args:
            dataset_id: Dataset 識別碼
            concept_vectors: 概念向量矩陣 (N, n_concepts)
            cluster_labels: HMM 分群標籤
            output_dir: 輸出目錄
            nmf_extractor: ConceptExtractor 物件（用於投影 MITRE 嵌入）
            
        Returns:
            標註結果字典，包含 'labels', 'output_path', 'result_df'
        """
        output_dir = output_dir or self.config.labeling_results_dir
        top_k = self.config.top_k_techniques
        
        if self.mitre_embeddings is None:
            print(f"    [Warning] MITRE 嵌入未載入，跳過標註")
            return None
        
        try:
            # 載入原始日誌資料
            original_log_path = os.path.join(self.config.input_logs_dir, f"{dataset_id}.csv")
            original_df = None
            if os.path.exists(original_log_path):
                try:
                    original_df = pd.read_csv(original_log_path)
                    if len(original_df) != len(cluster_labels):
                        print(f"    [Warning] 原始日誌長度 ({len(original_df)}) 與標籤數 ({len(cluster_labels)}) 不一致")
                        original_df = None
                except Exception as e:
                    print(f"    [Warning] 載入原始日誌失敗: {e}")
                    original_df = None
            
            # 載入異常分數（用於信心度計算）
            anomaly_scores = self._load_anomaly_scores(dataset_id, len(cluster_labels))
            
            # 計算每個 Cluster 的 Centroid（使用異常分數加權）
            unique_clusters = np.unique(cluster_labels)
            cluster_centroids = {}
            cluster_anomaly_scores = {}  # 每個 cluster 的平均異常分數
            
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                cluster_vectors = concept_vectors[mask]
                cluster_scores = anomaly_scores[mask]
                
                # 使用異常分數作為權重計算加權平均 Centroid
                if np.sum(cluster_scores) > 0:
                    weights = cluster_scores / np.sum(cluster_scores)
                    centroid = np.average(cluster_vectors, axis=0, weights=weights)
                else:
                    centroid = np.mean(cluster_vectors, axis=0)
                
                cluster_centroids[cluster_id] = centroid
                cluster_anomaly_scores[cluster_id] = np.mean(cluster_scores)
            
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
                    return self._generate_placeholder_result(dataset_id, cluster_labels, output_dir, original_df, top_k)
            elif mitre_dim != concept_dim:
                # 無 NMF 模型且維度不符
                print(f"    [Info] 維度不符 (MITRE={mitre_dim}, Concept={concept_dim})，無 NMF 模型，使用簡化標註")
                return self._generate_placeholder_result(dataset_id, cluster_labels, output_dir, original_df, top_k)
            else:
                # 維度匹配，直接計算相似度
                similarities = cosine_similarity(centroid_matrix, self.mitre_embeddings)
            
            # 為每個 cluster 找 Top-K 匹配（整合異常分數與閾值判斷）
            cluster_to_techniques = {}
            for i, cluster_id in enumerate(unique_clusters):
                top_k_indices = np.argsort(similarities[i])[-top_k:][::-1]
                avg_anomaly = cluster_anomaly_scores[cluster_id]
                
                techniques_list = []
                for idx in top_k_indices:
                    technique_id = self.mitre_technique_ids[idx] if self.mitre_technique_ids else f"T{idx}"
                    technique_name = self.mitre_technique_names[idx] if self.mitre_technique_names else "Unknown"
                    sim_score = float(similarities[i, idx])
                    
                    # 計算綜合信心度：結合異常分數與相似度
                    # confidence = w_a * anomaly_score + w_s * similarity
                    confidence = (
                        self.config.anomaly_weight * avg_anomaly +
                        self.config.similarity_weight * sim_score
                    )
                    
                    # 計算最終分數：similarity × confidence
                    final_score = sim_score * confidence
                    
                    # 閾值判斷：決定是否標記為 Benign
                    if sim_score < self.config.similarity_threshold or final_score < self.config.confidence_threshold:
                        label_name = "Benign"
                    else:
                        label_name = technique_name
                    
                    techniques_list.append({
                        "technique_id": technique_id,
                        "technique_name": technique_name,
                        "label": label_name,
                        "similarity": sim_score,
                        "anomaly_score": avg_anomaly,
                        "confidence": confidence,
                        "final_score": final_score,
                    })
                
                cluster_to_techniques[cluster_id] = techniques_list
            
            # 顯示 Top-3 Clusters 的最佳匹配技術
            print(f"    [Top-3 Cluster 匹配結果]")
            for cid in list(unique_clusters)[:3]:
                tech = cluster_to_techniques[cid][0]
                name_display = tech['technique_name'][:30] + "..." if len(tech['technique_name']) > 30 else tech['technique_name']
                label_display = tech['label']
                print(f"      Cluster {cid}: {label_display} (sim={tech['similarity']:.3f}, conf={tech['confidence']:.3f}, final={tech['final_score']:.3f})")
            
            # 建立結果 DataFrame
            result_data = []
            for log_idx in range(len(cluster_labels)):
                cluster_id = cluster_labels[log_idx]
                techniques = cluster_to_techniques[cluster_id]
                log_anomaly = anomaly_scores[log_idx]
                
                row = {"original_idx": log_idx}
                
                # 加入原始資料欄位
                if original_df is not None:
                    for col in original_df.columns:
                        row[col] = original_df.iloc[log_idx][col]
                
                # 加入該日誌的異常分數
                row["anomaly_score"] = log_anomaly
                
                # 加入 Top-K 技術預測（包含標籤、相似度、信心度）
                for k_idx, tech in enumerate(techniques, 1):
                    row[f"predicted_technique_{k_idx}_label"] = tech["label"]
                    row[f"predicted_technique_{k_idx}_name"] = tech["technique_name"]
                    row[f"predicted_technique_{k_idx}_similarity"] = tech["similarity"]
                    row[f"predicted_technique_{k_idx}_confidence"] = tech["confidence"]
                
                result_data.append(row)
            
            result_df = pd.DataFrame(result_data)
            
            # 儲存結果
            ensure_dir(output_dir)
            output_path = os.path.join(output_dir, f"{dataset_id}_Labeled.csv")
            result_df.to_csv(output_path, index=False)
            print(f"    標註結果已存至 {output_path}")
            
            # 統計最佳預測技術分布
            technique_counts = result_df["predicted_technique_1_label"].value_counts()
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
    
    def _generate_placeholder_result(
        self,
        dataset_id: str,
        cluster_labels: np.ndarray,
        output_dir: str,
        original_df: Optional[pd.DataFrame] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """生成佔位符標註結果（維度不符時使用）"""
        result_data = []
        for i in range(len(cluster_labels)):
            row = {"original_idx": i}
            
            if original_df is not None:
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
