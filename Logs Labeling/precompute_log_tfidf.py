"""TF-IDF Processing Module

負責統一處理專案中的 TF-IDF 計算：
1. 為 Reference Sources (MITRE Techniques) 建立 TF-IDF 指紋。
2. 將 Log Data 轉換為 TF-IDF 向量 (使用 Reference 的 Vectorizer 以確保空間一致性)。

Usage:
    from precompute_log_tfidf import run_tfidf_pipeline
    run_tfidf_pipeline()
"""

import os
import sys
import pickle
import pandas as pd
import scipy.sparse
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, Dict, Any, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from utils.path import ensure_dir

# =============================================================================
# Reference Source (MITRE) TF-IDF logic
# =============================================================================

def build_reference_tfidf(
    force_rebuild: bool = False,
    max_features: int = 5000
) -> Optional[Any]:
    """建立 MITRE Technique 的 TF-IDF Vectorizer 與 Matrix (指紋)"""
    
    out_dir = getattr(config, 'MITRE_TFIDF_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_TFIDF"))
    ensure_dir(out_dir)
    
    vectorizer_path = os.path.join(out_dir, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(out_dir, "mitre_tfidf_matrix.pkl")
    metadata_path = os.path.join(out_dir, "metadata.csv")

    if not force_rebuild and os.path.exists(vectorizer_path):
        print(f"  [TF-IDF] 載入現有 Reference Vectorizer: {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            return pickle.load(f)

    # Prefer the multi-source combined CSV if it exists, fall back to legacy files
    combined_csv = getattr(config, 'REFERENCE_COMBINED_CSV',
                           os.path.join(config.REFERENCE_RESOURCES_DIR, "combined.csv"))
    if os.path.exists(combined_csv):
        mitre_csv = combined_csv
    else:
        mitre_csv = getattr(config, 'MITRE_TECHNIQUES_CSV', None)
        if not mitre_csv or not os.path.exists(mitre_csv):
            mitre_csv = os.path.join(config.REFERENCE_RESOURCES_DIR, "MitreTechniquesTokens_V6_Sanitized.csv")
            if not os.path.exists(mitre_csv):
                mitre_csv = os.path.join(config.REFERENCE_RESOURCES_DIR, "MitreTechniquesTokens_V5.csv")
    
    if not os.path.exists(mitre_csv):
        print(f"[Error] 找不到 MITRE CSV: {mitre_csv}")
        return None

    print(f"  [TF-IDF] 建立 Reference 指紋 (來源: {os.path.basename(mitre_csv)})...")
    try:
        df = pd.read_csv(mitre_csv)
    except Exception as e:
        print(f"[Error] 讀取 CSV 失敗: {e}")
        return None
    
    # 決定使用哪個欄位作為描述
    desc_col = "description"
    if "cleaned_tokens" in df.columns:
        desc_col = "cleaned_tokens" # 優先使用清洗過的 token
    elif "description" not in df.columns:
        print(f"[Error] CSV 中找不到 'description' 或 'cleaned_tokens' 欄位")
        return None

    documents = df[desc_col].fillna("").astype(str).tolist()
    
    # 建立與訓練 Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # 儲存
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(matrix_path, "wb") as f:
        pickle.dump(tfidf_matrix, f) # 這是 Reference 的指紋矩陣
        
    # 保存 Metadata 以便查閱
    cols_to_save = [c for c in ["technique", "technique_id", "technique_name"] if c in df.columns]
    if cols_to_save:
        df[cols_to_save].to_csv(metadata_path, index=False)
        
    print(f"  [TF-IDF] Reference Vectorizer 已儲存至 {out_dir}")
    return vectorizer


# =============================================================================
# Log Data TF-IDF logic
# =============================================================================

def _find_source_csv(dataset_id: str) -> Optional[str]:
    """根據 dataset_id 定位原始 CSV 檔案"""
    # 清理後綴
    cleaned_id = dataset_id
    for suffix in ["_embeddings", "_concepts", "_vectors", "_logvectors"]:
        if cleaned_id.endswith(suffix):
            cleaned_id = cleaned_id[:-len(suffix)]
    
    # 候選檔名 pattern
    candidates = [
        f"{cleaned_id}.csv",
        f"{cleaned_id}_raw_events.csv",
        f"syslogs_{cleaned_id}_audit_log.csv"
    ]
    
    # 搜尋 Intermediate Data
    for cand in candidates:
        p = os.path.join(config.INTERMEDIATE_DATA_DIR, cand)
        if os.path.exists(p):
            return p
    
    # 搜尋 Input Logs
    input_path = os.path.join(config.INPUT_LOGS_DIR, f"{cleaned_id}.csv")
    if os.path.exists(input_path):
        return input_path
    
    return None

def _extract_text(df: pd.DataFrame) -> List[str]:
    """從 DataFrame 提取日誌文本"""
    # 優先順序: ConcatenatedLog > CleanedLog > Content > Event
    cols = ["ConcatenatedLog", "CleanedLog", "Content", "Event", "Template", "Message"]
    for c in cols:
        if c in df.columns:
            return df[c].fillna("").astype(str).tolist()
            
    # Fallback: Join all columns
    return df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()

def compute_log_tfidf(
    vectorizer,
    force_rebuild: bool = False
) -> Dict[str, int]:
    """使用給定的 Vectorizer 計算所有 Log Dataset 的 TF-IDF"""
    
    embeddings_dir = config.LOG_VECTORS_DIR
    if not os.path.exists(embeddings_dir):
        print(f"[Warning] Log Vectors 目錄不存在 ({embeddings_dir})，跳過 Log TF-IDF 計算")
        return {"success": 0, "failed": 0}
        
    subdirs = [d for d in os.listdir(embeddings_dir) 
               if os.path.isdir(os.path.join(embeddings_dir, d))]
    
    stats = {"success": 0, "skipped": 0, "failed": 0}
    
    for subdir in tqdm(subdirs, desc="[TF-IDF] Log Transformation"):
        dataset_path = os.path.join(embeddings_dir, subdir)
        out_path = os.path.join(dataset_path, "tfidf.npz")
        
        if not force_rebuild and os.path.exists(out_path):
            stats["skipped"] += 1
            continue
            
        csv_path = _find_source_csv(subdir)
        if not csv_path:
            stats["failed"] += 1
            continue
            
        try:
            df = pd.read_csv(csv_path)
            texts = _extract_text(df)
            if not texts:
                stats["failed"] += 1
                continue
            
            tfidf_matrix = vectorizer.transform(texts)
            scipy.sparse.save_npz(out_path, tfidf_matrix)
            stats["success"] += 1
        except Exception as e:
            stats["failed"] += 1
            
    return stats


# =============================================================================
# Sequence TF-IDF API (for Stage III Auto Labeling)
# =============================================================================

def compute_sequence_tfidf(
    log_texts: List[str],
    cluster_labels,
    vectorizer=None
) -> Optional[scipy.sparse.csr_matrix]:
    """
    計算 HMM Sequence (Cluster) 的 TF-IDF 向量
    
    將同一 cluster 內的所有 log 文本聚合，生成該 Sequence 的 TF-IDF 指紋。
    使用與 Reference 相同的 Vectorizer 確保向量空間一致。
    
    Args:
        log_texts: 原始日誌文本列表
        cluster_labels: HMM 分群標籤 (與 log_texts 等長)
        vectorizer: TF-IDF Vectorizer (若為 None 則自動載入)
        
    Returns:
        sequence_tfidf: [n_clusters, n_features] 稀疏矩陣
    """
    import numpy as np
    
    if vectorizer is None:
        vectorizer = load_reference_vectorizer()
        if vectorizer is None:
            return None
    
    cluster_labels = np.asarray(cluster_labels)
    unique_clusters = np.unique(cluster_labels)
    
    # 聚合每個 cluster 的文本
    cluster_texts = []
    for cid in unique_clusters:
        mask = cluster_labels == cid
        texts = [log_texts[i] for i in range(len(log_texts)) if mask[i]]
        cluster_texts.append(" ".join(texts))
    
    return vectorizer.transform(cluster_texts)


def load_reference_vectorizer():
    """載入 Reference TF-IDF Vectorizer"""
    tfidf_dir = getattr(config, 'MITRE_TFIDF_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_TFIDF"))
    vectorizer_path = os.path.join(tfidf_dir, "tfidf_vectorizer.pkl")
    
    if not os.path.exists(vectorizer_path):
        return None
        
    with open(vectorizer_path, "rb") as f:
        return pickle.load(f)


def load_reference_tfidf_matrix():
    """載入 MITRE Technique TF-IDF 指紋矩陣"""
    tfidf_dir = getattr(config, 'MITRE_TFIDF_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_TFIDF"))
    matrix_path = os.path.join(tfidf_dir, "mitre_tfidf_matrix.pkl")
    
    if not os.path.exists(matrix_path):
        return None
        
    with open(matrix_path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Main API
# =============================================================================

def run_tfidf_pipeline(force_rebuild: bool = False):
    """
    執行完整的 TF-IDF 流程：
    1. 準備 Reference Source 指紋 (Vectorizer)
    2. 計算 Log Data 的 TF-IDF 向量 (相似空間映射)
    """
    print("--- 啟動 TF-IDF Pipeline ---")
    
    # 1. Reference Sources
    vectorizer = build_reference_tfidf(force_rebuild=force_rebuild)
    if not vectorizer:
        print("[Error] 無法建立或載入 Reference Vectorizer，TF-IDF 流程中止。")
        return
    
    # 2. Log Data
    stats = compute_log_tfidf(vectorizer, force_rebuild=force_rebuild)
    print(f"--- TF-IDF 完成: Logs Success={stats['success']}, Skipped={stats['skipped']}, Failed={stats['failed']} ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    run_tfidf_pipeline(force_rebuild=args.force)
