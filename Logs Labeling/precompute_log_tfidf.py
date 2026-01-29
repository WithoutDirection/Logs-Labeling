"""Per-Log TF-IDF 預計算模組

將每個資料集的日誌文本轉換為 TF-IDF 稀疏向量，
使用 MITRE 預訓練的 TfidfVectorizer 以確保詞彙表一致性。

生成檔案: data/Embeddings/{dataset_id}_embeddings/tfidf.npz
依賴: 需先執行 external_sources/build_tfidf.py 生成 MITRE TF-IDF 模型

Usage:
    # Pipeline 整合
    from precompute_log_tfidf import run_log_tfidf_precompute
    result = run_log_tfidf_precompute(force_rebuild=False)
    
    # 命令列執行
    python precompute_log_tfidf.py [--force]
"""

import os
import sys
import pickle
from typing import Optional, Dict, Any, List
import pandas as pd
import scipy.sparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# 核心函數
# =============================================================================

def _load_vectorizer() -> Optional[Any]:
    """載入 MITRE TF-IDF Vectorizer"""
    tfidf_dir = getattr(config, 'MITRE_TFIDF_DIR', None)
    if not tfidf_dir:
        return None
    tfidf_path = os.path.join(tfidf_dir, "tfidf_vectorizer.pkl")
    if not os.path.exists(tfidf_path):
        return None
    with open(tfidf_path, "rb") as f:
        return pickle.load(f)


def _find_source_csv(dataset_id: str) -> Optional[str]:
    """根據 dataset_id 定位原始 CSV 檔案"""
    # 清理後綴
    cleaned_id = dataset_id
    for suffix in ["_embeddings", "_concepts", "_vectors"]:
        if cleaned_id.endswith(suffix):
            cleaned_id = cleaned_id[:-len(suffix)]
    
    # 候選檔名
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
    """從 DataFrame 提取日誌文本（按欄位優先順序）"""
    if "ConcatenatedLog" in df.columns:
        return df["ConcatenatedLog"].fillna("").astype(str).tolist()
    if "Template" in df.columns and "Parameters" in df.columns:
        return (df["Template"].fillna("") + " " + df["Parameters"].fillna("")).astype(str).tolist()
    if "Content" in df.columns:
        return df["Content"].fillna("").astype(str).tolist()
    if "Event" in df.columns:
        return df["Event"].fillna("").astype(str).tolist()
    return df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()


def _process_single_dataset(
    vectorizer, 
    dataset_id: str, 
    embeddings_path: str,
    force_rebuild: bool = False
) -> Optional[str]:
    """處理單一資料集，返回生成的 tfidf.npz 路徑或 None"""
    out_path = os.path.join(embeddings_path, "tfidf.npz")
    
    # 快取檢查
    if not force_rebuild and os.path.exists(out_path):
        return out_path
    
    # 尋找原始 CSV
    csv_path = _find_source_csv(dataset_id)
    if not csv_path:
        return None
    
    try:
        df = pd.read_csv(csv_path)
        texts = _extract_text(df)
        if not texts:
            return None
        
        tfidf_matrix = vectorizer.transform(texts)
        scipy.sparse.save_npz(out_path, tfidf_matrix)
        return out_path
    except Exception:
        return None


# =============================================================================
# Pipeline API
# =============================================================================

def run_log_tfidf_precompute(
    force_rebuild: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """Pipeline API：批次執行 Per-Log TF-IDF 預計算
    
    Args:
        force_rebuild: 強制重新計算（忽略快取）
        verbose: 輸出詳細日誌
        
    Returns:
        dict: {success, skipped, failed, total, enabled}
    """
    vectorizer = _load_vectorizer()
    if not vectorizer:
        return {
            "success": 0, "skipped": 0, "failed": 0, "total": 0,
            "enabled": False,
            "message": "TF-IDF Vectorizer 未找到"
        }
    
    embeddings_dir = config.LOG_VECTORS_DIR
    if not os.path.exists(embeddings_dir):
        return {"success": 0, "skipped": 0, "failed": 0, "total": 0, "enabled": True}
    
    subdirs = [d for d in os.listdir(embeddings_dir) 
               if os.path.isdir(os.path.join(embeddings_dir, d))]
    
    stats = {"success": 0, "skipped": 0, "failed": 0, "total": len(subdirs), "enabled": True}
    iterator = tqdm(subdirs, desc="TF-IDF 預計算", disable=not verbose)
    
    for subdir in iterator:
        dataset_path = os.path.join(embeddings_dir, subdir)
        out_path = os.path.join(dataset_path, "tfidf.npz")
        
        if not force_rebuild and os.path.exists(out_path):
            stats["skipped"] += 1
            continue
        
        result = _process_single_dataset(vectorizer, subdir, dataset_path, force_rebuild)
        if result:
            stats["success"] += 1
        else:
            stats["failed"] += 1
    
    return stats


# =============================================================================
# 命令列入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Per-Log TF-IDF 預計算")
    parser.add_argument("--force", action="store_true", help="強制重新計算")
    args = parser.parse_args()
    
    result = run_log_tfidf_precompute(force_rebuild=args.force, verbose=True)
    print(f"\n完成: 成功={result['success']} 快取={result['skipped']} 失敗={result['failed']}")
