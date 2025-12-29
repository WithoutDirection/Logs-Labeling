"""
Sequence Cluster 資料對應工具

將 data/SequenceClusters 中的分群結果對應回原始 data/input_logs，
並儲存結果至 data/clustered_logs。
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from collections import defaultdict
from tqdm import tqdm

# 將上層目錄加入路徑以便引入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 定義輸出目錄
CLUSTERED_LOGS_DIR = os.path.join(config.DATA_DIR, "clustered_logs")


def load_cluster_labels(dataset_id: str) -> Optional[np.ndarray]:
    """
    載入特定資料集的分群標籤。

    Args:
        dataset_id: 資料集 ID（不含 _embeddings 後綴）

    Returns:
        標籤陣列，若找不到則回傳 None
    """
    # 嘗試多種可能的資料夾命名格式
    possible_dirs = [
        os.path.join(config.CLUSTER_RESULTS_DIR, f"{dataset_id}_embeddings"),
        os.path.join(config.CLUSTER_RESULTS_DIR, dataset_id),
    ]
    
    for cluster_dir in possible_dirs:
        labels_path = os.path.join(cluster_dir, "labels.npy")
        if os.path.exists(labels_path):
            return np.load(labels_path)
    
    return None


def load_input_logs(dataset_id: str) -> Optional[pd.DataFrame]:
    """
    載入原始 input_logs 的 CSV 檔案。

    Args:
        dataset_id: 資料集 ID

    Returns:
        DataFrame，若找不到則回傳 None
    """
    csv_path = os.path.join(config.INTERMEDIATE_DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(csv_path):
        return None
    
    return pd.read_csv(csv_path)


def load_intermediate_data(dataset_id: str) -> Optional[pd.DataFrame]:
    """
    載入原始 Intermediate_data 的 CSV 檔案。

    Args:
        dataset_id: 資料集 ID

    Returns:
        DataFrame，若找不到則回傳 None
    """
    csv_path = os.path.join(config.INTERMEDIATE_DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(csv_path):
        return None
    
    return pd.read_csv(csv_path)


def list_available_datasets() -> List[str]:
    """
    列出所有有分群結果的資料集。

    Returns:
        資料集 ID 列表
    """
    if not os.path.exists(config.CLUSTER_RESULTS_DIR):
        return []
    
    datasets = []
    for subdir in os.listdir(config.CLUSTER_RESULTS_DIR):
        subdir_path = os.path.join(config.CLUSTER_RESULTS_DIR, subdir)
        if os.path.isdir(subdir_path):
            labels_path = os.path.join(subdir_path, "labels.npy")
            if os.path.exists(labels_path):
                # 移除 _embeddings 後綴（如果有）
                dataset_id = subdir.replace("_embeddings", "")
                datasets.append(dataset_id)
    
    return sorted(datasets)


def export_clustered_logs_to_csv(
    dataset_id: str,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    將分群結果對應回原始 input_logs 並匯出為 CSV 檔案。

    Args:
        dataset_id: 資料集 ID
        output_dir: 輸出目錄（預設為 data/clustered_logs）

    Returns:
        匯出的檔案路徑，若失敗則回傳 None
    """
    # 載入分群標籤
    labels = load_cluster_labels(dataset_id)
    if labels is None:
        print(f"[Warning] 找不到資料集 {dataset_id} 的分群結果，跳過")
        return None
    
    # 載入原始 input_logs
    df = load_input_logs(dataset_id)
    if df is None:
        print(f"[Warning] 找不到資料集 {dataset_id} 的原始 input_logs，跳過")
        return None
    
    # 驗證資料長度一致性
    if len(labels) != len(df):
        print(f"[Warning] 資料集 {dataset_id}: 標籤數量 ({len(labels)}) 與資料數量 ({len(df)}) 不一致")
    
    # 確保長度一致
    min_len = min(len(labels), len(df))
    df = df.iloc[:min_len].copy()
    df.insert(0, "cluster", labels[:min_len])
    
    # 設定輸出路徑
    if output_dir is None:
        output_dir = CLUSTERED_LOGS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_id}.csv")
    
    df.to_csv(output_path, index=False)
    
    return output_path


def export_all_clustered_logs(output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    將所有 SequenceClusters 中的分群結果對應回原始 input_logs，
    並儲存至指定目錄。

    Args:
        output_dir: 輸出目錄（預設為 data/clustered_logs）

    Returns:
        字典，key 為資料集 ID，value 為匯出的檔案路徑
    """
    if output_dir is None:
        output_dir = CLUSTERED_LOGS_DIR
    
    # 取得所有有分群結果的資料集
    datasets = list_available_datasets()
    
    if not datasets:
        print("[Error] 找不到任何分群結果")
        return {}
    
    print(f"[Info] 找到 {len(datasets)} 個資料集需要處理")
    print(f"[Info] 輸出目錄: {output_dir}")
    print("=" * 60)
    
    results = {}
    success_count = 0
    fail_count = 0
    
    for dataset_id in tqdm(datasets, desc="處理資料集"):
        output_path = export_clustered_logs_to_csv(dataset_id, output_dir)
        if output_path:
            results[dataset_id] = output_path
            success_count += 1
        else:
            fail_count += 1
    
    print("=" * 60)
    print(f"[Info] 處理完成!")
    print(f"       成功: {success_count} 個資料集")
    print(f"       失敗: {fail_count} 個資料集")
    print(f"       輸出目錄: {output_dir}")
    
    return results


def display_cluster_summary(dataset_id: str) -> None:
    """
    顯示特定資料集的分群摘要統計。

    Args:
        dataset_id: 資料集 ID
    """
    labels = load_cluster_labels(dataset_id)
    if labels is None:
        print(f"[Error] 找不到資料集 {dataset_id} 的分群結果")
        return
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print("=" * 50)
    print(f"資料集: {dataset_id}")
    print(f"總資料筆數: {len(labels)}")
    print(f"分群數量: {len(unique_labels)}")
    print("=" * 50)
    print(f"{'Cluster':^10} | {'數量':^10} | {'佔比':^10}")
    print("-" * 50)
    
    for label, count in zip(unique_labels, counts):
        ratio = count / len(labels) * 100
        print(f"{label:^10} | {count:^10} | {ratio:^9.1f}%")


def export_cluster_mapping_to_csv(
    dataset_id: str,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    將分群對應結果匯出為 CSV 檔案（使用 Intermediate_data）。

    Args:
        dataset_id: 資料集 ID
        output_path: 輸出路徑（預設為 result/cluster_mapping_{dataset_id}.csv）

    Returns:
        匯出的檔案路徑，若失敗則回傳 None
    """
    labels = load_cluster_labels(dataset_id)
    df = load_intermediate_data(dataset_id)
    
    if labels is None or df is None:
        print("[Error] 無法載入資料")
        return None
    
    # 確保長度一致
    min_len = min(len(labels), len(df))
    df = df.iloc[:min_len].copy()
    df["cluster"] = labels[:min_len]
    
    # 重新排列欄位順序
    cols = ["cluster"] + [c for c in df.columns if c != "cluster"]
    df = df[cols]
    
    # 設定輸出路徑
    if output_path is None:
        os.makedirs(config.RESULT_DIR, exist_ok=True)
        output_path = os.path.join(config.RESULT_DIR, f"cluster_mapping_{dataset_id}.csv")
    
    df.to_csv(output_path, index=False)
    print(f"[Info] 已匯出至: {output_path}")
    
    return output_path


# ======================== 主程式 ========================

if __name__ == "__main__":
    # 執行：匯出所有分群結果對應的原始 logs 至 data/clustered_logs
    export_all_clustered_logs()