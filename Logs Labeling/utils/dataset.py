"""Hugging Face Dataset I/O 工具模組"""
import os
import shutil
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datasets import Dataset

from utils.path import ensure_dir, get_parent_dir


def save_dataset(data_dict: Dict[str, Any], output_path: str) -> None:
    """將資料字典儲存為 Hugging Face Dataset 格式
    
    Args:
        data_dict: 資料字典，鍵為欄位名稱，值為資料列表
        output_path: 輸出路徑
    """
    ensure_dir(get_parent_dir(output_path))
    dataset = Dataset.from_dict(data_dict)
    dataset.save_to_disk(output_path)


def load_dataset(path: str) -> Dataset:
    """載入 Hugging Face Dataset"""
    return Dataset.load_from_disk(path)


def load_dataset_as_numpy(
    path: str, 
    columns: List[str]
) -> Dict[str, np.ndarray]:
    """載入 Dataset 並將指定欄位轉為 Numpy Array
    
    Args:
        path: Dataset 路徑
        columns: 要轉換的欄位名稱列表
        
    Returns:
        字典，鍵為欄位名稱，值為 Numpy Array
    """
    dataset = Dataset.load_from_disk(path)
    result = {}
    for col in columns:
        if col in dataset.column_names:
            result[col] = np.array(dataset[col])
    return result


def load_embeddings(
    path: str
) -> Tuple[List[str], np.ndarray, Optional[np.ndarray], bool]:
    """載入嵌入向量資料並自動判斷模式
    
    Returns:
        (log_ids, template_embeddings, param_embeddings, has_parsing)
        - has_parsing=True: 有 template_embedding 與 param_embedding
        - has_parsing=False: 僅有單一 embedding
    """
    dataset = Dataset.load_from_disk(path)
    log_ids = dataset['LogID']
    
    # * 判斷是否有雙欄位嵌入 (has_parsing)
    if 'template_embedding' in dataset.column_names and 'param_embedding' in dataset.column_names:
        template_embeddings = np.array(dataset['template_embedding'])
        param_embeddings = np.array(dataset['param_embedding'])
        return log_ids, template_embeddings, param_embeddings, True
    else:
        embeddings = np.array(dataset['embedding'])
        return log_ids, embeddings, None, False


# ======================== 資料檢查與清理 ========================


def _infer_vector_dim(dataset: Dataset) -> Tuple[Optional[str], Optional[int]]:
    """嘗試推斷向量欄位名稱與維度（僅取第一列，避免全量載入）。"""

    candidate_keys = ("embedding", "vector", "log_vector", "template_embedding", "param_embedding")
    for key in candidate_keys:
        if key not in dataset.column_names:
            continue
        sample = dataset[0][key]
        arr = np.array(sample)
        if arr.ndim == 0:
            dim = 1
        elif arr.ndim == 1:
            dim = arr.shape[0]
        else:
            dim = arr.shape[-1]
        return key, int(dim)
    return None, None


def prune_mismatched_datasets(
    root_dir: str,
    expected_dim: int,
    keep_suffix: str = "_embeddings",
    dry_run: bool = True,
) -> Dict[str, List[str]]:
    """檢查根目錄下的資料集，刪除維度不符或名稱不符的子目錄。

    Args:
        root_dir: HF dataset 子目錄所在的根目錄，例如 data/Embeddings
        expected_dim: 期望的向量維度（例如 384）
        keep_suffix: 只保留名稱結尾符合的資料夾（如 "_embeddings"），其他一律刪除
        dry_run: 為 True 時僅列出將被刪除的目錄，不實際刪除

    Returns:
        簡單的處理紀錄，含 kept / removed / failed 等清單。
    """

    root = Path(root_dir)
    summary: Dict[str, List[str]] = {
        "kept": [],
        "removed_suffix": [],
        "removed_dim": [],
        "failed_load": [],
    }

    if not root.exists():
        print(f"Root dir not found: {root}")
        return summary

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        name = subdir.name

        # 只保留指定結尾的資料夾
        if keep_suffix and not name.endswith(keep_suffix):
            summary["removed_suffix"].append(name)
            if not dry_run:
                shutil.rmtree(subdir, ignore_errors=True)
            continue

        try:
            ds = load_dataset(str(subdir))
        except Exception as exc:
            print(f"Failed to load dataset {name}: {exc}")
            summary["failed_load"].append(name)
            if not dry_run:
                shutil.rmtree(subdir, ignore_errors=True)
            continue

        _, dim = _infer_vector_dim(ds)
        if dim is None:
            print(f"Skip {name}: 無法推斷向量欄位")
            summary["failed_load"].append(name)
            if not dry_run:
                shutil.rmtree(subdir, ignore_errors=True)
            continue

        if dim != expected_dim:
            print(f"Remove {name}: dim {dim} != {expected_dim}")
            summary["removed_dim"].append(name)
            if not dry_run:
                shutil.rmtree(subdir, ignore_errors=True)
            continue

        summary["kept"].append(name)

    print("Prune summary:")
    print(f"  kept: {len(summary['kept'])}")
    print(f"  removed (suffix): {len(summary['removed_suffix'])}")
    print(f"  removed (dim): {len(summary['removed_dim'])}")
    print(f"  failed load: {len(summary['failed_load'])}")
    return summary
