"""Hugging Face Dataset I/O 工具模組"""
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
