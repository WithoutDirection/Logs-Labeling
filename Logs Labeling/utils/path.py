"""路徑操作工具模組"""
import os
from pathlib import Path
from typing import List, Union


def get_current_dir(file: str) -> str:
    """獲取指定檔案所在目錄的絕對路徑"""
    return os.path.dirname(os.path.abspath(file))


def get_parent_dir(path: str) -> str:
    """獲取路徑的父目錄"""
    return os.path.dirname(path)


def join_path(*paths: str) -> str:
    """連接多個路徑片段"""
    return os.path.join(*paths)


def get_basename(path: str) -> str:
    """獲取路徑中的檔案名稱 (含副檔名)"""
    return os.path.basename(path)


def get_stem(path: str) -> str:
    """獲取檔案名稱 (不含副檔名)"""
    return os.path.splitext(os.path.basename(path))[0]


def get_extension(path: str) -> str:
    """獲取檔案副檔名 (含點號)"""
    return os.path.splitext(path)[1]


def split_extension(path: str) -> tuple:
    """分離路徑與副檔名，返回 (路徑, 副檔名)"""
    return os.path.splitext(path)


def is_dir(path: str) -> bool:
    """判斷路徑是否為目錄"""
    return os.path.isdir(path)


def exists(path: str) -> bool:
    """判斷路徑是否存在"""
    return os.path.exists(path)


def ensure_dir(path: Union[str, Path]) -> None:
    """確保目錄存在，若不存在則建立 (包含父目錄)"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_files(directory: str, extension: str) -> List[str]:
    """獲取目錄下指定副檔名的所有檔案列表"""
    if not os.path.exists(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.endswith(extension))


def get_dirs(directory: str, suffix: str = "") -> List[str]:
    """獲取目錄下符合後綴條件的所有子目錄名稱"""
    if not os.path.exists(directory):
        return []
    return sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and d.endswith(suffix)
    ])


def get_filtered_files(
    directory: str, 
    extension: str, 
    num: int = None, 
    ratio: float = None
) -> List[str]:
    """獲取目錄下篩選後的檔案列表
    
    Args:
        directory: 目錄路徑
        extension: 副檔名 (如 ".csv")
        num: 取前 N 個檔案
        ratio: 取前 ratio 比例的檔案 (0-1)
        
    Returns:
        篩選後的檔案名稱列表
    """
    all_files = get_files(directory, extension)
    if num:
        return all_files[:num]
    elif ratio:
        return all_files[:int(len(all_files) * ratio)]
    return all_files


def get_filtered_dirs(
    directory: str, 
    suffix: str = "", 
    num: int = None, 
    ratio: float = None
) -> List[str]:
    """獲取目錄下篩選後的子目錄列表
    
    Args:
        directory: 目錄路徑
        suffix: 目錄後綴 (如 "_embeddings")
        num: 取前 N 個目錄
        ratio: 取前 ratio 比例的目錄 (0-1)
        
    Returns:
        篩選後的子目錄名稱列表
    """
    all_dirs = get_dirs(directory, suffix)
    if num:
        return all_dirs[:num]
    elif ratio:
        return all_dirs[:int(len(all_dirs) * ratio)]
    return all_dirs
