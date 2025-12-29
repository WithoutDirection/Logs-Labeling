"""路徑操作工具模組"""
import os
import shutil
from pathlib import Path
from typing import List, Union, Optional, Set


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
    return [f for f in os.listdir(directory) if f.endswith(extension)]


def get_dirs(directory: str, suffix: str = "") -> List[str]:
    """獲取目錄下符合後綴條件的所有子目錄名稱"""
    if not os.path.exists(directory):
        return []
    return [
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and d.endswith(suffix)
    ]


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


def clear_data_directories(
    data_dir: str = "data",
    preserve_dirs: Optional[Set[str]] = None,
    dry_run: bool = True
) -> dict:
    """清空資料目錄中的所有子目錄，但保留指定的目錄不刪除
    
    此函數用於清理管線執行過程中產生的中間資料和輸出結果，
    但會保留輸入資料（如 input_logs）和參考資源（如 reference_resources）。
    
    Args:
        data_dir: 資料根目錄路徑（預設為 "data"）
        preserve_dirs: 要保留的子目錄名稱集合。如果為 None，則預設保留 
                      {"input_logs", "reference_resources"}
        dry_run: 若為 True，只顯示將要刪除的目錄而不實際刪除（預設為 True）
    
    Returns:
        字典，包含清理統計資訊：
        {
            "removed": [...],  # 已刪除的目錄列表
            "preserved": [...],  # 已保留的目錄列表
            "failed": [...]  # 刪除失敗的目錄列表
        }
    
    Examples:
        >>> # 預覽將要刪除的目錄（不實際刪除）
        >>> result = clear_data_directories(dry_run=True)
        >>> 
        >>> # 實際執行清理
        >>> result = clear_data_directories(dry_run=False)
        >>> 
        >>> # 自訂要保留的目錄
        >>> result = clear_data_directories(
        ...     preserve_dirs={"input_logs", "reference_resources", "my_backup"},
        ...     dry_run=False
        ... )
    """
    # 預設保留 input_logs 和 reference_resources
    if preserve_dirs is None:
        preserve_dirs = {"input_logs", "reference_resources"}
    
    # 確保路徑存在
    if not exists(data_dir):
        print(f"[警告] 資料目錄不存在: {data_dir}")
        return {"removed": [], "preserved": [], "failed": []}
    
    # 統計資訊
    stats = {
        "removed": [],
        "preserved": [],
        "failed": []
    }
    
    # 列出所有子目錄
    try:
        subdirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    except Exception as e:
        print(f"[錯誤] 無法列出目錄 {data_dir}: {e}")
        return stats
    
    # 顯示操作模式
    mode_msg = "[預覽模式]" if dry_run else "[執行模式]"
    print(f"\n{mode_msg} 清理資料目錄: {data_dir}")
    print("=" * 60)
    
    # 處理每個子目錄
    for subdir in sorted(subdirs):
        subdir_path = os.path.join(data_dir, subdir)
        
        if subdir in preserve_dirs:
            stats["preserved"].append(subdir)
            print(f"[保留] {subdir}")
        else:
            if dry_run:
                stats["removed"].append(subdir)
                print(f"[將刪除] {subdir}")
            else:
                try:
                    shutil.rmtree(subdir_path, ignore_errors=False)
                    stats["removed"].append(subdir)
                    print(f"[已刪除] {subdir}")
                except Exception as e:
                    stats["failed"].append(subdir)
                    print(f"[失敗] {subdir}: {e}")
    
    # 顯示統計摘要
    print("=" * 60)
    print(f"總計:")
    print(f"  保留: {len(stats['preserved'])} 個目錄")
    print(f"  {'將刪除' if dry_run else '已刪除'}: {len(stats['removed'])} 個目錄")
    if stats["failed"]:
        print(f"  失敗: {len(stats['failed'])} 個目錄")
    
    if dry_run:
        print(f"\n提示: 使用 dry_run=False 來實際執行刪除操作")
    
    return stats
