"""
Preprocess 子模組

此模組提供日誌預處理的完整管道：
1. LogLoader - 載入與解析日誌
2. LogEmbedder - 計算 BERT 嵌入
3. LogChunker - 生成 Log Vector

主要 API:
    run_preprocessing(...) - 執行完整預處理流程
"""
from .drain import DrainParser
from .loader import LogLoader
from .embedder import LogEmbedder
from .chunker import LogChunker

__all__ = [
    'DrainParser',
    'LogLoader',
    'LogEmbedder', 
    'LogChunker',
    'run_preprocessing'
]


def run_preprocessing(
    n_datasets: int = None,
    enable_parser: bool = False,
    model_name: str = "sentence-bert",
    normalize: bool = False,
    enable_chunking: bool = False,
    verbose: bool = True,
) -> dict:
    """
    執行完整預處理流程的便捷 API
    
    此函式整合 LogLoader、LogEmbedder、LogChunker，提供一站式預處理。
    
    Args:
        n_datasets: 要處理的資料集數量（None 表示全部）
        enable_parser: 是否啟用日誌解析器（Drain）
        model_name: BERT 模型名稱
        normalize: 是否正規化嵌入向量
        enable_chunking: 是否執行 BiLSTM 區塊化
        verbose: 是否顯示詳細資訊
        
    Returns:
        包含處理結果摘要的字典
        
    Example:
        >>> # 處理前 10 個資料集
        >>> results = run_preprocessing(n_datasets=10)
        
        >>> # 啟用解析與區塊化
        >>> results = run_preprocessing(
        ...     n_datasets=50,
        ...     enable_parser=True,
        ...     enable_chunking=True
        ... )
    """
    results = {
        "n_loaded": 0,
        "n_embedded": 0,
        "n_chunked": 0,
        "embedding_dim": None,
        "model_name": model_name,
    }
    
    # Step 1: 載入日誌
    if verbose:
        print("[Step 1/3] 載入並解析日誌...")
    loader = LogLoader(enable_parser=enable_parser)
    parsed_dfs = loader.load_logs(num=n_datasets)
    results["n_loaded"] = len(parsed_dfs) if parsed_dfs else 0
    
    # Step 2: 計算嵌入
    if verbose:
        print("\n[Step 2/3] 計算 BERT 嵌入向量...")
    embedder = LogEmbedder(model_name=model_name, normalize=normalize)
    embedder.embed_logs(num=n_datasets)
    results["n_embedded"] = results["n_loaded"]
    results["embedding_dim"] = embedder.bert_model.get_embedding_dim() if embedder.bert_model else None
    
    # Step 3: 區塊化（可選）
    if enable_chunking:
        if verbose:
            print("\n[Step 3/3] 生成 Log Vector...")
        chunker = LogChunker()
        chunker.chunk_logs(num=n_datasets)
        results["n_chunked"] = results["n_loaded"]
    elif verbose:
        print("\n[Step 3/3] 跳過區塊化（enable_chunking=False）")
    
    return results
