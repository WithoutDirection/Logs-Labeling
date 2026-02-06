"""
Preprocess 子模組

此模組提供日誌預處理的完整管道：
1. LogLoader - 載入與解析日誌
2. LogEmbedder - 計算 BERT 嵌入
3. LogChunker - 生成 Log Vector

以及 Stage I 的統一入口：
4. process_all_inputs - 處理 Log Dataset 與 Reference Sources
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
    'run_preprocessing',
    'process_all_inputs'
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
    僅執行 Log Dataset 的預處理 (Parse -> Embed -> Chunk)
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
        print("[Log Process] 1. 載入並解析日誌...")
    loader = LogLoader(enable_parser=enable_parser)
    parsed_dfs = loader.load_logs(num=n_datasets, max_rows=15000)
    results["n_loaded"] = len(parsed_dfs) if parsed_dfs else 0
    
    # Step 2: 計算嵌入
    if verbose:
        print("[Log Process] 2. 計算 BERT 嵌入向量...")
    embedder = LogEmbedder(model_name=model_name, normalize=normalize)
    embedder.embed_logs(num=n_datasets)
    results["n_embedded"] = results["n_loaded"]
    results["embedding_dim"] = embedder.bert_model.get_embedding_dim() if embedder.bert_model else None
    
    # Step 3: 區塊化（可選）
    if enable_chunking:
        if verbose:
            print("[Log Process] 3. 生成 Log Vector...")
        chunker = LogChunker()
        chunker.chunk_logs(num=n_datasets)
        results["n_chunked"] = results["n_loaded"]
    
    return results


def process_all_inputs(
    n_datasets: int = None,
    enable_parser: bool = False,
    model_name: str = "sentence-bert",
    enable_chunking: bool = False,
    enable_tfidf: bool = True,
    verbose: bool = True
) -> dict:
    """
    Stage I 統一入口：處理所有輸入資料 (Log Datasets & Reference Sources)
    
    流程：
    1. Log Datasets 預處理 (Parse -> Embedding -> Chunkize)
    2. Reference Sources 預處理 (Embedding)
    3. TF-IDF Pipeline (Reference Fingerprints + Log Transformations)
    """
    results = {}
    
    if verbose:
        print("\n=== [Stage I] Processing Log Datasets ===")
    
    # 1. Log Preprocessing
    log_results = run_preprocessing(
        n_datasets=n_datasets,
        enable_parser=enable_parser,
        model_name=model_name,
        enable_chunking=enable_chunking,
        verbose=verbose
    )
    results.update(log_results)
    
    if verbose:
        print("\n=== [Stage I] Processing Reference Sources & TF-IDF ===")
        
    # 2. Reference Embedding (MITRE raw embeddings)
    # 動態 import 避免 circular dependencies
    from external_sources.build_mitre_raw_embeddings import build_mitre_raw_embeddings
    
    # 確保 Reference Embedding 與 Input 使用相同的 BERT 模型 (這裡假設 model_name 一致)
    print(f"[Ref Process] 1. 生成 Reference Embeddings ({model_name})...")
    ref_emb_path = build_mitre_raw_embeddings(bert_model=model_name, force_rebuild=True)
    results["reference_embedding_path"] = ref_emb_path
    
    # 3. TF-IDF Pipeline
    if enable_tfidf:
        print(f"[Ref Process] 2. 建立 Reference TF-IDF 指紋並轉換 Logs...")
        from precompute_log_tfidf import run_tfidf_pipeline
        run_tfidf_pipeline(force_rebuild=False)
        results["tfidf_enabled"] = True
    else:
        results["tfidf_enabled"] = False
        
    return results
