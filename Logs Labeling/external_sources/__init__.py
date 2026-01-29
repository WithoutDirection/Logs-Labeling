# External Sources Module for Log Labeling Pipeline
# This module provides tools for loading, processing, and extracting concepts
# from external threat intelligence sources (MITRE ATT&CK, CAPEC, CVE, etc.)
#
# Implements ConceptUML methodology:
# - Preprocessing with Zipf's law (filter top 5% high-frequency words)
# - BERT + BoW concatenation with min-max normalization
# - NMF topic modeling on embeddings
# - HMM sequence clustering
# - MITRE/CAPEC similarity mapping
#
# 主要 API:
#     build_knowledge_base() - 建立外部知識嵌入向量

from .source_manager import ExternalSourceManager
from .text_processor import (
    TextProcessor,
    preprocess_external_source,
    preprocess_log_corpus,
    DEFAULT_STOPWORDS,
    LOG_HIGH_FREQ_WORDS,
    SECURITY_TERMS,
)
from .build_mitre_raw_embeddings import build_mitre_raw_embeddings

# Optional modules (may not exist on this branch)
MitreFetcher = CapecFetcher = NvdFetcher = None
HMMClusterer = None
compute_cluster_similarity = None
rank_suspicious_clusters = None
rank_samples_in_cluster = None
ConceptUMLPipeline = None

try:
    from .fetchers import MitreFetcher, CapecFetcher, NvdFetcher
except Exception:
    pass

try:
    from .hmm_clustering import (
        HMMClusterer,
        compute_cluster_similarity,
        rank_suspicious_clusters,
        rank_samples_in_cluster,
    )
except Exception:
    pass

try:
    from .concept_uml_pipeline import ConceptUMLPipeline
except Exception:
    pass

__all__ = [
    # Core managers
    'ExternalSourceManager',
    'TextProcessor',
    
    # Preprocessing utilities
    'preprocess_external_source',
    'preprocess_log_corpus',
    'DEFAULT_STOPWORDS',
    'LOG_HIGH_FREQ_WORDS',
    'SECURITY_TERMS',
    
    # Pipeline API
    'build_mitre_raw_embeddings',
    'build_knowledge_base',
]

if MitreFetcher is not None:
    __all__ += ['MitreFetcher', 'CapecFetcher', 'NvdFetcher']

if HMMClusterer is not None:
    __all__ += [
        'HMMClusterer',
        'compute_cluster_similarity',
        'rank_suspicious_clusters',
        'rank_samples_in_cluster',
    ]

if ConceptUMLPipeline is not None:
    __all__ += ['ConceptUMLPipeline']


def build_knowledge_base(
    force_rebuild: bool = False,
    verbose: bool = True
) -> dict:
    """
    建立外部知識嵌入向量
    
    將 MITRE ATT&CK 技術描述轉換為 BERT 嵌入向量，
    供後續概念提取與自動標註使用。
    
    Args:
        force_rebuild: 是否強制重建（即使已存在）
        verbose: 是否顯示詳細資訊
        
    Returns:
        包含處理結果的字典：
        - output_dir: 輸出目錄路徑
        - n_techniques: 技術數量
        - embedding_dim: 嵌入維度
        
    Example:
        >>> from external_sources import build_knowledge_base
        >>> result = build_knowledge_base()
        >>> print(f"已建立 {result['n_techniques']} 個技術嵌入")
    """
    import os
    import config
    
    mitre_csv = getattr(config, "MITRE_TECHNIQUES_CSV", None)
    out_dir = getattr(config, "MITRE_EXTERNAL_KNOWLEDGE_DIR", None)
    bert_model = getattr(config, "BERT_MODEL_NAME", "sentence-bert")
    
    # 檢查是否已存在
    if not force_rebuild and out_dir and os.path.exists(out_dir):
        state_file = os.path.join(out_dir, "state.json")
        if os.path.exists(state_file):
            if verbose:
                print(f"外部知識已存在，跳過建立: {out_dir}")
            # 讀取統計資訊
            try:
                from datasets import load_from_disk
                ds = load_from_disk(out_dir)
                return {
                    "output_dir": out_dir,
                    "n_techniques": len(ds),
                    "embedding_dim": len(ds['embedding'][0]) if len(ds) > 0 else 0,
                    "cached": True
                }
            except Exception:
                pass
    
    # 建立嵌入
    output_dir = build_mitre_raw_embeddings(
        mitre_csv=mitre_csv,
        out_dir=out_dir,
        bert_model=bert_model,
        force_rebuild=force_rebuild,
    )
    
    # * 建立 TF-IDF 向量（用於混合評分）
    try:
        from .build_tfidf import build_mitre_tfidf
        tfidf_dir = getattr(config, 'MITRE_TFIDF_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_TFIDF"))
        build_mitre_tfidf(out_dir=tfidf_dir, mitre_csv=mitre_csv, force_rebuild=force_rebuild)
        if verbose:
            print(f"TF-IDF 向量已建立: {tfidf_dir}")
    except Exception as e:
        if verbose:
            print(f"[Warning] TF-IDF 建立失敗: {e}")
    
    # 讀取統計資訊
    try:
        from datasets import load_from_disk
        ds = load_from_disk(output_dir)
        n_techniques = len(ds)
        embedding_dim = len(ds['embedding'][0]) if len(ds) > 0 else 0
    except Exception:
        n_techniques = 0
        embedding_dim = 0
    
    return {
        "output_dir": output_dir,
        "n_techniques": n_techniques,
        "embedding_dim": embedding_dim,
        "cached": False
    }