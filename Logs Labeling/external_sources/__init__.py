# External Sources Module for Log Labeling Pipeline (Refactored)
# 
# 本模組提供 MITRE 等外部知識來源的處理工具。
#
# Stage I 整合後，原本分開的 `build_knowledge_base()` 已被移除，
# Reference 處理整合至 `preprocess.process_all_inputs()` 統一入口。
#
# 主要 API:
#     build_mitre_raw_embeddings() - 建立 MITRE 嵌入 (被 preprocess 呼叫)

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

# Optional modules
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


# =============================================================================
# Legacy Compatibility Wrapper (Deprecated)
# =============================================================================

def build_knowledge_base(
    force_rebuild: bool = False,
    verbose: bool = True
) -> dict:
    """
    [DEPRECATED] 此函式已整合至 preprocess.process_all_inputs()
    
    保留此 API 以維持向下相容，內部轉發至新函式。
    """
    import warnings
    warnings.warn(
        "build_knowledge_base() is deprecated. Use preprocess.process_all_inputs() instead.",
        DeprecationWarning
    )
    
    import os
    import config
    
    mitre_csv = getattr(config, "MITRE_TECHNIQUES_CSV", None)
    out_dir = getattr(config, "MITRE_EXTERNAL_KNOWLEDGE_DIR", None)
    bert_model = getattr(config, "BERT_MODEL_NAME", "sentence-bert")
    
    # 檢查快取
    if not force_rebuild and out_dir and os.path.exists(out_dir):
        state_file = os.path.join(out_dir, "state.json")
        if os.path.exists(state_file):
            if verbose:
                print(f"[Deprecated API] 使用快取: {out_dir}")
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