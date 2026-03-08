# External Sources Module for Log Labeling Pipeline
#
# Main API:
#     ReferenceBuilder           - multi-source CSV scan, normalise, merge, tokenise
#     build_mitre_raw_embeddings - build MITRE embeddings (called by preprocess)

from .reference_builder import ReferenceBuilder
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
    # Reference source management
    'ReferenceBuilder',
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
