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

from .source_manager import ExternalSourceManager
from .text_processor import (
    TextProcessor,
    preprocess_external_source,
    preprocess_log_corpus,
    DEFAULT_STOPWORDS,
    LOG_HIGH_FREQ_WORDS,
    SECURITY_TERMS
)
from .fetchers import MitreFetcher, CapecFetcher, NvdFetcher
from .hmm_clustering import (
    HMMClusterer,
    compute_cluster_similarity,
    rank_suspicious_clusters,
    rank_samples_in_cluster
)
from .concept_uml_pipeline import ConceptUMLPipeline

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
    
    # Data fetchers
    'MitreFetcher',
    'CapecFetcher', 
    'NvdFetcher',
    
    # HMM clustering
    'HMMClusterer',
    'compute_cluster_similarity',
    'rank_suspicious_clusters',
    'rank_samples_in_cluster',
    
    # Full pipeline
    'ConceptUMLPipeline',
]
