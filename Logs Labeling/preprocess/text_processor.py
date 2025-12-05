"""
Unified Text Processor for Log Labeling Pipeline

This module provides a centralized text processing interface that integrates
with both the preprocessing pipeline and external source analysis.

Re-exports from external_sources.text_processor for backwards compatibility,
and adds log-specific preprocessing utilities.
"""

import sys
import os

# Ensure external_sources is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Re-export everything from external_sources.text_processor
try:
    from external_sources.text_processor import (
        TextProcessor,
        preprocess_external_source,
        preprocess_log_corpus,
        DEFAULT_STOPWORDS,
        LOG_HIGH_FREQ_WORDS,
        SECURITY_TERMS,
    )
except ImportError:
    # Fallback for standalone usage
    from typing import List, Optional, Set, Dict
    from collections import Counter
    import re
    import numpy as np
    
    DEFAULT_STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
        'because', 'until', 'while', 'this', 'that', 'these', 'those', 'am',
        'it', 'its', 'itself', 'they', 'them', 'their', 'what', 'which', 'who',
        'whom', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself'
    }
    
    LOG_HIGH_FREQ_WORDS = {
        'authority', 'system', 'users', 'microsoft', 'path', 'currentversion',
        'windows', 'software', 'program', 'files', 'local', 'appdata', 'temp',
        'administrator', 'default', 'desktop', 'documents', 'c', 'c32', 'd',
        'user', 'application', 'data', 'roaming', 'programdata', 'nt'
    }
    
    SECURITY_TERMS = {
        'attack', 'exploit', 'vulnerability', 'malware', 'threat', 'adversary',
        'persistence', 'execution', 'privilege', 'escalation', 'defense', 'evasion',
        'credential', 'access', 'discovery', 'lateral', 'movement', 'collection',
        'exfiltration', 'command', 'control', 'impact', 'injection', 'bypass',
        'reconnaissance', 'resource', 'development', 'initial', 'access'
    }
    
    class TextProcessor:
        """Minimal TextProcessor fallback when external_sources not available."""
        
        def __init__(
            self, 
            stopwords: Optional[Set[str]] = None,
            keep_security_terms: bool = True,
            bert_model_name: str = 'bert-base-nli-mean-tokens',
            zipf_percentile: float = 0.05,
            use_log_high_freq: bool = True
        ):
            self.stopwords = stopwords or DEFAULT_STOPWORDS.copy()
            self.bert_model_name = bert_model_name
            self.zipf_percentile = zipf_percentile
            self._bert_model = None
            self._word_frequencies: Counter = Counter()
            self._zipf_filter_words: Set[str] = set()
            self._corpus_fitted = False
            
            if use_log_high_freq:
                self._zipf_filter_words.update(LOG_HIGH_FREQ_WORDS)
            if keep_security_terms:
                self.stopwords -= SECURITY_TERMS
        
        @property
        def bert_model(self):
            if self._bert_model is None:
                from sentence_transformers import SentenceTransformer
                self._bert_model = SentenceTransformer(self.bert_model_name)
            return self._bert_model
        
        def clean_text(self, text: str) -> str:
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r'\(citation:[^)]+\)', '', text)
            text = re.sub(r'https?://\S+', '', text)
            text = re.sub(r'<[^>]+>', '', text)
            return ' '.join(text.split()).strip()
        
        def tokenize(self, text: str, remove_stopwords: bool = True, 
                     apply_zipf_filter: bool = True) -> List[str]:
            text = self.clean_text(text)
            tokens = re.findall(r'[a-z0-9_\-\.]+', text)
            tokens = [t for t in tokens if len(t) > 1]
            if remove_stopwords:
                tokens = [t for t in tokens if t not in self.stopwords]
            if apply_zipf_filter and self._zipf_filter_words:
                tokens = [t for t in tokens if t not in self._zipf_filter_words]
            return tokens
        
        def fit_zipf_filter(self, texts: List[str], percentile: Optional[float] = None) -> Set[str]:
            percentile = percentile or self.zipf_percentile
            self._word_frequencies = Counter()
            for text in texts:
                tokens = re.findall(r'[a-z0-9_\-\.]+', text.lower())
                self._word_frequencies.update([t for t in tokens if len(t) > 1])
            
            if not self._word_frequencies:
                return set()
            
            cutoff = max(1, int(len(self._word_frequencies) * percentile))
            high_freq = {w for w, _ in self._word_frequencies.most_common(cutoff)}
            self._zipf_filter_words.update(high_freq)
            self._corpus_fitted = True
            return high_freq
        
        def generate_embeddings(self, texts: List[str], show_progress: bool = True,
                                batch_size: int = 32) -> np.ndarray:
            return self.bert_model.encode(texts, show_progress_bar=show_progress,
                                          batch_size=batch_size)
        
        def get_zipf_filtered_words(self) -> Set[str]:
            return self._zipf_filter_words.copy()
        
        def is_fitted(self) -> bool:
            return self._corpus_fitted
    
    def preprocess_log_corpus(texts, processor=None, zipf_percentile=0.05):
        if processor is None:
            processor = TextProcessor(zipf_percentile=zipf_percentile)
        processor.fit_zipf_filter(texts, zipf_percentile)
        processed = [' '.join(processor.tokenize(t)) for t in texts]
        return processed, processor
    
    def preprocess_external_source(df, description_col='description', 
                                   processor=None, fit_zipf=True, zipf_percentile=0.05):
        import pandas as pd
        if processor is None:
            processor = TextProcessor(zipf_percentile=zipf_percentile)
        df = df.copy()
        df['description_clean'] = df[description_col].apply(processor.clean_text)
        if fit_zipf:
            processor.fit_zipf_filter(df[description_col].dropna().tolist(), zipf_percentile)
        df['tokens'] = df[description_col].apply(
            lambda x: processor.tokenize(x, remove_stopwords=False, apply_zipf_filter=False))
        df['cleaned_tokens'] = df[description_col].apply(
            lambda x: processor.tokenize(x, remove_stopwords=True, apply_zipf_filter=True))
        return df


# Import config for default settings
try:
    import config
    DEFAULT_ZIPF_PERCENTILE = getattr(config, 'ZIPF_PERCENTILE', 0.05)
    DEFAULT_USE_LOG_HIGH_FREQ = getattr(config, 'USE_LOG_HIGH_FREQ', True)
except ImportError:
    DEFAULT_ZIPF_PERCENTILE = 0.05
    DEFAULT_USE_LOG_HIGH_FREQ = True


def create_text_processor(
    bert_model_name: str = None,
    zipf_percentile: float = None,
    use_log_high_freq: bool = None
) -> TextProcessor:
    """
    Factory function to create a TextProcessor with config defaults.
    
    Args:
        bert_model_name: BERT model for embeddings (default from config or 'bert-base-nli-mean-tokens')
        zipf_percentile: Top % of words to filter (default from config or 0.05)
        use_log_high_freq: Include common log words in filter (default from config or True)
    
    Returns:
        Configured TextProcessor instance
    """
    return TextProcessor(
        bert_model_name=bert_model_name or 'bert-base-nli-mean-tokens',
        zipf_percentile=zipf_percentile if zipf_percentile is not None else DEFAULT_ZIPF_PERCENTILE,
        use_log_high_freq=use_log_high_freq if use_log_high_freq is not None else DEFAULT_USE_LOG_HIGH_FREQ
    )


__all__ = [
    'TextProcessor',
    'preprocess_external_source', 
    'preprocess_log_corpus',
    'create_text_processor',
    'DEFAULT_STOPWORDS',
    'LOG_HIGH_FREQ_WORDS',
    'SECURITY_TERMS',
    'DEFAULT_ZIPF_PERCENTILE',
    'DEFAULT_USE_LOG_HIGH_FREQ',
]
