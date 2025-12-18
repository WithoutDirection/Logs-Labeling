"""
Text Processing Utilities for External Sources
Provides tokenization, cleaning, and embedding generation for threat intelligence text.
"""

import re
from typing import List, Optional, Set, Dict
from collections import Counter
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Stopwords for text cleaning
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

# Log-specific high-frequency words to filter (common in Windows logs)
LOG_HIGH_FREQ_WORDS = {
    'authority', 'system', 'users', 'microsoft', 'path', 'currentversion',
    'windows', 'software', 'program', 'files', 'local', 'appdata', 'temp',
    'administrator', 'default', 'desktop', 'documents', 'c', 'c32', 'd',
    'user', 'application', 'data', 'roaming', 'programdata', 'nt'
}

# Security-specific terms to keep (not remove as stopwords)
SECURITY_TERMS = {
    'attack', 'exploit', 'vulnerability', 'malware', 'threat', 'adversary',
    'persistence', 'execution', 'privilege', 'escalation', 'defense', 'evasion',
    'credential', 'access', 'discovery', 'lateral', 'movement', 'collection',
    'exfiltration', 'command', 'control', 'impact', 'injection', 'bypass',
    'reconnaissance', 'resource', 'development', 'initial', 'access'
}


class TextProcessor:
    """
    Text processing utilities for threat intelligence sources.
    
    Preprocessing pipeline (following ConceptUML paper):
    1. Lowercasing all words
    2. Tokenizing sentences  
    3. Removing stopwords
    4. Zipf's law filtering - removing top 5% high-frequency words
    
    Provides methods for:
    - Tokenization (standard and command-line specific)
    - Text cleaning and normalization
    - Stopword removal
    - Zipf's law high-frequency word filtering
    - BERT embedding generation
    """
    
    def __init__(
        self, 
        stopwords: Optional[Set[str]] = None,
        keep_security_terms: Optional[bool] = None,
        bert_model_name: Optional[str] = None,
        zipf_percentile: Optional[float] = None,
        use_log_high_freq: Optional[bool] = None,
        bert_cache_dir: Optional[str] = None,
        embedding_normalize: Optional[bool] = None,
        embedding_batch_size: Optional[int] = None,
    ):
        """
        Initialize TextProcessor.
        
        Args:
            stopwords: Custom stopwords set (uses default if None)
            keep_security_terms: If True, don't remove security-related terms
            bert_model_name: SentenceTransformer model name for embeddings
            zipf_percentile: Top percentile of high-frequency words to filter (default 5%)
            use_log_high_freq: If True, include common log high-frequency words
        """
        self.stopwords = stopwords or DEFAULT_STOPWORDS.copy()
        self.keep_security_terms = (
            keep_security_terms
            if keep_security_terms is not None
            else getattr(config, "EXTERNAL_SOURCES_KEEP_SECURITY_TERMS", True)
        )
        self.bert_model_name = (
            bert_model_name
            if bert_model_name is not None
            else getattr(config, "EXTERNAL_SOURCES_BERT_MODEL_NAME", "sentence-bert")
        )
        self.bert_cache_dir = (
            bert_cache_dir
            if bert_cache_dir is not None
            else getattr(config, "EXTERNAL_SOURCES_BERT_CACHE_DIR", None)
        )
        self.embedding_normalize = (
            embedding_normalize
            if embedding_normalize is not None
            else getattr(config, "EXTERNAL_SOURCES_EMBED_NORMALIZE", True)
        )
        self.embedding_batch_size = (
            int(embedding_batch_size)
            if embedding_batch_size is not None
            else int(getattr(config, "EXTERNAL_SOURCES_EMBED_BATCH_SIZE", 32))
        )
        self.zipf_percentile = (
            float(zipf_percentile)
            if zipf_percentile is not None
            else float(getattr(config, "EXTERNAL_SOURCES_ZIPF_PERCENTILE", 0.05))
        )
        self._bert_model = None  # Lazy loading (BaseBERTModel)
        
        # Zipf's law filtering state
        self._word_frequencies: Counter = Counter()
        self._zipf_filter_words: Set[str] = set()
        self._corpus_fitted = False
        
        # Include predefined log high-frequency words
        resolved_use_log_high_freq = (
            use_log_high_freq
            if use_log_high_freq is not None
            else (getattr(config, "EXTERNAL_SOURCES_USE_LOG_HIGH_FREQ", True) if config else True)
        )

        if resolved_use_log_high_freq:
            self._zipf_filter_words.update(LOG_HIGH_FREQ_WORDS)
        
        if keep_security_terms:
            self.stopwords -= SECURITY_TERMS
    
    @property
    def bert_model(self):
        """Lazy-load the unified BERT embedding model (models.bert)."""
        if self._bert_model is None:
            # Make project root importable when running as a script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from models.bert import get_bert_model

            self._bert_model = get_bert_model(
                self.bert_model_name,
                cache_dir=self.bert_cache_dir,
                auto_load=True,
            )
        return self._bert_model

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True,
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """Generate embeddings for texts using the unified BERT backend."""
        if texts is None:
            return np.empty((0, 0), dtype=float)

        batch_size = batch_size if batch_size is not None else self.embedding_batch_size
        normalize = normalize if normalize is not None else self.embedding_normalize
        return self.bert_model.embed(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=normalize,
        )
    
    def fit_zipf_filter(self, texts: List[str], percentile: Optional[float] = None) -> Set[str]:
        """
        Fit Zipf's law filter on corpus to identify high-frequency words.
        
        According to Zipf's law, the frequency of a word is inversely proportional
        to its rank. The top words are typically common, domain-agnostic terms that
        don't carry meaningful information for classification.
        
        Args:
            texts: List of text documents to analyze
            percentile: Top percentile of words to filter (default: self.zipf_percentile)
            
        Returns:
            Set of high-frequency words to filter
        """
        percentile = percentile or self.zipf_percentile
        
        # Count word frequencies across corpus
        self._word_frequencies = Counter()
        for text in texts:
            tokens = self._raw_tokenize(text)
            self._word_frequencies.update(tokens)
        
        # Calculate cutoff for top percentile
        if not self._word_frequencies:
            return set()
        
        total_unique = len(self._word_frequencies)
        cutoff_count = max(1, int(total_unique * percentile))
        
        # Get most common words
        most_common = self._word_frequencies.most_common(cutoff_count)
        high_freq_words = {word for word, _ in most_common}
        
        # Don't filter security-related terms even if they're frequent
        if self.keep_security_terms:
            high_freq_words -= SECURITY_TERMS
        
        self._zipf_filter_words.update(high_freq_words)
        self._corpus_fitted = True
        
        return high_freq_words
    
    def _raw_tokenize(self, text: str) -> List[str]:
        """
        Basic tokenization for frequency counting (lowercased, no filtering).
        
        Args:
            text: Input text
            
        Returns:
            List of lowercase tokens
        """
        if not isinstance(text, str):
            return []
        
        text = text.lower()
        # Split on non-alphanumeric characters
        tokens = re.findall(r'[a-z0-9_\-\.]+', text)
        # Filter very short tokens
        return [t for t in tokens if len(t) > 1]
    
    def get_zipf_filtered_words(self) -> Set[str]:
        """Get the current set of high-frequency words being filtered."""
        return self._zipf_filter_words.copy()
    
    def get_word_frequencies(self) -> Dict[str, int]:
        """Get word frequency counts from fitted corpus."""
        return dict(self._word_frequencies)
    
    def is_fitted(self) -> bool:
        """Check if Zipf filter has been fitted on a corpus."""
        return self._corpus_fitted
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove citations like (Citation: xxx)
        text = re.sub(r'\(citation:[^)]+\)', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove markdown links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove code blocks
        text = re.sub(r'`[^`]+`', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def tokenize(
        self, 
        text: str, 
        remove_stopwords: bool = True,
        apply_zipf_filter: bool = True
    ) -> List[str]:
        """
        Tokenize text into words with full preprocessing pipeline.
        
        Preprocessing steps (ConceptUML paper):
        1. Lowercasing (in clean_text)
        2. Tokenization
        3. Stopword removal
        4. Zipf's law high-frequency word filtering
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            apply_zipf_filter: Whether to filter high-frequency words via Zipf's law
            
        Returns:
            List of tokens
        """
        # Clean first (includes lowercasing)
        text = self.clean_text(text)
        
        # Split on non-alphanumeric characters but keep meaningful ones
        tokens = re.findall(r'[a-z0-9_\-\.]+', text)
        
        # Filter tokens
        tokens = [t for t in tokens if len(t) > 1]  # Remove single chars
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        # Apply Zipf's law filtering (remove high-frequency words)
        if apply_zipf_filter and self._zipf_filter_words:
            tokens = [t for t in tokens if t not in self._zipf_filter_words]
        
        return tokens
    
    def tokenize_command(self, command: str) -> List[str]:
        """
        Tokenize command-line strings with special handling.
        
        Preserves important patterns like:
        - File paths and extensions
        - Registry keys
        - Executable names
        - Flags and parameters
        
        Args:
            command: Command-line string
            
        Returns:
            List of tokens
        """
        if not isinstance(command, str):
            return []
        
        tokens = []
        
        # Extract file extensions
        extensions = re.findall(r'\.[a-zA-Z0-9]{1,4}\b', command)
        tokens.extend([ext.lower() for ext in extensions])
        
        # Extract executable names
        exes = re.findall(r'[a-zA-Z0-9_\-]+\.exe', command, re.IGNORECASE)
        tokens.extend([exe.lower() for exe in exes])
        
        # Extract DLL names
        dlls = re.findall(r'[a-zA-Z0-9_\-]+\.dll', command, re.IGNORECASE)
        tokens.extend([dll.lower() for dll in dlls])
        
        # Extract registry hives
        reg_hives = re.findall(r'HKLM|HKCU|HKCR|HKU|HKEY_[A-Z_]+', command, re.IGNORECASE)
        tokens.extend([h.lower() for h in reg_hives])
        
        # Extract path components
        path_parts = re.findall(r'\\([a-zA-Z0-9_\-]+)\\', command)
        tokens.extend([p.lower() for p in path_parts])
        
        # Extract flags (e.g., -flag, /flag)
        flags = re.findall(r'[\-\/]([a-zA-Z][a-zA-Z0-9]*)', command)
        tokens.extend([f.lower() for f in flags])
        
        # Standard word tokenization for remaining content
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]{2,}\b', command)
        tokens.extend([w.lower() for w in words])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for t in tokens:
            # Apply stopword and Zipf filtering
            if t not in seen and t not in self.stopwords and t not in self._zipf_filter_words:
                seen.add(t)
                unique_tokens.append(t)
        
        return unique_tokens
    
    def extract_attack_patterns(self, text: str) -> List[str]:
        """
        Extract common attack pattern keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of detected attack patterns
        """
        patterns = []
        text_lower = text.lower()
        
        # Common attack patterns to look for
        attack_keywords = {
            'injection': ['sql injection', 'command injection', 'code injection', 
                         'dll injection', 'process injection', 'ldap injection'],
            'overflow': ['buffer overflow', 'stack overflow', 'heap overflow', 
                        'integer overflow'],
            'bypass': ['authentication bypass', 'authorization bypass', 
                      'security bypass', 'filter bypass'],
            'execution': ['remote code execution', 'arbitrary code execution',
                         'command execution', 'script execution'],
            'escalation': ['privilege escalation', 'elevation of privilege'],
            'exfiltration': ['data exfiltration', 'data theft'],
            'persistence': ['persistence mechanism', 'backdoor'],
            'lateral_movement': ['lateral movement', 'pivoting'],
            'credential': ['credential theft', 'credential dumping', 
                          'credential harvesting', 'password stealing'],
        }
        
        for category, keywords in attack_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    patterns.append(keyword)
        
        return patterns
    
    def create_bow_vocabulary(self, token_lists: List[List[str]], min_freq: int = 2) -> List[str]:
        """
        Create a bag-of-words vocabulary from token lists.
        
        Args:
            token_lists: List of token lists
            min_freq: Minimum frequency for inclusion
            
        Returns:
            List of vocabulary words
        """
        from collections import Counter
        
        # Count all tokens
        token_counts = Counter()
        for tokens in token_lists:
            token_counts.update(tokens)
        
        # Filter by minimum frequency
        vocab = [token for token, count in token_counts.items() if count >= min_freq]
        
        return sorted(vocab)
    
    def tokens_to_bow(self, tokens: List[str], vocabulary: List[str]) -> np.ndarray:
        """
        Convert token list to bag-of-words vector.
        
        Args:
            tokens: List of tokens
            vocabulary: Vocabulary list
            
        Returns:
            BoW vector as numpy array
        """
        vocab_set = set(vocabulary)
        vocab_idx = {word: i for i, word in enumerate(vocabulary)}
        
        bow = np.zeros(len(vocabulary))
        for token in tokens:
            if token in vocab_set:
                bow[vocab_idx[token]] += 1
        
        return bow


def preprocess_external_source(
    df,
    description_col: str = 'description',
    processor: Optional[TextProcessor] = None,
    fit_zipf: bool = True,
    zipf_percentile: float = 0.05
):
    """
    Preprocess an external source DataFrame with full ConceptUML pipeline.
    
    Preprocessing pipeline:
    1. Lowercasing all words
    2. Tokenizing sentences
    3. Removing stopwords
    4. Zipf's law filtering (top 5% high-frequency words)
    
    Adds columns:
    - description_clean: Cleaned description text
    - tokens: Tokenized description (raw)
    - cleaned_tokens: Tokens with full preprocessing applied
    
    Args:
        df: Input DataFrame
        description_col: Column containing description text
        processor: TextProcessor instance (creates new if None)
        fit_zipf: Whether to fit Zipf filter on this corpus
        zipf_percentile: Top percentile of words to filter (default 5%)
        
    Returns:
        DataFrame with added columns
    """
    if processor is None:
        processor = TextProcessor(zipf_percentile=zipf_percentile)
    
    import pandas as pd
    
    df = df.copy()
    
    # Clean descriptions
    df['description_clean'] = df[description_col].apply(processor.clean_text)
    
    # Fit Zipf filter on corpus if requested
    if fit_zipf:
        texts = df[description_col].dropna().tolist()
        high_freq_words = processor.fit_zipf_filter(texts, zipf_percentile)
        print(f"Zipf's law filter: identified {len(high_freq_words)} high-frequency words to filter")
    
    # Tokenize (raw, no filtering)
    df['tokens'] = df[description_col].apply(
        lambda x: processor.tokenize(x, remove_stopwords=False, apply_zipf_filter=False)
    )
    
    # Cleaned tokens (full preprocessing: stopwords + Zipf filtering)
    df['cleaned_tokens'] = df[description_col].apply(
        lambda x: processor.tokenize(x, remove_stopwords=True, apply_zipf_filter=True)
    )
    
    return df


def preprocess_log_corpus(
    texts: List[str],
    processor: Optional[TextProcessor] = None,
    zipf_percentile: float = 0.05
) -> tuple:
    """
    Preprocess a log corpus with Zipf's law filtering.
    
    This function fits the Zipf filter on the provided corpus and returns
    processed tokens along with the fitted processor.
    
    Args:
        texts: List of log text entries
        processor: TextProcessor instance (creates new if None)
        zipf_percentile: Top percentile of words to filter (default 5%)
        
    Returns:
        Tuple of (processed_texts, processor):
        - processed_texts: List of joined token strings
        - processor: Fitted TextProcessor instance (for reuse)
    """
    if processor is None:
        processor = TextProcessor(zipf_percentile=zipf_percentile)
    
    # Fit Zipf filter on log corpus
    high_freq_words = processor.fit_zipf_filter(texts, zipf_percentile)
    print(f"Zipf's law filter: identified {len(high_freq_words)} high-frequency words")
    print(f"Sample filtered words: {list(high_freq_words)[:10]}")
    
    # Process each text with full pipeline
    processed_texts = []
    for text in texts:
        tokens = processor.tokenize(text, remove_stopwords=True, apply_zipf_filter=True)
        processed_texts.append(' '.join(tokens))
    
    return processed_texts, processor
