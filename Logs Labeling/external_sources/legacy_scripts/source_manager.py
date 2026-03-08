"""
External Source Manager

Unified interface for loading, processing, and querying multiple 
external threat intelligence sources for concept extraction.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Optional fetchers (may not exist in minimal installs)
MitreFetcher = None
CapecFetcher = None
NvdFetcher = None

try:
    from .fetchers import MitreFetcher, CapecFetcher, NvdFetcher
except Exception:
    try:
        from fetchers import MitreFetcher, CapecFetcher, NvdFetcher
    except Exception:
        pass

# Import BERT API from models
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.bert import get_bert_model, BaseBERTModel
except Exception as e:
    raise ImportError(
        "Failed to import `models.bert`. Ensure the project root 'Logs Labeling' "
        "is on PYTHONPATH (or run from the repository root)."
    ) from e


class ExternalSourceManager:
    """
    Unified manager for external threat intelligence sources.
    
    Provides:
    - Loading and preprocessing of multiple sources (MITRE, CAPEC, CVE, etc.)
    - BERT embedding generation for semantic similarity
    - NMF topic modeling for concept extraction
    - Similarity computation against external knowledge bases
    
    Usage:
        manager = ExternalSourceManager()
        manager.load_source('MITRE', 'path/to/mitre.csv')
        manager.load_source('CAPEC', 'path/to/capec.csv')
        manager.prepare_embeddings()
        
        # Query similarity
        results = manager.compute_similarity(query_embedding, 'MITRE')
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        bert_model: Optional[str] = None,
        nmf_components: Optional[int] = None,
        bert_cache_dir: Optional[str] = None
    ):
        """
        Initialize ExternalSourceManager.
        
        Args:
            data_dir: Base directory for external source data
            bert_model: BERT model key or custom model name
            nmf_components: Number of NMF components for topic modeling
            bert_cache_dir: Cache directory for BERT models
        """

        self.data_dir = data_dir or getattr(config, "REFERENCE_RESOURCES_DIR", None) or os.path.join('data', 'reference_resources')

        self.bert_model_name = bert_model or getattr(config, "EXTERNAL_SOURCES_BERT_MODEL_NAME", None) or 'sentence-bert'
        self.nmf_components = int(
            nmf_components
            if nmf_components is not None
            else getattr(config, "NMF_COMPONENTS", 10)
        )

        self.bert_cache_dir = bert_cache_dir if bert_cache_dir is not None else (
            getattr(config, "EXTERNAL_SOURCES_BERT_CACHE_DIR", None)
        )
        
        # Storage
        self.sources: Dict[str, pd.DataFrame] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.nmf_matrices: Dict[str, np.ndarray] = {}
        self.vocabularies: Dict[str, List[str]] = {}
        
        # Components
        self.bert: BaseBERTModel = get_bert_model(
            self.bert_model_name,
            cache_dir=self.bert_cache_dir,
            auto_load=True
        )
        self._global_vocabulary: Optional[List[str]] = None
        self._vectorizer: Optional[CountVectorizer] = None
        
        # Fetchers
        # Fetchers (optional)
        self.fetchers = {}
        cache_dir = (
            getattr(config, "EXTERNAL_SOURCES_CACHE_DIR", None) if config else None
        ) or os.path.join(self.data_dir, 'cache')
        if MitreFetcher is not None:
            self.fetchers['MITRE'] = MitreFetcher(cache_dir=cache_dir)
        if CapecFetcher is not None:
            self.fetchers['CAPEC'] = CapecFetcher(cache_dir=cache_dir)
        if NvdFetcher is not None:
            self.fetchers['NVD'] = NvdFetcher(cache_dir=cache_dir)
    
    # ==================== Loading Methods ====================
    
    def load_source(
        self,
        name: str,
        filepath: str,
        description_col: str = 'description',
        preprocess: bool = True
    ) -> pd.DataFrame:
        """
        Load an external source from CSV file.
        
        Args:
            name: Source identifier (e.g., 'MITRE', 'CAPEC')
            filepath: Path to CSV file
            description_col: Column containing description text
            preprocess: Whether to preprocess the text
            
        Returns:
            Loaded DataFrame
        """
        print(f"Loading {name} from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        if preprocess:
            # Simple preprocessing: fill NaN and strip whitespace
            if description_col in df.columns:
                df[description_col] = df[description_col].fillna('').astype(str).str.strip()
        
        self.sources[name] = df
        print(f"  Loaded {len(df)} entries from {name}")
        
        return df
    
    def fetch_source(self, name: str, **kwargs) -> pd.DataFrame:
        """
        Fetch an external source using its fetcher.
        
        Args:
            name: Source name ('MITRE', 'CAPEC', 'NVD')
            **kwargs: Additional arguments for the fetcher
            
        Returns:
            Fetched DataFrame
        """
        if name not in self.fetchers:
            raise ValueError(f"Unknown source: {name}. Available: {list(self.fetchers.keys())}")
        
        fetcher = self.fetchers[name]
        df = fetcher.fetch(**kwargs)
        
        if not df.empty:
            # Simple preprocessing
            if 'description' in df.columns:
                df['description'] = df['description'].fillna('').astype(str).str.strip()
            self.sources[name] = df
        
        return df
    
    def load_all_sources(self, sources_dir: Optional[str] = None):
        """
        Load all CSV files from a directory as sources.
        
        Args:
            sources_dir: Directory containing source CSV files
        """
        sources_dir = sources_dir or self.data_dir
        
        if not os.path.exists(sources_dir):
            print(f"Sources directory not found: {sources_dir}")
            return
        
        for filename in os.listdir(sources_dir):
            if filename.endswith('.csv'):
                name = filename.replace('.csv', '').replace('Tokens_V5', '')
                name = name.replace('_', ' ').title().replace(' ', '')
                filepath = os.path.join(sources_dir, filename)
                
                try:
                    self.load_source(name, filepath)
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
    
    # ==================== Embedding Methods ====================
    
    def compute_embeddings(
        self, 
        name: str,
        description_col: str = 'auto'
    ) -> np.ndarray:
        """
        Compute BERT embeddings for a source.
        
        Args:
            name: Source name
            description_col: Column to embed. Use 'auto' to auto-detect best column.
            
        Returns:
            Numpy array of embeddings
        """
        if name not in self.sources:
            raise ValueError(f"Source '{name}' not loaded. Use load_source() first.")
        
        df = self.sources[name]
        
        # Auto-detect best description column for embeddings
        # Prefer raw/original text over preprocessed lowercase text
        if description_col == 'auto':
            # Priority: description_raw > all_text > Description > description > description_clean
            col_priority = ['description_raw', 'all_text', 'Description', 'description', 'description_clean']
            description_col = None
            for col in col_priority:
                if col in df.columns:
                    # Check if column has meaningful content
                    sample = df[col].dropna().head(1)
                    if len(sample) > 0 and len(str(sample.iloc[0])) > 10:
                        description_col = col
                        break
            
            if description_col is None:
                raise ValueError(f"No suitable description column found in {name}. Available: {df.columns.tolist()}")
            
            print(f"  Auto-selected column: '{description_col}'")
        
        # Get texts for embedding
        if description_col in df.columns:
            texts = df[description_col].fillna('').tolist()
        else:
            raise ValueError(f"Column '{description_col}' not found in {name}")
        
        print(f"Computing BERT embeddings for {name} ({len(texts)} entries)...")
        embeddings = self.bert.embed(texts, show_progress=True)
        
        self.embeddings[name] = embeddings
        print(f"  Embedding shape: {embeddings.shape}")
        
        return embeddings
    
    def prepare_all_embeddings(self):
        """Compute embeddings for all loaded sources."""
        for name in self.sources:
            if name not in self.embeddings:
                self.compute_embeddings(name)
    
    # ==================== NMF Topic Modeling ====================
    
    def build_vocabulary(self, min_freq: int = 2) -> List[str]:
        """
        Build a global vocabulary from all sources.
        
        Args:
            min_freq: Minimum token frequency
            
        Returns:
            List of vocabulary words
        """
        all_tokens = []
        
        for name, df in self.sources.items():
            if 'cleaned_tokens' in df.columns:
                for tokens in df['cleaned_tokens']:
                    if isinstance(tokens, str):
                        # Parse string representation of list
                        import ast
                        try:
                            tokens = ast.literal_eval(tokens)
                        except:
                            tokens = tokens.split()
                    if isinstance(tokens, list):
                        all_tokens.extend(tokens)
        
        # Build vocabulary from token frequency
        from collections import Counter
        token_counts = Counter(all_tokens)
        self._global_vocabulary = [token for token, count in token_counts.items() if count >= min_freq]
        
        print(f"Built vocabulary with {len(self._global_vocabulary)} words")
        return self._global_vocabulary
    
    def compute_nmf(
        self,
        name: str,
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute NMF topic matrix for a source.
        
        Args:
            name: Source name
            n_components: Number of NMF components (uses default if None)
            
        Returns:
            NMF topic matrix (n_samples x n_components)
        """
        if name not in self.sources:
            raise ValueError(f"Source '{name}' not loaded")
        
        n_components = n_components or self.nmf_components
        df = self.sources[name]
        
        # Get tokens
        if 'cleaned_tokens' in df.columns:
            token_col = 'cleaned_tokens'
        elif 'tokens' in df.columns:
            token_col = 'tokens'
        else:
            raise ValueError(f"No token column in {name}")
        
        # Convert tokens to text for vectorizer
        texts = []
        for tokens in df[token_col]:
            if isinstance(tokens, str):
                import ast
                try:
                    tokens = ast.literal_eval(tokens)
                except:
                    tokens = tokens.split()
            if isinstance(tokens, list):
                texts.append(' '.join(tokens))
            else:
                texts.append('')
        
        # Vectorize
        if self._vectorizer is None:
            self._vectorizer = CountVectorizer(max_features=5000)
            bow_matrix = self._vectorizer.fit_transform(texts)
        else:
            bow_matrix = self._vectorizer.transform(texts)
        
        # Apply NMF
        print(f"Computing NMF for {name} ({n_components} components)...")
        nmf = NMF(n_components=n_components, random_state=42, max_iter=300)
        nmf_matrix = nmf.fit_transform(bow_matrix)
        
        self.nmf_matrices[name] = nmf_matrix
        print(f"  NMF matrix shape: {nmf_matrix.shape}")
        
        return nmf_matrix
    
    def prepare_all_nmf(self, n_components: Optional[int] = None):
        """Compute NMF for all loaded sources."""
        # First, fit vectorizer on all data
        all_texts = []
        for name, df in self.sources.items():
            token_col = 'cleaned_tokens' if 'cleaned_tokens' in df.columns else 'tokens'
            if token_col in df.columns:
                for tokens in df[token_col]:
                    if isinstance(tokens, str):
                        import ast
                        try:
                            tokens = ast.literal_eval(tokens)
                        except:
                            tokens = tokens.split()
                    if isinstance(tokens, list):
                        all_texts.append(' '.join(tokens))
        
        self._vectorizer = CountVectorizer(max_features=5000)
        self._vectorizer.fit(all_texts)
        
        # Then compute NMF for each source
        for name in self.sources:
            self.compute_nmf(name, n_components)
    
    # ==================== Similarity Methods ====================
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        source_name: str,
        method: str = 'bert',
        top_k: int = 5,
        query_text: Optional[str] = None,
        keyword_boost: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Compute similarity between query and a source.
        
        Args:
            query_embedding: Query embedding vector
            source_name: Source to compare against
            method: 'bert' for BERT similarity, 'nmf' for topic similarity
            top_k: Number of top matches to return
            query_text: Original query text (for keyword boosting)
            keyword_boost: Weight for keyword matching (0-1). 
                          Final score = (1-boost)*semantic + boost*keyword
            
        Returns:
            List of dicts with match info
        """
        if source_name not in self.sources:
            raise ValueError(f"Source '{source_name}' not loaded")
        
        df = self.sources[source_name]
        
        if method == 'bert':
            if source_name not in self.embeddings:
                self.compute_embeddings(source_name)
            source_embeddings = self.embeddings[source_name]
        elif method == 'nmf':
            if source_name not in self.nmf_matrices:
                self.compute_nmf(source_name)
            source_embeddings = self.nmf_matrices[source_name]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute semantic similarity
        similarities = cosine_similarity(query_embedding, source_embeddings)[0]
        
        # Apply keyword boosting if requested
        if keyword_boost > 0 and query_text:
            keyword_scores = self._compute_keyword_scores(query_text, source_name)
            similarities = (1 - keyword_boost) * similarities + keyword_boost * keyword_scores
        
        # Get top-k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            result = {
                'index': int(idx),
                'similarity': float(similarities[idx]),
                'source': source_name,
            }
            
            # Add source-specific info
            row = df.iloc[idx]
            if 'technique' in df.columns:
                result['technique'] = row.get('technique', '')
                result['technique_id'] = row.get('technique_id', '')
            if 'description' in df.columns:
                result['description'] = row.get('description', '')[:200] + '...'
            if 'name' in df.columns:
                result['name'] = row.get('name', '')
            
            results.append(result)
        
        return results
    
    def _compute_keyword_scores(self, query_text: str, source_name: str) -> np.ndarray:
        """
        Compute keyword-based matching scores.
        
        Boosts techniques whose names appear in the query.
        """
        df = self.sources[source_name]
        scores = np.zeros(len(df))
        
        query_lower = query_text.lower()
        # Simple tokenization
        query_tokens = set(query_lower.split())
        
        # Check technique/name column
        name_col = 'technique' if 'technique' in df.columns else 'name' if 'name' in df.columns else None
        
        for idx, row in df.iterrows():
            score = 0.0
            
            # Boost if technique name words appear in query
            if name_col and pd.notna(row.get(name_col)):
                name = str(row[name_col]).lower()
                name_tokens = set(name.split())
                
                # Count matching tokens
                matching = len(query_tokens & name_tokens)
                if matching > 0:
                    score = matching / max(len(name_tokens), 1)
                
                # Extra boost if full name is in query
                if name in query_lower:
                    score = 1.0
            
            scores[idx] = score
        
        # Normalize scores to 0-1 range
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def compute_similarity_with_keywords(
        self,
        query_text: str,
        source_name: str,
        top_k: int = 5,
        semantic_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Compute similarity using both semantic embeddings and keyword matching.
        
        This is the recommended method for matching user queries to techniques.
        
        Args:
            query_text: Query text string
            source_name: Source to compare against
            top_k: Number of top matches to return
            semantic_weight: Weight for semantic similarity (1-weight for keywords)
            
        Returns:
            List of dicts with match info
        """
        # Compute query embedding
        query_emb = self.bert.embed([query_text], show_progress=False)[0]
        
        return self.compute_similarity(
            query_emb, 
            source_name, 
            method='bert',
            top_k=top_k,
            query_text=query_text,
            keyword_boost=1.0 - semantic_weight
        )
    
    def identify_source(
        self,
        query_embedding: np.ndarray,
        method: str = 'bert',
        threshold: float = 0.3
    ) -> Dict[str, Dict[str, Any]]:
        """
        Identify which external source best matches a query.
        
        Args:
            query_embedding: Query embedding vector
            method: Similarity method ('bert' or 'nmf')
            threshold: Minimum similarity threshold
            
        Returns:
            Dict mapping source names to match info
        """
        results = {}
        
        for source_name in self.sources:
            matches = self.compute_similarity(
                query_embedding, source_name, method=method, top_k=1
            )
            
            if matches:
                best_match = matches[0]
                if best_match['similarity'] >= threshold:
                    results[source_name] = best_match
        
        return results
    
    def hybrid_similarity(
        self,
        query_bert: np.ndarray,
        query_nmf: np.ndarray,
        source_name: str,
        alpha: float = 0.5,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Compute hybrid similarity using both BERT and NMF.
        
        Args:
            query_bert: BERT embedding of query
            query_nmf: NMF topic vector of query
            source_name: Source to compare against
            alpha: Weight for BERT similarity (1-alpha for NMF)
            top_k: Number of top matches
            
        Returns:
            List of match results
        """
        # Ensure embeddings exist
        if source_name not in self.embeddings:
            self.compute_embeddings(source_name)
        if source_name not in self.nmf_matrices:
            self.compute_nmf(source_name)
        
        # Get source data
        bert_emb = self.embeddings[source_name]
        nmf_emb = self.nmf_matrices[source_name]
        df = self.sources[source_name]
        
        # Reshape queries if needed
        if query_bert.ndim == 1:
            query_bert = query_bert.reshape(1, -1)
        if query_nmf.ndim == 1:
            query_nmf = query_nmf.reshape(1, -1)
        
        # Compute both similarities
        bert_sim = cosine_similarity(query_bert, bert_emb)[0]
        nmf_sim = cosine_similarity(query_nmf, nmf_emb)[0]
        
        # Combine
        combined_sim = alpha * bert_sim + (1 - alpha) * nmf_sim
        
        # Get top-k
        top_indices = np.argsort(combined_sim)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            result = {
                'index': int(idx),
                'combined_similarity': float(combined_sim[idx]),
                'bert_similarity': float(bert_sim[idx]),
                'nmf_similarity': float(nmf_sim[idx]),
                'source': source_name,
            }
            
            row = df.iloc[idx]
            if 'technique' in df.columns:
                result['technique'] = row.get('technique', '')
            if 'description' in df.columns:
                result['description'] = row.get('description', '')[:200] + '...'
            
            results.append(result)
        
        return results
    
    # ==================== Utility Methods ====================
    
    def get_source_info(self) -> Dict[str, Dict[str, Any]]:
        """Get summary information about loaded sources."""
        info = {}
        
        for name, df in self.sources.items():
            info[name] = {
                'entries': len(df),
                'columns': list(df.columns),
                'has_embeddings': name in self.embeddings,
                'has_nmf': name in self.nmf_matrices,
            }
            if name in self.embeddings:
                info[name]['embedding_dim'] = self.embeddings[name].shape[1]
            if name in self.nmf_matrices:
                info[name]['nmf_dim'] = self.nmf_matrices[name].shape[1]
        
        return info
    
    def save_processed(self, output_dir: str):
        """
        Save processed sources and embeddings.
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in self.sources.items():
            df.to_csv(os.path.join(output_dir, f'{name}_processed.csv'), index=False)
        
        for name, emb in self.embeddings.items():
            np.save(os.path.join(output_dir, f'{name}_embeddings.npy'), emb)
        
        for name, nmf in self.nmf_matrices.items():
            np.save(os.path.join(output_dir, f'{name}_nmf.npy'), nmf)
        
        print(f"Saved processed data to {output_dir}")
    
    def load_processed(self, input_dir: str):
        """
        Load previously processed sources and embeddings.
        
        Args:
            input_dir: Directory with processed files
        """
        for filename in os.listdir(input_dir):
            if filename.endswith('_processed.csv'):
                name = filename.replace('_processed.csv', '')
                self.sources[name] = pd.read_csv(os.path.join(input_dir, filename))
            elif filename.endswith('_embeddings.npy'):
                name = filename.replace('_embeddings.npy', '')
                self.embeddings[name] = np.load(os.path.join(input_dir, filename))
            elif filename.endswith('_nmf.npy'):
                name = filename.replace('_nmf.npy', '')
                self.nmf_matrices[name] = np.load(os.path.join(input_dir, filename))
        
        print(f"Loaded: {list(self.sources.keys())}")
