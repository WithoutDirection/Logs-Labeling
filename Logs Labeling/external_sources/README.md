# External Sources Module

This module provides tools for loading, processing, and extracting concepts from external threat intelligence sources for the Log Labeling pipeline.

## Overview

The module supports:
- **MITRE ATT&CK**: Adversary techniques and tactics
- **CAPEC**: Common attack patterns
- **NVD/CVE**: Vulnerability descriptions
- **Sigma Rules**: Detection rule keywords (optional)

## Preprocessing Pipeline (ConceptUML Paper)

1. **Lowercasing** all words
2. **Tokenization** of sentences
3. **Stopword removal** (English stopwords)
4. **Zipf's law filtering** - remove top 5% high-frequency words

Example filtered words: `authority`, `system`, `users`, `microsoft`, `path`, `currentversion`, `windows`, `software`, `program`, `files`

```python
from external_sources import TextProcessor, preprocess_log_corpus

# Initialize with Zipf's law
processor = TextProcessor(
    zipf_percentile=0.05,  # Filter top 5%
    use_log_high_freq=True  # Include common log terms
)

# Preprocess corpus (fits Zipf filter)
processed_texts, processor = preprocess_log_corpus(
    raw_texts, 
    processor=processor,
    zipf_percentile=0.05
)

# Check filtered words
print(processor.get_zipf_filtered_words())
```

## Directory Structure

```
external_sources/
├── __init__.py           # Module exports
├── source_manager.py     # Main ExternalSourceManager class
├── text_processor.py     # Text tokenization, Zipf's law, and embedding
├── hmm_clustering.py     # HMM clustering with optimal state search
├── concept_uml_pipeline.py # Full ConceptUML pipeline
├── fetchers.py           # Data fetchers for online sources
├── examples.py           # Usage examples
└── README.md             # This file

data/reference_resources/
├── MitreTechniquesTokens_V5.csv   # MITRE ATT&CK data
├── CapecTokens_V5.csv             # CAPEC data
└── cache/                          # Cached fetched data
```

## Quick Start

### 1. Basic Loading

```python
from external_sources import ExternalSourceManager

# Initialize
manager = ExternalSourceManager(
    data_dir='data/reference_resources',
    nmf_components=10
)

# Load sources
manager.load_source('MITRE', 'data/reference_resources/MitreTechniquesTokens_V5.csv')
manager.load_source('CAPEC', 'data/reference_resources/CapecTokens_V5.csv')
```

### 2. Compute Embeddings

```python
# BERT embeddings for semantic similarity
manager.prepare_all_embeddings()

# NMF for topic modeling
manager.prepare_all_nmf(n_components=10)
```

### 3. Query Similarity

```python
from external_sources import TextProcessor

processor = TextProcessor()

# Embed a log/query
query = "schtasks.exe /create scheduled task"
query_emb = processor.generate_embeddings([query])[0]

# Find similar techniques
results = manager.compute_similarity(query_emb, 'MITRE', top_k=5)

for r in results:
    print(f"{r['technique']}: {r['similarity']:.3f}")
```

### 4. Source Identification

```python
# Which source best matches?
results = manager.identify_source(query_emb, threshold=0.3)

for source, match in results.items():
    print(f"{source}: {match['similarity']:.3f}")
```

## Main Classes

### `ExternalSourceManager`

Central manager for all external sources.

| Method | Description |
|--------|-------------|
| `load_source(name, path)` | Load CSV file as source |
| `fetch_source(name)` | Fetch from online API |
| `prepare_all_embeddings()` | Compute BERT embeddings |
| `prepare_all_nmf()` | Compute NMF topic matrices |
| `compute_similarity(query, source)` | Find similar entries |
| `identify_source(query)` | Find best matching source |
| `hybrid_similarity(bert, nmf, source)` | Combined similarity |

### `TextProcessor`

Text preprocessing with Zipf's law and embedding utilities.

| Method | Description |
|--------|-------------|
| `fit_zipf_filter(texts)` | Fit Zipf's law on corpus, identify top N% words |
| `clean_text(text)` | Remove citations, URLs, lowercase |
| `tokenize(text, apply_zipf_filter)` | Full preprocessing pipeline |
| `tokenize_command(cmd)` | Command-line tokenization |
| `generate_embeddings(texts)` | BERT embeddings |
| `extract_attack_patterns(text)` | Find attack keywords |
| `get_zipf_filtered_words()` | Get filtered high-frequency words |
| `is_fitted()` | Check if Zipf filter has been fitted |



### Fetchers

| Class | Source | API |
|-------|--------|-----|
| `MitreFetcher` | MITRE ATT&CK | GitHub STIX |
| `CapecFetcher` | CAPEC | Local XML |
| `NvdFetcher` | NVD/CVE | NVD API 2.0 |
| `SigmaRulesFetcher` | Sigma | Local YAML |

## Integration with Pipeline (Three-Stage Architecture)

> **注意：** 自 v2.0 起，Pipeline 已重構為三階段架構。
> - Stage I: 統一輸入處理（Log + Reference + TF-IDF）
> - Stage II: 異常偵測
> - Stage III: Per-Dataset 處理（NMF → HMM → Auto Labeling）
>
> 原本獨立的 `build_knowledge_base()` 已被整合到 Stage I 的 `process_all_inputs()` 中。

### Pipeline Stage I: Reference Knowledge Preprocessing

在 Stage I 中，MITRE 等 Reference 資料與 Log 資料一起進行前處理：

```python
# In Pipeline.py STAGE_I
from preprocess import process_all_inputs
from precompute_log_tfidf import run_tfidf_pipeline

def STAGE_I():
    """Stage I: 統一處理所有輸入"""
    # 1. 處理 Log + Reference 資料
    process_all_inputs()
    
    # 2. TF-IDF 特徵建立（Reference + Log）
    run_tfidf_pipeline(force_rebuild=False)
```

### Pipeline Stage III: Per-Dataset Processing

在 Stage III 中，使用 Reference 知識進行 NMF 聯合訓練與自動標註（含 Hybrid Scoring）：

```python
# In Pipeline.py STAGE_III
from conception_extraction import ConceptExtractor
from sequence_clustering import SequenceClusterer
from auto_labeling import AutoLabeler

def STAGE_III():
    """Stage III: Per-Dataset 處理"""
    # Step a: NMF 概念提取（載入 Reference 知識）
    extractor = ConceptExtractor(n_concepts=config.NMF_COMPONENTS)
    
    # Step b: HMM Sequence 分群
    clusterer = SequenceClusterer()
    
    # Step c: 自動標註（載入 MITRE embeddings + Hybrid Scoring）
    labeler = AutoLabeler()
    labeler.load_mitre_embeddings()
    
    # 對每個 Dataset 進行處理
    for dataset_id in all_datasets:
        concept_vectors = extractor.process_single_dataset(...)
        cluster_labels = clusterer.process_single_dataset(...)
        
        # 使用 Hybrid Scoring: Embedding × 0.6 + TF-IDF × 0.3 + DualBoost × 0.1
        labeling_result = labeler.process_single_dataset(
            dataset_id=dataset_id,
            concept_vectors=concept_vectors,
            cluster_labels=cluster_labels,
            nmf_extractor=extractor,
        )
```

### In `concept_extraction.py`:

```python
from external_sources import ExternalSourceManager

def extract_concepts_with_external(log_embeddings, manager):
    """Match log concepts against external sources."""
    
    results = []
    for log_emb in log_embeddings:
        # Find best matching technique
        matches = manager.compute_similarity(log_emb, 'MITRE', top_k=1)
        if matches:
            results.append({
                'technique': matches[0]['technique'],
                'confidence': matches[0]['similarity'],
            })
    
    return results
```

### In `auto_labeling.py`:

```python
def auto_label_with_sources(cluster_embeddings, manager, threshold=0.5):
    """Auto-label clusters using external source similarity."""
    
    labels = []
    for cluster_emb in cluster_embeddings:
        source_matches = manager.identify_source(cluster_emb)
        
        if source_matches:
            best_source = max(source_matches.items(), 
                            key=lambda x: x[1]['similarity'])
            
            if best_source[1]['similarity'] >= threshold:
                labels.append(best_source[1].get('technique', 'Unknown'))
            else:
                labels.append('Benign')
        else:
            labels.append('Benign')
    
    return labels
```

## Configuration (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MITRE_TECHNIQUES_CSV` | `data/reference_resources/MitreTechniquesTokens_V5.csv` | MITRE data CSV path |
| `MITRE_EXTERNAL_KNOWLEDGE_DIR` | `data/ExternalKnowledge/MITRE_RAW_EMBEDDINGS` | Output embeddings directory |
| `EXTERNAL_SOURCES_BERT_MODEL_NAME` | Same as `BERT_MODEL_NAME` | BERT model for embedding |
| `EXTERNAL_SOURCES_EMBED_BATCH_SIZE` | `32` | Embedding batch size |
| `EXTERNAL_SOURCES_EMBED_NORMALIZE` | `True` | Normalize embeddings |
| `FETCHER_REQUEST_TIMEOUT_SECONDS` | `60` | Online fetch timeout |

## Dependencies

Required:
- `pandas`, `numpy`
- `scikit-learn` (NMF, cosine_similarity)
- `sentence-transformers` (BERT embeddings)
- `requests` (online fetching)

Optional:
- `pyyaml` (Sigma rules parsing)

Install all:
```bash
pip install -r requirements.txt
```

## Data Format

### MITRE CSV Format

| Column | Description |
|--------|-------------|
| technique | Technique name |
| technique_id | MITRE ID (e.g., T1053) |
| tactics_id | Associated tactics |
| description | Full description |
| tokens | Tokenized description |
| cleaned_tokens | Stopwords removed |

### CAPEC CSV Format

| Column | Description |
|--------|-------------|
| Description | Attack pattern description |
| description | Lowercase version |
| tokens | Tokenized |
| cleaned_tokens | Stopwords removed |

## Run Examples

```bash
cd "Logs Labeling/external_sources"
python examples.py
```

## Author

Part of the Logs-Labeling pipeline project.
