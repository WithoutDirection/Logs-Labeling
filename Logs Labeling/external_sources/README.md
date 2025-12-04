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

## Integration with Pipeline

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
