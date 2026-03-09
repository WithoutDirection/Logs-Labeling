# TF-IDF 模組

本專案採用**雙層 TF-IDF 架構**，結合 BERT 語義嵌入與詞彙層級匹配，並在 Stage III 自動標註時實現**雙高加分機制**。

---

## 1. 概述

### 設計動機

| 方法 | 優勢 | 限制 |
|------|------|------|
| **BERT 嵌入** | 捕捉語義相似性 | 可能忽略關鍵詞彙 |
| **TF-IDF** | 精確的詞彙匹配 | 無法理解語義 |
| **混合評分** | 結合兩者優勢 | — |
| **雙高加分** | 雙重確認提升信心度 | — |

### 三層架構

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TF-IDF 三層架構                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Layer 1: Reference TF-IDF (Stage I)                                │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │  輸入: MITRE ATT&CK 技術描述                               │      │
│  │  處理: TfidfVectorizer.fit_transform()                    │      │
│  │  輸出: tfidf_vectorizer.pkl + mitre_tfidf_matrix.pkl      │      │
│  │        = MITRE Technique 指紋                              │      │
│  └───────────────────────────────────────────────────────────┘      │
│                              │                                      │
│                              │ 共享 Vectorizer                      │
│              ┌───────────────┴───────────────┐                      │
│              ▼                               ▼                      │
│  Layer 2: Log TF-IDF (Stage I)    Layer 3: Sequence TF-IDF (III)    │
│  ┌──────────────────────────┐    ┌──────────────────────────┐       │
│  │ 輸入: 原始日誌文本        │    │ 輸入: HMM Cluster 聚合   │       │
│  │ 處理: vectorizer.transform│    │ 處理: vectorizer.transform│      │
│  │ 輸出: tfidf.npz          │    │ 輸出: Sequence 指紋       │       │
│  └──────────────────────────┘    └──────────────────────────┘       │
│                                               │                     │
│                                               ▼                     │
│                              ┌───────────────────────────────────┐  │
│                              │ Hybrid Scoring (Stage III)        │  │
│                              │ Emb×0.6 + TF-IDF×0.3 + Boost×0.1  │  │
│                              └───────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1: Reference TF-IDF

> 模組: `precompute_log_tfidf.py`
> 
> 觸發: Stage I (`process_all_inputs()`)

### 目的

為 MITRE ATT&CK 技術描述建立 TF-IDF 指紋，作為後續匹配的基準。輸入來源由 **ReferenceBuilder** 統一管理（`data/reference_resources/combined.csv`），可整合多個外部知識來源（MITRE 原始描述、程式碼 tokens、資料庫文字等）。

### API

```python
from precompute_log_tfidf import build_reference_tfidf

vectorizer = build_reference_tfidf(
    force_rebuild=False,    # 強制重建
    max_features=5000,      # 詞彙表大小
)
```

### 處理流程

1. **讀取 MITRE CSV**：優先使用 `combined.csv`（ReferenceBuilder 輸出），fallback 順序為 `MitreTechniquesTokens_V6_Sanitized.csv` → `MitreTechniquesTokens_V5.csv`；載入 `description_raw` / `description` / `cleaned_tokens` 欄位
2. **訓練 Vectorizer**：`TfidfVectorizer(stop_words='english', max_features=5000)`
3. **生成指紋矩陣**：`fit_transform()` 生成 `(n_techniques, 5000)` 稀疏矩陣
4. **儲存成品**：
   - `tfidf_vectorizer.pkl`：訓練好的 Vectorizer（供後續 Layer 使用）
   - `mitre_tfidf_matrix.pkl`：MITRE 技術的 TF-IDF 指紋
   - `metadata.csv`：技術 ID 與名稱對照表

### 輸出結構

```
data/ExternalKnowledge/MITRE_TFIDF/
├── tfidf_vectorizer.pkl     # sklearn TfidfVectorizer
├── mitre_tfidf_matrix.pkl   # scipy.sparse.csr_matrix
└── metadata.csv             # technique, technique_id
```

---

## 3. Layer 2: Log TF-IDF

> 模組: `precompute_log_tfidf.py`
> 
> 觸發: Stage I (`run_tfidf_pipeline()`)

### 目的

使用 Layer 1 的 Vectorizer，將每個資料集的日誌文本轉換為 TF-IDF 向量。

### API

```python
from precompute_log_tfidf import compute_log_tfidf

stats = compute_log_tfidf(
    vectorizer=vectorizer,  # Layer 1 產出
    force_rebuild=False,
)
# stats: {"success": 10, "skipped": 5, "failed": 0}
```

### 輸出結構

```
data/Embeddings/{dataset_id}_embeddings/
├── data-00000-of-00001.arrow    # BERT 嵌入
└── tfidf.npz                    # TF-IDF 稀疏矩陣
```

---

## 4. Layer 3: Sequence TF-IDF

> 模組: `auto_labeling.py` (`_compute_sequence_tfidf()`)
> 
> 觸發: Stage III (自動標註)

### 目的

將 HMM 分群結果（Sequence）聚合為 TF-IDF 向量，與 MITRE 指紋進行匹配。

### 邏輯

```python
def _compute_sequence_tfidf(self, log_texts, cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    cluster_texts = []
    
    # 聚合每個 cluster 內所有 log 的文本
    for cid in unique_clusters:
        mask = cluster_labels == cid
        texts = [log_texts[i] for i in range(len(log_texts)) if mask[i]]
        cluster_texts.append(" ".join(texts))
    
    # 使用 Reference Vectorizer 轉換
    cluster_tfidf = self.tfidf_vectorizer.transform(cluster_texts)
    
    # 計算與 MITRE 指紋的相似度
    tfidf_similarities = cosine_similarity(cluster_tfidf, self.mitre_tfidf_matrix)
    
    return cluster_tfidf, tfidf_similarities, unique_clusters
```

### 獨立 API

```python
from precompute_log_tfidf import compute_sequence_tfidf, load_reference_vectorizer

vectorizer = load_reference_vectorizer()
sequence_tfidf = compute_sequence_tfidf(
    log_texts=log_texts,
    cluster_labels=cluster_labels,
    vectorizer=vectorizer,
)
```

---

## 5. 混合評分與威脅信心度

> 模組: `auto_labeling.py` (`_compute_hybrid_score()`, `_compute_threat_confidence()`)

### 雙層評分架構

混合評分用於計算 **Similarity Score**，再結合 **Anomaly Score** 產生最終的 **Threat Confidence**。

```
Layer 1: Similarity Score
─────────────────────────
Embedding Sim ──┐
                ├──► w_emb × Emb + w_tfidf × TF-IDF + Boost ──► Similarity
TF-IDF Sim ─────┘

Layer 2: Threat Confidence
─────────────────────────
Similarity Score ──┐
(α = 0.7)          │
                   ├──► α × Similarity + β × Anomaly ──► Threat Confidence
Anomaly Score ─────┘
(β = 0.3)
```

### Similarity Score 公式

$$
\text{Similarity Score} = w_{emb} \times \text{Sim}_{embedding} + w_{tfidf} \times \text{Sim}_{tfidf} + \text{Boost}
$$

### Threat Confidence 公式

$$
\text{Threat Confidence} = \alpha \times \text{Similarity Score} + \beta \times \text{Anomaly Score}
$$

### 雙高加分邏輯

```python
def _compute_hybrid_score(self, embedding_sim, tfidf_sim):
    w_emb = 0.6    # LABELING_WEIGHT_EMBEDDING
    w_tfidf = 0.3  # LABELING_WEIGHT_TFIDF
    
    # 基礎混合分數
    base_score = w_emb * embedding_sim + w_tfidf * tfidf_sim
    
    # 雙高加分
    if self.config.enable_dual_boost:
        threshold = 0.5  # LABELING_DUAL_BOOST_THRESHOLD
        boost_weight = 0.1  # LABELING_DUAL_BOOST_WEIGHT
        
        # 識別雙高情況
        dual_high_mask = (embedding_sim >= threshold) & (tfidf_sim >= threshold)
        
        # 加分 = boost_weight × min(embedding_sim, tfidf_sim)
        boost_score = np.where(
            dual_high_mask,
            boost_weight * np.minimum(embedding_sim, tfidf_sim),
            0.0
        )
        
        return base_score + boost_score
    
    return base_score
```

### 設計理念

| 情境 | Embedding | TF-IDF | Boost | 說明 |
|------|-----------|--------|-------|------|
| 雙高 | ≥0.5 | ≥0.5 | +0.1×min | 語義+詞彙都吻合，高信心度 |
| 單高 (Emb) | ≥0.5 | <0.5 | 0 | 可能語義相近但詞彙不同 |
| 單高 (TF-IDF) | <0.5 | ≥0.5 | 0 | 可能關鍵字匹配但語義不同 |
| 雙低 | <0.5 | <0.5 | 0 | 不太可能是該技術 |

> **注意**：Similarity Score 只是評分的一部分。最終的 **Threat Confidence** 還會結合 Stage II 的 **Anomaly Score**，確保異常程度低的日誌不會被錯誤標註為攻擊。

---

## 6. 配置參數彙整

### config.py 設定

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `MITRE_TFIDF_DIR` | `data/ExternalKnowledge/MITRE_TFIDF` | TF-IDF 輸出目錄 |
| `LABELING_USE_TFIDF` | `True` | 是否啟用 TF-IDF 混合評分 |
| `LABELING_WEIGHT_EMBEDDING` | `0.6` | Embedding 權重 |
| `LABELING_WEIGHT_TFIDF` | `0.3` | TF-IDF 權重 |
| `LABELING_ENABLE_DUAL_BOOST` | `True` | 是否啟用雙高加分 |
| `LABELING_DUAL_BOOST_THRESHOLD` | `0.5` | 雙高判定閾值 |
| `LABELING_DUAL_BOOST_WEIGHT` | `0.1` | 雙高加分權重 |
| `LABELING_SIMILARITY_WEIGHT` | `0.7` | Similarity 權重 ($\alpha$) |
| `LABELING_ANOMALY_WEIGHT` | `0.3` | Anomaly 權重 ($\beta$) |

---

## 7. API 總覽

### precompute_log_tfidf 模組

| 函數 | 說明 |
|------|------|
| `build_reference_tfidf()` | 建立 MITRE TF-IDF 指紋 (Layer 1) |
| `compute_log_tfidf()` | 轉換 Log TF-IDF (Layer 2) |
| `compute_sequence_tfidf()` | 計算 Sequence TF-IDF (Layer 3) |
| `load_reference_vectorizer()` | 載入 Reference Vectorizer |
| `load_reference_tfidf_matrix()` | 載入 MITRE 指紋矩陣 |
| `run_tfidf_pipeline()` | 統一入口：Layer 1 + Layer 2 |

### auto_labeling 模組

| 方法 | 說明 |
|------|------|
| `load_mitre_tfidf()` | 載入 MITRE TF-IDF 資料 |
| `_compute_sequence_tfidf()` | 計算 Sequence TF-IDF 相似度 |
| `_compute_hybrid_score()` | 計算 Similarity Score（含雙高加分） |
| `_compute_threat_confidence()` | 計算最終 Threat Confidence |
| `_load_anomaly_scores()` | 載入 Stage II 異常分數 |

---

## 8. 相關模組

- [Auto_Labeling.md](./Auto_Labeling.md) - Stage III-c：自動標註
- [Preprocessing.md](./Preprocessing.md) - Stage I：輸入處理
