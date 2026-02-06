# Auto Labeling（自動標註）

## 概述

自動標註模組 (`auto_labeling.py`) 是 Logs Labeling Pipeline 的最終階段（Stage III-c），負責將 HMM 序列分群結果與 MITRE ATT&CK 外部知識進行比對，結合**異常偵測分數 (Stage II)**，計算每筆日誌的**威脅信心度 (Threat Confidence)**。

---

## 核心概念

### 雙層評分架構

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Threat Confidence Architecture                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │  Layer 1: Similarity Score (與 Technique 的相似程度)          │     │
│   │                                                               │     │
│   │  Cluster Centroid ───┐                                        │     │
│   │                      ├──► Embedding Sim ──┐                   │     │
│   │  MITRE Embedding ────┘                    │                   │     │
│   │                                           ├──► Similarity     │     │
│   │  Sequence TF-IDF ────┐                    │      Score        │     │
│   │                      ├──► TF-IDF Sim ─────┤                   │     │
│   │  MITRE TF-IDF 指紋 ──┘                    │                   │     │
│   │                                           │                   │     │
│   │  Dual-High Boost ─────────────────────────┘                   │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                    │                                    │
│                                    ▼                                    │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │  Layer 2: Threat Confidence (最終威脅信心度)                   │     │
│   │                                                               │     │
│   │  Similarity Score ────┐                                       │     │
│   │  (α 權重)             ├──► Threat Confidence ──► Top-K Label  │     │
│   │  Anomaly Score ───────┘                                       │     │
│   │  (β 權重, Stage II)                                           │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 評分公式

#### Layer 1: Similarity Score

$$
\text{Similarity Score} = w_{emb} \times \text{Sim}_{embedding} + w_{tfidf} \times \text{Sim}_{tfidf} + \text{Boost}
$$

其中**雙高加分 (Dual-High Boost)**：

$$
\text{Boost} = 
\begin{cases}
w_{boost} \times \min(\text{Sim}_{emb}, \text{Sim}_{tfidf}) & \text{若 } \text{Sim}_{emb} \geq \theta \text{ 且 } \text{Sim}_{tfidf} \geq \theta \\
0 & \text{否則}
\end{cases}
$$

#### Layer 2: Threat Confidence

$$
\text{Threat Confidence} = \alpha \times \text{Similarity Score} + \beta \times \text{Anomaly Score}
$$

其中：
- **Similarity Score**：Sequence 與 MITRE Technique 的語義/詞彙相似度
- **Anomaly Score**：Stage II 異常偵測結果 (0~1)，代表 raw event 的惡意可能性
- $\alpha$, $\beta$：權重參數（預設 $\alpha = 0.7$, $\beta = 0.3$）

**設計理念**：
- 即使一個 Sequence 與某 Technique 相似度高，若其 Anomaly Score 低（正常行為），最終 Threat Confidence 也會較低
- 反之，若 Anomaly Score 高但 Similarity 低，可能是未知攻擊型態

### Sequence TF-IDF 計算

**Step 1: 載入原始日誌文本**
```python
# 從原始 CSV 載入文本（優先使用 ConcatenatedLog, Template, Content 欄位）
log_texts = labeler._load_log_texts(dataset_id)
# 搜尋路徑：
#   1. data/Intermediate_data/{dataset_id}.csv
#   2. data/input_logs/{dataset_id}.csv
```

**Step 2: 聚合 Cluster 文本**
```python
# 聚合 Cluster 內所有日誌文本
for cluster_id in unique_clusters:
    mask = cluster_labels == cluster_id
    cluster_text = " ".join([log_texts[i] for i in range(len(log_texts)) if mask[i]])
    cluster_texts.append(cluster_text)
```

**Step 3: TF-IDF 轉換與相似度計算**
```python
# 使用 Reference Vectorizer 轉換（與 MITRE 相同向量空間）
sequence_tfidf = vectorizer.transform(cluster_texts)

# 計算與 MITRE 指紋的相似度
tfidf_similarities = cosine_similarity(sequence_tfidf, mitre_tfidf_matrix)
```

> **關鍵**：Sequence TF-IDF 使用與 MITRE TF-IDF 相同的 Vectorizer（Stage I 產出），確保向量空間一致性。

---

## 配置參數

### Similarity Score 權重

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `LABELING_WEIGHT_EMBEDDING` | 0.6 | Embedding 相似度權重 ($w_{emb}$) |
| `LABELING_WEIGHT_TFIDF` | 0.3 | TF-IDF 相似度權重 ($w_{tfidf}$) |
| `LABELING_ENABLE_DUAL_BOOST` | True | 是否啟用雙高加分 |
| `LABELING_DUAL_BOOST_THRESHOLD` | 0.5 | 雙高判定閾值 ($\theta$) |
| `LABELING_DUAL_BOOST_WEIGHT` | 0.1 | 雙高加分權重 ($w_{boost}$) |

### Threat Confidence 權重

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `LABELING_SIMILARITY_WEIGHT` | 0.7 | Similarity Score 權重 ($\alpha$) |
| `LABELING_ANOMALY_WEIGHT` | 0.3 | Anomaly Score 權重 ($\beta$) |
| `LABELING_SIMILARITY_THRESHOLD` | 0.3 | 相似度下界，低於此值標記為 Benign |
| `LABELING_TOP_K` | 3 | 輸出的候選技術數量 |
| `LABELING_RESULTS_DIR` | `result/Labeling_Results/` | 標註結果輸出目錄 |

---

## 使用方式

### 在 Pipeline 中執行

```python
# 在 Pipeline Stage III 中自動執行
from Pipeline import STAGE_III

STAGE_III()  # 執行 Per-Dataset 處理：NMF → HMM → 自動標註
```

### 獨立執行

```python
from auto_labeling import AutoLabeler

# 建立標註器
labeler = AutoLabeler()

# 載入 MITRE 嵌入與 TF-IDF（Stage I 產出）
labeler.load_mitre_embeddings()
labeler.load_mitre_tfidf()

# 標註單一資料集
result = labeler.process_single_dataset(
    dataset_id="dataset_001",
    concept_vectors=concept_vectors,  # NMF 概念向量
    cluster_labels=cluster_labels,    # HMM 分群標籤
    output_dir="result/Labeling_Results/",
    nmf_extractor=extractor,
    log_vectors_path=input_path,      # 用於載入 log vectors
    anomaly_scores=anomaly_scores,    # Stage II 異常分數（可選，會自動載入）
)
```

---

## 輸出格式

標註結果為 CSV 檔案，儲存於 `result/Labeling_Results/{dataset_id}_Labeled.csv`。

### 欄位說明

| 欄位 | 型別 | 說明 |
|------|------|------|
| `original_idx` | int | 日誌在原始資料集中的索引 |
| `anomaly_score` | float | Stage II 異常分數 (0~1) |
| `groundtruth_tid` | str | Ground Truth 技術 ID（若有載入） |
| `groundtruth_t_name` | str | Ground Truth 技術名稱（若有載入） |
| `{original_columns}` | various | 原始日誌的所有欄位 |
| `predicted_technique_1_name` | str | Top-1 預測的 MITRE 技術名稱 |
| `predicted_technique_1_threat_confidence` | float | Top-1 的威脅信心度 |
| `predicted_technique_1_similarity` | float | Top-1 的相似度分數 |
| `predicted_technique_K_*` | various | Top-K 預測的相應欄位 |

### 輸出範例

**主要標註結果** (`{dataset_id}_Labeled.csv`):
```csv
original_idx,anomaly_score,groundtruth_tid,groundtruth_t_name,timestamp,event_type,predicted_technique_1_name,predicted_technique_1_threat_confidence,predicted_technique_1_similarity
0,0.85,T1059,Command and Scripting Interpreter,2025-01-08 10:30:00,Process Create,PowerShell,0.78,0.82
1,0.12,T1059,Command and Scripting Interpreter,2025-01-08 10:30:02,Network Connect,Application Layer Protocol,0.35,0.45
```

**摘要檔案** (`{dataset_id}_Summary.csv`, `Summary_All.csv`):

| 欄位 | 說明 |
|------|------|
| `dataset_id` | 資料集 ID |
| `groundtruth` | Ground Truth 技術 (tid \| t_name) |
| `embedding_top1~5` | Embedding-only 模式的 Top-5 技術分布 |
| `tfidf_top1~5` | TF-IDF-only 模式的 Top-5 技術分布 |
| `hybrid_top1~5` | Hybrid 混合模式的 Top-5 技術分布 |
| `*_gt_avg_rank` | Ground Truth 在各模式的平均排名 |
| `*_gt_best_rank` | Ground Truth 在各模式的最佳排名 |

---

## 資料流依賴（三階段架構）

```
STAGE_I (Input Processing)
    │
    ├─── Log Datasets ───────────────────────────────────┐
    │    Parse → Embed → TF-IDF                          │
    │                                                    │
    └─── Reference Sources ───────────────────────────┐  │
         MITRE Embedding + TF-IDF 指紋                │  │
                                                      │  │
                                                      ▼  ▼
STAGE_II (Anomaly Detection) ────────────────────────────┤
    │                                                    │
    ▼                                                    │
┌─────────────────────┐                                  │
│ Detection_Results/  │                                  │
│ anomaly_scores      │                                  │
└─────────────────────┘                                  │
                                                         │
                     ┌───────────────────────────────────┘
                     ▼
         STAGE_III (Per-Dataset 處理)
         ┌───────────────────────────────────┐
         │  NMF → HMM → Auto Labeling        │
         │                                   │
         │  ┌───────────┐  ┌───────────────┐ │
         │  │ConceptVec │→ │SequenceCluster│ │
         │  └───────────┘  └───────────────┘ │
         │                       │           │
         │                       ▼           │
         │              ┌───────────────────┐│
         │              │ Hybrid Scoring    ││
         │              │ Emb + TF-IDF      ││
         │              │ + Dual-High Boost ││
         │              └───────────────────┘│
         └───────────────────┬───────────────┘
                             ▼
                 ┌─────────────────────┐
                 │ Labeling_Results/   │
                 │ {id}_Labeled.csv    │
                 └─────────────────────┘
```

---

## 核心 API

### AutoLabeler 類別

| 方法 | 說明 |
|------|------|
| `load_mitre_embeddings()` | 載入 MITRE 嵌入向量 |
| `load_mitre_tfidf()` | 載入 MITRE TF-IDF 矩陣與 Vectorizer |
| `process_single_dataset()` | 標註單一資料集（Per-Dataset API） |
| `_compute_sequence_tfidf()` | 計算 Sequence TF-IDF 向量 |
| `_compute_hybrid_score()` | 計算 Similarity Score（含雙高加分） |
| `_compute_threat_confidence()` | 計算最終 Threat Confidence |
| `_load_anomaly_scores()` | 載入 Stage II 異常偵測分數 |

### precompute_log_tfidf 模組

| 函數 | 說明 |
|------|------|
| `compute_sequence_tfidf()` | 計算 HMM Sequence 的 TF-IDF 向量 |
| `load_reference_vectorizer()` | 載入 Reference TF-IDF Vectorizer |
| `load_reference_tfidf_matrix()` | 載入 MITRE TF-IDF 指紋矩陣 |

---

## 注意事項

1. **Stage I 依賴**：自動標註需要 Stage I 產出的 MITRE Embedding 與 TF-IDF 指紋
2. **Stage II 依賴**：Anomaly Score 來自 Stage II 異常偵測結果；若無提供則僅使用 Similarity Score
3. **向量空間一致性**：Log TF-IDF 與 MITRE TF-IDF 使用相同的 Vectorizer 確保空間一致
4. **權重總和**：
   - Similarity Score: $w_{emb} + w_{tfidf} + w_{boost} = 0.6 + 0.3 + 0.1 = 1.0$
   - Threat Confidence: $\alpha + \beta = 0.7 + 0.3 = 1.0$
5. **Top-K 輸出**：每筆日誌輸出 K 個候選技術及其威脅信心度

---

## 相關模組

- [TF-IDF.md](./TF-IDF.md) - TF-IDF 雙層架構與混合評分機制
- [Preprocessing.md](./Preprocessing.md) - Stage I：輸入處理
- [Anomaly_Detection.md](./Anomaly_Detection.md) - Stage II：異常偵測
- [Concept_Extraction.md](./Concept_Extraction.md) - Stage III-a：概念提取（NMF）
- [Sequence_Clustering.md](./Sequence_Clustering.md) - Stage III-b：序列分群（HMM）
