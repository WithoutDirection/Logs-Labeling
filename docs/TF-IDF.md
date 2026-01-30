# TF-IDF 模組

本專案採用**雙層 TF-IDF 架構**，結合 BERT 語義嵌入與詞彙層級匹配，提升自動標註的準確性。

---

## 1. 概述

### 設計動機

| 方法 | 優勢 | 限制 |
|------|------|------|
| **BERT 嵌入** | 捕捉語義相似性 | 可能忽略關鍵詞彙 |
| **TF-IDF** | 精確的詞彙匹配 | 無法理解語義 |
| **混合評分** | 結合兩者優勢 | — |

### 雙層架構

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TF-IDF 雙層架構                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Layer 1: MITRE TF-IDF (外部知識)                                   │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │  輸入: MITRE ATT&CK 技術描述 (623 筆)                      │      │
│  │  處理: TfidfVectorizer.fit_transform()                    │      │
│  │  輸出: tfidf_vectorizer.pkl + mitre_tfidf_matrix.pkl      │      │
│  │        (623, 5000) 稀疏矩陣                                │      │
│  └───────────────────────────────────────────────────────────┘      │
│                              │                                      │
│                              │ 共享 Vectorizer                      │
│                              ▼                                      │
│  Layer 2: Per-Log TF-IDF (日誌向量)                                 │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │  輸入: 各資料集日誌文本                                    │      │
│  │  處理: vectorizer.transform() (使用 Layer 1 的 Vectorizer) │      │
│  │  輸出: data/Embeddings/{dataset_id}/tfidf.npz             │      │
│  │        (n_logs, 5000) 稀疏矩陣                             │      │
│  └───────────────────────────────────────────────────────────┘      │
│                              │                                      │
│                              ▼                                      │
│  Stage IV: 混合評分                                                 │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │  Score = α × Sim_embedding + (1-α) × Sim_tfidf            │      │
│  │  α = 0.7 (預設 Embedding 權重)                             │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1: MITRE TF-IDF

> 模組: `external_sources/build_tfidf.py`

### 目的

將 MITRE ATT&CK 技術描述轉換為 TF-IDF 稀疏向量，建立詞彙表供後續日誌轉換使用。

### API

```python
from external_sources.build_tfidf import build_mitre_tfidf

build_mitre_tfidf(
    out_dir="data/ExternalKnowledge/MITRE_TFIDF",
    mitre_csv="data/reference_resources/MitreTechniquesTokens_V5.csv",
    max_features=5000,      # 詞彙表大小
    force_rebuild=False,    # 強制重建
)
```

### 處理流程

1. **讀取 MITRE CSV**：載入 `description` 欄位作為文本語料
2. **訓練 Vectorizer**：使用 `TfidfVectorizer(stop_words='english', max_features=5000)` 建立詞彙表
3. **轉換矩陣**：`fit_transform()` 生成 `(623, 5000)` 稀疏矩陣
4. **儲存成品**：
   - `tfidf_vectorizer.pkl`：訓練好的 Vectorizer（供 Layer 2 使用）
   - `mitre_tfidf_matrix.pkl`：MITRE 技術的 TF-IDF 向量
   - `metadata.csv`：技術 ID 與名稱對照表

### 輸出結構

```
data/ExternalKnowledge/MITRE_TFIDF/
├── tfidf_vectorizer.pkl     # sklearn TfidfVectorizer 物件
├── mitre_tfidf_matrix.pkl   # scipy.sparse.csr_matrix (623, 5000)
└── metadata.csv             # technique, technique_id
```

### 配置參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `MITRE_TFIDF_DIR` | `data/ExternalKnowledge/MITRE_TFIDF` | 輸出目錄 |
| `max_features` | `5000` | TF-IDF 詞彙表大小 |

---

## 3. Layer 2: Per-Log TF-IDF

> 模組: `precompute_log_tfidf.py`
> 
> 對應 Pipeline: **Stage I (Step 2)**

### 目的

使用 Layer 1 建立的 Vectorizer，將每個資料集的日誌文本轉換為 TF-IDF 稀疏向量。

### API

```python
from precompute_log_tfidf import run_log_tfidf_precompute

result = run_log_tfidf_precompute(
    force_rebuild=False,    # 強制重建（忽略快取）
    verbose=True,           # 輸出進度
)

# result: {
#     "success": 10,      # 新生成數量
#     "skipped": 5,       # 使用快取數量
#     "failed": 0,        # 失敗數量
#     "total": 15,        # 總資料集數
#     "enabled": True,    # TF-IDF 是否啟用
# }
```

### 處理流程

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 載入 Vectorizer │ ──► │  提取日誌文本   │ ──► │  轉換並儲存     │
│ (Layer 1 產出)  │     │ (按欄位優先順序) │     │  tfidf.npz      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **載入 Vectorizer**：讀取 `MITRE_TFIDF_DIR/tfidf_vectorizer.pkl`
2. **定位原始 CSV**：依序搜尋 `Intermediate_data/` → `input_logs/`
3. **提取文本**：按優先順序讀取欄位
   - `ConcatenatedLog`
   - `Template + Parameters`
   - `Content`
   - `Event`
   - 全欄位串接
4. **轉換**：`vectorizer.transform(texts)` 生成稀疏矩陣
5. **儲存**：`scipy.sparse.save_npz(tfidf.npz)`

### 內部函數

| 函數 | 說明 |
|------|------|
| `_load_vectorizer()` | 載入 MITRE TF-IDF Vectorizer |
| `_find_source_csv(dataset_id)` | 根據 dataset_id 定位原始 CSV |
| `_extract_text(df)` | 從 DataFrame 提取日誌文本 |
| `_process_single_dataset(...)` | 處理單一資料集 |

### 輸出結構

```
data/Embeddings/{dataset_id}_embeddings/
├── data-00000-of-00001.arrow    # BERT 嵌入 (Stage I Step 1)
├── dataset_info.json
└── tfidf.npz                    # TF-IDF 稀疏矩陣 (n_logs, 5000)
```

### 命令列執行

```bash
# 預設執行（使用快取）
python precompute_log_tfidf.py

# 強制重建
python precompute_log_tfidf.py --force
```

---

## 4. Stage IV: 混合評分

> 模組: `auto_labeling.py`

### 載入 TF-IDF 資料

```python
from auto_labeling import AutoLabeler

labeler = AutoLabeler()
labeler.load_mitre_embeddings()  # BERT 嵌入
labeler.load_mitre_tfidf()       # TF-IDF 矩陣與 Vectorizer
```

### 相關方法

| 方法 | 說明 |
|------|------|
| `load_mitre_tfidf()` | 載入 MITRE TF-IDF 向量與 Vectorizer |
| `_compute_tfidf_similarity(log_texts, cluster_labels)` | 計算日誌與 MITRE 的 TF-IDF 相似度 |
| `_load_log_texts(dataset_id)` | 載入原始日誌文本 |

### 混合評分公式

$$
\text{Score}_{final} = \alpha \times \text{Sim}_{embedding} + (1 - \alpha) \times \text{Sim}_{tfidf}
$$

其中：
- $\alpha = 0.7$（`LABELING_WEIGHT_EMBEDDING`）
- $1 - \alpha = 0.3$（`LABELING_WEIGHT_TFIDF`）

### 評分流程

```
1. 計算 Cluster Centroid 與 MITRE 嵌入的餘弦相似度 (Sim_embedding)

2. 聚合 Cluster 內所有日誌文本
   ↓
3. 使用 Vectorizer.transform() 轉換為 TF-IDF 向量
   ↓
4. 計算與 MITRE TF-IDF 矩陣的餘弦相似度 (Sim_tfidf)
   ↓
5. 加權混合: Score = 0.7 × Sim_embedding + 0.3 × Sim_tfidf
```

### TF-IDF 相似度計算

```python
def _compute_tfidf_similarity(self, log_texts, cluster_labels):
    # 1. 按 Cluster 聚合文本
    unique_clusters = np.unique(cluster_labels)
    cluster_texts = []
    for cid in unique_clusters:
        mask = cluster_labels == cid
        cluster_text = " ".join([log_texts[i] for i in range(len(log_texts)) if mask[i]])
        cluster_texts.append(cluster_text)
    
    # 2. 轉換為 TF-IDF 向量
    cluster_tfidf = self.tfidf_vectorizer.transform(cluster_texts)
    
    # 3. 計算與 MITRE 的餘弦相似度
    tfidf_similarities = cosine_similarity(cluster_tfidf, self.mitre_tfidf_matrix)
    
    return tfidf_similarities  # shape: (n_clusters, n_techniques)
```

---

## 5. 配置參數彙整

### config.py 設定

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `MITRE_TFIDF_DIR` | `data/ExternalKnowledge/MITRE_TFIDF` | MITRE TF-IDF 輸出目錄 |
| `LABELING_USE_TFIDF` | `True` | 是否啟用 TF-IDF 混合評分 |
| `LABELING_WEIGHT_EMBEDDING` | `0.7` | Embedding 相似度權重 |
| `LABELING_WEIGHT_TFIDF` | `0.3` | TF-IDF 相似度權重 |

### Pipeline 參數

```python
# Stage I 呼叫
STAGE_I(N=10, enable_tfidf=True)   # 啟用 Per-Log TF-IDF 預計算
STAGE_I(N=10, enable_tfidf=False)  # 跳過 TF-IDF

# 命令列
python Pipeline.py --skip-tfidf    # 跳過 TF-IDF 預計算
```

---

## 6. 資料流總覽

```
┌────────────────────────────────────────────────────────────────────┐
│                           TF-IDF 資料流                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [MITRE ATT&CK CSV]                                                │
│        │                                                           │
│        ▼                                                           │
│  build_tfidf.py ───────────────────────────────────────────┐       │
│        │                                                   │       │
│        ▼                                                   ▼       │
│  ┌──────────────────────┐                    ┌────────────────────┐│
│  │ tfidf_vectorizer.pkl │                    │mitre_tfidf_matrix  ││
│  │ (sklearn Vectorizer) │                    │ (623, 5000)        ││
│  └──────────┬───────────┘                    └─────────┬──────────┘│
│             │                                          │           │
│             │ Stage I                                  │ Stage IV  │
│             ▼                                          ▼           │
│  precompute_log_tfidf.py                       auto_labeling.py    │
│        │                                               │           │
│        ▼                                               ▼           │
│  ┌──────────────────────┐                    ┌────────────────────┐│
│  │ tfidf.npz            │ ─────────────────► │ 混合評分           ││
│  │ (n_logs, 5000)       │                    │ Sim_embedding×0.7  ││
│  │ Per-Dataset          │                    │ + Sim_tfidf×0.3    ││
│  └──────────────────────┘                    └────────────────────┘│
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 7. 常見問題

### Q1: 為什麼需要共用 Vectorizer？

**原因**：TF-IDF 的向量維度由詞彙表決定。若 MITRE 和日誌使用不同 Vectorizer，產生的向量維度不同，無法計算餘弦相似度。

**解決**：Layer 1 建立 Vectorizer 並 `fit()`，Layer 2 僅使用 `transform()`，確保詞彙表一致。

### Q2: TF-IDF 未找到時會發生什麼？

**行為**：
1. `run_log_tfidf_precompute()` 返回 `{"enabled": False}`
2. Stage IV 自動跳過 TF-IDF 評分，僅使用 Embedding 相似度

**檢查**：確認 `MITRE_TFIDF_DIR/tfidf_vectorizer.pkl` 存在

### Q3: 如何調整 Embedding/TF-IDF 權重？

修改 `config.py`：

```python
LABELING_WEIGHT_EMBEDDING = 0.8  # 增加語義權重
LABELING_WEIGHT_TFIDF = 0.2      # 降低詞彙權重
```

### Q4: 如何強制重建所有 TF-IDF？

```bash
# 重建 MITRE TF-IDF
python external_sources/build_tfidf.py

# 重建 Per-Log TF-IDF
python precompute_log_tfidf.py --force
```

---

## 8. 視覺化：詞彙覆蓋率分析

> 模組: `visualization/tfidf_coverage.py`

### 功能

分析日誌詞彙與 MITRE ATT&CK 高權重詞彙的重疊情況，評估 TF-IDF 模組的有效性。

### 輸出

```
result/tfidf_coverage/
├── vocabulary_coverage_bar.png    # 各資料集覆蓋率長條圖
├── vocabulary_venn.png            # 詞彙交集 Venn 圖
├── top_overlapping_terms.png      # 高重疊詞彙權重分佈
├── coverage_by_dataset.csv        # 各資料集覆蓋率統計
├── top_overlapping_terms.csv      # 高重疊詞彙清單
└── coverage_summary.json          # 整體摘要
```

### API

```python
from visualization.tfidf_coverage import run_tfidf_coverage_analysis

# 基本用法
result = run_tfidf_coverage_analysis()

# 自訂參數
result = run_tfidf_coverage_analysis(
    top_n_mitre=500,        # 取 MITRE 前 N 個高權重詞彙
    output_dir="result/my_analysis",
    max_datasets=10,        # 限制處理資料集數
    verbose=True
)

# 回傳值
# {
#     "n_datasets": 50,
#     "avg_log_coverage": 0.12,      # 12% 日誌詞彙在 MITRE 詞彙中
#     "avg_mitre_coverage": 0.35,    # 35% MITRE 詞彙在日誌中出現
#     "total_overlap_terms": 180,
#     "output_dir": "result/tfidf_coverage",
#     "enabled": True
# }
```

### 命令列

```bash
cd "Logs Labeling"

# 預設執行
python visualization/tfidf_coverage.py

# 自訂參數
python visualization/tfidf_coverage.py --top-n 1000 --max-datasets 20
```

---

## 9. 相關模組

- [Preprocessing.md](./Preprocessing.md) - Stage I：包含 TF-IDF 預計算
- [External_Sources.md](./External_Sources.md) - MITRE 外部知識建構
- [Auto_Labeling.md](./Auto_Labeling.md) - Stage IV：混合評分機制
- [Embedding.md](./Embedding.md) - BERT 嵌入（與 TF-IDF 互補）

