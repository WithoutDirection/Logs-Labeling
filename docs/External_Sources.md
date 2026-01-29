# 外部情資來源模組使用說明 (External Sources Module)

本模組 (`external_sources`) 提供了整合外部威脅情資（如 MITRE ATT&CK、CAPEC）的工具，用於增強日誌標註與分析的能力。

## 1. 模組概述

此模組的主要功能包括：
- **載入外部知識庫**：支援 MITRE ATT&CK (戰術與技術)、CAPEC (攻擊模式)、NVD/CVE (漏洞資料) 等。
- **文本預處理**：實作 ConceptUML 論文中的預處理流程，包含 Zipf's Law 過濾。
- **語義檢索**：利用 BERT Embedding 與 NMF 主題模型進行相似度比對。

## 2. 核心類別說明

### 2.1 `ExternalSourceManager` (情資管理器)

這是管理所有外部來源的核心類別，位於 `source_manager.py`。

**主要方法：**

| 方法 | 說明 |
|------|------|
| `load_source(name, path)` | 載入 CSV 格式的情資檔案 (如 MITRE 數據) |
| `fetch_source(name)` | 從線上 API 抓取最新資料 |
| `prepare_all_embeddings()` | 為所有已載入的來源計算 BERT 嵌入向量 |
| `prepare_all_nmf(n_components)` | 計算 NMF 主題矩陣 (用於主題建模) |
| `compute_similarity(query_emb, source)` | 計算查詢向量與指定來源的相似度 |
| `identify_source(query_emb)` | 識別查詢向量最匹配的來源 |
| `hybrid_similarity(bert, nmf, source)` | 結合 BERT 與 NMF 的混合相似度計算 |

**使用範例：**

```python
from external_sources import ExternalSourceManager

# 初始化
manager = ExternalSourceManager(
    data_dir='data/reference_resources',
    nmf_components=10
)

# 1. 載入資料
manager.load_source('MITRE', 'data/reference_resources/MitreTechniquesTokens_V5.csv')
manager.load_source('CAPEC', 'data/reference_resources/CapecTokens_V5.csv')

# 2. 準備嵌入向量 (需先設定好 BERT 模型)
manager.prepare_all_embeddings()

# 3. 準備 NMF 主題模型
manager.prepare_all_nmf(n_components=10)
```

### 2.2 `TextProcessor` (文本處理器)

位於 `text_processor.py`，負責文本的清洗、分詞與過濾，特別是實作了基於 Zipf's Law 的高頻詞過濾。

**預處理流程 (ConceptUML Pipeline)：**
1. **轉小寫** (Lowercasing)
2. **分詞** (Tokenization)
3. **移除停用詞** (Stopword removal)
4. **Zipf's Law 過濾**：移除出現頻率最高的前 5% 單詞 (如 `system`, `windows`, `file`, `microsoft` 等通用詞)，以保留具鑑別度的概念詞。

**主要方法：**

| 方法 | 說明 |
|------|------|
| `fit_zipf_filter(texts)` | 分析語料庫詞頻，建立過濾清單 |
| `clean_text(text)` | 移除引用、URL，轉小寫 |
| `tokenize(text)` | 執行完整的預處理流程 (清洗 -> 分詞 -> 過濾) |
| `generate_embeddings(texts)` | 呼叫 BERT 模型生成向量 |
| `get_zipf_filtered_words()` | 取得被過濾掉的高頻詞列表 |

**使用範例：**

```python
from external_sources import TextProcessor, preprocess_log_corpus

# 初始化處理器 (設定過濾前 5% 高頻詞)
processor = TextProcessor(
    zipf_percentile=0.05,
    use_log_high_freq=True  # 包含日誌常見詞
)

# 對日誌語料進行預處理並適配 Zipf 過濾器
processed_texts, processor = preprocess_log_corpus(
    raw_logs, 
    processor=processor,
    zipf_percentile=0.05
)

# 查看被過濾掉的高頻詞
print(processor.get_zipf_filtered_words())
```

### 2.3 Fetchers (資料擷取器)

位於 `fetchers.py`，用於從外部來源獲取資料。

| 類別 | 來源 | 方式 |
|------|------|------|
| `MitreFetcher` | MITRE ATT&CK | GitHub STIX 數據 |
| `CapecFetcher` | CAPEC | 本地 XML 解析 |
| `NvdFetcher` | NVD/CVE | NVD API 2.0 |
| `SigmaRulesFetcher` | Sigma Rules | 本地 YAML 解析 |

### 2.4 build_mitre_raw_embeddings (MITRE 嵌入建構器)

位於 `build_mitre_raw_embeddings.py`，用於將 MITRE ATT&CK 技術描述轉換為 BERT 嵌入向量。這是 **Pipeline Stage III** 的核心模組。

**使用方式：**

```python
from external_sources.build_mitre_raw_embeddings import build_mitre_raw_embeddings

out_dir = build_mitre_raw_embeddings(
    mitre_csv="data/reference_resources/MitreTechniquesTokens_V5.csv",
    out_dir="data/ExternalKnowledge/MITRE_RAW_EMBEDDINGS",
    bert_model="sentence-bert",
    force_rebuild=False,  # 若已存在則跳過
)
```

## 3. Pipeline 整合

### 3.1 Stage III：外部知識嵌入

在 Pipeline 的 Stage III 中，此模組負責建立 MITRE ATT&CK 的 BERT 嵌入向量：

```python
# Pipeline.py STAGE_III
from external_sources import build_knowledge_base

def STAGE_III():
    """建立外部知識嵌入"""
    result = build_knowledge_base(force_rebuild=False, verbose=True)
    
    status = "使用快取" if result.get("cached") else "新建完成"
    print(f"[Stage III 完成] {status} | 技術數量: {result['n_techniques']}")
    return result
```

### 3.2 Stage IV：自動標註整合

在 Stage IV 的自動標註步驟中，概念提取器會載入外部知識進行聯合訓練：

```python
# 在 STAGE_IV 中
extractor = ConceptExtractor(n_concepts=config.NMF_COMPONENTS)
extractor.load_external_knowledge(config.EXTERNAL_KNOWLEDGE_DIR)

# 標註時 AutoLabeler 會載入 MITRE 嵌入進行比對
labeler = AutoLabeler()
labeler.load_mitre_embeddings()
```

### 3.3 查詢相似威脅技術

將日誌或查詢語句轉換為向量，並在 MITRE ATT&CK 中尋找最相似的技術。

```python
from external_sources import TextProcessor

processor = TextProcessor()

# 1. 將查詢語句向量化
query = "schtasks.exe /create scheduled task"
query_emb = processor.generate_embeddings([query])[0]

# 2. 計算相似度 (Top 5)
results = manager.compute_similarity(query_emb, 'MITRE', top_k=5)

for r in results:
    print(f"技術: {r['technique']}")
    print(f"相似度: {r['similarity']:.3f}")
```

## 4. 配置參數

在 `config.py` 中的相關設定：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `MITRE_TECHNIQUES_CSV` | `data/reference_resources/MitreTechniquesTokens_V5.csv` | MITRE 技術資料 CSV 路徑 |
| `MITRE_EXTERNAL_KNOWLEDGE_DIR` | `data/ExternalKnowledge/MITRE_RAW_EMBEDDINGS` | MITRE 嵌入向量輸出目錄 |
| `EXTERNAL_SOURCES_BERT_MODEL_NAME` | 與 `BERT_MODEL_NAME` 相同 | 嵌入使用的 BERT 模型 |
| `EXTERNAL_SOURCES_EMBED_BATCH_SIZE` | `32` | 嵌入計算批次大小 |
| `EXTERNAL_SOURCES_EMBED_NORMALIZE` | `True` | 是否正規化嵌入向量 |
| `FETCHER_REQUEST_TIMEOUT_SECONDS` | `60` | 線上資料抓取超時時間 |

## 5. 資料格式需求

若要匯入自訂的 CSV 來源，建議包含以下欄位：

**MITRE CSV 格式範例：**
- `technique`: 技術名稱
- `technique_id`: MITRE ID (如 T1053)
- `tactics_id`: 關聯戰術 ID
- `description`: 完整描述
- `tokens`: 分詞後的內容
- `cleaned_tokens`: 移除停用詞後的內容

**CAPEC CSV 格式範例：**
- `Description`: 攻擊模式描述
- `description`: 小寫版本描述
- `tokens`: 分詞後的內容
- `cleaned_tokens`: 移除停用詞後的內容

## 6. 依賴套件

- `pandas`, `numpy`: 資料處理
- `scikit-learn`: NMF, 餘弦相似度
- `sentence-transformers`: BERT 向量化
- `requests`: 線上資料抓取
- `pyyaml`: (可選) 解析 Sigma 規則

## 7. 相關模組

- [Preprocessing](./Preprocessing.md) - Stage I：日誌預處理與嵌入
- [Anomaly_Detection](./Anomaly_Detection.md) - Stage II：異常偵測
- [Concept_Extraction](./Concept_Extraction.md) - Stage IV-a：概念提取（NMF）
- [Sequence_Clustering](./Sequence_Clustering.md) - Stage IV-b：序列分群（HMM）
- [Auto_Labeling](./Auto_Labeling.md) - Stage IV-c：自動標註

---
此文件說明基於 `external_sources/README.md` 及其程式碼實作。
