# Logs Labeling

透過無監督式機器學習，自動將日誌序列標註為 MITRE ATT&CK 攻擊技術。

---

## 目錄

- [專案簡介](#專案簡介)
- [核心流程概覽](#核心流程概覽)
- [Pipeline 四階段詳解](#pipeline-四階段詳解)
  - [Stage I：預處理與嵌入 (Preprocessing & Embedding)](#stage-i預處理與嵌入-preprocessing--embedding)
  - [Stage II：異常偵測 (Anomaly Detection)](#stage-ii異常偵測-anomaly-detection)
  - [Stage III：外部知識嵌入 (External Knowledge)](#stage-iii外部知識嵌入-external-knowledge)
  - [Stage IV：Per-Dataset 處理 (NMF → HMM → Auto Labeling)](#stage-ivper-dataset-處理-nmf--hmm--auto-labeling)
- [專案目錄結構](#專案目錄結構)
- [快速開始](#快速開始)
- [更新日誌](#更新日誌)

---

## 專案簡介

### 解決的問題

在資安事件分析中，面對數十萬筆日誌資料，人工判讀不僅耗時且容易遺漏關鍵攻擊行為。本專案建立一套**端到端的自動化標註系統**，將原始日誌自動對應至 MITRE ATT&CK 攻擊技術。

### 核心價值

| 特性 | 說明 |
|------|------|
| **無監督學習** | 無需人工標註的訓練資料 |
| **語義理解** | 透過 BERT 捕捉日誌的深層語義 |
| **異常導向** | 優先標註行為異常的日誌，降低誤報 |
| **可解釋性** | 每個標籤都附帶相似度與信心度分數 |

### 技術棧

```
BERT Embedding → NMF 降維 → HMM 分群 → 餘弦相似度 → MITRE ATT&CK 標籤
```

---

## 核心流程概覽

![Structure](./docs/assests/LogsLabeling%20Structure.png)

本專案採用 **Per-Dataset 策略**，將日誌標註拆解為 **四個獨立階段**，確保每個 Technique 的標註不會被其他 Technique 混淆：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Logs Labeling Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   [原始日誌]                                                             │
│       │                                                                 │
│       ▼                                                                 │
│   ┌──────────────┐                                                      │
│   │  STAGE I     │  預處理與嵌入：日誌解析 → BERT 嵌入 → 向量化             │
│   │ Preprocessing│  產出：768 維語義向量                                  │
│   └──────┬───────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │  STAGE II    │  異常偵測：Ensemble 模型識別異常日誌                    │
│   │   Anomaly    │  產出：每筆日誌的異常分數 (0~1)                         │
│   └──────┬───────┘                                                      │
│          │                                                              │
│          │       ┌───────────────────────────────────────┐              │
│          │       │  STAGE III: 外部知識嵌入               │              │
│          │       │  將 MITRE ATT&CK 描述轉換為 BERT 向量  │              │
│          │       └───────────────┬───────────────────────┘              │
│          │                       │                                      │
│          ▼                       ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  STAGE IV: Per-Dataset 處理 (每個 Dataset 獨立執行)           │      │
│   │                                                              │      │
│   │  ┌────────────┐   ┌────────────┐   ┌────────────┐           │      │
│   │  │ 概念提取   │ → │ 序列分群   │ → │ 自動標註   │           │      │
│   │  │   (NMF)    │   │   (HMM)    │   │ (Labeling) │           │      │
│   │  └────────────┘   └────────────┘   └────────────┘           │      │
│   │                                                              │      │
│   │  • NMF: 與外部知識聯合訓練，降維至 k 維概念空間               │      │
│   │  • HMM: 識別攻擊演變階段，產出群集標籤                        │      │
│   │  • Labeling: 比對 MITRE 技術，分配攻擊技術標籤                │      │
│   └──────────────────────────────────────────────────────────────┘      │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │   輸出結果   │  帶標籤的 CSV 檔案                                    │
│   └──────────────┘  result/Labeling_Results/{dataset_id}_Labeled.csv    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline 四階段詳解

### Stage I：預處理與嵌入 (Preprocessing & Embedding)

>  詳細文件：[Preprocessing.md](./docs/Preprocessing.md)、[Embedding.md](./docs/Embedding.md)、[Templatize.md](./docs/Templatize.md)

#### 目的
將非結構化的原始日誌轉換為**固定維度的語義向量**，並預計算 TF-IDF 稀疏向量供後續混合評分使用。

#### 處理流程

```
原始 CSV 日誌
     │
     ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  日誌解析   │ ──► │  BERT 嵌入  │ ──► │  TF-IDF     │ ──► │  向量儲存   │
│ (Drain)     │     │ (384 dim)   │     │ (稀疏矩陣)  │     │ (Arrow)     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

| 子步驟 | 說明 | 輸出 |
|--------|------|------|
| **日誌解析** | 使用 Drain 演算法將日誌拆解為「模板」+「參數」（可透過 `ENABLE_PARSER` 設定啟用/停用） | `Template`: `CreateFile <*> SUCCESS` |
| **BERT 嵌入** | 將模板轉換為語義向量（支援 Sentence-BERT、SecBERT 等，透過 `BERT_MODEL_NAME` 設定） | `embedding`: `[0.23, -0.15, ...]` |
| **Per-Log TF-IDF** | 使用 MITRE 預訓練 Vectorizer 計算詞彙權重向量（用於 Stage IV 混合評分） | `tfidf.npz`: 稀疏矩陣 |
| **向量儲存** | 以 Arrow 格式高效儲存 | `data/Embeddings/` |

#### API 呼叫

```python
from Pipeline import STAGE_I

# 執行 Stage I（含 TF-IDF 預計算）
result = STAGE_I(N=10, enable_tfidf=True, enable_comparison=False)

# result: {n_loaded, embedding_dim, tfidf_stats}
```

**內部 API**：

| 模組 | 函數 | 說明 |
|------|------|------|
| `preprocess` | `run_preprocessing()` | BERT 嵌入計算 |
| `precompute_log_tfidf` | `run_log_tfidf_precompute()` | Per-Log TF-IDF 預計算 |
| `visualization.bert_comparison` | `BertEmbeddingComparator` | BERT 模型比較（可選） |

#### 配置參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `ENABLE_PARSER` | `True` | 是否啟用日誌解析 |
| `DEFAULT_PARSER` | `"drain"` | 預設解析器 |
| `BERT_MODEL_NAME` | `"sentence-bert"` | BERT 模型名稱 |
| `ZIPF_PERCENTILE` | `0.05` | Zipf 法則高頻詞過濾比例 |
| `MITRE_TFIDF_DIR` | `data/ExternalKnowledge/MITRE_TFIDF` | TF-IDF Vectorizer 目錄 |

#### 輸入輸出

- **輸入**：`data/input_logs/*.csv`（原始日誌 CSV）
- **輸出**：
  - `data/Embeddings/{dataset_id}/data-*.arrow`（BERT 嵌入向量）
  - `data/Embeddings/{dataset_id}/tfidf.npz`（TF-IDF 稀疏矩陣）

---

### Stage II：異常偵測 (Anomaly Detection)

>  詳細文件：[Anomaly_Detection.md](./docs/Anomaly_Detection.md)

#### 目的
識別行為異常的日誌，作為後續標註的**優先權重**——異常分數越高，越可能是攻擊行為。

#### 核心機制

```
                    ┌─── Isolation Forest ───┐
                    │                        │
嵌入向量 ───►         ├─── COPOD ──────────────┼───► Ensemble ───► 異常分數
                    │                        │      加權平均     (0~1)
                    ├─── AutoEncoder ────────┤
                    │                        │
                    └─── PCA + GMM ──────────┘
```

| 演算法 | 原理 | 優勢 |
|--------|------|------|
| **Isolation Forest** | 樹型結構隔離異常點 | 高效處理高維資料 |
| **COPOD** | 機率密度估計 | 對極端值敏感 |
| **AutoEncoder** | 重建誤差檢測 | 捕捉非線性異常 |
| **PCA + GMM** | 降維後高斯混合 | 識別分布外樣本 |

#### 配置參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `DETECTION_MODELS` | `["isolation_forest", "copod", "autoencoder", "pca_gmm"]` | 啟用的偵測模型 |
| `IF_CONTAMINATION` | `0.05` | Isolation Forest 預期異常比例 |
| `MAD_THRESHOLD_MULTIPLIER` | `3.0` | MAD 自適應閾值乘數 |
| `ENSEMBLE_WEIGHTS` | 各模型 0.25 | 模型加權平均權重 |

#### 輸入輸出

- **輸入**：`data/Embeddings/{dataset_id}/`
- **輸出**：`data/Detection_Results/`（含 `ensemble_score` 欄位）、`result/Anomaly_Detection/`（視覺化報告）

---

### Stage III：外部知識嵌入 (External Knowledge)

>  詳細文件：[External_Sources.md](./docs/External_Sources.md)

#### 目的
將 MITRE ATT&CK 攻擊技術描述轉換為 **BERT 嵌入向量**，作為後續標註的比對基準。

#### 處理流程

```
MITRE ATT&CK 技術描述 (CSV)
        │
        ▼
   BERT 嵌入 (768 維)
        │
        ▼
   儲存為外部知識向量
        │
        ▼
   供 Stage IV NMF 聯合訓練使用
```

#### 支援的外部來源

| 來源 | 用途 |
|------|------|
| **MITRE ATT&CK** | 攻擊技術標籤（主要） |
| **CAPEC** | 攻擊模式補充 |
| **NVD/CVE** | 漏洞關聯 |

#### 配置參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `MITRE_TECHNIQUES_CSV` | `data/reference_resources/MitreTechniquesTokens_V5.csv` | MITRE 技術資料路徑 |
| `MITRE_EXTERNAL_KNOWLEDGE_DIR` | `data/ExternalKnowledge/MITRE_RAW_EMBEDDINGS` | 輸出嵌入向量目錄 |

#### 輸入輸出

- **輸入**：`data/reference_resources/MitreTechniquesTokens_V5.csv`
- **輸出**：`data/ExternalKnowledge/MITRE_RAW_EMBEDDINGS/`

---

### Stage IV：Per-Dataset 處理 (NMF → HMM → Auto Labeling)

>  詳細文件：[Concept_Extraction.md](./docs/Concept_Extraction.md)、[Sequence_Clustering.md](./docs/Sequence_Clustering.md)、[Auto_Labeling.md](./docs/Auto_Labeling.md)

#### 目的
對每個 Dataset **獨立執行**完整的概念提取、序列分群與自動標註流程，確保每個攻擊技術的標註不會被其他技術混淆。

#### 核心機制：Per-Dataset 策略

```
對每個 Dataset 獨立執行：

┌─────────────────────────────────────────────────────────────────┐
│  Step 4a: 概念提取 (NMF)                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  X (768 維嵌入) ≈ H (概念權重) × W (概念基矩陣)          │    │
│  │  • 與外部知識聯合訓練統一的概念基矩陣 W                  │    │
│  │  • 產出：k 維概念向量 (k=30)                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                    │
│                            ▼                                    │
│  Step 4b: 序列分群 (HMM)                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • 雙軌特徵：常態化訓練 + 原始資料應用                   │    │
│  │  • 一階差分特徵識別階段轉換點                            │    │
│  │  • BIC 自動選擇最佳狀態數                                │    │
│  │  • 產出：群集標籤序列                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                    │
│                            ▼                                    │
│  Step 4c: 自動標註 (Auto Labeling)                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Cluster Centroid 與 MITRE 向量餘弦相似度比對           │    │
│  │  • 異常分數加權計算信心度                                 │    │
│  │  • 閾值判斷：低於閾值標記為 Benign                        │    │
│  │  • 產出：帶標籤的 CSV 檔案                                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### Step 4a：概念提取 (NMF)

| 策略 | 說明 |
|------|------|
| **全域聯合訓練** | 聚合多資料集 + 外部知識訓練統一的概念基矩陣 W |
| **L1 稀疏約束** | 強制每筆日誌只屬於少數概念，提升可解釋性 |
| **GPU 加速** | 支援 CUDA GPU 加速大規模矩陣運算 |

**配置參數**：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `NMF_COMPONENTS` | `30` | 概念數量（潛在空間維度） |
| `NMF_L1_REG` | `0.01` | L1 正則化強度 |
| `NMF_USE_GPU` | `True` | 是否使用 GPU 加速 |

#### Step 4b：序列分群 (HMM)

| 策略 | 說明 |
|------|------|
| **Per-Dataset 訓練** | 每個資料集獨立訓練 HMM，捕捉特定攻擊的獨有階段 |
| **雙軌特徵** | 訓練用常態化資料（確保收斂）、應用用原始資料（保留語義） |
| **一階差分** | 額外計算變化率特徵，識別攻擊階段的轉換點 |
| **BIC 選擇 K** | 使用貝葉斯資訊量準則自動選擇最佳狀態數 |

**配置參數**：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `HMM_K_MIN` | `1` | 隱藏狀態下界 |
| `HMM_K_MAX` | `15` | 隱藏狀態上界 |
| `HMM_COVARIANCE_TYPE` | `"diag"` | 共變異數型式 |
| `HMM_N_ITER` | `100` | Baum-Welch 最大迭代次數 |

#### Step 4c：自動標註 (Auto Labeling)

**標註流程**：

```
       Cluster Centroid ─────┐
      (異常加權平均)         │
                             ├──► 餘弦相似度 ──► 閾值判斷 ──► 技術標籤
       MITRE 概念向量 ───────┘                      │           或
         (NMF 投影)                                │        "Benign"
       異常分數 ─────────────────► 信心度計算 ──────┘
```

**信心度計算**：

$$\text{confidence} = w_a \times \text{anomaly\_score} + w_s \times \text{similarity}$$

$$\text{final\_score} = \text{similarity} \times \text{confidence}$$

- 若 $\text{similarity} < \text{threshold}$ 或 $\text{final\_score} < \text{threshold}$，標記為 `Benign`
- 否則標記為最相似的 MITRE 技術名稱（如 "PowerShell"、"Video Capture"）

**配置參數**：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `LABELING_SIMILARITY_THRESHOLD` | `0.3` | 相似度下界 |
| `LABELING_CONFIDENCE_THRESHOLD` | `0.2` | 最終分數閾值 |
| `LABELING_ANOMALY_WEIGHT` | `0.3` | 異常分數權重 |
| `LABELING_SIMILARITY_WEIGHT` | `0.7` | 相似度權重 |
| `LABELING_TOP_K` | `3` | 候選技術數量 |

#### 輸入輸出

- **輸入**：
  - `data/Embeddings/{dataset_id}/`（Stage I 產出）
  - `data/Detection_Results/`（Stage II 產出）
  - `data/ExternalKnowledge/`（Stage III 產出）
- **輸出**：
  - `data/ConceptVectors/{dataset_id}/`（概念向量）
  - `data/SequenceClusters/{dataset_id}/`（群集標籤）
  - `result/Labeling_Results/{dataset_id}_Labeled.csv`（最終標註結果）

#### 輸出範例

| original_idx | timestamp | anomaly_score | predicted_technique_1_label | predicted_technique_1_similarity | predicted_technique_1_confidence |
|--------------|-----------|---------------|-----------------------------|---------------------------------|----------------------------------|
| 0 | 2025-01-08 10:30:00 | 0.85 | PowerShell | 0.78 | 0.80 |
| 1 | 2025-01-08 10:30:02 | 0.12 | Benign | 0.45 | 0.35 |

---

## 專案目錄結構

```
Logs-Labeling/
├── Logs Labeling/           # 核心程式碼
│   ├── Pipeline.py          # 主流程控制（四階段）
│   ├── config.py            # 參數配置
│   ├── preprocess/          # Stage I：預處理與嵌入
│   │   ├── preprocess.py    # LogLoader, LogEmbedder
│   │   └── drain.py         # Drain 日誌解析器
│   ├── precompute_log_tfidf.py  # Stage I：Per-Log TF-IDF 預計算
│   ├── anomaly_dection/     # Stage II：異常偵測
│   │   └── log_detector.py  # Ensemble 異常偵測
│   ├── external_sources/    # Stage III：外部知識嵌入
│   │   └── build_mitre_raw_embeddings.py
│   ├── conception_extraction.py  # Stage IV-a：概念提取 (NMF)
│   ├── sequence_clustering.py    # Stage IV-b：序列分群 (HMM)
│   ├── auto_labeling.py     # Stage IV-c：自動標註
│   ├── models/              # 模型相關
│   │   └── bert.py          # BERT 嵌入 API
│   └── visualization/       # 視覺化工具
│
├── data/                    # 資料目錄
│   ├── input_logs/          # 原始日誌
│   ├── reference_resources/ # 外部知識資源（MITRE CSV 等）
│   ├── Embeddings/          # Stage I 輸出（BERT 嵌入向量）
│   ├── Detection_Results/   # Stage II 輸出（異常分數）
│   ├── ExternalKnowledge/   # Stage III 輸出（MITRE 嵌入向量）
│   ├── ConceptVectors/      # Stage IV-a 輸出（概念向量）
│   └── SequenceClusters/    # Stage IV-b 輸出（群集標籤）
│
├── result/                  # 最終結果
│   ├── Anomaly_Detection/   # Stage II 視覺化報告
│   └── Labeling_Results/    # Stage IV-c 輸出（標註結果）
│
├── docs/                    # 詳細文件
│   ├── Preprocessing.md
│   ├── Anomaly_Detection.md
│   ├── Concept_Extraction.md
│   ├── Sequence_Clustering.md
│   ├── External_Sources.md
│   └── Auto_Labeling.md
│
└── requirements.txt         # 依賴套件
```

---

## 快速開始

### 環境安裝

```bash
# 建立 Conda 環境
conda create -n LogsLabeling python=3.10
conda activate LogsLabeling

# 安裝依賴
pip install -r requirements.txt
```

### 準備資料

1. 將原始日誌 CSV 檔案放入 `data/input_logs/`
2. 確保每個 CSV 包含 `Operation`、`Path`、`Result` 等欄位

### 執行 Pipeline

```python
from Pipeline import main

# 執行完整流程（預設處理 5 個資料集）
main()

# 指定處理數量
main(n_datasets=100)
```

或逐階段執行：

```python
from Pipeline import STAGE_I, STAGE_II, STAGE_III, STAGE_IV

# Stage I: 預處理前 50 個資料集並計算 BERT 嵌入 + TF-IDF
STAGE_I(N=50, enable_tfidf=True)

STAGE_II()       # Stage II: 異常偵測
STAGE_III()      # Stage III: 建立 MITRE 外部知識嵌入
STAGE_IV()       # Stage IV: Per-Dataset 處理 (NMF → HMM → 標註)
```

### 查看結果

標註結果位於 `result/Labeling_Results/`，每個資料集產生一個 `{dataset_id}_Labeled.csv`。

---

## 更新日誌

| 日期 | 更新內容 |
|------|---------|
| 2026-01-15 | 重構 Pipeline 為四階段架構，採用 Per-Dataset 策略整合 NMF → HMM → Auto Labeling |
| 2025-12-05 | 新增 BERT 嵌入模組 (`models/bert.py`)、整合 BERT API |
| 2025-11-25 | 更新外部知識爬蟲架構 |
| 2025-11-21 | 建立專案架構、新增 Drain 解析器 |
| 2025-11-16 | 專案初始化 |

---

