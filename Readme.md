# Logs Labeling

透過無監督式機器學習，自動將日誌序列標註為 MITRE ATT&CK 攻擊技術。

---

## 目錄

- [專案簡介](#專案簡介)
- [核心流程概覽](#核心流程概覽)
- [Pipeline 三階段詳解](#pipeline-三階段詳解)
  - [Stage I：輸入處理 (Input Processing)](#stage-i輸入處理-input-processing)
  - [Stage II：異常偵測 (Anomaly Detection)](#stage-ii異常偵測-anomaly-detection)
  - [Stage III：Per-Dataset 處理 (NMF → HMM → Auto Labeling)](#stage-iiiper-dataset-處理-nmf--hmm--auto-labeling)
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

本專案採用 **Per-Dataset 策略**，將日誌標註拆解為 **三個獨立階段**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Logs Labeling Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   [原始日誌] + [MITRE ATT&CK]                                           │
│       │                                                                 │
│       ▼                                                                 │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  STAGE I: 輸入處理 (Input Processing)                        │      │
│   │                                                              │      │
│   │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │      │
│   │  │ Log Datasets │   │  Reference   │   │   TF-IDF     │     │      │
│   │  │ Parse→Embed  │   │  Embedding   │   │  Pipeline    │     │      │
│   │  └──────────────┘   └──────────────┘   └──────────────┘     │      │
│   │  產出：Log Vectors + Reference Vectors + TF-IDF 指紋        │      │
│   └──────────────────────────────────────────────────────────────┘      │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │  STAGE II    │  異常偵測：Ensemble 模型識別異常日誌                    │
│   │   Anomaly    │  產出：每筆日誌的異常分數 (0~1)                         │
│   └──────┬───────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  STAGE III: Per-Dataset 處理 (每個 Dataset 獨立執行)          │      │
│   │                                                              │      │
│   │  ┌────────────┐   ┌────────────┐   ┌─────────────────┐      │      │
│   │  │ 概念提取   │ → │ 序列分群   │ → │  自動標註       │      │      │
│   │  │   (NMF)    │   │   (HMM)    │   │(Threat Confid.) │      │      │
│   │  └────────────┘   └────────────┘   └─────────────────┘      │      │
│   │                                                              │      │
│   │  • NMF: 與外部知識聯合訓練，降維至 k 維概念空間               │      │
│   │  • HMM: 識別攻擊演變階段，產出群集標籤                        │      │
│   │  • Labeling: Similarity + Anomaly = Threat Confidence         │      │
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

## Pipeline 三階段詳解

### Stage I：輸入處理 (Input Processing)

>  詳細文件：[Preprocessing.md](./docs/Preprocessing.md)、[Embedding.md](./docs/Embedding.md)、[TF-IDF.md](./docs/TF-IDF.md)

#### 目的
統一處理所有輸入資料：將原始日誌轉換為語義向量，同時為 MITRE ATT&CK 技術建立嵌入與 TF-IDF 指紋。

#### 處理流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Stage I: 輸入處理                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Log Datasets                        Reference Sources (MITRE)      │
│  ┌─────────────────────────┐        ┌─────────────────────────┐     │
│  │ Parse → Embed → Chunk   │        │ Embedding + TF-IDF 指紋 │     │
│  └───────────┬─────────────┘        └───────────┬─────────────┘     │
│              │                                  │                   │
│              ▼                                  ▼                   │
│  ┌─────────────────┐                ┌─────────────────┐             │
│  │ Log Vectors     │                │ MITRE Embedding │             │
│  │ + tfidf.npz     │                │ + TF-IDF Matrix │             │
│  └─────────────────┘                └─────────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| 子步驟 | 說明 | 輸出 |
|--------|------|------|
| **日誌解析** | 使用 Drain 演算法拆解為「模板」+「參數」 | `Template`: `CreateFile <*> SUCCESS` |
| **BERT 嵌入** | 將日誌轉換為語義向量 | `embedding`: `[0.23, -0.15, ...]` |
| **Reference Embedding** | 將 MITRE 技術描述轉換為 BERT 向量 | `data/ExternalKnowledge/MITRE_RAW_EMBEDDINGS/` |
| **TF-IDF 指紋** | 建立 MITRE 技術的詞彙指紋 | `mitre_tfidf_matrix.pkl` |
| **Log TF-IDF** | 使用共享 Vectorizer 轉換日誌 | `tfidf.npz`: 稀疏矩陣 |

> 詳細配置參數請參考：[Preprocessing.md](./docs/Preprocessing.md)、[TF-IDF.md](./docs/TF-IDF.md)

#### 輸入輸出

- **輸入**：
  - `data/input_logs/*.csv`（原始日誌）
  - `data/reference_resources/MitreTechniquesTokens_V5.csv`（MITRE 技術）
- **輸出**：
  - `data/Embeddings/{dataset_id}/`（Log Vectors + TF-IDF）
  - `data/ExternalKnowledge/MITRE_RAW_EMBEDDINGS/`（Reference Embedding）
  - `data/ExternalKnowledge/MITRE_TFIDF/`（TF-IDF 指紋）

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

> 詳細配置參數與各模型說明請參考：[Anomaly_Detection.md](./docs/Anomaly_Detection.md)、[docs/unsupervised model/](./docs/unsupervised%20model/)

#### 輸入輸出

- **輸入**：`data/Embeddings/{dataset_id}/`
- **輸出**：`data/Detection_Results/`（含 `ensemble_score` 欄位）、`result/Anomaly_Detection/`（視覺化報告）

---

### Stage III：Per-Dataset 處理 (NMF → HMM → Auto Labeling)

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
│  │  • Per-Dataset 策略：每個 Dataset 獨立訓練 NMF 模型        │    │
│  │  • 與外部知識聯合訓練，外部知識作為「語義錨點」        │    │
│  │  • 產出：k 維概念向量                                     │    │
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
│  │  • 混合評分：Embedding + TF-IDF + 雙高加分                 │    │
│  │  • 結合異常分數計算威脅信心度                             │    │
│  │  • 產出：帶標籤的 CSV 檔案                                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### Step 4a：概念提取 (NMF)

| 策略 | 說明 |
|------|------|
| **Per-Dataset 局部訓練** | 每個 Dataset 獨立訓練 NMF 模型，與外部知識聯合訓練 |
| **語義錨點機制** | 外部知識作為「語義錨點」引導 NMF 學習與已知攻擊模式相關的概念 |
| **L1 稀疏約束** | 強制每筆日誌只屬於少數概念，提升可解釋性 |
| **GPU 加速** | 支援 CUDA GPU 加速大規模矩陣運算 |

> 詳細配置請參考：[Concept_Extraction.md](./docs/Concept_Extraction.md)

#### Step 4b：序列分群 (HMM)

| 策略 | 說明 |
|------|------|
| **Per-Dataset 訓練** | 每個資料集獨立訓練 HMM，捕捉特定攻擊的獨有階段 |
| **雙軌特徵** | 訓練用常態化資料（確保收斂）、應用用原始資料（保留語義） |
| **一階差分** | 額外計算變化率特徵，識別攻擊階段的轉換點 |
| **BIC 選擇 K** | 使用貝葉斯資訊量準則自動選擇最佳狀態數 |

> 詳細配置請參考：[Sequence_Clustering.md](./docs/Sequence_Clustering.md)

#### Step 4c：自動標註 (Auto Labeling)

**雙層評分架構**：

```
Layer 1: Similarity Score (與 Technique 的相似程度)
─────────────────────────────────────────────────────
       Cluster Centroid ─────┐
                             ├──► Embedding Sim ──┐
       MITRE Embedding ──────┘                    │
                                                  ├──► Similarity Score
       Sequence TF-IDF ──────┐                    │
                             ├──► TF-IDF Sim ─────┤
       MITRE TF-IDF 指紋 ────┘                    │
                                                  │
       Dual-High Boost ───────────────────────────┘

Layer 2: Threat Confidence (最終威脅信心度)
─────────────────────────────────────────────────────
       Similarity Score ────┐
       (α = 0.7)            │
                            ├──► Threat Confidence ──► Top-K Label
       Anomaly Score ───────┘
       (β = 0.3, Stage II)
```

**評分公式**：

1. **Similarity Score**（與 Technique 的相似程度）：

$$
\text{Similarity} = w_{emb} \times \text{Sim}_{embedding} + w_{tfidf} \times \text{Sim}_{tfidf} + \text{Boost}
$$

2. **Threat Confidence**（最終威脅信心度）：

$$
\text{Threat Confidence} = \alpha \times \text{Similarity Score} + \beta \times \text{Anomaly Score}
$$

- **Similarity Score**：Sequence 與 Technique 的語義/詞彙相似度
- **Anomaly Score**：Stage II 異常偵測結果 (0~1)，代表 raw event 的惡意可能性

**預設權重**：
- Similarity: $w_{emb} = 0.6$, $w_{tfidf} = 0.3$, $w_{boost} = 0.1$
- Threat: $\alpha = 0.7$, $\beta = 0.3$

> 詳細配置請參考：[Auto_Labeling.md](./docs/Auto_Labeling.md)

#### 輸入輸出

- **輸入**：
  - `data/Embeddings/{dataset_id}/`（Stage I 產出）
  - `data/Detection_Results/`（Stage II 產出）
  - `data/ExternalKnowledge/`（Stage I 產出：MITRE Embedding + TF-IDF）
- **輸出**：
  - `data/ConceptVectors/{dataset_id}/`（概念向量）
  - `data/SequenceClusters/{dataset_id}/`（群集標籤）
  - `result/Labeling_Results/{dataset_id}_Labeled.csv`（最終標註結果）

#### 輸出範例

| original_idx | anomaly_score | groundtruth_tid | predicted_technique_1_name | predicted_technique_1_similarity | predicted_technique_1_threat_confidence |
|--------------|---------------|-----------------|----------------------------|---------------------------------|-----------------------------------------|
| 0 | 0.85 | T1059 | PowerShell | 0.82 | 0.78 |
| 1 | 0.12 | T1059 | Application Layer Protocol | 0.45 | 0.35 |

---

## 專案目錄結構

```
Logs-Labeling/
├── Logs Labeling/           # 核心程式碼
│   ├── Pipeline.py          # 主流程控制（三階段）
│   ├── config.py            # 參數配置
│   ├── preprocess/          # Stage I：輸入處理
│   │   ├── __init__.py      # process_all_inputs(), run_preprocessing()
│   │   ├── loader.py        # LogLoader
│   │   ├── embedder.py      # LogEmbedder
│   │   └── drain.py         # Drain 日誌解析器
│   ├── precompute_log_tfidf.py  # Stage I：TF-IDF Pipeline
│   ├── anomaly_dection/     # Stage II：異常偵測
│   │   └── log_detector.py  # Ensemble 異常偵測
│   ├── external_sources/    # Reference Embedding 工具
│   │   └── build_mitre_raw_embeddings.py
│   ├── conception_extraction.py  # Stage III-a：概念提取 (NMF)
│   ├── sequence_clustering.py    # Stage III-b：序列分群 (HMM)
│   ├── auto_labeling.py     # Stage III-c：自動標註（混合評分）
│   ├── models/              # 模型相關
│   │   └── bert.py          # BERT 嵌入 API
│   └── visualization/       # 視覺化工具
│
├── data/                    # 資料目錄
│   ├── input_logs/          # 原始日誌
│   ├── reference_resources/ # 外部知識資源（MITRE CSV 等）
│   ├── Embeddings/          # Stage I 輸出（Log Vectors + TF-IDF）
│   ├── Detection_Results/   # Stage II 輸出（異常分數）
│   ├── ExternalKnowledge/   # Stage I 輸出（MITRE Embedding + TF-IDF 指紋）
│   ├── ConceptVectors/      # Stage III-a 輸出（概念向量）
│   └── SequenceClusters/    # Stage III-b 輸出（群集標籤）
│
├── result/                  # 最終結果
│   ├── Anomaly_Detection/   # Stage II 視覺化報告
│   └── Labeling_Results/    # Stage III-c 輸出（標註結果）
│
├── docs/                    # 詳細文件
│   ├── Preprocessing.md
│   ├── Anomaly_Detection.md
│   ├── Concept_Extraction.md
│   ├── Sequence_Clustering.md
│   ├── TF-IDF.md
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
from Pipeline import STAGE_I, STAGE_II, STAGE_III

# Stage I: 處理所有輸入（Log Datasets + Reference + TF-IDF）
STAGE_I(N=50, enable_tfidf=True)

STAGE_II()       # Stage II: 異常偵測
STAGE_III()      # Stage III: Per-Dataset 處理 (NMF → HMM → 標註)
```

### 查看結果

標註結果位於 `result/Labeling_Results/`，每個資料集產生一個 `{dataset_id}_Labeled.csv`。

---

## 更新日誌

| 日期 | 更新內容 |
|------|---------|
| 2026-01-30 | **重構為三階段架構**：Stage I 統一處理 Log + Reference + TF-IDF；移除獨立的 Stage III (External Knowledge)；新增混合評分機制（Embedding + TF-IDF + 雙高加分） |
| 2026-01-15 | 重構 Pipeline 為四階段架構，採用 Per-Dataset 策略整合 NMF → HMM → Auto Labeling |
| 2025-12-05 | 新增 BERT 嵌入模組 (`models/bert.py`)、整合 BERT API |
| 2025-11-25 | 更新外部知識爬蟲架構 |
| 2025-11-21 | 建立專案架構、新增 Drain 解析器 |
| 2025-11-16 | 專案初始化 |

---

