# Logs Labeling

透過無監督式機器學習，自動將日誌序列標註為 MITRE ATT&CK 攻擊技術。

---

## 目錄

- [專案簡介](#專案簡介)
- [核心流程概覽](#核心流程概覽)
- [Pipeline 六階段詳解](#pipeline-六階段詳解)
  - [Stage I：預處理 (Preprocessing)](#stage-i預處理-preprocessing)
  - [Stage II：異常偵測 (Anomaly Detection)](#stage-ii異常偵測-anomaly-detection)
  - [Stage III：概念提取 (Concept Extraction)](#stage-iii概念提取-concept-extraction)
  - [Stage IV：序列分群 (Sequence Clustering)](#stage-iv序列分群-sequence-clustering)
  - [Stage V：外部知識嵌入 (External Knowledge)](#stage-v外部知識嵌入-external-knowledge)
  - [Stage VI：自動標註 (Auto Labeling)](#stage-vi自動標註-auto-labeling)
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

本專案將日誌標註拆解為 **六個獨立階段**，每個階段各司其職：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Logs Labeling Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   [原始日誌]                                                             │
│       │                                                                 │
│       ▼                                                                 │
│   ┌──────────────┐                                                      │
│   │  STAGE I     │  預處理：日誌解析 → BERT 嵌入 → 向量化                   │
│   │ Preprocessing│  產出：768 維語義向量                                  │
│   └──────┬───────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │  STAGE II    │  異常偵測：Ensemble 模型識別異常日誌                    │
│   │   Anomaly    │  產出：每筆日誌的異常分數 (0~1)                         │
│   └──────┬───────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │  STAGE III   │  概念提取：NMF 降維至潛在概念空間                       │
│   │   Concept    │  產出：k 維概念向量 (如 k=50)                          │
│   └──────┬───────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │  STAGE IV    │  序列分群：HMM 識別攻擊演變階段                         │
│   │  Clustering  │  產出：每筆日誌的群集標籤                               │
│   └──────┬───────┘                                                      │
│          │                                                              │
│          ├──────────────────────────────────┐                           │
│          │                                  │                           │
│          ▼                                  ▼                           │
│   ┌──────────────┐                   ┌──────────────┐                   │
│   │  STAGE V     │                   │ MITRE ATT&CK │                   │
│   │ External KB  │ ◄─────────────────│   知識庫     │                   │
│   └──────┬───────┘                   └──────────────┘                   │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │  STAGE VI    │  自動標註：群集 × MITRE 相似度 → 攻擊技術標籤            │
│   │ Auto Labeling│  產出：帶標籤的 CSV 檔案                               │
│   └──────────────┘                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline 六階段詳解

### Stage I：預處理 (Preprocessing)

> 📄 詳細文件：[Preprocessing.md](./docs/Preprocessing.md)、[Embedding.md](./docs/Embedding.md)、[Templatize.md](./docs/Templatize.md)

#### 目的
將非結構化的原始日誌轉換為**固定維度的語義向量**，使後續機器學習模型能夠處理。

#### 處理流程

```
原始 CSV 日誌
     │
     ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  日誌解析   │ ──► │  BERT 嵌入  │ ──► │  向量儲存   │
│ (Drain)     │     │ (768 dim)   │     │ (Arrow)     │
└─────────────┘     └─────────────┘     └─────────────┘
```

| 子步驟 | 說明 | 輸出 |
|--------|------|------|
| **日誌解析** | 使用 Drain 演算法將日誌拆解為「模板」+「參數」 | `Template`: `CreateFile <*> SUCCESS` |
| **BERT 嵌入** | 將模板轉換為 768 維向量（支援 SecBERT、Sentence-BERT） | `embedding`: `[0.23, -0.15, ...]` |
| **向量儲存** | 以 Arrow 格式高效儲存 | `data/Embeddings/` |

#### 輸入輸出

- **輸入**：`data/input_logs/*.csv`（原始日誌 CSV）
- **輸出**：`data/Embeddings/{dataset_id}/`（嵌入向量資料集）

---

### Stage II：異常偵測 (Anomaly Detection)

> 📄 詳細文件：[Anomaly_Detection.md](./docs/Anomaly_Detection.md)

#### 目的
識別行為異常的日誌，作為後續標註的**優先權重**——異常分數越高，越可能是攻擊行為。

#### 核心機制

```
                    ┌─── Isolation Forest ───┐
                    │                        │
嵌入向量 ───►      ├─── COPOD ──────────────┼───► Ensemble ───► 異常分數
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

#### 輸入輸出

- **輸入**：`data/Embeddings/{dataset_id}/`
- **輸出**：`data/Detection_Results/`（含 `ensemble_score` 欄位）

---

### Stage III：概念提取 (Concept Extraction)

> 📄 詳細文件：[Concept_Extraction.md](./docs/Concept_Extraction.md)

#### 目的
將高維嵌入（768 維）降維至**潛在概念空間**（如 50 維），使日誌與 MITRE 技術可在同一空間進行比較。

#### 核心機制：NMF（非負矩陣分解）

```
X (768 維嵌入)  ≈  H (概念權重)  ×  W (概念基矩陣)
   n × 768           n × k            k × 768

X：原始嵌入矩陣
H：每筆日誌在 k 個概念上的權重分佈
W：k 個概念的定義（由全域訓練學習）
```

#### 關鍵設計

| 策略 | 說明 |
|------|------|
| **全域聯合訓練** | 聚合多資料集 + 外部知識訓練統一的概念基矩陣 W |
| **L1 稀疏約束** | 強制每筆日誌只屬於少數概念，提升可解釋性 |
| **獨立批次轉換** | 使用固定的 W 對各資料集進行轉換，確保可比性 |

#### 輸入輸出

- **輸入**：`data/Embeddings/`、`data/ExternalKnowledge/`
- **輸出**：`data/ConceptVectors/{dataset_id}/`、`models/nmf_concept_model.pkl`

---

### Stage IV：序列分群 (Sequence Clustering)

> 📄 詳細文件：[Sequence_Clustering.md](./docs/Sequence_Clustering.md)

#### 目的
識別日誌序列中的**攻擊演變階段**（如：初始存取 → 執行 → 持久化 → 清理），將同一階段的日誌歸為同一群集。

#### 核心機制：HMM（隱馬可夫模型）

```
觀測序列：[概念向量 1] → [概念向量 2] → [概念向量 3] → ...
              ↓              ↓              ↓
隱藏狀態：  [狀態 A]   →   [狀態 A]   →   [狀態 B]   → ...
           (初始存取)     (初始存取)     (執行階段)
```

#### 關鍵設計

| 策略 | 說明 |
|------|------|
| **Per-Dataset 訓練** | 每個資料集獨立訓練 HMM，捕捉特定攻擊的獨有階段 |
| **雙軌特徵** | 訓練用常態化資料（確保收斂）、應用用原始資料（保留語義） |
| **一階差分** | 額外計算變化率特徵，識別攻擊階段的轉換點 |
| **BIC 選擇 K** | 使用貝葉斯資訊量準則自動選擇最佳狀態數 |

#### 輸入輸出

- **輸入**：`data/ConceptVectors/{dataset_id}/`
- **輸出**：`data/SequenceClusters/{dataset_id}/`（`labels.npy`、`model.pkl`）

---

### Stage V：外部知識嵌入 (External Knowledge)

> 📄 詳細文件：[External_Sources.md](./docs/External_Sources.md)

#### 目的
將 MITRE ATT&CK 攻擊技術描述轉換為**與日誌相同格式的概念向量**，作為標註的比對基準。

#### 處理流程

```
MITRE ATT&CK 技術描述
        │
        ▼
   BERT 嵌入 (768 維)
        │
        ▼
   NMF 轉換 (使用 Stage III 訓練的 W)
        │
        ▼
   概念向量 (k 維)
```

#### 支援的外部來源

| 來源 | 用途 |
|------|------|
| **MITRE ATT&CK** | 攻擊技術標籤（主要） |
| **CAPEC** | 攻擊模式補充 |
| **NVD/CVE** | 漏洞關聯 |

#### 輸入輸出

- **輸入**：`data/reference_resources/MitreTechniques.csv`
- **輸出**：`data/ExternalKnowledge/MITRE_ATTACK/`

---

### Stage VI：自動標註 (Auto Labeling)

> 📄 詳細文件：[Auto_Labeling.md](./docs/Auto_Labeling.md)

#### 目的
將 HMM 分群結果與 MITRE ATT&CK 比對，自動為每筆日誌分配攻擊技術標籤。

#### 標註流程

```
       Cluster Centroid ─────┐
      (異常加權平均)         │
                             ├──► 餘弦相似度 ──► 閾值判斷 ──► 技術標籤
       MITRE 概念向量 ───────┘                      │           或
                                                    │        "Benign"
       異常分數 ─────────────────► 信心度調整 ──────┘
```

#### 信心度計算

$$\text{confidence} = w_a \times \text{anomaly\_score} + w_s \times \text{similarity}$$

$$\text{final\_score} = \text{similarity} \times \text{confidence}$$

- 若 $\text{final\_score} < \text{threshold}$，標記為 `Benign`
- 否則標記為最相似的 MITRE 技術 ID（如 `T1059.001`）

#### 輸出範例

| log_index | cluster_id | predicted_technique | similarity | confidence |
|-----------|------------|---------------------|------------|------------|
| 0 | 2 | T1059.001 (PowerShell) | 0.78 | 0.80 |
| 1 | 0 | Benign | 0.45 | 0.33 |

#### 輸入輸出

- **輸入**：`data/ConceptVectors/`、`data/SequenceClusters/`、`data/Detection_Results/`、`data/ExternalKnowledge/`
- **輸出**：`result/Labeling_Results/{dataset_id}_Labeled.csv`

---

## 專案目錄結構

```
Logs-Labeling/
├── Logs Labeling/           # 核心程式碼
│   ├── Pipeline.py          # 主流程控制
│   ├── config.py            # 參數配置
│   ├── preprocess/          # Stage I：預處理
│   │   ├── preprocess.py    # LogLoader, LogEmbedder
│   │   └── drain.py         # Drain 日誌解析器
│   ├── anomaly_dection/     # Stage II：異常偵測
│   │   └── log_detector.py  # Ensemble 異常偵測
│   ├── conception_extraction.py  # Stage III：概念提取
│   ├── sequence_clustering.py    # Stage IV：序列分群
│   ├── external_sources/    # Stage V：外部知識
│   │   └── build_mitre_raw_embeddings.py
│   ├── auto_labeling.py     # Stage VI：自動標註
│   ├── models/              # 模型相關
│   │   └── bert.py          # BERT 嵌入 API
│   └── visualization/       # 視覺化工具
│
├── data/                    # 資料目錄
│   ├── input_logs/          # 原始日誌
│   ├── Embeddings/          # Stage I 輸出
│   ├── Detection_Results/   # Stage II 輸出
│   ├── ConceptVectors/      # Stage III 輸出
│   ├── SequenceClusters/    # Stage IV 輸出
│   └── ExternalKnowledge/   # Stage V 輸出
│
├── result/                  # 最終結果
│   └── Labeling_Results/    # Stage VI 輸出
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

# 執行完整流程
main()
```

或逐階段執行：

```python
from Pipeline import STAGE_I, STAGE_II, STAGE_III, STAGE_IV, STAGE_V, STAGE_VI

STAGE_I(N=50)    # 處理前 50 個資料集
STAGE_II()       # 異常偵測
STAGE_III()      # 概念提取
STAGE_IV()       # 序列分群
STAGE_V()        # 建立 MITRE 嵌入
STAGE_VI()       # 自動標註
```

### 查看結果

標註結果位於 `result/Labeling_Results/`，每個資料集產生一個 `{dataset_id}_Labeled.csv`。

---

## 更新日誌

| 日期 | 更新內容 |
|------|---------|
| 2025-12-05 | 新增 BERT 嵌入模組 (`models/bert.py`)、整合 BERT API |
| 2025-11-25 | 更新外部知識爬蟲架構 |
| 2025-11-21 | 建立專案架構、新增 Drain 解析器 |
| 2025-11-16 | 專案初始化 |

---

