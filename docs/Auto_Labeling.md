# Auto Labeling（自動標註）

## 概述

自動標註模組 (`auto_labeling.py`) 是 Logs Labeling Pipeline 的最終階段，負責將 HMM 序列分群結果與 MITRE ATT&CK 外部知識進行比對，自動為每筆日誌標註對應的攻擊技術。

## 核心概念

### 標註流程

```
Cluster Centroid ──┐
  (異常加權平均)  │
                   ├──► Cosine Similarity ──► Thresholding ──► Technique Label
MITRE Concept    ──┘                              ▲              或 "Benign"
   (NMF 投影)                                     │
Anomaly Score ─────────────────► Confidence ──────┘
```

1. **Cluster Centroid 計算**：對每個 HMM 隱藏狀態（Cluster），使用異常分數作為權重計算加權平均 Centroid
2. **概念空間映射**：使用相同的 NMF 模型將 MITRE ATT&CK 嵌入投影至概念空間
3. **相似度計算**：計算 Cluster Centroid 與各 MITRE 技術向量的餘弦相似度
4. **信心度整合**：結合異常分數與相似度計算綜合信心度
5. **閾值決策**：根據 `similarity × confidence` 決定最終標籤，低於閾值則標記為 `Benign`

### 信心度計算

$$
\text{confidence} = w_a \times \text{anomaly\_score} + w_s \times \text{similarity}
$$

$$
\text{final\_score} = \text{similarity} \times \text{confidence}
$$

其中：
- $w_a$：異常分數權重（預設 0.3）
- $w_s$：相似度權重（預設 0.7）
- 若 $\text{similarity} < \text{similarity\_threshold}$ 或 $\text{final\_score} < \text{confidence\_threshold}$，則標記為 `Benign`

### 異常分數的作用

- **高異常分數**：表示該日誌在行為上與正常模式差異較大，更可能是攻擊行為
- **低異常分數**：表示該日誌較接近正常模式，即使與某攻擊技術相似度高，也應降低信心度
- **加權 Centroid**：計算 Cluster Centroid 時，高異常分數的樣本會被賦予更高的權重

### 混合評分機制（Embedding + TF-IDF）

自動標註支援**雙軌評分**，結合語義嵌入與詞彙匹配：

$$
\text{Score}_{hybrid} = \alpha \times \text{Sim}_{embedding} + (1 - \alpha) \times \text{Sim}_{tfidf}
$$

其中：
- $\alpha$：Embedding 權重（預設 0.7）
- $\text{Sim}_{embedding}$：Cluster Centroid 與 MITRE 嵌入的餘弦相似度
- $\text{Sim}_{tfidf}$：Cluster 平均 TF-IDF 向量與 MITRE TF-IDF 的餘弦相似度

**啟用條件**：
- Stage I 需啟用 `enable_tfidf=True`（生成 `tfidf.npz`）
- `data/ExternalKnowledge/MITRE_TFIDF/` 需存在 `tfidf_matrix.npz`

若 TF-IDF 資料不存在，系統自動回退至純 Embedding 評分。

## 配置參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `LABELING_SIMILARITY_THRESHOLD` | 0.3 | 相似度下界，低於此值視為不匹配，標記為 Benign |
| `LABELING_CONFIDENCE_THRESHOLD` | 0.2 | 最終分數閾值，低於此值標記為 Benign |
| `LABELING_ANOMALY_WEIGHT` | 0.3 | 異常分數在信心度計算中的權重 |
| `LABELING_SIMILARITY_WEIGHT` | 0.7 | 相似度在信心度計算中的權重 |
| `LABELING_EMBEDDING_WEIGHT` | 0.7 | 混合評分中 Embedding 權重（$\alpha$） |
| `LABELING_TOP_K` | 3 | 輸出的候選技術數量 |
| `LABELING_RESULTS_DIR` | `result/Labeling_Results/` | 標註結果輸出目錄 |
| `MITRE_EXTERNAL_KNOWLEDGE_DIR` | `data/ExternalKnowledge/MITRE_RAW_EMBEDDINGS` | MITRE 嵌入向量目錄 |
| `MITRE_TFIDF_DIR` | `data/ExternalKnowledge/MITRE_TFIDF` | MITRE TF-IDF 向量目錄 |

## 使用方式

### 在 Pipeline 中執行

```python
# 在 Pipeline Stage IV 中自動執行
from Pipeline import STAGE_IV

STAGE_IV()  # 執行 Per-Dataset 處理：NMF → HMM → 自動標註
```

### 獨立執行

```python
from auto_labeling import AutoLabeler

# 建立標註器
labeler = AutoLabeler()

# 載入 MITRE 嵌入與 TF-IDF
labeler.load_mitre_embeddings()
labeler.load_mitre_tfidf()  # 用於混合評分

# 標註單一資料集（需要事先準備好 concept_vectors 和 cluster_labels）
result = labeler.process_single_dataset(
    dataset_id="dataset_001",
    concept_vectors=concept_vectors,  # NMF 轉換後的概念向量
    cluster_labels=cluster_labels,    # HMM 分群標籤
    output_dir="result/Labeling_Results/",
    nmf_extractor=extractor,          # NMF 提取器實例
)
```

### Pipeline 整合方式

在 Pipeline.py 的 STAGE_IV 中，自動標註作為 Per-Dataset 處理的最後一步執行：

```python
# 在 STAGE_IV 中的處理流程
for dataset_id in all_datasets:
    # Step 4a: NMF 概念提取
    concept_vectors = extractor.process_single_dataset(...)
    
    # Step 4b: HMM 序列分群
    cluster_labels = clusterer.process_single_dataset(...)
    
    # Step 4c: 自動標註
    labeling_result = labeler.process_single_dataset(
        dataset_id=dataset_id,
        concept_vectors=concept_vectors,
        cluster_labels=cluster_labels,
        output_dir=config.LABELING_RESULTS_DIR,
        nmf_extractor=extractor,
    )
```

## 輸出格式

標註結果為 CSV 檔案，儲存於 `result/Labeling_Results/{dataset_id}_Labeled.csv`。

### 欄位說明

| 欄位 | 型別 | 說明 |
|------|------|------|
| `original_idx` | int | 日誌在原始資料集中的索引 |
| `{original_columns}` | various | 原始日誌的所有欄位（如 timestamp, event_type, message 等） |
| `anomaly_score` | float | 該日誌的異常偵測分數（0-1） |
| `predicted_technique_1_label` | str | Top-1 預測標籤（技術名稱或 "Benign"） |
| `predicted_technique_1_name` | str | Top-1 預測的 MITRE 技術名稱（如 "PowerShell"、"Video Capture"） |
| `predicted_technique_1_similarity` | float | Top-1 的餘弦相似度（0-1） |
| `predicted_technique_1_confidence` | float | Top-1 的綜合信心度（結合異常分數與相似度） |
| `predicted_technique_K_*` | various | Top-K 預測的相應欄位 |

> **注意**：
> - `label` 欄位根據閾值判斷決定，可能為技術名稱或 "Benign"
> - `name` 欄位始終為 MITRE 技術名稱（可讀格式，如 "Video Capture"）

### 輸出範例

```csv
original_idx,timestamp,event_type,message,anomaly_score,predicted_technique_1_label,predicted_technique_1_name,predicted_technique_1_similarity,predicted_technique_1_confidence,predicted_technique_2_label,predicted_technique_2_name,predicted_technique_2_similarity,predicted_technique_2_confidence
0,2025-01-08 10:30:00,Process Create,powershell.exe -enc...,0.85,PowerShell,PowerShell,0.78,0.80,Command and Scripting Interpreter,Command and Scripting Interpreter,0.72,0.76
1,2025-01-08 10:30:01,Process Create,powershell.exe -nop...,0.62,PowerShell,PowerShell,0.78,0.75,Command and Scripting Interpreter,Command and Scripting Interpreter,0.72,0.73
2,2025-01-08 10:30:02,Network Connect,svchost.exe connecting...,0.12,Benign,Application Layer Protocol,0.45,0.35,Benign,Web Protocols,0.42,0.33
```

## 資料流依賴（四階段架構）

```
STAGE_I (Preprocessing & Embedding)
    │
    ▼
┌─────────────────────┐
│ data/Embeddings/    │  ← BERT 嵌入向量
└─────────────────────┘
    │
    ├─────────────────────────────────────────┐
    ▼                                         ▼
STAGE_II (Anomaly Detection)           STAGE_III (External Knowledge)
    │                                         │
    ▼                                         ▼
┌─────────────────────┐               ┌─────────────────────┐
│ Detection_Results/  │               │ ExternalKnowledge/  │
│ ensemble_scores     │               │ MITRE_RAW_EMBEDDINGS│
└─────────────────────┘               └─────────────────────┘
    │                                         │
    └────────────────┬────────────────────────┘
                     ▼
         STAGE_IV (Per-Dataset 處理)
         ┌───────────────────────────────────┐
         │  NMF → HMM → Auto Labeling        │
         │                                   │
         │  ┌───────────┐  ┌───────────┐     │
         │  │ConceptVec │→ │SequenceCluster│ │
         │  └───────────┘  └───────────┘     │
         │                       │           │
         │                       ▼           │
         │              ┌───────────────┐    │
         │              │ Auto Labeling │    │
         │              └───────────────┘    │
         └───────────────────┬───────────────┘
                             ▼
                 ┌─────────────────────┐
                 │ Labeling_Results/   │
                 │ {id}_Labeled.csv    │
                 └─────────────────────┘
```

## MITRE ATT&CK 整合

### 嵌入向量來源

自動標註模組支援多種 MITRE 嵌入格式：

1. **NumPy 格式**：`embeddings.npy` + `metadata.csv`
2. **Arrow 格式**：`data-00000-of-00001.arrow`（包含 `embedding`, `technique_id`, `technique` 欄位）

### 建立 MITRE 嵌入（Stage III）

```bash
cd "Logs Labeling/external_sources"
python build_mitre_raw_embeddings.py --bert-model sentence-bert
```

這會將 MITRE ATT&CK 技術描述轉換為 BERT 嵌入向量，儲存於 `data/ExternalKnowledge/MITRE_RAW_EMBEDDINGS/`。

## 注意事項

1. **Pipeline 整合**：自動標註是 STAGE_IV Per-Dataset 處理的最後一步，無需單獨執行
2. **概念空間一致性**：MITRE 嵌入會使用與日誌相同的 NMF 模型進行投影，確保向量空間一致
3. **異常分數整合**：異常分數用於 Centroid 加權平均及信心度計算，若無異常偵測結果則使用預設值 0.5
4. **閾值判斷**：低於相似度閾值或信心度閾值的預測會被標記為 Benign
5. **Top-K 輸出**：每筆日誌輸出 K 個候選技術及其標籤、相似度、信心度
6. **原始日誌合併**：標註結果會自動合併原始日誌欄位，需確保 `input_logs/` 中存在對應的 CSV 檔案
7. **技術名稱格式**：輸出使用可讀的技術名稱（如 "Video Capture"），而非 UUID 格式

## 相關模組

- [Preprocessing](./Preprocessing.md) - Stage I：日誌預處理與嵌入
- [Anomaly_Detection](./Anomaly_Detection.md) - Stage II：異常偵測
- [External_Sources](./External_Sources.md) - Stage III：外部知識整合
- [Concept_Extraction](./Concept_Extraction.md) - Stage IV-a：概念提取（NMF）
- [Sequence_Clustering](./Sequence_Clustering.md) - Stage IV-b：序列分群（HMM）
