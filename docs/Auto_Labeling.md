# Auto Labeling（自動標註）

## 概述

自動標註模組 (`auto_labeling.py`) 是 Logs Labeling Pipeline 的最終階段，負責將 HMM 序列分群結果與 MITRE ATT&CK 外部知識進行比對，自動為每筆日誌標註對應的攻擊技術。

## 核心概念

### 標註流程

```
Cluster Centroid ──┐
                   ├──► Cosine Similarity ──► Thresholding ──► Technique Label
MITRE Concept    ──┘                              ▲
                                                  │
Anomaly Score ─────────────────► Confidence ──────┘
```

1. **Cluster Centroid 計算**：對每個 HMM 隱藏狀態（Cluster），計算其概念向量的加權平均（權重為異常分數）
2. **概念空間映射**：使用相同的 NMF 模型將 MITRE ATT&CK 嵌入轉換至概念空間
3. **相似度計算**：計算 Cluster Centroid 與各 MITRE 技術向量的餘弦相似度
4. **信心度整合**：結合異常分數與相似度計算最終信心度
5. **閾值決策**：根據 `similarity × confidence` 決定最終標籤

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
- 若 $\text{final\_score} < \text{threshold}$，則標記為 `Benign`

### 異常分數的作用

- **高異常分數**：表示該日誌在行為上與正常模式差異較大，更可能是攻擊行為
- **低異常分數**：表示該日誌較接近正常模式，即使與某攻擊技術相似度高，也應降低信心度
- **加權 Centroid**：計算 Cluster Centroid 時，高異常分數的樣本會被賦予更高的權重

## 配置參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `LABELING_SIMILARITY_THRESHOLD` | 0.3 | 相似度下界，低於此值視為不匹配 |
| `LABELING_CONFIDENCE_THRESHOLD` | 0.2 | 最終分數閾值，低於此值標記為 Benign |
| `LABELING_ANOMALY_WEIGHT` | 0.3 | 異常分數在信心度計算中的權重 |
| `LABELING_SIMILARITY_WEIGHT` | 0.7 | 相似度在信心度計算中的權重 |
| `LABELING_TOP_K` | 3 | 輸出的候選技術數量 |
| `LABELING_RESULTS_DIR` | `result/Labeling_Results/` | 標註結果輸出目錄 |

## 使用方式

### 在 Pipeline 中執行

```python
from Pipeline import STAGE_V

STAGE_V()  # 執行自動標註
```

### 獨立執行

```python
from auto_labeling import run_auto_labeling

# 標註所有資料集
results = run_auto_labeling()

# 標註指定資料集
results = run_auto_labeling(
    dataset_ids=["dataset_001", "dataset_002"],
    output_dir="custom_output/",
)
```

### 進階使用

```python
from auto_labeling import AutoLabeler, LabelingConfig

# 自訂配置
config = LabelingConfig(
    similarity_threshold=0.4,
    confidence_threshold=0.25,
    anomaly_weight=0.4,
    similarity_weight=0.6,
    top_k_techniques=5,
)

# 建立標註器
labeler = AutoLabeler(config)

# 載入資料
labeler.load_nmf_model()
labeler.load_concept_vectors()
labeler.load_cluster_labels()
labeler.load_anomaly_scores()
labeler.load_mitre_embeddings()
labeler.transform_mitre_to_concepts()

# 標註單一資料集
result_df = labeler.label_dataset("dataset_001")

# 批次標註
results = labeler.batch_label_all()
```

## 輸出格式

標註結果為 CSV 檔案，儲存於 `result/Labeling_Results/{dataset_id}_Labeled.csv`。

### 欄位說明

| 欄位 | 型別 | 說明 |
|------|------|------|
| `log_index` | int | 日誌在原始資料集中的索引 |
| `cluster_id` | int | HMM 分群的隱藏狀態 ID |
| `predicted_technique` | str | 預測的 MITRE ATT&CK 技術 ID（如 T1059.001）或 "Benign" |
| `technique_name` | str | 技術名稱 |
| `similarity_score` | float | 與該技術的餘弦相似度（0-1） |
| `anomaly_score` | float | 該日誌的異常偵測分數（0-1） |
| `confidence` | float | 綜合信心度指標（0-1） |
| `{original_columns}` | various | 原始日誌的所有欄位 |

### 輸出範例

```csv
log_index,cluster_id,predicted_technique,technique_name,similarity_score,anomaly_score,confidence,timestamp,event_type,message
0,2,T1059.001,PowerShell,0.78,0.85,0.80,2025-01-08 10:30:00,Process Create,powershell.exe -enc...
1,2,T1059.001,PowerShell,0.78,0.62,0.80,2025-01-08 10:30:01,Process Create,powershell.exe -nop...
2,0,Benign,Benign,0.45,0.12,0.33,2025-01-08 10:30:02,Network Connect,svchost.exe connecting...
```

## 資料流依賴

```
STAGE_I (Preprocessing)
    │
    ▼
┌─────────────────────┐
│ data/Embeddings/    │  ← Log Vectors
└─────────────────────┘
    │
    ├─────────────────────────────────────────┐
    ▼                                         ▼
STAGE_II (Anomaly Detection)           STAGE_III (Concept Extraction)
    │                                         │
    ▼                                         ▼
┌─────────────────────┐               ┌─────────────────────┐
│ Detection_Results/  │               │ ConceptVectors/     │
│ ensemble_scores.npy │               │ nmf_model.pkl       │
└─────────────────────┘               └─────────────────────┘
    │                                         │
    │                 STAGE_IV (Sequence Clustering)
    │                         │
    │                         ▼
    │                 ┌─────────────────────┐
    │                 │ SequenceClusters/   │
    │                 │ labels.npy          │
    │                 └─────────────────────┘
    │                         │
    └────────────┬────────────┘
                 ▼
         STAGE_V (Auto Labeling)
                 │
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

### 建立 MITRE 嵌入

```bash
cd "Logs Labeling/external_sources"
python build_mitre_raw_embeddings.py --bert-model sentence-bert
```

這會將 MITRE ATT&CK 技術描述轉換為 BERT 嵌入向量，儲存於 `data/ExternalKnowledge/MITRE_ATTACK/`。

## 注意事項

1. **前置步驟**：執行 STAGE_V 前必須完成 STAGE_I ~ STAGE_IV
2. **概念空間一致性**：MITRE 嵌入必須使用與日誌相同的 NMF 模型進行轉換
3. **異常分數可選**：若無異常偵測結果，將使用預設分數（0.5）
4. **原始日誌對應**：標註結果會嘗試合併原始日誌欄位，需確保 `input_logs/` 中存在對應的 CSV 檔案

## 相關模組

- [Preprocessing](./Preprocessing.md) - 日誌預處理與嵌入
- [Anomaly_Detection](./Anomaly_Detection.md) - 異常偵測
- [Concept_Extraction](./Concept_Extraction.md) - 概念提取（NMF）
- [Sequence_Clustering](./Sequence_Clustering.md) - 序列分群（HMM）
- [External_Sources](./External_Sources.md) - 外部知識整合
