# 日誌預處理管線

日誌預處理的三個主要階段：**日誌解析與模板化**、**文字嵌入**、以及 **Log Vector 生成**。

---

## 階段一：日誌解析與模板化 (LogLoader)

> 詳見 [Templateize](Templatize.md)

### 輸入
- **原始日誌檔案**：位於 `data/input_logs/` 目錄下的 CSV 檔案
- **選定欄位**：預設為 `Operation`、`Path`、`Result`、`Command Line`

### 輸出
- **解析後的中間資料**：儲存於 `data/Intermediate_data/` 目錄
- 若啟用解析：輸出包含 `LogID`、`Template`（模板）、`Parameters`（參數）、`OriginalLog`
- 若停用解析：輸出僅包含 `LogID`、`ConcatenatedLog`（串接的原始日誌）

### 處理流程

1. **解析器初始化**：根據設定動態載入解析器模組（如 Drain、Spell、Lenma），並區分「標準解析器」與「註冊表專用解析器」。

2. **逐行解析**：
   - 若啟用解析，系統會根據 `Operation` 欄位判斷該行是否為註冊表操作：
     - **是**：使用註冊表專用解析器（較深的解析樹，專門處理複雜的註冊表路徑）
     - **否**：使用標準解析器
   - 解析器會將日誌訊息拆解為**模板**（固定結構）與**參數**（變動部分），例如：
     - 原始：`Process Create C:\Windows\System32\cmd.exe SUCCESS`
     - 模板：`Process Create <*> SUCCESS`
     - 參數：`C:\Windows\System32\cmd.exe`

3. **停用解析時**：系統僅將選定欄位的值串接成單一字串，保留原始日誌內容。

### 機制說明
解析器採用插件式架構，透過動態模組載入實現可擴展性。解析的核心目的是將日誌「正規化」——相同結構但參數不同的日誌會產生相同的模板，這有助於後續的聚類與異常偵測。

---

## 階段二：文字嵌入 (LogEmbedder)

> 詳見 [Embedding.md](/docs/Embedding.md)

### 輸入
- **中間資料檔案**：階段一輸出的 CSV 檔案（位於 `data/Intermediate_data/`）

### 輸出
- **嵌入向量資料集**：以 Hugging Face Dataset 格式儲存於 `data/Embeddings/`
- 若有解析：包含 `LogID`、`template_embedding`、`param_embedding`
- 若無解析：包含 `LogID`、`embedding`

### 處理流程

1. **模型載入**：根據設定載入預訓練的 BERT 模型（如 Sentence-BERT）。

2. **文字準備**：
   - 若資料含 `Template` + `Parameters`：分別提取兩者的文字
   - 若資料僅含 `ConcatenatedLog`：直接使用該欄位

3. **批次嵌入計算**：
   - 將文字分批送入 BERT 模型，取得固定維度的向量表示
   - 可選擇是否對向量進行正規化（L2 Normalize）

4. **儲存嵌入**：將結果以 Arrow 格式儲存，便於讀取。

### 機制說明
嵌入的目的是將離散的文字轉換為連續的向量空間表示。BERT 模型能夠捕捉文字的語義資訊，使得語義相近的日誌在向量空間中距離較近。分別對模板與參數進行嵌入的設計，讓系統能分開處理「日誌結構」與「具體內容」的語義資訊。

---

## 階段三：Log Vector 生成 (LogChunker)

### 輸入
- **嵌入向量資料集**：階段二輸出的資料（位於 `data/Embeddings/`）

### 輸出
- **Log Vector 資料集**：以 Hugging Face Dataset 格式儲存於 `data/LogVectors/`
- 包含 `window_idx`、`start_idx`、`end_idx`、`start_log_id`、`end_log_id`、`log_vector`

### 處理流程

1. **滑動視窗切割**：
   - 將日誌序列以固定大小的視窗切割，視窗之間有重疊
   - 例如：視窗大小 50、步長 25，則每次移動 25 筆日誌，重疊 25 筆
   - 尾部不足一個視窗的部分會額外處理，確保不遺漏

2. **BiLSTM**：
   - 將視窗內的嵌入序列送入雙向 LSTM
   - BiLSTM 能同時捕捉序列的**前向**與**後向**時序關係
   - 輸出為每段時間的隱藏狀態

3. **Attention 機制**：
   - 對 BiLSTM 的輸出應用Attention機制
   - 學習每個時間步的重要性權重
   - 加權聚合產生單一向量（Context Vector）

4. **雙流處理**（若有parse）：
   - Template 與 Parameters 分別經過獨立的 BiLSTM + Attention
   - 最終將兩者的 Context Vector 串接

5. **融合層**（可選）：
   - 通過全連接層將串接的向量映射到指定維度
   - 加入 ReLU 激活與 Dropout 正則化

### 機制說明
Log Vector 的目的是將**一段時間內的日誌序列**壓縮為單一向量表示。滑動視窗確保相鄰時間段有重疊，避免邊界效應。BiLSTM 能夠建模序列之間的時序依賴關係，而 Attention 機制則讓模型自動學習「哪些序列更重要」，產生更具鑑別力的表示。

---

## 附錄：超參數設定說明

以下為 `config.py` 中與預處理管線相關的超參數：

### 解析器設定

| 參數 | 預設值 | 說明 | 修改影響 |
|------|--------|------|----------|
| `ENABLE_PARSER` | `True` | 是否啟用日誌解析 | 設為 `False` 時保留原始日誌，不進行模板化 |
| `DEFAULT_PARSER` | `"drain"` | 預設使用的解析器 | 可選 `drain`、`spell`、`lenma`，不同解析器的解析策略與效果不同 |

### BERT 模型設定

| 參數 | 預設值 | 說明 | 修改影響 |
|------|--------|------|----------|
| `BERT_MODEL_NAME` | `"sentence-bert"` | BERT 模型名稱 | 可選其他預訓練模型，影響嵌入品質與計算速度 |
| `BERT_AUTO_LOAD` | `True` | 是否自動載入模型 | 設為 `False` 時需手動觸發載入 |

### LogChunker 設定

| 參數 | 預設值 | 說明 | 修改影響 |
|------|--------|------|----------|
| `SEQUENCE_WINDOW_SIZE` | `50` | 滑動視窗大小 | **增大**：捕捉更長的時序關係，但可能模糊局部特徵；**減小**：更精細但可能遺失長程依賴 |
| `SEQUENCE_STRIDE` | `25` | 滑動步長 | **減小**：更多重疊，結果更平滑但計算量增加；**增大**：較少重疊，可能遺漏邊界資訊 |
| `BILSTM_HIDDEN_SIZE` | `128` | BiLSTM 隱藏層維度 | **增大**：模型容量更大，能學習更複雜的模式，但更耗資源 |
| `BILSTM_NUM_LAYERS` | `2` | BiLSTM 層數 | **增加**：更深的網路，可能學習更抽象的特徵，但訓練更困難 |
| `BILSTM_DROPOUT` | `0.3` | Dropout 比例 | **增大**：正則化效果更強，防止過擬合；**減小**：模型更容易記住訓練資料 |
| `FUSION_ENABLE` | `True` | 是否啟用融合層 | 設為 `False` 時直接輸出 BiLSTM 的結果，維度較高 |
| `FUSION_OUTPUT_DIM` | `256` | 融合層輸出維度 | 決定最終 Log Vector 的維度，影響下游任務的輸入大小 |

---

## 處理流程總覽

```
原始日誌 (CSV)
     │
     ▼
┌─────────────────┐
│  階段一：解析   │  LogLoader
│  模板化 / 串接  │
└────────┬────────┘
         │
         ▼
   中間資料 (CSV)
         │
         ▼
┌─────────────────┐
│  階段二：嵌入   │  LogEmbedder
│  BERT 向量化   │
└────────┬────────┘
         │
         ▼
 嵌入向量 (Dataset)
         │
         ▼
┌─────────────────┐
│ 階段三：序列編碼│  LogChunker
│ BiLSTM+Attention│
└────────┬────────┘
         │
         ▼
  Log Vector (Dataset)
```
