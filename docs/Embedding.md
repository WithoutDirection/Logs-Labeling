# BERT 嵌入 (BERT Embedding)

> **對應 Pipeline Stage I (Step 1)**：`STAGE_I(N, enable_tfidf=True)`

## 概述

BERT 嵌入模組 (`models/bert.py`) 提供了一個統一的介面，用於載入和使用不同的 BERT 模型進行文本嵌入（text embedding）。該模組支援多種 BERT 變體，並特別整合了針對網路安全領域優化的模型（如 SecBERT），旨在將非結構化的日誌文本或威脅情報轉換為高維度的語義向量，以利後續的相似度計算與分類。

## Pipeline 整合

```python
# Stage I 內部呼叫
from preprocess import run_preprocessing

results = run_preprocessing(
    n_datasets=N,
    enable_parser=False,          # 是否啟用日誌解析
    model_name='sentence-bert',   # BERT 模型名稱
    normalize=False,              # 是否正規化向量
    enable_chunking=False,        # 是否啟用序列切分
)
# results: {n_loaded, embedding_dim}
```

---

## 1. 輸入 (Input)

### 資料來源
- **格式**: 字串 (String) 或 字串列表 (List[String])
- **內容**: 任意文本，例如：
  - 原始日誌訊息 (`OriginalLog`)
  - 解析後的日誌模板 (`Template`)
  - 威脅情報描述 (如 MITRE ATT&CK 技術描述)

### 配置參數
可於 `config.py` 或初始化時指定：
- `model_key`: 選擇使用的模型 (如 `'secbert'`, `'sentence-bert'`)
- `cache_dir`: 模型權重下載後的快取目錄
- `batch_size`: 批次處理的大小

### 支援的模型類型
系統內建多種預定義模型，可透過 `MODEL_REGISTRY` 存取：

| 模型 | 類型 | 維度 | 說明 |
|---------|------|------|------|
| `secbert` | TransformerBERT | 768 | **(推薦)** 針對安全文本訓練的 BERT，對資安術語理解較佳 |
| `sentence-bert` | SentenceBERT | 384 | 快速高效的通用句子嵌入 |
| `sentence-bert-large` | SentenceBERT | 768 | 高品質通用句子嵌入 |
| `bert-base` | TransformerBERT | 768 | 原始 BERT 基礎模型 |
| `cti-bert` | SentenceBERT | 384 | 威脅情報專用 (需驗證) |

---

## 2. 處理流程 (Process)

### 步驟 1: 模型初始化
- **1.1 獲取實例**: 使用 `get_bert_model(model_key)` 獲取模型物件。
- **1.2 自動載入**: 若 `auto_load=True`，系統會自動檢查快取並載入模型權重。
- **1.3 依賴檢查**: 自動偵測 `sentence-transformers` 或 `transformers` 庫是否存在，並選擇對應的實作類別。

### 步驟 2: 文本嵌入
- **2.1 預處理**: 將輸入文本轉換為模型可接受的格式 (Tokenization)。
- **2.2 批次處理**: 支援 `batch_size` 參數，分批處理大量文本以控制記憶體使用。
- **2.3 推論**:
  - **SentenceBERT**: 使用 `encode()` 方法。
  - **TransformerBERT**: 使用 `model()` 前向傳播並進行池化 (Pooling，預設為 Mean Pooling)。
- **2.4 正規化**: 可選 `normalize=True` 將向量正規化至單位長度（便於餘弦相似度計算）。

---

## 3. 輸出 (Output)

### 嵌入向量
**格式**: NumPy 陣列 (`np.ndarray`)

**形狀**: `(樣本數, 嵌入維度)`
- **樣本數**: 輸入文本的數量
- **嵌入維度**: 384 或 768 (取決於模型架構)

**範例**:
```python
embeddings = bert.embed(["Malware detected", "Login failed"])
# embeddings.shape -> (2, 768)  # 若使用 SecBERT
```

### 模型資訊
透過 `get_info()` 獲取：
- 模型名稱與類型
- 載入狀態
- 嵌入維度
- 運算裝置 (CPU/CUDA)

---

## 4. 擴展模型 (Extending Models)

### 快速摘要

系統採用**註冊表模式**，新增 BERT 模型只需修改 `models/bert.py`，無需更動核心邏輯。

### 修改步驟

1. ✅ 開啟 `models/bert.py`
2. ✅ 找到 `MODEL_REGISTRY` 字典
3. ✅ 新增模型配置項目 (Key-Value 對)

```python
MODEL_REGISTRY = {
    # ... 既有模型 ...
    
    'my-new-model': {
        'class': TransformerBERTModel,  # 選擇基底類別: SentenceBERTModel 或 TransformerBERTModel
        'model_name': 'organization/new-bert-model', # HuggingFace Model ID
        'description': '新模型的描述 (768 dim)'
    },
}
```

**參數說明**:
- `class`: 
  - `SentenceBERTModel`: 適用於 `sentence-transformers` 庫的模型，適合語義相似度。
  - `TransformerBERTModel`: 適用於標準 `transformers` 庫的模型，適合一般 BERT 變體。
- `model_name`: HuggingFace Hub 上的模型 ID 或本地路徑。
- `description`: 供 `list_available_models()` 顯示的描述。

---

## 5. 外部介面說明 (External Interface)

本模組主要提供給 `ExternalSourceManager` 或其他分析模組使用。

### 主要函數

#### `get_bert_model(model_key, cache_dir=None, auto_load=True, **kwargs)`
工廠函數，用於獲取模型實例。
- **輸入**: 
  - `model_key`: 模型鍵值 (如 `'secbert'`)
  - `cache_dir`: 指定快取路徑
- **輸出**: `BaseBERTModel` 的子類實例

#### `list_available_models()`
列出所有已註冊的模型。
- **輸出**: `Dict[str, str]` (鍵值對描述)

#### `compare_models(texts, model_keys)`
比較多個模型在相同文本上的效能與維度。

### 模型物件方法 (`BaseBERTModel`)

#### `embed(texts, batch_size=32, show_progress=False, normalize=True)`
核心方法，將文本轉換為向量。
- **輸入**: 文本列表
- **輸出**: NumPy 嵌入矩陣

#### `load()`
手動載入模型（當初始化時 `auto_load=False`）。

#### `get_embedding_dim()`
獲取當前模型的向量維度 (如 768)。

---

## 使用範例

### 範例 1: 基本使用 (SecBERT)
```python
from models.bert import get_bert_model

# 1. 獲取模型 (自動載入)
bert = get_bert_model('secbert')

# 2. 準備文本
logs = [
    "EventID: 4625, Account: Admin, Status: Failed",
    "Suspicious powershell script execution detected"
]

# 3. 產生嵌入
embeddings = bert.embed(logs)
print(f"維度: {embeddings.shape}")
```

### 範例 2: 在 SourceManager 中使用
```python
from external_sources.source_manager import ExternalSourceManager

# 指定使用 SecBERT 初始化管理器
manager = ExternalSourceManager(bert_model='secbert')

# 計算外部來源的嵌入
manager.compute_embeddings('MITRE_ATTACK')
```

### 範例 3: 新增並使用自訂模型
無需修改程式碼，也可直接指定模型類別與名稱：
```python
from models.bert import get_bert_model, TransformerBERTModel

# 直接指定 HuggingFace ID
custom_bert = get_bert_model(
    'microsoft/codebert-base', 
    model_class=TransformerBERTModel
)
custom_bert.embed(["def function(): pass"])
```
