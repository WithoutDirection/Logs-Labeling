# 日誌模板化 (Templatize)

## 概述

日誌模板化流程透過可插拔的解析器將原始日誌事件轉換為結構化的模板與參數，(可)支援多種解析演算法（如 Drain、LenMa...）以及可選的解析器停用模式。

---

## 1. 輸入 (Input)

### 資料來源
- **位置**: `data/input_logs/` 目錄
- **格式**: CSV 檔案
- **內容欄位**: 
  - `Operation`: 操作類型（如 CreateFile、RegQueryValue）
  - `Path`: 檔案路徑或註冊表路徑
  - `Result`: 操作結果（SUCCESS、ACCESS DENIED 等）
  - `Command Line`: 命令列參數（可選）
  - 其他自訂欄位

### 配置參數
```python
# config.py
ENABLE_PARSER = True          # 是否啟用解析器
DEFAULT_PARSER = "drain"      # 預設解析器類型
PARSER_LIST = ["drain", "spell", "lenma"]  # 可用解析器
```

### 執行參數
```python
loader = LogLoader(
    enable_parser=True,        # 啟用/停用解析
    parser_name="drain",       # 選擇解析器
    use_registry_parser=True,  # 使用註冊表專用解析器
    parser_config={            # 自訂解析器配置
        'standard': {'depth': 4, 'st': 0.5},
        'registry': {'depth': 6, 'st': 0.5}
    }
)
loader.load_logs(
    columns=["Operation", "Path", "Result", "Command Line"],
    num=100,        # 處理檔案數量
    ratio=0.3       # 或處理檔案比例
)
```

---

## 2. 處理流程 (Process)

### 步驟 1: 解析器初始化
- **1.1 動態載入**: 使用 `importlib` 根據 `parser_name` 動態導入解析器模組
- **1.2 類別發現**: 搜尋 `DrainParser`/`Parser` 作為標準解析器
- **1.3 註冊表支援**: 搜尋 `RegistryDrainParser`/`RegistryParser` 處理註冊表事件
- **1.4 配置注入**: 將 `parser_config` 參數傳遞給解析器

### 步驟 2: 逐行解析
**解析器啟用時**:
- 從指定欄位構建日誌訊息
- 判斷是否為註冊表操作（`operation.startswith('Reg')`）
- 選擇適當的解析器（標準 vs 註冊表）
- 呼叫 `parse_log_row()` 提取模板與參數

**解析器停用時**:
- 直接合併選定欄位為單一字串
- 不進行模板提取，保留原始內容

### 步驟 3: 檔案處理
- 讀取 CSV 檔案
- 遍歷每一行進行解析
- 生成 `LogID`（格式: `{檔名}_{行號}`）
- 建立結果 DataFrame

### 步驟 4: 批次處理
- 支援處理指定數量（`num`）或比例（`ratio`）的檔案
- 使用 `tqdm` 顯示進度條
- 將結果儲存至中間資料目錄

---

## 3. 輸出 (Output)

### 解析器啟用時
**位置**: `data/Intermediate_data/`

**格式**: CSV 檔案，包含以下欄位：

| 欄位 | 說明 | 範例 |
|------|------|------|
| `LogID` | 日誌唯一識別碼 | `00e4883d-a791-4e87-8786-0c9fc7ba2478_0` |
| `Template` | 提取的日誌模板 | `CreateFile <PATH> SUCCESS` |
| `Parameters` | 提取的參數（\|分隔） | `C:\Windows\System32\cmd.exe` |
| `OriginalLog` | 原始日誌訊息 | `CreateFile C:\Windows\System32\cmd.exe SUCCESS` |

**特殊處理 - 註冊表事件**:
- Template 保留結構化標籤：`<ROOT:HKLM> <OP:QUERY> <RESULT:SUCCESS>`
- Parameters 保留完整原始欄位：`RegQueryValue|HKLM\System|SUCCESS|Type: REG_DWORD`

### 解析器停用時
**格式**: CSV 檔案，簡化欄位：

| 欄位 | 說明 | 範例 |
|------|------|------|
| `LogID` | 日誌唯一識別碼 | `00e4883d-a791-4e87-8786-0c9fc7ba2478_0` |
| `ConcatenatedLog` | 合併的選定欄位 | `CreateFile C:\Windows\System32\cmd.exe SUCCESS` |

### 統計資訊
處理完成後顯示：
- 已處理檔案數量
- 輸出位置
- 解析器名稱（若啟用）
- 標準模板數量（若啟用）
- 註冊表模板數量（若啟用且使用）

---

## 4. 擴展解析器類型

### 快速摘要

系統採用**插件式架構**，新增解析器只需：

1. ✅ 在 `Logs Labeling/preprocess/` 建立 `<parser_name>.py`
2. ✅ 實作必要方法：
   - `__init__(**kwargs)`
   - `parse(log_message: str) -> Tuple[str, List[str]]`
   - `parse_log_row(row: pd.Series, columns: List[str]) -> Tuple[str, List[str], str]`
   - `get_clusters() -> List`
   - `is_registry_operation(operation: str) -> bool` (靜態方法)
3. ✅ 使用 `LogLoader(parser_name="<parser_name>")` 載入

### 類別命名
- **標準解析器**: `Parser` 或 `DrainParser`
- **註冊表解析器**: `RegistryParser` 或 `RegistryDrainParser` (可選)

### 詳細文檔

完整的解析器開發請參考：**[如何撰寫新解析器](./Add%20parser.md)**

內容包括：
- 完整 API 規格與方法簽章
- LenMa 演算法完整實作範例
- 測試與除錯方法
- 常見問題解答

---

# 補充

## 使用範例

### 範例 1: 使用預設 Drain 解析器
```python
from preprocess.preprocess import LogLoader

loader = LogLoader()
loader.load_logs(ratio=0.3)
```

### 範例 2: 切換至自訂解析器
```python
loader = LogLoader(parser_name="spell")
loader.load_logs(num=50)
```

### 範例 3: 停用解析（僅合併欄位）
```python
loader = LogLoader(enable_parser=False)
loader.load_logs()
```

### 範例 4: 自訂配置
```python
loader = LogLoader(
    parser_name="drain",
    parser_config={
        'standard': {'depth': 5, 'st': 0.6},
        'registry': {'depth': 7, 'st': 0.5}
    }
)
loader.load_logs(columns=["Operation", "Path", "Result"])
```

---

## Detail

### 動態載入機制
系統使用 `importlib` 實現解析器的動態載入：

```python
parser_module = importlib.import_module(f'preprocess.{parser_name}')
```

### 解析器選擇邏輯
```
是否為註冊表操作？
├─ 是 → 使用 RegistryParser (若可用)
└─ 否 → 使用標準 Parser
```

### 錯誤處理
- 解析器模組不存在 → 自動使用 `drain`
- 類別名稱不符 → 拋出 `AttributeError`
- CSV 讀取失敗 → 跳過該檔案並記錄錯誤
