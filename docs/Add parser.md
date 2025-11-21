# 如何撰寫新解析器

## 概述

本文檔說明如何撰寫自訂解析器並整合至預處理流程。

---

## Quick Start

### 1. 建立解析器檔案

在 `Logs Labeling/preprocess/` 目錄下建立新的 `.py` 檔案，例如 `spell.py` 或 `lenma.py`。

### 2. 實作必要類別與方法

```python
# spell.py
import pandas as pd
from typing import List, Tuple, Dict

class Parser:
    """你的自訂解析器"""
    
    def __init__(self, depth: int = 4, st: float = 0.5, **kwargs):
        """初始化解析器，接收任意配置參數"""
        self.depth = depth
        self.st = st
        self.clusters = []
    
    @staticmethod
    def is_registry_operation(operation: str) -> bool:
        """判斷是否為註冊表操作"""
        return operation.startswith('Reg') if operation else False
    
    def parse(self, log_message: str) -> Tuple[str, List[str]]:
        """解析單行日誌
        
        Args:
            log_message: 日誌文字
            
        Returns:
            (模板, 參數列表)
        """
        # 你的解析邏輯
        template = "Process <OP> <PATH>"
        params = ["Start", "C:\\test.exe"]
        return template, params
    
    def parse_log_row(self, row: pd.Series, columns: List[str]) -> Tuple[str, List[str], str]:
        """解析 DataFrame 行
        
        Args:
            row: DataFrame 的一行
            columns: 要解析的欄位列表
            
        Returns:
            (模板, 參數列表, 原始日誌)
        """
        # 從欄位建構日誌訊息
        log_parts = []
        for col in columns:
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    log_parts.append(str(val).strip())
        log_message = " ".join(log_parts)
        
        # 解析
        template, params = self.parse(log_message)
        return template, params, log_message
    
    def get_clusters(self) -> List:
        """返回所有模板群集"""
        return self.clusters
```

### 3. 使用新解析器

```python
from preprocess.preprocess import LogLoader

# 方法 1: 直接指定解析器名稱
loader = LogLoader(parser_name="spell")
loader.load_logs()

# 方法 2: 傳入自訂配置
loader = LogLoader(
    parser_name="spell",
    parser_config={
        'standard': {'depth': 5, 'st': 0.6},
    }
)
loader.load_logs()
```

---

## API 規格

### 必要方法

| 方法 | 簽章 | 說明 |
|------|------|------|
| `__init__` | `(**kwargs)` | 初始化，接收任意配置參數 |
| `parse` | `(log_message: str) -> Tuple[str, List[str]]` | 解析單行日誌，返回 (模板, 參數) |
| `parse_log_row` | `(row: pd.Series, columns: List[str]) -> Tuple[str, List[str], str]` | 解析 DataFrame 行 |
| `get_clusters` | `() -> List` | 返回所有模板群集 |
| `is_registry_operation` | `(operation: str) -> bool` | 判斷是否為註冊表操作 (靜態方法) |

### 可選方法（支援註冊表事件）

若要支援註冊表事件的特殊處理，可額外實作：

```python
class RegistryParser(Parser):
    """註冊表專用解析器"""
    
    def __init__(self, registry_mode: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.registry_mode = registry_mode
    
    def parse_from_row(self, row_dict: Dict) -> Tuple[str, List[str]]:
        """從字典解析註冊表事件
        
        Args:
            row_dict: 包含 Operation, Path, Result, Detail 的字典
            
        Returns:
            (模板, 原始欄位列表)
        """
        # 你的註冊表解析邏輯
        template = "<ROOT:HKLM> <OP:QUERY> <RESULT:SUCCESS>"
        params = [row_dict['Operation'], row_dict['Path'], row_dict['Result']]
        return template, params
```

---

## 類別命名規則

系統會依序搜尋以下類別名稱：

### 標準解析器
1. `DrainParser` (優先)
2. `Parser` (備選)

### 註冊表解析器（可選）
1. `RegistryDrainParser` (優先)
2. `RegistryParser` (備選)

**建議命名方式：**
- 新演算法使用 `Parser` 作為標準類別名稱
- 如需註冊表支援，使用 `RegistryParser`

---

## 配置參數

### 在 config.py 設定

```python
# config.py
ENABLE_PARSER = True  # 是否啟用解析（False 則保留原始日誌）
DEFAULT_PARSER = "spell"  # 預設解析器
PARSER_LIST = ["drain", "spell", "lenma"]  # 可用解析器列表
```

### 傳入自訂配置

```python
loader = LogLoader(
    enable_parser=True,        # 啟用解析
    parser_name="spell",       # 使用 spell 解析器
    use_registry_parser=True,  # 啟用註冊表解析器
    parser_config={
        'standard': {          # 標準解析器配置
            'depth': 5,
            'st': 0.6,
            'max_children': 200
        },
        'registry': {          # 註冊表解析器配置
            'depth': 7,
            'st': 0.5,
            'registry_mode': True
        }
    }
)
```

---

## 範例：完整解析器實作

```python
# lenma.py
import re
import pandas as pd
from typing import List, Tuple, Dict

class Parser:
    """LenMa 演算法解析器範例"""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        self.threshold = threshold
        self.templates = {}
        
    @staticmethod
    def is_registry_operation(operation: str) -> bool:
        return operation.startswith('Reg') if operation else False
    
    def parse(self, log_message: str) -> Tuple[str, List[str]]:
        """使用 LenMa 演算法解析"""
        # 簡化範例：用正則取代變數部分
        params = []
        
        # 提取路徑
        template = re.sub(r'[A-Z]:\\[\w\\.-]+', lambda m: (params.append(m.group()), '<PATH>')[1], log_message)
        # 提取數字
        template = re.sub(r'\b\d+\b', lambda m: (params.append(m.group()), '<NUM>')[1], template)
        
        return template, params
    
    def parse_log_row(self, row: pd.Series, columns: List[str]) -> Tuple[str, List[str], str]:
        """解析 DataFrame 行"""
        log_parts = []
        for col in columns:
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    log_parts.append(str(val).strip())
        
        log_message = " ".join(log_parts)
        template, params = self.parse(log_message)
        
        # 記錄模板
        if template not in self.templates:
            self.templates[template] = 0
        self.templates[template] += 1
        
        return template, params, log_message
    
    def get_clusters(self) -> List:
        """返回所有模板"""
        return list(self.templates.keys())


class RegistryParser(Parser):
    """LenMa 註冊表解析器"""
    
    def __init__(self, registry_mode: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.registry_mode = registry_mode
    
    def parse_from_row(self, row_dict: Dict) -> Tuple[str, List[str]]:
        """註冊表事件保留完整欄位"""
        operation = row_dict.get('Operation', '')
        path = row_dict.get('Path', '')
        result = row_dict.get('Result', '')
        detail = row_dict.get('Detail', '')
        
        # 建立模板（保留結構化標籤）
        parts = []
        if operation:
            parts.append(f"<OP:{operation}>")
        if path:
            parts.append(f"<PATH>")
        if result:
            parts.append(f"<RESULT:{result}>")
        
        template = " ".join(parts)
        params = [operation, path, result, detail] if detail else [operation, path, result]
        
        return template, params
```

---

## 測試你的解析器

```python
# test_my_parser.py
from preprocess.preprocess import LogLoader

# 測試 1: 基本功能
loader = LogLoader(parser_name="lenma", enable_parser=True)
loader.load_logs(num=5)

# 測試 2: 停用解析
loader = LogLoader(enable_parser=False)
loader.load_logs(num=5)

# 測試 3: 自訂配置
loader = LogLoader(
    parser_name="lenma",
    parser_config={'standard': {'threshold': 0.7}}
)
loader.load_logs(ratio=0.1)
```

---

## 常見問題

### Q1: 解析器找不到？
確認檔案放在 `Logs Labeling/preprocess/` 目錄下，且檔名為**小寫**（如 `spell.py`）。

### Q2: 需要實作所有方法嗎？
是的，除了 `RegistryParser` 是可選的，其他方法都是必要的。

### Q3: 如何除錯？
```python
# 直接測試解析器
from preprocess.spell import Parser

parser = Parser()
template, params = parser.parse("CreateFile C:\\test.exe SUCCESS")
print(f"Template: {template}")
print(f"Params: {params}")
```

### Q4: 解析器可以有狀態嗎？
可以！在 `__init__` 中初始化任何需要的資料結構，例如模板快取、統計資訊等。

---

## 參考範例

完整實作請參考：
- **`drain.py`** - Drain 演算法實作（支援註冊表）
- **`parser_template.py`** - 解析器模板（英文註解）
- **`test_enable_parser.py`** - 測試範例

---

## 總結

撰寫新解析器只需三步驟：
1. ✅ 建立 `.py` 檔案於 `preprocess/` 目錄
2. ✅ 實作必要的 5 個方法
3. ✅ 使用 `LogLoader(parser_name="your_parser")` 載入

系統會自動：
- 動態載入你的解析器模組
- 傳遞配置參數
- 處理標準與註冊表事件
- 產生輸出檔案
