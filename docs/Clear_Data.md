# 資料目錄清理工具

## 概述

此工具用於清理管線執行過程中產生的中間資料和輸出結果，但會保留輸入資料（`input_logs`）和參考資源（`reference_resources`）。

## 功能

- 清空 `data/` 目錄下的所有子目錄，除了 `input_logs` 和 `reference_resources`
- 提供預覽模式，可以在實際刪除前查看將要刪除的目錄
- 支援自訂要保留的目錄
- 提供命令列工具和 Python API 兩種使用方式

## 使用方式

### 方式一：命令列工具

```bash
# 預覽將要刪除的目錄（不實際刪除）
cd "Logs Labeling/utils"
python clear_data.py

# 實際執行清理
python clear_data.py --execute

# 自訂要保留的目錄
python clear_data.py --execute --preserve input_logs reference_resources my_backup
```

### 方式二：Python API

```python
from utils.path import clear_data_directories

# 預覽模式（預設）
result = clear_data_directories(dry_run=True)

# 實際執行清理
result = clear_data_directories(dry_run=False)

# 自訂要保留的目錄
result = clear_data_directories(
    preserve_dirs={"input_logs", "reference_resources", "my_backup"},
    dry_run=False
)

# 查看結果
print(f"已刪除: {result['removed']}")
print(f"已保留: {result['preserved']}")
print(f"失敗: {result['failed']}")
```

## 預設行為

預設情況下，以下目錄會被**保留**：
- `input_logs` - 原始輸入日誌
- `reference_resources` - 參考資源（如 MITRE ATT&CK 資料）

以下目錄會被**刪除**：
- `Intermediate_data` - 中間處理資料
- `processed_logs` - 處理後的日誌
- `Embeddings` - 嵌入向量
- `LogVectors` - 日誌向量
- `ConceptVectors` - 概念向量
- `SequenceClusters` - 序列分群結果
- `Detection_Results` - 異常檢測結果
- `clustered_logs` - 分群後的日誌
- 以及其他在 `data/` 目錄下的子目錄

## 安全特性

1. **預覽模式**: 預設為預覽模式（`dry_run=True`），不會實際刪除任何檔案
2. **確認提示**: 使用 `--execute` 時會要求使用者確認
3. **錯誤處理**: 即使某個目錄刪除失敗，也會繼續處理其他目錄
4. **詳細日誌**: 顯示每個目錄的處理狀態

## 注意事項

⚠️ **警告**: 
- 此操作會永久刪除中間資料和輸出結果
- 刪除前請確保已備份重要資料
- 建議先使用預覽模式確認將要刪除的目錄

## 使用場景

1. **重新開始管線**: 清理所有中間資料，從頭開始處理
2. **節省空間**: 清理不再需要的中間結果
3. **問題排查**: 清除可能損壞的中間資料
4. **測試**: 在乾淨的環境中測試管線

## 範例輸出

```
[預覽模式] 清理資料目錄: data
============================================================
[將刪除] ConceptVectors
[將刪除] Embeddings
[將刪除] Intermediate_data
[將刪除] SequenceClusters
[保留] input_logs
[將刪除] processed_logs
[保留] reference_resources
============================================================
總計:
  保留: 2 個目錄
  將刪除: 5 個目錄

提示: 使用 dry_run=False 來實際執行刪除操作
```
