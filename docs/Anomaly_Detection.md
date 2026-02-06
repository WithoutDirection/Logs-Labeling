# 異常偵測 (Anomaly Detection)

## 概述

`anomaly_dection/log_detector.py` 整合多種無監督學習演算法，對日誌嵌入向量進行異常分析。
採用 Ensemble 機制結合四種偵測模型的分數，透過自適應閾值與時間序列平滑化技術，識別異常日誌事件。

---

## 1. 輸入 (Input)

### 資料來源
- **格式**: Hugging Face Dataset 資料夾
- **必要欄位**: `embedding`、`log_vector` 或類似嵌入向量欄位
- **資料型態**: 浮點數向量陣列 (n_samples × n_features)

### 配置參數（`config.py`）

| 參數名稱 | 預設值 | 簡介 | 改動影響 | 補充 |
|---------|-------|------|---------|------|
| **模型選擇** |
| `DETECTION_MODELS` | `["isolation_forest", "copod", "autoencoder", "pca_gmm"]` | 啟用的偵測模型列表 | 減少模型 → 速度 ↑、準確度可能 ↓ | 建議至少保留 2 個模型以維持 Ensemble 效益 |
| `IF_N_ESTIMATORS` | `100` | Isolation Forest 樹數量 | 增加 → 準確度 ↑、訓練時間 ↑ | 100-200 為合理範圍 |
| `IF_CONTAMINATION` | `0.05` | 預期異常比例 | 提高 → 更多樣本被標記為異常 | 設為 `"auto"` 可自動估計 |
| `AE_LATENT_DIM` | `32` | AutoEncoder 潛在空間維度 | 降低 → 壓縮更強、異常更易偵測 | 建議為嵌入維度的 1/4 至 1/10 |
| `AE_EPOCHS` | `50` | AutoEncoder 訓練輪數 | 增加 → 可能過擬合正常樣本 | 監控驗證損失以避免過擬合 |
| `PCA_EXPLAINED_VAR` | `0.95` | PCA 保留變異量 | 降低 → 降維更激進、異常更顯著 | 0.90-0.99 為常用範圍 |
| **分數處理** |
| `SCORE_SCALER` | `"minmax"` | 分數正規化方法 | `"rank"` 抗離群值、`"zscore"` 適合常態分布 | MinMax 保留原始分布形狀 |
| `THRESHOLDING_METHOD` | `"mad"` | 閾值決策策略 | MAD 抗極端值、Percentile 需已知異常率 | **(推薦 MAD)** 自適應性最佳 |
| `THRESHOLDING_PARAMS` | `{"percentile": 98}` | 閾值方法參數 | 依選擇的 METHOD 而異 | Percentile: 95-99、STD: n_std=2-3 |
| **Ensemble** |
| `ENSEMBLE_WEIGHTS` | 各 `0.25` | 各模型權重 | 提高某模型權重 → 強調該模型特性 | 權重和自動正規化，可不等於 1 |
| **MAD 閾值** |
| `MAD_THRESHOLD_MULTIPLIER` | `3.0` | MAD 乘數（k 值） | `↑` 更嚴格（少異常）、`↓` 更寬鬆 | 2.5=1%、3.0=0.3%、3.5=0.05% 異常率 |
| `MAD_USE_MODIFIED` | `True` | 使用修正版 MAD | 啟用 → MAD×1.4826 接近標準差 | 常態分布下建議啟用 |
| **時間序列平滑化** |
| `ENABLE_TIME_SERIES_SMOOTHING` | `True` | 啟用平滑化 | 停用 → 保留瞬時峰值、延遲 ↓ | 短序列（<100）建議停用 |
| `SMOOTHING_WINDOW_SIZE` | `5` | 滑動視窗大小 | 增大 → 平滑 ↑、峰值鈍化 ↑ | 建議奇數，範圍 3-9 |
| `SMOOTHING_METHOD` | `"mean"` | 平滑方法 | `"median"` 抗噪聲、`"gaussian"` 更平滑 | Gaussian 計算成本較高 |
| **相關性分析** |
| `ENABLE_CORRELATION_ANALYSIS` | `True` | 啟用模型相關性分析 | 停用 → 無熱圖輸出 | 用於評估模型互補性 |
| `CORRELATION_METHOD` | `"pearson"` | 相關性計算方法 | Spearman 適合非線性關係 | Pearson 最常用 |
| **路徑設定** |
| `DETECTION_RESULTS_DIR` | `"data/Detection_Results"` | 偵測結果輸出目錄 | - | 自動建立目錄 |

---

## 2. 處理流程 (Process)

### 階段一：模型訓練與預測
```
[嵌入向量] → [四種模型並行偵測] → [原始異常分數]
             ├─ Isolation Forest (樹型集成)
             ├─ COPOD (概率密度)
             ├─ AutoEncoder (重建誤差)
             └─ PCA+GMM (降維+高斯混合)
```

### 階段二：分數正規化
- **MinMax**: 線性縮放至 [0, 1]，保留原始分布形狀
- **Rank**: 百分位數正規化，適合處理離群值
- **Z-Score + Sigmoid**: 標準化後經 Sigmoid 映射，適合常態分布

### 階段三：Ensemble 整合
```
Ensemble_Score = Σ(weight_i × normalized_score_i)
```
- 加權平均各模型分數
- 權重總和自動正規化至 1.0

### 階段四：時間序列平滑化（可選）
- **移動平均 (Mean)**: 減少短期波動
- **移動中位數 (Median)**: 抵抗極端值影響
- **高斯濾波 (Gaussian)**: 平滑且保留趨勢

### 階段五：閾值決策
| 方法 | 計算方式 | 適用情境 |
|------|---------|---------|
| **Percentile** | 取分數前 N% | 已知異常比例 |
| **STD** | mean + k×std | 常態分布資料 |
| **TOP_N** | 固定取前 N 個 | 需固定異常數量 |
| **MAD** | median + k×MAD | **(推薦)** 抵抗極端值 |

**MAD 公式**:
```
MAD = median(|score - median(score)|)
修正版 MAD = MAD × 1.4826  # 接近標準差
閾值 = median + k × MAD
```

---

## 3. 輸出 (Output)

### Dataset 新增欄位
```python
{
    # 各模型分數與標籤
    "{model}_raw_score": List[float],     # 原始分數
    "{model}_score": List[float],         # 正規化分數
    "{model}_label": List[int],           # 0/1 標籤
    
    # Ensemble 結果
    "ensemble_raw_score": List[float],    # 平滑化前分數
    "ensemble_raw_label": List[int],      # 平滑化前標籤
    "ensemble_score": List[float],        # 最終分數
    "ensemble_label": List[int]           # 最終標籤 (主要使用)
}
```

### 視覺化輸出
- **相關性熱圖**: `{dataset_name}_correlation.png`
  - 顯示各模型分數的 Pearson/Spearman 相關性
  - 用於評估模型互補性與冗餘度

---

## 4. 核心概念

### 4.1 Ensemble 策略
- **互補性**: 四種模型基於不同假設（距離、密度、重建、分布），捕捉不同類型異常
- **穩健性**: 單一模型失效時，其他模型可補償
- **可調性**: 透過權重調整強調特定模型

### 4.2 MAD 自適應閾值
**優勢**:
- 對極端值穩健（使用中位數而非平均數）
- 自動適應不同資料集的分數分布
- 修正版 MAD 與標準差等價（常態分布下）

**參數影響**:
- `k=2.5`: 偵測約 1% 異常（寬鬆）
- `k=3.0`: 偵測約 0.3% 異常（平衡）**(推薦)**
- `k=3.5`: 偵測約 0.05% 異常（嚴格）

### 4.3 時間序列平滑化
**目的**: 降低瞬時波動，提取趨勢性異常

**權衡考量**:
- **視窗大小 ↑**: 平滑效果 ↑，但延遲 ↑、峰值鈍化
- **Gaussian vs Mean**: Gaussian 更平滑但計算成本高

**建議**:
- 日誌序列長度 < 100: 停用平滑化
- 日誌序列長度 100-1000: `window=5`，`method="mean"`
- 日誌序列長度 > 1000: `window=9`，`method="gaussian"`

### 4.4 模型相關性分析
**解讀指南**:
- **高相關 (r > 0.7)**: 模型行為相似，可能存在冗餘
- **中相關 (0.3 < r < 0.7)**: 模型互補，理想狀態
- **低相關/負相關 (r < 0.3)**: 模型捕捉不同異常特徵

**應用建議**:
- 若兩模型高度相關，考慮降低其中一個的 Ensemble 權重
- 若某模型與其他模型負相關，檢查該模型是否適用於資料

---

## 5. 參數調整指南

根據不同情境調整參數：

| 情境 | 調整建議 | 效果 |
|------|---------|------|
| **誤報過多** | `MAD_THRESHOLD_MULTIPLIER = 3.5`<br>`SMOOTHING_METHOD = "median"`<br>`SMOOTHING_WINDOW_SIZE = 7` | 降低誤報率，但可能漏掉弱異常 |
| **漏報過多** | `MAD_THRESHOLD_MULTIPLIER = 2.5`<br>提高敏感模型權重（如 COPOD） | 提升偵測率，但誤報可能增加 |
| **實時偵測（低延遲）** | `DETECTION_MODELS = ["isolation_forest", "copod"]`<br>`ENABLE_TIME_SERIES_SMOOTHING = False` | 速度提升 50%+，犧牲部分準確度 |
| **少樣本（< 100 筆）** | `DETECTION_MODELS = ["isolation_forest"]`<br>`THRESHOLDING_METHOD = "percentile"`<br>`ENABLE_TIME_SERIES_SMOOTHING = False` | 避免複雜模型過擬合 |
| **高噪聲日誌** | `SMOOTHING_METHOD = "gaussian"`<br>`SMOOTHING_WINDOW_SIZE = 9`<br>`SCORE_SCALER = "rank"` | 抵抗噪聲干擾，平滑效果最佳 |
| **已知異常率（如 5%）** | `THRESHOLDING_METHOD = "percentile"`<br>`THRESHOLDING_PARAMS = {"percentile": 95}` | 精確控制異常數量比例 |

**快速診斷**：
- 相關性熱圖顯示模型高度相關（r > 0.8） → 降低冗餘模型權重或移除
- Ensemble 分數分布雙峰 → 考慮調整閾值方法為 MAD
- 平滑化後異常數量大幅減少 → 減小 `SMOOTHING_WINDOW_SIZE`

---

## 6. 使用範例

### 基本使用
```python
from anomaly_dection import run_detection

# 對所有資料集執行偵測（推薦）
result = run_detection(
    input_dir="data/Embeddings",
    output_dir="data/Detection_Results",
    generate_viz=True,
    verbose=True
)
print(f"處理了 {result['n_datasets']} 個資料集")
```

### 使用底層 Pipeline API
```python
from anomaly_dection.log_detector import run_detection_pipeline

# 僅執行偵測（不含視覺化）
results = run_detection_pipeline(
    input_dir="data/Embeddings",
    output_dir="data/Detection_Results",
    models=["isolation_forest", "copod"],  # 自訂模型
    verbose=True
)
```

### 生成摘要與視覺化
```python
from visualization.anomaly_comparison import generate_detection_summary

# 在執行偵測後生成摘要
generate_detection_summary(
    results,
    output_dir="result/Anomaly_Detection",
    generate_visualizations=True,
    enable_advanced_plots=True
)
```

---


