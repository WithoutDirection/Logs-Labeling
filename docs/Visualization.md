# 視覺化模組說明

## 概述

本模組提供多規模異常偵測實驗的視覺化功能，用於分析資料規模對模型效能的影響。

## 模組架構

```
visualization/
├── __init__.py          # 模組匯出
├── aggregator.py        # 結果聚合與標準化
├── trend_analysis.py    # 效能趨勢分析
└── distribution_plot.py # 分佈演變圖
```

---

## aggregator.py - 結果聚合器

將多次實驗結果標準化並聚合，使其具備可比較性。

### 類別

| 類別 | 說明 |
|------|------|
| `AggregatedResult` | 單次實驗的聚合結果，包含 `model_name`、`dataset_size`、`normalized_scores`、`labels` |
| `ResultAggregator` | 管理多次實驗結果的聚合器 |

### 使用範例

```python
from visualization.aggregator import ResultAggregator

aggregator = ResultAggregator(scaler_type="minmax")

# 加入實驗結果
aggregator.add_experiment(results_1, dataset_size=1)
aggregator.add_experiment(results_5, dataset_size=5)
aggregator.add_experiment(results_10, dataset_size=10)

# 取得摘要
summary = aggregator.get_metrics_summary()
```

### 關鍵屬性

- `score_gap`: 異常與正常分數的間距
- `anomaly_ratio`: 異常比例
- `n_samples`: 樣本數

---

## trend_analysis.py - 趨勢分析

繪製效能指標隨資料規模變化的趨勢圖。

### 函數

| 函數 | 說明 |
|------|------|
| `plot_trend_analysis()` | 繪製 Score Gap、Anomaly Ratio、Score Std 趨勢 |
| `plot_anomaly_count_trend()` | 繪製異常數量趨勢 |

### 輸出範例

- `trend_analysis.png`: 三個指標的趨勢折線圖
- `anomaly_count_trend.png`: 各模型異常數量折線圖

---

## distribution_plot.py - 分佈圖

展示正常與異常分數分佈的演變。

### 函數

| 函數 | 說明 |
|------|------|
| `plot_score_histogram()` | 單次實驗的分數直方圖 |
| `plot_distribution_evolution()` | Ridge Plot 風格的分佈演變圖 |
| `plot_comparison_violin()` | 小提琴對比圖 |

### 輸出範例

- `score_histogram.png`: 各模型分數直方圖
- `evolution_{model}.png`: 各模型的分佈演變圖
- `comparison_violin.png`: 跨規模小提琴對比圖

---

## 整合使用

在 `log_detector.py` 的 `__main__` 區段中，自動執行多規模實驗並生成所有視覺化：

```bash
python -m anomaly_dection.log_detector
```

### 實驗流程

1. 掃描所有 Log Vector Dataset
2. 依序執行 1, 5, 10, 15, 20... 規模的實驗
3. 將結果加入 `ResultAggregator`
4. 生成視覺化報告

### 輸出目錄

所有圖表儲存於 `result/unsupervised_anomaly_dection/`

---

## 顏色配置

| 類別 | 顏色 |
|------|------|
| Normal | `#3498db` (藍) |
| Anomaly | `#e74c3c` (紅) |
| Isolation Forest | `#2ecc71` (綠) |
| COPOD | `#3498db` (藍) |
| AutoEncoder | `#9b59b6` (紫) |
| PCA + GMM | `#e74c3c` (紅) |
| Ensemble | `#34495e` (深灰) |
