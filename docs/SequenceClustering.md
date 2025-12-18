# 序列分群模組 (Sequence Clustering)

## 模組概述

本模組採用 **隱馬可夫模型 (HMM)** 對事件日誌進行分群，自動識別系統行為中的同質群體。

---

## 核心流程

```
概念向量 (H_E) → Grid Search 優化 → Viterbi 解碼 → 分群標籤
```

### 階段 A：全域模型優化

1. **載入資料**：從 `data/ConceptVectors` 讀取所有概念矩陣
2. **聚合向量**：將多個 Dataset 合併為單一訓練集
3. **網格搜索**：遍歷 K ∈ [2, 10]，每個 K 執行 10 次隨機初始化
4. **擇優保留**：選取 Log-Likelihood 最高的模型

### 階段 B：序列標註

1. **載入最佳模型**
2. **Viterbi 解碼**：對每個 Dataset 預測隱藏狀態序列
3. **儲存結果**：輸出至 `data/ClusterResults`

---

## API 說明

| 方法 | 功能 |
|------|------|
| `optimize_global_hmm(X)` | 執行 Grid Search 找最佳 HMM |
| `decode_sequences(X)` | 使用 Viterbi 演算法產生分群標籤 |
| `fit_predict(X)` | 一步完成訓練與預測 |
| `save_model() / load_model()` | 模型持久化 |

---

## 超參數配置

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `HMM_K_MIN` | 2 | 隱藏狀態數下界 |
| `HMM_K_MAX` | 10 | 隱藏狀態數上界 |
| `HMM_N_STARTS` | 10 | 每個 K 的初始化次數 |
| `HMM_N_ITER` | 100 | Baum-Welch 最大迭代次數 |
| `HMM_TOL` | 1e-3 | 收斂閾值 |
| `HMM_COVARIANCE_TYPE` | diag | 共變異數類型 |

---

## 使用範例

```python
from sequence_clustering import SequenceClustering, load_concept_vectors, aggregate_vectors

# 載入與聚合
vectors = load_concept_vectors()
aggregated = aggregate_vectors(vectors)

# 訓練與預測
clusterer = SequenceClustering()
labels = clusterer.fit_predict(aggregated)
clusterer.save_model()
```
