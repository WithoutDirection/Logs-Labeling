
# Sequence Clustering

## 1. Overview

**目標**：針對每一個獨立的攻擊行為（Dataset），精確識別其內部的演變階段（如：初始存取 -> 執行 -> 清理）。
**策略核心**：採用 **「單一資料集獨立訓練」** 搭配 **「雙軌特徵機制」**。前者解決不同攻擊手法互相干擾的問題，後者解決 NMF 特徵不易收斂的數學難題。

### 核心輸入與輸出

* **輸入 (Input)**：單一資料集的「概念矩陣」()，來自 Concept Extraction 步驟。
* **輸出 (Output)**：
1. **專屬模型**：針對該 Dataset 最佳化的 HMM 模型（包含狀態定義與轉移機率）。
2. **分群標籤**：長度與原始資料嚴格對應的狀態序列（Labels），可直接用於後續與 MITRE 文件比對。



---

## 2. Core Concepts

### A. 雙軌特徵機制 (Dual-Track Feature Strategy)

將資料分為兩條路徑使用：

1. **訓練軌道 (Training Track)**：使用 **常態化預處理數據** $X_{normalized}$
   * **預處理流程**（三階段）：
     1. **Log-Transform**：$\log(1 + |X|)$ 壓縮數值範圍
     2. **Percentile Clipping**：裁剪至 [0.5%, 99.5%] 去除極端值
     3. **Z-Score Normalization**：$(X - \mu) / (\sigma + \epsilon)$ 標準化為均值 0、標準差 1
   * **目的**：HMM 的高斯假設要求數據接近常態分佈。此三階段強迫 NMF 輸出的偏態分佈轉為對稱分佈，讓模型在數十次迭代內快速收斂，避免 `Not Converging` 錯誤。

2. **應用軌道 (Application Track)**：保留 **原始 NMF 數據** $X_{raw}$
   * **目的**：NMF 的非負特性代表「概念強度」。分群後的標籤必須對應回原始數據，才能計算 Cluster Centroid 並與 MITRE 向量進行餘弦相似度比對。

### B. 一階差分特徵 (Delta Features)

在標準化特徵基礎上，額外計算 **一階差分** $\Delta = x_t - x_{t-1}$：

* **數學定義**：對 $X_{normalized}$ 執行 `np.diff(X, axis=0)`，第一筆補零（無前一筆可比較）
* **物理意義**：捕捉「變化率」或「速度」，識別攻擊階段的轉換點
* **實際效果**：
  * 攻擊開始：概念權重突然上升 → 正向大幅差分
  * 攻擊結束：概念權重突然下降 → 負向大幅差分
  * 穩定期：概念權重平穩 → 差分接近零
* **特徵擴增**：最終輸入 HMM 的特徵為 $[X_{normalized}, \Delta]$，維度翻倍



### C. Per-Dataset 策略

* 不訓練全域模型。
* 將每一個 Dataset 資料夾視為獨立事件，為其量身定做一個 HMM，使其能捕捉到該特定攻擊手法獨有的細微階段。
* **動態 K 值調整**：根據樣本數自動收縮 K 上界（$K_{max} = \min(K_{config}, n_{samples} / 5)$），避免過擬合。
* **變異度保護**：若特徵變異度 < $10^{-8}$，自動鎖定 $K = K_{min}$，避免無意義的多狀態嘗試。

### D. 模型選擇指標 (BIC)

* 使用 **貝葉斯資訊量準則 (BIC)** 而非單純的 Log-Likelihood 來選擇最佳狀態數，防止模型為了衝高分數而無腦將 K 選到最大，平衡模型的複雜度與精確度。

---

## 3. Workflow

此流程針對**每一個** Dataset 資料夾獨立執行一次：

### 步驟一：資料載入與預處理 (Load & Preprocessing)

* **讀取**：載入原始概念矩陣 $X_{raw}$（來自 ConceptVectors）
* **清理**：使用 `np.nan_to_num` 處理 NaN/Inf 值
* **三階段常態化**：
  1. **Log-Transform**：$X_{log} = \log(1 + |X_{raw}|)$
  2. **Clipping**：裁剪至 [0.5%, 99.5%] 百分位數，記為 $X_{clip}$
  3. **Z-Score**：$X_{norm} = (X_{clip} - \mu) / (\sigma + 10^{-8})$
* **一階差分**：計算 $\Delta = \text{diff}(X_{norm})$，第一筆補零
* **特徵拼接**：$X_{augmented} = [X_{norm}, \Delta]$（特徵維度翻倍）


### 步驟二：局部模型優化 (Local Optimization)

* **輸入**：使用 $X_{augmented}$（含原始標準化特徵 + 一階差分）
* **動態 K 範圍**：
  * 基礎範圍：$[K_{min}, K_{max}] = [2, 10]$
  * 樣本數調整：$K_{effective} = \min(K_{max}, \max(2, n_{samples} / 5))$
  * 變異度保護：若 $\text{Var}(X) < 10^{-8}$，強制 $K_{effective} = K_{min}$
* **Grid Search**：
  * 並行測試 $K \in [K_{min}, K_{effective}]$
  * 每個 K 執行 20 次隨機初始化（比全域策略更高，確保穩定性）
  * 失敗保護：單一 K 連續失敗 2 次則跳過該 K
* **擇優**：選出 Log-Likelihood 最高的模型與 K 值（注：此處使用 LL 而非 BIC）

### 步驟三：序列解碼 (Decoding)

* **輸入**：使用 $X_{augmented}$ 與訓練好的模型
* **機制**：執行 **Viterbi 演算法**（`model.predict()`），計算最可能的隱藏狀態路徑
* **產出**：得到整數序列 `labels`，長度與 $X_{raw}$ 完全一致

### 步驟四：追溯性驗證與存檔 (Validation & Save)

* **追溯性驗證**：嚴格檢查 `len(labels) == len(X_raw)`，確保無資料遺失
* **存檔內容**：
  * `labels.npy`：分群標籤（整數序列）
  * `model.pkl`：包含五個元素的字典
    - `model`：訓練好的 `GaussianHMM` 物件
    - `best_k`：最佳狀態數
    - `best_score`：Log-Likelihood 分數
    - `scaler_mean`：標準化參數 $\mu$（用於未來推論）
    - `scaler_std`：標準化參數 $\sigma$（用於未來推論）
* **警告摘要**：統計該 Dataset 的訓練失敗/警告次數並輸出



---

## 4. Configuration

針對 Per-Dataset 策略的設定：

### A. 基礎設定

| 參數名稱 | 建議值 | 說明 |
| --- | --- | --- |
| **`HMM_K_MIN` / `HMM_K_MAX**` | `2` / `10` | **搜尋範圍**。通常單一攻擊行為包含 3~5 個階段，保留 10 以應對複雜案例。 |
| **`HMM_COVARIANCE_TYPE`** | `'diag'` | **形狀設定**。維持對角矩陣，運算最快且穩定。 |
| **`HMM_MIN_COVAR`** | `1e-3` | **穩定性保護**。防止因特徵稀疏導致矩陣計算崩潰 (Singular Matrix)。 |

### B. 穩定性與效能

| 參數名稱 | 建議值 | 說明 |
| --- | --- | --- |
| **`HMM_ENABLE_PARALLEL`** | `True` | **並行開關**。務必開啟，在單一 Dataset 內部並行測試不同種子。 |
| **`HMM_N_STARTS`** | `20` | **重試次數**。比全域模型更高 (原為 10)。單一資料集運算快，多試幾次能顯著降低陷入局部最佳解的機率。 |
| **`HMM_TOL`** | `1e-4` | **收斂門檻**。設定得更嚴格（更小），要求模型收斂到更精確的位置。 |
| **`failure_warning_limit`** | `4` | **警告閾值**。累計失敗次數超過此值時觸發降級策略（回退至 $K_{min}$ 單次重試）。 |
| **`failure_per_k_limit`** | `2` | **單 K 失敗容忍**。單一 K 值連續失敗超過此次數時，跳過該 K 並嘗試下一個 K。 |

---

## 5. 常見問題排除 (Troubleshooting)

1. **出現 "Model is not converging"**
   * **檢查預處理**：確認是否正確執行三階段常態化（Log-Transform → Clipping → Z-Score）。這是解決不收斂的最強解法。
   * **檢查特徵**：執行 `np.isnan(X).any()` 和 `np.isinf(X).any()` 確認無異常值。
   * **調整參數**：若已標準化仍報錯，請增大 `HMM_MIN_COVAR`（如 `1e-2`）或稍微放寬 `HMM_TOL`。

2. **K 值總是選到最大值（例如 10）**
   * **樣本數檢查**：當前版本已內建動態 K 調整，若仍選最大 K，檢查 `n_samples / 5` 是否合理。
   * **變異度檢查**：執行 `np.var(X_normalized)` 確認不是常數特徵（< $10^{-8}$ 會自動鎖定 $K_{min}$）。
   * **特徵品質**：若動態調整後仍選最大 K，可能代表特徵雜訊過大，建議回到 Concept Extraction 調整 L1 正則化強度。


3. **分群結果與原始 Log 對不上**
   * **追溯性驗證**：程式碼已內建 `len(labels) != n_rows` 檢查，若通過檢查仍對不上，檢查是否在外部分析時不慎打亂順序。
   * **時序保證**：整個流程嚴格禁止 Shuffle 或 Drop，確保 `labels[i]` 對應 `X_raw[i]`。

4. **訓練速度慢**
   * 檢查 `HMM_ENABLE_PARALLEL` 是否生效。
   * 確認並行化是否實作在「扁平化任務列表（Flattened Tasks）」上，以最大化 CPU 利用率。

5. **訓練過程出現大量 Warning**
   * **正常現象**：Per-Dataset 策略下，單一 K 值可能因隨機初始化不佳而失敗，但只要有成功的就能繼續。
   * **檢查閾值**：若超過 `failure_warning_limit`（預設 4），會觸發降級策略。
   * **根本解法**：檢查 `HMM_MIN_COVAR` 是否過小（建議 $\geq 10^{-3}$），或特徵是否存在常數欄位。