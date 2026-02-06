# 概念提取 (ConceptExtractor)

## 1\. 概述 (Overview)

`ConceptExtractor` 負責將高維度的稠密向量（Dense Vectors）映射到低維度的「潛在概念空間」（Latent Concept Space）。
無論輸入是來自內部系統日誌的 `Log Vector`，還是來自外部知識庫（E.g. MITRE ATT\&CK）的 `Technique Vector`，皆可以透過矩陣分解或機率模型找出數據中隱含的「**概念**」，將其轉化為概念權重分佈
> （例如：某段日誌 80% 屬於「資料竊取概念」、20% 屬於「網路掃描概念」）。

### 核心概念

本模組採用 **Per-Dataset 策略** — 每個 Dataset 各自訓練一個獨立的 NMF 模型，並結合外部知識作為「語義錨點」來凸顯該 Dataset 的特異性：

1.  **Per-Dataset 局部訓練 (Per-Dataset Local Training)**：針對每個 Dataset 單獨訓練一個 NMF 模型。訓練時將該 Dataset 的 Log Vectors 與外部知識庫（MITRE ATT&CK）的 Technique Vectors 垂直堆疊，聯合進行矩陣分解。
2.  **語義錨點機制 (Semantic Anchor Mechanism)**：外部知識向量作為「語義錨點」，引導 NMF 學習出與已知攻擊模式相關的概念。萃取出的概念會同時反映：
    * 該 Dataset 與已知攻擊模式的相似性
    * 該 Dataset 的獨特行為模式
3.  **L1 稀疏性約束 (L1 Sparsity Constraint)**：在 NMF 更新規則中加入 L1 正則化項，強制概念權重矩陣 $H$ 更加稀疏，使每個日誌傾向於只屬於少數幾個明確的概念，提升解釋性。
4.  **僅轉換 Dataset 部分 (Dataset-Only Transform)**：訓練時使用 Dataset + External Knowledge，但轉換時僅對 Dataset 部分進行投影，排除外部知識的干擾。
5.  **GPU 併發控制 (GPU Concurrency Control)**：當使用 GPU 加速時，自動強制單執行緒處理（`n_jobs=1`），避免多 Process 競爭 VRAM 導致 OOM。

## 2. 輸入與輸出 (I/O Specification)

### 輸入 (Input)
本模組需處理來自不同來源的嵌入矩陣，支援以下輸入:
1.  **日誌向量集合 (Log Vectors Collection)**：
    * 路徑結構：`data/LogVectors/{LogID}/data.arrow`
    * 內容：該 Log ID 對應的時序日誌向量矩陣 $X_{local}$。
2.  **外部知識向量 (External Knowledge Vectors)**：
    * 路徑：`data/ExternalKnowledge/MITRE_ATTACK/data.arrow`
    * 內容：MITRE ATT&CK Technique 描述的嵌入向量 $X_{M}$。

> **資料格式**：輸入為非負特徵矩陣 $X \in \mathbb{R}^{n \times m}_+$，其中 $n$ 為樣本數，$m$ 為原始特徵維度（如 768）。

### 輸出 (Output)
本模組將產生以下兩類輸出:
1.  **概念分佈資料集 (Concept Distribution Datasets)**：
    * 路徑結構：`data/ConceptVectors/{DatasetID}_concepts/data-00000-of-00001.arrow`（與輸入結構對應）。
    * 內容：轉換後的概念權重矩陣 $H_{local}$，維度為 $N \times k$（$N$=樣本數, $k$=概念數）。
    * Metadata：需同步複製原始的 `state.json` 或 `dataset_info.json` 以保留標籤資訊。
2.  **Per-Dataset 概念模型 (Per-Dataset Concept Model)**：
    * 路徑：`data/ConceptVectors/{DatasetID}_concepts/nmf_model.pkl`
    * 內容：每個 Dataset 獨立訓練的模型，包含基矩陣 $W$、Scaler、模型參數等。

---

## 3. 核心實作機制 (Implementation Mechanisms)

本模組以 **NMF** 為主要實作標準，並保留 **LDA** 作為機率模型的替代路徑。

### A. 非負矩陣分解 (NMF) - 主要方法
NMF 因其出色的降維能力與直觀的「基於組件（Part-based）」表示形式，被選為處理高維度稀疏日誌資料的首選技術。

- **數學原理**：
    - 目標是將輸入矩陣 $X$ 分解為兩個較小的非負矩陣 $W$ 與 $H$，使得 $X \approx HW$。
    - 此過程將高維空間 ($m$) 映射到低維度的潛在語義空間 ($k \ll m$)。
- **配置細節 (Configuration)**：
    - **初始化 (Initialization)**：採用 **NNDSVD** (Non-Negative Double Singular Value Decomposition)。此方法能增強模型收斂速度並提升分解的穩定性。
    - **優化求解器 (Solver)**：使用 **座標下降法 (Coordinate Descent)** 進行迭代優化。
    - **L1 稀疏性約束 (L1 Sparsity)**：在更新規則中加入 L1 正則化項（預設強度 0.01），強迫 $H$ 矩陣更加稀疏，讓每個日誌傾向於只屬於少數幾個明確的概念，大幅提升解釋性。
- **更新規則 (含 L1 正則化)**：
    - $H = H \cdot (W^T X) / (W^T W H + \lambda_{L1} + \epsilon)$
    - $W = W \cdot (X H^T) / (W H H^T + \epsilon)$
    - 其中 $\lambda_{L1}$ 為 L1 正則化強度，$\epsilon$ 為數值穩定常數
- **解釋性**：
    - NMF 產生的權重皆為非負值，這與 PCA/SVD 不同，使得特徵具有加法性質，**更容易解釋為「該日誌由哪些概念組成」**。
    - L1 正則化進一步強化解釋性，避免單一日誌同時屬於過多概念的模糊狀況。

### B. 隱含狄利克雷分佈 (LDA) - 替代方法
雖然 NMF 是主要路徑，但在需要探索機率分佈或處理離散特徵時，可提供 LDA 作為選項。
- **機制**：假設每個日誌是由多個潛在主題混合生成，透過機率生成模型逆向推導主題分佈。
- **應用**：提供軟分群 (Soft Clustering) 的機率觀點。

---


## 4. 處理流程詳解 (Workflow)

採用 **Per-Dataset 策略**，每個 Dataset 執行獨立的完整流程：**載入 → 訓練 → 轉換 → 存檔**。

### 主入口：`process_single_dataset()`
此方法是 Per-Dataset 策略的核心入口點，封裝了完整的處理流程。

### 步驟 1：載入外部知識 (Load External Knowledge)
* 呼叫 `load_external_knowledge()` 載入 MITRE ATT&CK Technique Vectors
* 外部知識向量快取於 `_external_vectors`，避免重複載入
* 作用：作為「語義錨點」引導 NMF 學習與已知攻擊模式相關的概念

### 步驟 2：載入 Dataset 向量 (Load Dataset Vectors)
* 讀取該 Dataset 的 `data-00000-of-00001.arrow`
* 取出 Log Embedding Vectors ($X_{dataset}$)

### 步驟 3：Per-Dataset NMF 訓練 (Fit Local Model)
由 `fit_local_model()` 執行：
1.  **資料聯合 (Data Union)**：
    * 將 Dataset Vectors ($X_{dataset}$) 與 External Vectors ($X_{external}$) 垂直堆疊
    * $X_{train} = [X_{dataset}; X_{external}]$
2.  **動態調整概念數**：
    * 確保 $k < \min(n_{samples}, n_{features})$
    * 若樣本過少則自動降低概念數
3.  **前處理 (Preprocessing)**：
    * Min-Max Scaling 確保非負性
    * `np.clip(X, 0, None)` 額外保護
4.  **模型擬合 (Model Fitting)**：
    * 初始化 NMF 模型（或 `NMFGpu` 若啟用 GPU）
    * 配置 NNDSVD 初始化與座標下降求解器
    * 執行分解，獲得基矩陣 $W$ 與係數矩陣 $H$

### 步驟 4：轉換至概念空間 (Transform to Concept Space)
由 `transform_dataset_only()` 執行：
* 僅對 Dataset 部分進行投影（排除 External Knowledge）
* 產生 Dataset 的概念向量 $H_{dataset}$

### 步驟 5：結構化輸出 (Structured Output)
1.  建立輸出目錄：`{output_dir}/{DatasetID}_concepts/`
2.  儲存概念向量：`data-00000-of-00001.arrow`
3.  儲存模型：`nmf_model.pkl`（每個 Dataset 獨立一份）
4.  複製 Metadata：`state.json`, `dataset_info.json`

---

## 5. 實作機制與演算法 (Implementation Mechanisms)

### 演算法選擇
* **首選：非負矩陣分解 (NMF)**
    * **原因**：提供基於組件（Part-based）的加法特徵，解釋性高，適合稀疏且高維的日誌資料。
    * **配置**：不施加稀疏約束（Sparse Constraint），以學習密集的潛在結構 。
    * **GPU 加速**：支援 PyTorch GPU 後端加速（透過 `NMFGpu` 類別），自動偵測 CUDA 相容性並在不支援時回退至 CPU。
* **備選：隱含狄利克雷分佈 (LDA)**
    * **場景**：若需要機率分佈解釋（Soft Clustering）時使用。需注意輸入特徵需轉換為適合 LDA 的格式（如虛擬詞頻）。

### GPU 加速實作 (`NMFGpu`)
當資料量龐大時（如百萬級樣本），可啟用 GPU 加速以大幅縮短訓練時間。`NMFGpu` 類別採用 **乘法更新規則 (Multiplicative Update Rules)** 並支援 **L1 稀疏性約束**，完全基於矩陣運算，適合 GPU 平行加速。

* **演算法（含 L1 正則化）**：
    * 更新規則：$H = H \cdot (W^T X) / (W^T W H + \lambda_{L1} + \epsilon)$，$W = W \cdot (X H^T) / (W H H^T + \epsilon)$
    * 自動保持非負性，無需額外 Clamping
    * L1 正則化項直接整合於更新規則中，無額外計算開銷
* **OOM 保護**：
    * 自動偵測 GPU 記憶體，動態計算安全的 batch size
    * 若資料超過記憶體限制，自動切換至 Mini-batch 訓練模式
* **相容性檢查**：
    * 啟動時執行 CUDA kernel 測試運算，確認 GPU compute capability 相容性
    * 若 GPU 不相容（如 GTX 1080 Ti 搭配 CUDA 12+ PyTorch），自動回退至 CPU 模式
* **併發控制**：
    * 當 `use_gpu=True` 且 CUDA 可用時，自動強制 `n_jobs=1`
    * 避免多個 Python Process 同時嘗試使用 GPU，導致 VRAM 競爭和 OOM

### API 設計(`ConceptExtractor`)
下列方法為主要 API call，方便其他模組直接引用：

**外部知識載入**：
* `load_external_knowledge(external_dir)`: 載入外部知識庫（MITRE ATT&CK）的 Embedding Vectors 作為語義錨點。向量快取於 `_external_vectors`，僅需載入一次。

**Per-Dataset 訓練**：
* `fit_local_model(dataset_vectors, external_vectors=None, dataset_id="unknown")`: 針對單一 Dataset 訓練局部 NMF 模型。將 Dataset + External Knowledge 聯合訓練，External 作為語義錨點凸顯 Dataset 特異性。當 `use_gpu=True` 且 CUDA 可用時，使用 `NMFGpu` 加速訓練。

**資料轉換**：
* `transform_local(X)`: 使用局部模型將向量投影至概念空間。
* `transform_dataset_only(dataset_vectors)`: **[核心]** 僅轉換 Dataset 部分（排除 External Knowledge）。這是 Per-Dataset 策略的核心 — 訓練時使用聯合資料，轉換時僅對 Dataset 進行投影。

**模型存取**：
* `save_local_model(output_dir, dataset_id)`: 儲存 Per-Dataset 模型至 `{output_dir}/{dataset_id}_concepts/nmf_model.pkl`。
* `load_local_model(model_path)`: 載入已儲存的 Per-Dataset 模型。

**完整流程封裝**：
* `process_single_dataset(dataset_id, input_path, output_dir, ...)`: **[主入口]** 處理單一 Dataset 的完整流程：載入 → 訓練 → 轉換 → 存檔。支援選用 TF-IDF 加權 (`use_tfidf_weighting=True`)。

**資料載入**：
* `load_dataset_vectors(dataset_path)`: 載入單一 Dataset 的 Log Vectors（Arrow 格式）。

**Pipeline 整合**：
在 `Pipeline.py` 中的典型呼叫模式：
```python
extractor = ConceptExtractor(n_concepts=config.NMF_COMPONENTS)
extractor.load_external_knowledge(config.EXTERNAL_KNOWLEDGE_DIR)

for dataset_id, input_path in datasets:
    # 每個 Dataset 重置模型，執行獨立訓練
    extractor.model = None
    extractor._is_fitted = False
    concept_vectors = extractor.process_single_dataset(
        dataset_id=dataset_id,
        input_path=input_path,
        output_dir=config.CONCEPT_VECTORS_DIR,
    )
```

---

## 6. 超參數配置 (Hyperparameters)

### 核心訓練參數

| 參數角色                    | Config Key / 名稱              | 型別   | 說明                                                             | 建議預設值              |
|-----------------------------|--------------------------------|--------|------------------------------------------------------------------|-------------------------|
| 概念數量                    | `NMF_COMPONENTS`               | int    | 潛在概念空間的維度 $k$，即 NMF 的組件數。                       | `75`                    |
| L1 正則化強度               | `NMF_L1_REG`                   | float  | L1 正則化強度，控制概念稀疏度。值越大越稀疏，0 表示無正則化。 | `0.01`                  |
| Embedding 根目錄            | `EMBEDDINGS_DIR`               | str    | 載入 `data/Embeddings/{DatasetID}_embeddings/` 的根路徑         | `data/Embeddings`       |
| 概念向量輸出目錄            | `CONCEPT_VECTORS_DIR`          | str    | 輸出 `data/ConceptVectors/{DatasetID}_concepts/` 的根路徑       | `data/ConceptVectors`   |
| 外部知識目錄                | `EXTERNAL_KNOWLEDGE_DIR`       | str    | MITRE ATT&CK Embedding 所在目錄                                  | `data/ExternalKnowledge`|
| 訓練隨機種子                | `SEED`                         | int    | 控制模型初始化的隨機性，方便重現結果。                           | `42`                    |
| 最大訓練迭代次數            | `NMF_MAX_ITER`                 | int    | NMF 的最大迭代次數，避免訓練時間過長。                           | `500`                   |
| 收斂容忍度                  | `NMF_TOL`                      | float  | NMF 收斂條件，控制訓練精度與時間的權衡。                         | `1e-3`                  |
| 初始化方法                  | `NMF_INIT`                     | str    | NMF 初始化策略，例如 `nndsvd`。                                  | `"nndsvd"`              |

### GPU 加速設定

| 參數角色                    | Config Key / 名稱              | 型別         | 說明                                                             | 建議預設值   |
|-----------------------------|--------------------------------|--------------|------------------------------------------------------------------|--------------|
| 啟用 GPU 加速               | `NMF_USE_GPU`                  | bool         | 是否嘗試使用 GPU 加速 NMF 訓練                                   | `True`       |
| GPU Batch Size              | `NMF_GPU_BATCH_SIZE`           | int \| None  | Mini-batch 大小；`None` 表示自動根據 GPU 記憶體計算              | `None`       |
| 數值穩定常數                | `NMF_GPU_EPSILON`              | float        | 防止除零的小常數 ($\epsilon$)                                    | `1e-8`       |
| 收斂檢查間隔                | `NMF_GPU_CHECK_INTERVAL`       | int          | 每隔多少次迭代檢查收斂狀態                                       | `10`         |
| 詳細輸出                    | `NMF_GPU_VERBOSE`              | bool         | 是否顯示 GPU NMF 訓練進度                                        | `True`       |


