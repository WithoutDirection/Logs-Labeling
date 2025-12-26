# 概念提取 (ConceptExtractor)

## 1\. 概述 (Overview)

`ConceptExtractor` 負責將高維度的稠密向量（Dense Vectors）映射到低維度的「潛在概念空間」（Latent Concept Space）。
無論輸入是來自內部系統日誌的 `Log Vector`，還是來自外部知識庫（E.g. MITRE ATT\&CK）的 `Technique Vector`，皆可以透過矩陣分解或機率模型找出數據中隱含的「**概念**」，將其轉化為概念權重分佈
> （例如：某段日誌 80% 屬於「資料竊取概念」、20% 屬於「網路掃描概念」）。

### 核心概念

為了確保不同 Dataset 資料夾（對應不同 Technique）產出的概念向量具有可比性，我們採用以下策略：

1.  **全域聯合訓練 (Joint Global Training)**：不針對單一 Dataset 訓練模型，而是聚合「外部知識庫」與「多個 Log Dataset 的採樣」來訓練一個共用的 NMF 模型。這確保了基矩陣 $W$（概念定義）在所有資料中是一致的。
2.  **獨立批次轉換 (Independent Batch Transformation)**：利用訓練好的共用 $W$，對每一個原始資料夾進行獨立的轉換，生成對應的概念權重矩陣 $H$，並維持原始的檔案目錄結構。

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
    * 路徑結構：`data/ConceptVectors/{LogID}/data.arrow`（與輸入結構對應）。
    * 內容：轉換後的概念權重矩陣 $H_{local}$，維度為 $N \times k$（$N$=樣本數, $k$=概念數）。
    * Metadata：需同步複製原始的 `state.json` 或 `dataset.info` 以保留標籤資訊。
2.  **概念模型 (Concept Model)**：
    * 路徑：`models/nmf_concept_model.pkl`
    * 內容：包含訓練好的基矩陣 $W$ 與模型參數。

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
    - **稀疏性 (Sparsity)**：**不施加稀疏約束**。目的是學習一種密集的表示形式 (Dense Representation)，以完整捕捉嵌入向量中的緊湊潛在結構。
- **解釋性**：
    - NMF 產生的權重皆為非負值，這與 PCA/SVD 不同，使得特徵具有加法性質，**更容易解釋為「該日誌由哪些概念組成」**。

### B. 隱含狄利克雷分佈 (LDA) - 替代方法
雖然 NMF 是主要路徑，但在需要探索機率分佈或處理離散特徵時，可提供 LDA 作為選項。
- **機制**：假設每個日誌是由多個潛在主題混合生成，透過機率生成模型逆向推導主題分佈。
- **應用**：提供軟分群 (Soft Clustering) 的機率觀點。

---


## 4. 處理流程詳解 (Workflow)

流程分為兩個獨立階段：**模型訓練 (A)** 與 **資料轉換 (B)**。

### 階段 A：訓練全域概念模型 (Train Global Model)
此階段目標是學習出 $X \approx HW$ 中的 $W$。

1.  **資料採樣與聚合 (Sampling & Aggregation)**：
    * 讀取所有外部知識向量 ($X_M$)。
    * 遍歷 `data/LogVectors/` 下的所有子資料夾，從每個 Dataset 中隨機採樣一定比例（例如 10%）的 Log Vector。
    * 將上述所有向量垂直堆疊（Stacking），構建出一個具代表性的全域訓練矩陣 $X_{train}$。
2.  **前處理 (Preprocessing)**：
    * 檢查 $X_{train}$ 數值。若使用 NMF，需執行平移（Shifting）或 Min-Max Scaling 確保非負性（Non-negative）。
3.  **模型擬合 (Model Fitting)**：
    * 初始化 NMF 模型，設定概念數 $k$（超參數）。
    * 配置 **NNDSVD** 初始化與 **座標下降 (Coordinate Descent)** 求解器以優化收斂。
    * 執行分解，獲得並凍結基矩陣 $W$。
4.  **模型持久化 (Save)**：
    * 將訓練好的模型物件儲存至硬碟。

### 階段 B：批次轉換與映射 (Batch Transform & Map)
此階段利用固定的 $W$ 將各別資料集映射至概念空間。

1.  **載入模型**：讀取 `models/nmf_concept_model.pkl`。
2.  **資料夾迭代 (Directory Iteration)**：
    * 掃描 `data/LogVectors/` 下的所有 Log ID 資料夾。
3.  **單一資料集轉換 (Per-Dataset Transform)**：
    * 讀取：載入該資料夾的 `data.arrow` ($X_{local}$)。
    * 投影：呼叫模型的 `transform()` 方法。數學上即固定 $W$，求解 $H_{local}$ 使得 $X_{local} \approx H_{local}W$。
    * **注意**：此步驟**不**更新 $W$。
4.  **結構化寫入 (Structured Write)**：
    * 建立對應的輸出目錄 `data/ConceptVectors/{LogID}/`。
    * 將 $H_{local}$ 存為 `data.arrow`。
    * 複製原始資料夾中的 Metadata 檔案。

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
當資料量龐大時（如百萬級樣本），可啟用 GPU 加速以大幅縮短訓練時間。`NMFGpu` 類別採用 **乘法更新規則 (Multiplicative Update Rules)**，完全基於矩陣運算，適合 GPU 平行加速。

* **演算法**：
    * 更新規則：$H = H \cdot (W^T X) / (W^T W H + \epsilon)$，$W = W \cdot (X H^T) / (W H H^T + \epsilon)$
    * 自動保持非負性，無需額外 Clamping
* **OOM 保護**：
    * 自動偵測 GPU 記憶體，動態計算安全的 batch size
    * 若資料超過記憶體限制，自動切換至 Mini-batch 訓練模式
* **相容性檢查**：
    * 啟動時執行 CUDA kernel 測試運算，確認 GPU compute capability 相容性
    * 若 GPU 不相容（如 GTX 1080 Ti 搭配 CUDA 12+ PyTorch），自動回退至 CPU 模式

### API 設計(`ConceptExtractor`)
下列方法為主要API call，方便其他模組直接引用：

* `prepare_training_data(log_vectors_dir, external_knowledge_dir, sample_ratio)`: 依據設定的日誌向量目錄與外部知識目錄載入向量，逐資料集採樣後垂直堆疊成訓練矩陣。
* `fit_global_model(X_train)`: 以 Min-Max 縮放後的 `X_train` 擬合 NMF/LDA 模型並凍結基矩陣；概念數、收斂條件等超參數來自初始化。當 `use_gpu=True` 且 CUDA 可用時，使用 `NMFGpu` 加速訓練。
* `transform_dataset(input_path, output_path, copy_metadata=True)`: 讀取單一資料集向量、投影至概念空間並輸出 Feather，必要時一併複製 `state.json`/`dataset_info.json`。
* `batch_transform(log_vectors_dir, concept_vectors_dir)`: 對 `log_vectors_dir` 下所有子資料夾批次轉換，維持 `{LogID}_logvectors -> {LogID}_concepts` 目錄對應。
* `get_concept_basis()`: 回傳已訓練的 $W$ 基矩陣（或 LDA 主題-詞彙分佈），供分析或視覺化使用。
> train_concept_extractor(...)` 執行資料準備、模型訓練與儲存；`transform_all_datasets(...)` 讀取既有模型後批次轉換所有資料集。

---

## 6. 模型超參數與設定 (Hyperparameters & Config)

結合現有 [Logs Labeling/config.py](Logs%20Labeling/config.py) 設定，`ConceptExtractor` 相關與建議的主要超參數如下：

| 參數角色                    | Config Key / 名稱              | 型別   | 說明                                               | 建議預設值              |
|-----------------------------|--------------------------------|--------|----------------------------------------------------|-------------------------|
| 概念數 (latent concepts k)  | `NMF_COMPONENTS`               | int    | NMF 分解的概念維度，決定 $W$ 與 $H$ 的寬度。       | `10`                    |
| 日誌向量根目錄              | `LOG_VECTORS_DIR`             | str    | 載入 `data/LogVectors/{LogID}/data.arrow` 的根路徑 | `data/LogVectors`       |
| 訓練隨機種子                | `SEED`                         | int    | 控制採樣與模型初始化的隨機性，方便重現結果。      | `42`                    |
| 訓練資料採樣比例            | `CONCEPT_SAMPLE_RATIO` | float  | 從各資料集抽樣的比例，用於建構 $X_{train}$。      | `0.1`                   |
| 最大訓練迭代次數            | `NMF_MAX_ITER`         | int    | NMF 的最大迭代次數，避免訓練時間過長。            | `200`                   |
| 收斂容忍度                  |`NMF_TOL`              | float  | NMF 收斂條件，控制訓練精度與時間的權衡。          | `1e-4`                  |
| 初始化方法                  |`NMF_INIT`             | str    | NMF 初始化策略，例如 `nndsvd`。                    | `"nndsvd"`            |
| 概念模型輸出路徑            | `NMF_MODEL_PATH`       | str    | 儲存 `nmf_concept_model.pkl` 的完整路徑。         | `models/nmf_concept_model.pkl` |

### GPU 加速設定

| 參數角色                    | Config Key / 名稱              | 型別         | 說明                                                             | 建議預設值   |
|-----------------------------|--------------------------------|--------------|------------------------------------------------------------------|--------------|
| 啟用 GPU 加速               | `NMF_USE_GPU`                  | bool         | 是否嘗試使用 GPU 加速 NMF 訓練                                   | `True`       |
| GPU Batch Size              | `NMF_GPU_BATCH_SIZE`           | int \| None  | Mini-batch 大小；`None` 表示自動根據 GPU 記憶體計算              | `None`       |
| 數值穩定常數                | `NMF_GPU_EPSILON`              | float        | 防止除零的小常數 ($\epsilon$)                                    | `1e-8`       |
| 收斂檢查間隔                | `NMF_GPU_CHECK_INTERVAL`       | int          | 每隔多少次迭代檢查收斂狀態                                       | `10`         |
| 詳細輸出                    | `NMF_GPU_VERBOSE`              | bool         | 是否顯示 GPU NMF 訓練進度                                        | `True`       |

**注意**：GPU 加速需要安裝與 GPU compute capability 相容的 PyTorch 版本。若 GPU 不相容（例如舊版 GPU 搭配新版 PyTorch），系統會自動回退至 sklearn CPU NMF。
