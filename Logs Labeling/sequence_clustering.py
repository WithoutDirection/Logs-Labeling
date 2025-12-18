"""
SequenceClustering：以隱馬可夫模型 (HMM) 進行事件日誌序列分群。
使用概念矩陣 H_E 作為輸入，辨識同質群體。
支援並行化 Grid Search、兩階段分層訓練與批次推論。
"""

import os
import pickle
import numpy as np
from typing import Optional, Tuple, List, Dict
from hmmlearn import hmm
import pyarrow.feather as feather

import config


class SequenceClustering:
    """基於 HMM 的事件日誌序列分群，支援加速優化機制。"""

    def __init__(
        self,
        k_min: int = config.HMM_K_MIN,
        k_max: int = config.HMM_K_MAX,
        n_starts: int = config.HMM_N_STARTS,
        n_iter: int = config.HMM_N_ITER,
        tol: float = config.HMM_TOL,
        covariance_type: str = config.HMM_COVARIANCE_TYPE,
        random_state: int = config.SEED,
        enable_parallel: bool = config.HMM_ENABLE_PARALLEL,
        n_jobs: int = config.HMM_PARALLEL_N_JOBS,
        enable_two_stage: bool = config.HMM_ENABLE_TWO_STAGE,
        two_stage_threshold: int = config.HMM_TWO_STAGE_THRESHOLD,
        two_stage_sample_ratio: float = config.HMM_TWO_STAGE_SAMPLE_RATIO,
        max_train_samples: int = config.HMM_MAX_TRAIN_SAMPLES,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.n_starts = n_starts
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        # 加速優化設定
        self.enable_parallel = enable_parallel
        self.n_jobs = n_jobs
        self.enable_two_stage = enable_two_stage
        self.two_stage_threshold = two_stage_threshold
        self.two_stage_sample_ratio = two_stage_sample_ratio
        self.max_train_samples = max_train_samples
        
        self.best_model: Optional[hmm.GaussianHMM] = None
        self.best_k: Optional[int] = None
        self.best_score: float = float("-inf")

    def _train_single_hmm(
        self,
        X: np.ndarray,
        lengths: Optional[List[int]],
        n_components: int,
        seed: int,
    ) -> Tuple[Optional[hmm.GaussianHMM], float]:
        """訓練單一 HMM 模型並回傳其對數概似度。"""
        try:
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=seed,
            )
            # * 使用 lengths 參數防止跨序列轉移學習
            model.fit(X, lengths=lengths)
            score = model.score(X, lengths=lengths)
            return model, score
        except Exception as e:
            return None, float("-inf")

    def _grid_search(
        self,
        X: np.ndarray,
        lengths: Optional[List[int]],
        k_min: int,
        k_max: int,
        n_starts: int,
    ) -> Tuple[Optional[hmm.GaussianHMM], int, float]:
        """
        統一的 Grid Search 實作。
        joblib 在 n_jobs=1 時自動切換為串行模式，無需額外判斷。
        """
        from joblib import Parallel, delayed
        
        # * 建立所有 (K, seed) 組合的任務列表
        tasks = [
            (k, self.random_state + k * n_starts + i)
            for k in range(k_min, k_max + 1)
            for i in range(n_starts)
        ]
        
        n_jobs = self.n_jobs if self.enable_parallel else 1
        
        # * 並行/串行訓練（由 n_jobs 決定）
        results = Parallel(n_jobs=n_jobs, backend=config.HMM_PARALLEL_BACKEND)(
            delayed(self._train_single_hmm)(X, lengths, k, seed)
            for k, seed in tasks
        )
        
        # * 選出最佳模型
        best_model, best_k, best_score = None, None, float("-inf")
        for (model, score), (k, _) in zip(results, tasks):
            if score > best_score:
                best_score = score
                best_model = model
                best_k = k
        
        return best_model, best_k, best_score

    def _sample_sequences(
        self,
        X: np.ndarray,
        lengths: List[int],
        sample_ratio: float,
    ) -> Tuple[np.ndarray, List[int]]:
        """序列感知採樣：隨機選取部分序列而非隨機樣本。"""
        np.random.seed(self.random_state)
        n_sequences = len(lengths)
        n_sample = max(1, int(n_sequences * sample_ratio))
        
        # * 隨機選取完整序列，保持序列結構
        selected_indices = np.random.choice(n_sequences, n_sample, replace=False)
        selected_indices = np.sort(selected_indices)
        
        cumsum = np.cumsum([0] + lengths)
        
        sampled_X_list = []
        sampled_lengths = []
        for idx in selected_indices:
            start, end = cumsum[idx], cumsum[idx + 1]
            sampled_X_list.append(X[start:end])
            sampled_lengths.append(lengths[idx])
        
        sampled_X = np.vstack(sampled_X_list)
        return sampled_X, sampled_lengths

    def _prepare_search_data(
        self,
        X: np.ndarray,
        lengths: Optional[List[int]],
    ) -> Tuple[np.ndarray, Optional[List[int]]]:
        """準備搜尋用資料：根據設定決定是否採樣。"""
        n_samples = len(X)
        use_two_stage = (
            self.enable_two_stage 
            and n_samples > self.two_stage_threshold 
            and lengths is not None
        )
        
        if use_two_stage:
            print(f"[Phase 1] 子集採樣，快速搜索最佳 K...")
            X_search, lengths_search = self._sample_sequences(
                X, lengths, self.two_stage_sample_ratio
            )
            print(f"  採樣後: {len(X_search)} 樣本, {len(lengths_search)} 序列")
            return X_search, lengths_search
        
        return X, lengths

    def optimize_global_hmm(
        self,
        X_train: np.ndarray,
        lengths: Optional[List[int]] = None,
        k_min: Optional[int] = None,
        k_max: Optional[int] = None,
        n_starts: Optional[int] = None,
    ) -> hmm.GaussianHMM:
        """
        透過網格搜索與穩定性迴圈尋找最佳 HMM。
        流程：資料準備 -> Grid Search -> 最終訓練（若兩階段）。

        Args:
            X_train: 聚合後的概念矩陣 H_E，形狀 (n_samples, n_features)
            lengths: 每個序列的長度列表，用於防止跨序列轉移學習
            k_min: 隱藏狀態數量下界
            k_max: 隱藏狀態數量上界
            n_starts: 每個 K 的隨機初始化次數

        Returns:
            最佳的 GaussianHMM 模型
        """
        k_min = k_min or self.k_min
        k_max = k_max or self.k_max
        n_starts = n_starts or self.n_starts
        X = np.asarray(X_train, dtype=np.float64)
        n_samples = len(X)
        
        # * Step 1: 資料前處理（截斷過大資料集）
        if n_samples > self.max_train_samples:
            print(f"[Warning] 樣本數 {n_samples} 超過上限，進行截斷")
            if lengths:
                X, lengths = self._truncate_to_limit(X, lengths, self.max_train_samples)
            else:
                X = X[:self.max_train_samples]
            n_samples = len(X)
        
        use_two_stage = (
            self.enable_two_stage 
            and n_samples > self.two_stage_threshold 
            and lengths is not None
        )
        print(f"[Info] 樣本數={n_samples}, 並行={self.enable_parallel}, 兩階段={use_two_stage}")
        
        # * Step 2: 準備搜尋資料（若兩階段則採樣）
        X_search, lengths_search = self._prepare_search_data(X, lengths)
        
        # * Step 3: 執行 Grid Search
        _, best_k, _ = self._grid_search(X_search, lengths_search, k_min, k_max, n_starts)
        print(f"  Grid Search 最佳 K = {best_k}")
        
        # * Step 4: 最終訓練（兩階段時用全量資料重訓練）
        if use_two_stage:
            print(f"[Phase 2] 使用 K={best_k} 在全量資料上訓練...")
            self.best_model, self.best_score = self._train_single_hmm(
                X, lengths, best_k, self.random_state
            )
        else:
            self.best_model, self.best_score = self._train_single_hmm(
                X, lengths, best_k, self.random_state
            )
        self.best_k = best_k
        
        print(f"[Info] 最佳 K={self.best_k}, Log-Likelihood={self.best_score:.4f}")
        return self.best_model

    def _truncate_to_limit(
        self,
        X: np.ndarray,
        lengths: List[int],
        limit: int,
    ) -> Tuple[np.ndarray, List[int]]:
        """截斷資料至樣本數上限，保持序列完整性。"""
        cumsum = np.cumsum(lengths)
        valid_idx = np.searchsorted(cumsum, limit, side='right')
        if valid_idx == 0:
            valid_idx = 1
        
        total_samples = cumsum[valid_idx - 1]
        return X[:total_samples], lengths[:valid_idx]

    def decode_sequences(
        self,
        concept_vectors: np.ndarray,
        lengths: Optional[List[int]] = None,
        model: Optional[hmm.GaussianHMM] = None,
    ) -> np.ndarray:
        """
        使用 Viterbi 演算法預測隱藏狀態序列。

        Args:
            concept_vectors: 單一資料集的概念矩陣，形狀 (n_samples, n_features)
            lengths: 序列長度列表（通常單一資料集為 None）
            model: 已訓練的 HMM 模型（若為 None 則使用 best_model）

        Returns:
            分群標籤（隱藏狀態序列）
        """
        model = model or self.best_model
        if model is None:
            raise ValueError("尚無可用模型，請先執行 optimize_global_hmm")

        X = np.asarray(concept_vectors, dtype=np.float64)
        
        # * Viterbi 演算法：預測最可能的隱藏狀態路徑
        labels = model.predict(X, lengths=lengths)
        return labels

    def save_model(self, path: str = config.HMM_MODEL_PATH) -> None:
        """將最佳 HMM 模型儲存到磁碟。"""
        if self.best_model is None:
            raise ValueError("尚無模型可儲存，請先執行 optimize_global_hmm")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.best_model,
                "best_k": self.best_k,
                "best_score": self.best_score,
            }, f)
        print(f"[Info] 模型已儲存至 {path}")

    def load_model(self, path: str = config.HMM_MODEL_PATH) -> hmm.GaussianHMM:
        """從磁碟載入已訓練的 HMM 模型。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.best_model = data["model"]
        self.best_k = data["best_k"]
        self.best_score = data["best_score"]
        print(f"[Info] 已載入模型: K={self.best_k}, Score={self.best_score:.4f}")
        return self.best_model

    def fit_predict(
        self,
        X_train: np.ndarray,
        lengths: Optional[List[int]] = None,
    ) -> np.ndarray:
        """便捷方法：先優化模型再解碼序列。"""
        self.optimize_global_hmm(X_train, lengths)
        return self.decode_sequences(X_train, lengths)


# ======================== 資料載入與處理函式 ========================

def load_concept_vectors(
    dataset_ids: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    從 ConceptVectors 目錄載入概念向量。

    Args:
        dataset_ids: 要載入的資料集 ID 清單，None 則載入全部。

    Returns:
        dataset_id 對應的概念矩陣字典
    """
    vectors_dir = config.CONCEPT_VECTORS_DIR
    if not os.path.exists(vectors_dir):
        raise FileNotFoundError(f"找不到概念向量目錄: {vectors_dir}")

    result = {}
    
    # * 遍歷子目錄（每個資料集儲存為獨立資料夾）
    for subdir in os.listdir(vectors_dir):
        subdir_path = os.path.join(vectors_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        dataset_id = subdir.replace("_concepts", "")
        if dataset_ids is not None and dataset_id not in dataset_ids:
            continue
        
        # * 載入 Arrow/Feather 格式的概念向量
        arrow_path = os.path.join(subdir_path, "data-00000-of-00001.arrow")
        if not os.path.exists(arrow_path):
            print(f"[Warning] 找不到 Arrow 檔案: {arrow_path}")
            continue
        
        try:
            table = feather.read_table(arrow_path)
            if "concept_vector" in table.column_names:
                vectors = np.array(table["concept_vector"].to_pylist())
            else:
                vectors = table.to_pandas().values
            result[dataset_id] = vectors
        except Exception as e:
            print(f"[Warning] 載入失敗 {arrow_path}: {e}")
            continue
    
    return result


def prepare_training_data(
    vectors_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[int]]:
    """
    準備訓練資料：堆疊矩陣並建立 lengths 列表。

    Args:
        vectors_dict: dataset_id 對應的概念矩陣字典

    Returns:
        X_train: 聚合後的訓練矩陣
        lengths: 每個序列的長度列表
    """
    # * 建立 lengths 列表，記錄每個序列長度以防止跨序列轉移
    X_list = []
    lengths = []
    
    for dataset_id, vectors in vectors_dict.items():
        X_list.append(vectors)
        lengths.append(len(vectors))
    
    X_train = np.vstack(X_list)
    return X_train, lengths


def save_cluster_results(
    dataset_id: str,
    labels: np.ndarray,
    original_length: int,
    output_dir: str = config.CLUSTER_RESULTS_DIR,
) -> None:
    """
    將分群標籤儲存至資料集輸出，並驗證長度一致性。

    Args:
        dataset_id: 資料集 ID
        labels: 分群標籤陣列
        original_length: 原始資料集長度（用於驗證）
        output_dir: 輸出目錄
    """
    # * 驗證分群結果與原始資料集長度一致
    if len(labels) != original_length:
        raise ValueError(
            f"長度不一致: 標籤數={len(labels)}, 原始長度={original_length}"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_id}_clusters.npy")
    np.save(output_path, labels)
    print(f"  已儲存: {output_path}")


# ======================== 主程式 ========================

if __name__ == "__main__":
    print("載入概念向量...")
    vectors = load_concept_vectors()
    
    if not vectors:
        print("[Error] 找不到概念向量，請先執行 ConceptExtractor")
        exit(1)
    
    print(f"已載入 {len(vectors)} 個資料集")
    
    # ===== 階段 A：全域模型優化訓練 =====
    print("\n=== 階段 A：全域模型優化訓練 ===")
    
    # * 準備訓練資料並建立 lengths 列表
    X_train, lengths = prepare_training_data(vectors)
    print(f"訓練資料: {X_train.shape}, 序列數: {len(lengths)}")
    
    clusterer = SequenceClustering()
    clusterer.optimize_global_hmm(X_train, lengths)
    clusterer.save_model()
    
    # ===== 階段 B：獨立序列標註（批次推論） =====
    print("\n=== 階段 B：獨立序列標註 ===")
    
    # * 批次推論：逐個 Dataset 載入解碼，避免 OOM
    for dataset_id, concept_matrix in vectors.items():
        original_length = len(concept_matrix)
        labels = clusterer.decode_sequences(concept_matrix)
        
        save_cluster_results(dataset_id, labels, original_length)
        n_clusters = len(np.unique(labels))
        print(f"  {dataset_id}: {original_length} 條目 -> {n_clusters} 個群集")
    
    print("\n[完成] 序列分群已完成。")
