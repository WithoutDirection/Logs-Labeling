"""
SequenceClustering：以隱馬可夫模型 (HMM) 進行事件日誌序列分群。
採用 Per-Dataset HMM 策略：針對每一個 Dataset 獨立訓練 HMM 模型，
精確識別該特定攻擊行為內部的演變階段（如：初始存取 → 執行 → 清理）。
"""

import os
import pickle
import numpy as np
from typing import Optional, Tuple, List, Dict
from hmmlearn import hmm
import pyarrow.feather as feather

import config


class SequenceClustering:
    """
    基於 HMM 的事件日誌序列分群（Per-Dataset 策略）。
    
    核心策略：
    - 每個 Dataset 獨立訓練專屬 HMM 模型
    - 高頻隨機初始化確保收斂穩定性
    - 數值穩定性保護（min_covar）防止矩陣奇異
    """

    def __init__(
        self,
        k_min: int = config.HMM_K_MIN,
        k_max: int = config.HMM_K_MAX,
        n_starts: int = config.HMM_N_STARTS,
        n_iter: int = config.HMM_N_ITER,
        tol: float = config.HMM_TOL,
        covariance_type: str = config.HMM_COVARIANCE_TYPE,
        min_covar: float = config.HMM_MIN_COVAR,
        random_state: int = config.SEED,
        enable_parallel: bool = config.HMM_ENABLE_PARALLEL,
        n_jobs: int = config.HMM_PARALLEL_N_JOBS,
        failure_warning_limit: int = 4,
        failure_per_k_limit: int = 2,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.n_starts = n_starts
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.min_covar = min_covar  # 數值穩定性保護
        self.random_state = random_state
        
        # 並行優化設定
        self.enable_parallel = enable_parallel
        self.n_jobs = n_jobs
        self.failure_warning_limit = failure_warning_limit
        self.failure_per_k_limit = failure_per_k_limit
        self._warning_counter: Dict[str, int] = {}
        
        # Per-Dataset 模型儲存
        self.current_model: Optional[hmm.GaussianHMM] = None
        self.current_k: Optional[int] = None
        self.current_score: float = float("-inf")
        
        # 標準化參數（雙軌策略）
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

    def _train_single_hmm(
        self,
        dataset_id: str,
        X: np.ndarray,
        n_components: int,
        seed: int,
    ) -> Tuple[Optional[hmm.GaussianHMM], float, Optional[str]]:
        """
        訓練單一 HMM 模型並回傳其對數概似度。
        
        注意：Per-Dataset 策略下，單一資料集視為完整序列，
        不使用 lengths 參數（無跨資料集問題）。
        """
        try:
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=seed,
                min_covar=self.min_covar,  # 防止特徵稀疏導致矩陣奇異
            )
            model.fit(X)
            score = model.score(X)
            return model, score, None
        except Exception as e:
            key = f"{dataset_id}|K={n_components}"
            self._warning_counter[key] = self._warning_counter.get(key, 0) + 1
            msg = f"HMM 訓練失敗 dataset={dataset_id} (K={n_components}, seed={seed}): {e}"
            # 僅前幾次打印，之後靜音避免刷屏
            if self._warning_counter[key] <= 2:
                print(f"    [Warning] {msg}")
            return None, float("-inf"), msg

    def _grid_search(
        self,
        dataset_id: str,
        X: np.ndarray,
        k_min: int,
        k_max: int,
        n_starts: int,
    ) -> Tuple[Optional[hmm.GaussianHMM], int, float]:
        """
        執行 Grid Search 尋找最佳 K 值。
        針對單一 Dataset 內部並行化，加速優化過程。
        """
        best_model, best_k, best_score = None, None, float("-inf")
        failure_logs: List[str] = []

        for k in range(k_min, k_max + 1):
            consecutive_fail = 0
            for i in range(n_starts):
                seed = self.random_state + k * n_starts + i
                model, score, err_msg = self._train_single_hmm(dataset_id, X, k, seed)
                if model is not None and score > best_score:
                    best_score = score
                    best_model = model
                    best_k = k
                if err_msg:
                    failure_logs.append(err_msg)
                    consecutive_fail += 1
                    if consecutive_fail >= self.failure_per_k_limit:
                        # 這個 K 多次失敗，提早跳下一個 K
                        break
                else:
                    consecutive_fail = 0

            if best_model is not None and best_k == k:
                # 若該 K 已獲得成功模型且得分最佳，繼續嘗試下一個 K 以尋找更佳結果
                continue

        if failure_logs and len(failure_logs) >= self.failure_warning_limit and best_model is None:
            print(f"    [Warning] {dataset_id}: 超過 {self.failure_warning_limit} 次失敗，嘗試縮減 K={k_min} 單次重試")
            fallback_model, fallback_score, err = self._train_single_hmm(
                dataset_id,
                X,
                k_min,
                self.random_state,
            )
            if fallback_model is not None:
                return fallback_model, k_min, fallback_score
            raise RuntimeError(f"{dataset_id}: HMM 訓練多次失敗，已跳過。最後錯誤: {err}")

        return best_model, best_k, best_score

    def optimize_local_hmm(
        self,
        X: np.ndarray,
        k_min: Optional[int] = None,
        k_max: Optional[int] = None,
        n_starts: Optional[int] = None,
        dataset_id: str = "unknown_dataset",
    ) -> hmm.GaussianHMM:
        """
        針對單一 Dataset 執行局部模型優化。
        
        與 optimize_global_hmm 不同，此方法：
        - 不使用 lengths 參數（單一資料集為完整序列）
        - 不進行兩階段訓練（單一資料集資料量小，直接全量訓練）
        - 使用更高的 n_starts 確保穩定性

        Args:
            X: 單一 Dataset 的概念矩陣，形狀 (n_samples, n_features)
            k_min: 隱藏狀態數量下界
            k_max: 隱藏狀態數量上界
            n_starts: 每個 K 的隨機初始化次數

        Returns:
            該 Dataset 的最佳 GaussianHMM 模型
        """
        k_min = k_min or self.k_min
        k_max = k_max or self.k_max
        n_starts = n_starts or self.n_starts
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        
        # * 根據資料量動態收斂 K 上界，避免樣本過少仍嘗試高 K
        dynamic_k_max = min(k_max, max(2, n_samples // 5))
        effective_k_max = min(dynamic_k_max, n_samples - 1)

        # * 變異度過低時，鎖定最低 K，避免無意義的多狀態嘗試
        variance = np.var(X)
        if variance < 1e-8:
            print(f"    [Warning] 特徵變異度極低，鎖定 K={k_min} (var={variance:.2e})")
            effective_k_max = k_min

        if effective_k_max < k_min:
            print(f"    [Warning] 樣本數過少 ({n_samples})，使用 K={k_min}")
            effective_k_max = k_min
        
        print(f"    [Grid Search] K 範圍=[{k_min}, {effective_k_max}], "
              f"n_starts={n_starts}, 並行={self.enable_parallel}")
        
        # * 執行 Grid Search
        best_model, best_k, best_score = self._grid_search(
            dataset_id,
            X,
            k_min,
            effective_k_max,
            n_starts,
        )
        
        if best_model is None:
            raise RuntimeError("所有 HMM 訓練均失敗，請檢查資料品質或調整 min_covar")
        
        self.current_model = best_model
        self.current_k = best_k
        self.current_score = best_score
        
        print(f"    [結果] 最佳 K={best_k}, Log-Likelihood={best_score:.4f}")
        return best_model

    def decode_sequences(
        self,
        concept_vectors: np.ndarray,
        model: Optional[hmm.GaussianHMM] = None,
    ) -> np.ndarray:
        """
        使用 Viterbi 演算法預測隱藏狀態序列。

        Args:
            concept_vectors: 單一資料集的概念矩陣，形狀 (n_samples, n_features)
            model: 已訓練的 HMM 模型（若為 None 則使用 current_model）

        Returns:
            分群標籤（隱藏狀態序列）
        """
        model = model or self.current_model
        if model is None:
            raise ValueError("尚無可用模型，請先執行 optimize_local_hmm")

        X = np.asarray(concept_vectors, dtype=np.float64)
        
        # * Viterbi 演算法：預測最可能的隱藏狀態路徑
        labels = model.predict(X)
        return labels

    def process_single_dataset(
        self,
        dataset_id: str,
        concept_matrix: np.ndarray,
        output_dir: str = config.CLUSTER_RESULTS_DIR,
    ) -> np.ndarray:
        """
        處理單一 Dataset 的完整流程：優化、訓練、解碼、存檔。
        
        這是 Per-Dataset 策略的主控函式，包含追溯性驗證。

        Args:
            dataset_id: 資料集 ID
            concept_matrix: 概念矩陣，形狀 (n_samples, n_features)
            output_dir: 輸出目錄

        Returns:
            分群標籤陣列
        """
        n_rows = len(concept_matrix)
        print(f"\n[Processing] {dataset_id} ({n_rows} 筆資料)")
        
        # * Step 1: 資料清理與標準化（雙軌策略）
        clean_matrix = np.nan_to_num(concept_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # * 先對極端值做截斷，避免過大值拉高標準差
        lower = np.percentile(clean_matrix, 0.5)
        upper = np.percentile(clean_matrix, 99.5)
        clipped_matrix = np.clip(clean_matrix, lower, upper)
        
        # * 計算 Z-Score 標準化參數（用於 HMM 訓練軌道）
        self._scaler_mean = np.mean(clipped_matrix, axis=0)
        self._scaler_std = np.std(clipped_matrix, axis=0) + 1e-8
        X_scaled = (clipped_matrix - self._scaler_mean) / self._scaler_std

        # * Step 2: 使用標準化資料進行 HMM 訓練
        model = self.optimize_local_hmm(X_scaled, dataset_id=dataset_id)
        
        # * Step 3: 使用標準化資料進行 Viterbi 解碼
        labels = self.decode_sequences(X_scaled, model)
        
        # * Step 4: 追溯性驗證 (Traceability Check)
        if len(labels) != n_rows:
            raise ValueError(
                f"追溯性驗證失敗: 標籤數={len(labels)}, 原始長度={n_rows}"
            )
        
        # * Step 5: 存檔（模型 + 標籤 + 標準化參數）
        dataset_output_dir = os.path.join(output_dir, dataset_id)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 儲存標籤
        labels_path = os.path.join(dataset_output_dir, "labels.npy")
        np.save(labels_path, labels)
        
        # 儲存模型（含標準化參數，供未來預測新資料使用）
        model_path = os.path.join(dataset_output_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model,
                "best_k": self.current_k,
                "best_score": self.current_score,
                "scaler_mean": self._scaler_mean,
                "scaler_std": self._scaler_std,
            }, f)
        
        n_clusters = len(np.unique(labels))
        print(f"    [完成] {n_clusters} 個群集，已存至 {dataset_output_dir}")

        # * Step 6: 警告摘要
        warning_total = sum(
            count for key, count in self._warning_counter.items()
            if key.startswith(f"{dataset_id}|")
        )
        if warning_total:
            print(f"    [Warning-Summary] {dataset_id}: 累計 {warning_total} 次訓練失敗/警告")
        else:
            print(f"    [Warning-Summary] {dataset_id}: 無訓練警告")
        
        return labels

    def batch_process_all(
        self,
        vectors_dict: Dict[str, np.ndarray],
        output_dir: str = config.CLUSTER_RESULTS_DIR,
    ) -> Dict[str, np.ndarray]:
        """
        批次處理所有 Dataset。

        Args:
            vectors_dict: dataset_id 對應的概念矩陣字典
            output_dir: 輸出目錄

        Returns:
            dataset_id 對應的標籤字典
        """
        results = {}
        total = len(vectors_dict)
        
        for idx, (dataset_id, concept_matrix) in enumerate(vectors_dict.items(), 1):
            print(f"\n=== [{idx}/{total}] ===")
            try:
                labels = self.process_single_dataset(
                    dataset_id, concept_matrix, output_dir
                )
                results[dataset_id] = labels
            except Exception as e:
                print(f"    [Error] 處理失敗: {e}")
                continue
        # ===== 統計摘要 =====
        print("\n" + "=" * 60)
        print("處理摘要")
        print("=" * 60)
        
        success_count = len(results)
        total_count = len(vectors_dict)
        print(f"成功處理: {success_count}/{total_count} 個資料集")
        
        return results

    def load_model(
        self,
        dataset_id: str,
        input_dir: str = config.CLUSTER_RESULTS_DIR,
    ) -> hmm.GaussianHMM:
        """
        載入特定 Dataset 的已訓練模型。

        Args:
            dataset_id: 資料集 ID
            input_dir: 模型儲存目錄

        Returns:
            該 Dataset 的 GaussianHMM 模型
        """
        model_path = os.path.join(input_dir, dataset_id, "model.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型: {model_path}")
        
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        
        self.current_model = data["model"]
        self.current_k = data["best_k"]
        self.current_score = data["best_score"]
        self._scaler_mean = data.get("scaler_mean")
        self._scaler_std = data.get("scaler_std")
        
        print(f"[Info] 已載入 {dataset_id} 模型: K={self.current_k}, "
              f"Score={self.current_score:.4f}")
        return self.current_model


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
    print("\n載入概念向量...")
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
        
    if not result:
        print("[Error] 找不到概念向量，請先執行 ConceptExtractor")
        exit(1)
    
    print(f"已載入 {len(result)} 個資料集")
    
    return result


# ======================== 主程式 ========================

if __name__ == "__main__":
    print("=" * 60)
    print("序列分群 - Per-Dataset HMM 策略")
    print("=" * 60)
    
    vectors = load_concept_vectors()
    
    
    
    # ===== 批次處理所有資料集 =====
    clusterer = SequenceClustering()
    results = clusterer.batch_process_all(vectors)
    
    if results:
        avg_clusters = np.mean([len(np.unique(labels)) for labels in results.values()])
        print(f"平均群集數: {avg_clusters:.2f}")
    
    print("\n[完成] 序列分群已完成。")
