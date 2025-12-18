"""
SequenceClustering：以隱馬可夫模型 (HMM) 進行事件日誌序列分群。
使用概念矩陣 H_E 作為輸入，辨識同質群體。
"""

import os
import pickle
import numpy as np
from typing import Optional, Tuple, List, Dict
from hmmlearn import hmm

import config


class SequenceClustering:
    """基於 HMM 的事件日誌序列分群。"""

    def __init__(
        self,
        k_min: int = config.HMM_K_MIN,
        k_max: int = config.HMM_K_MAX,
        n_starts: int = config.HMM_N_STARTS,
        n_iter: int = config.HMM_N_ITER,
        tol: float = config.HMM_TOL,
        covariance_type: str = config.HMM_COVARIANCE_TYPE,
        random_state: int = config.SEED,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.n_starts = n_starts
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.best_model: Optional[hmm.GaussianHMM] = None
        self.best_k: Optional[int] = None
        self.best_score: float = float("-inf")

    def _train_single_hmm(
        self, X: np.ndarray, n_components: int, seed: int
    ) -> Tuple[hmm.GaussianHMM, float]:
        """訓練單一 HMM 模型並回傳其對數概似度。"""
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=seed,
        )
        model.fit(X)
        score = model.score(X)
        return model, score

    def optimize_global_hmm(
        self,
        concept_vectors: np.ndarray,
        k_min: Optional[int] = None,
        k_max: Optional[int] = None,
        n_starts: Optional[int] = None,
    ) -> hmm.GaussianHMM:
        """
        透過網格搜索與穩定性迴圈尋找最佳 HMM。

        Args:
            concept_vectors: 聚合後的概念矩陣 H_E，形狀 (n_samples, n_features)
            k_min: 隱藏狀態數量下界
            k_max: 隱藏狀態數量上界
            n_starts: 每個 K 的隨機初始化次數

        Returns:
            最佳的 GaussianHMM 模型
        """
        k_min = k_min or self.k_min
        k_max = k_max or self.k_max
        n_starts = n_starts or self.n_starts

        X = np.asarray(concept_vectors, dtype=np.float64)
        
        # * Grid Search: iterate over K values from k_min to k_max
        for k in range(k_min, k_max + 1):
            # * Stability Loop: run n_starts times per K to mitigate random initialization
            for i in range(n_starts):
                seed = self.random_state + k * n_starts + i
                try:
                    model, score = self._train_single_hmm(X, k, seed)
                    # * Model Selection: keep model with highest Log-Likelihood
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = model
                        self.best_k = k
                except Exception as e:
                    print(f"[Warning] K={k}, iter={i} failed: {e}")
                    continue

        print(f"[Info] Best K={self.best_k}, Log-Likelihood={self.best_score:.4f}")
        return self.best_model

    def decode_sequences(
        self, 
        concept_vectors: np.ndarray, 
        model: Optional[hmm.GaussianHMM] = None
    ) -> np.ndarray:
        """
        使用 Viterbi 演算法預測隱藏狀態序列。

        Args:
            concept_vectors: 單一資料集的概念矩陣，形狀 (n_samples, n_features)
            model: 已訓練的 HMM 模型（若為 None 則使用 best_model）

        Returns:
            分群標籤（隱藏狀態序列）
        """
        model = model or self.best_model
        if model is None:
            raise ValueError("No model available. Run optimize_global_hmm first.")

        X = np.asarray(concept_vectors, dtype=np.float64)
        # * Viterbi Algorithm: predict most likely hidden state path
        labels = model.predict(X)
        return labels

    def save_model(self, path: str = config.HMM_MODEL_PATH) -> None:
        """將最佳 HMM 模型儲存到磁碟。"""
        if self.best_model is None:
            raise ValueError("No model to save. Run optimize_global_hmm first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.best_model,
                "best_k": self.best_k,
                "best_score": self.best_score,
            }, f)
        print(f"[Info] Model saved to {path}")

    def load_model(self, path: str = config.HMM_MODEL_PATH) -> hmm.GaussianHMM:
        """從磁碟載入已訓練的 HMM 模型。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.best_model = data["model"]
        self.best_k = data["best_k"]
        self.best_score = data["best_score"]
        print(f"[Info] Loaded model: K={self.best_k}, Score={self.best_score:.4f}")
        return self.best_model

    def fit_predict(self, concept_vectors: np.ndarray) -> np.ndarray:
        """便捷方法：先優化模型再解碼序列。"""
        self.optimize_global_hmm(concept_vectors)
        return self.decode_sequences(concept_vectors)


def load_concept_vectors(dataset_ids: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    從 ConceptVectors 目錄載入概念向量。

    Args:
        dataset_ids: 要載入的資料集 ID 清單，None 則載入全部。

    Returns:
        dataset_id 對應的概念矩陣字典
    """
    vectors_dir = config.CONCEPT_VECTORS_DIR
    if not os.path.exists(vectors_dir):
        raise FileNotFoundError(f"Concept vectors directory not found: {vectors_dir}")

    result = {}
    files = os.listdir(vectors_dir)
    
    for filename in files:
        if not filename.endswith(".npy"):
            continue
        dataset_id = filename.replace("_concepts.npy", "").replace(".npy", "")
        if dataset_ids is not None and dataset_id not in dataset_ids:
            continue
        
        filepath = os.path.join(vectors_dir, filename)
        result[dataset_id] = np.load(filepath)
    
    return result


def aggregate_vectors(vectors_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """將多個概念矩陣合併為全域訓練資料。"""
    return np.vstack(list(vectors_dict.values()))


def save_cluster_results(
    dataset_id: str, 
    labels: np.ndarray, 
    output_dir: str = config.CLUSTER_RESULTS_DIR
) -> None:
    """將分群標籤儲存至資料集輸出。"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_id}_clusters.npy")
    np.save(output_path, labels)


if __name__ == "__main__":
    # Example usage
    print("Loading concept vectors...")
    vectors = load_concept_vectors()
    
    if not vectors:
        print("[Error] No concept vectors found. Run ConceptExtractor first.")
        exit(1)
    
    print(f"Loaded {len(vectors)} datasets")
    
    # Phase A: Global Model Optimization
    print("\n=== Phase A: Global Model Optimization ===")
    aggregated = aggregate_vectors(vectors)
    print(f"Aggregated shape: {aggregated.shape}")
    
    clusterer = SequenceClustering()
    clusterer.optimize_global_hmm(aggregated)
    clusterer.save_model()
    
    # Phase B: Independent Sequence Labeling
    print("\n=== Phase B: Independent Sequence Labeling ===")
    for dataset_id, concept_matrix in vectors.items():
        labels = clusterer.decode_sequences(concept_matrix)
        save_cluster_results(dataset_id, labels)
        print(f"  {dataset_id}: {len(labels)} entries -> {len(np.unique(labels))} clusters")
    
    print("\n[Done] Sequence clustering completed.")
