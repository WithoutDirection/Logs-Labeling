"""
ConceptExtractor：使用 NMF/LDA 將日誌向量映射至潛在概念空間

# * Per-Dataset NMF 策略：每個 Dataset 獨立訓練專屬模型
# * 結合 External Knowledge 作為對比基準（語義錨點）
# * GPU 加速的 NMF（乘法更新規則 + L1 稀疏性）
# * 自動處理 OOM（動態調整 batch size）
# * GPU 不可用時自動回退至 sklearn CPU
"""

import sys
import shutil
import pickle
from pathlib import Path
from typing import Optional, List, Literal, Dict

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler

# * 調整匯入路徑
CURRENT_DIR = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(CURRENT_DIR).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.path import (
    join_path, get_parent_dir, ensure_dir, exists, get_dirs
)
from utils.dataset import load_dataset, _infer_vector_dim
from config import (
    LOG_VECTORS_DIR,
    NMF_COMPONENTS,
    NMF_L1_REG,
    NMF_MAX_ITER,
    NMF_TOL,
    NMF_INIT,
    NMF_MODEL_PATH,
    CONCEPT_VECTORS_DIR,
    EXTERNAL_KNOWLEDGE_DIR,
    SEED,
    NMF_USE_GPU,
    NMF_GPU_BATCH_SIZE,
    NMF_GPU_EPSILON,
    NMF_GPU_CHECK_INTERVAL,
    NMF_GPU_VERBOSE,
)

# * 匯入 GPU NMF 模組
sys.path.insert(0, str(Path(CURRENT_DIR) / "models"))
from models.NMF_gpu import NMFGpu, _check_cuda_available


class ConceptExtractor:
    """
    使用 NMF/LDA 從日誌向量中萃取潛在概念（Per-Dataset 策略）
    
    # * Per-Dataset 策略：每個 Dataset 獨立訓練專屬 NMF 模型
    # * External Knowledge 作為語義錨點，凸顯 Dataset 的特異性
    # * 流程：fit_local_model() → transform_local() → save_local_model()
    # * GPU 加速：NMF 優先使用 PyTorch GPU
    """
    
    def __init__(
        self,
        n_concepts: int = NMF_COMPONENTS,
        l1_reg: float = NMF_L1_REG,
        method: Literal["nmf", "lda"] = "nmf",
        max_iter: int = NMF_MAX_ITER,
        tol: float = NMF_TOL,
        init: str = NMF_INIT,
        random_state: int = SEED,
        use_gpu: bool = NMF_USE_GPU,
        gpu_batch_size: Optional[int] = NMF_GPU_BATCH_SIZE,
        gpu_epsilon: float = NMF_GPU_EPSILON,
        gpu_check_interval: int = NMF_GPU_CHECK_INTERVAL,
        gpu_verbose: bool = NMF_GPU_VERBOSE,
    ):
        self.n_concepts = n_concepts
        self.l1_reg = l1_reg
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        
        # * GPU 設定
        self.use_gpu = use_gpu
        self.gpu_batch_size = gpu_batch_size
        self.gpu_epsilon = gpu_epsilon
        self.gpu_check_interval = gpu_check_interval
        self.gpu_verbose = gpu_verbose
        
        # * Per-Dataset 模型儲存
        self.model = None
        self.scaler = MinMaxScaler()
        self._is_fitted = False
        self._is_gpu_model = False
        self._dataset_sample_count = 0  # 記錄 Dataset 的樣本數量
        self._effective_n_concepts = n_concepts  # 實際使用的概念數
        
        # * 外部知識快取
        self._external_vectors: Optional[np.ndarray] = None

    def _resolve_path(self, path: str) -> str:
        """轉相對路徑為專案根目錄下的絕對路徑"""
        return str(Path(path)) if Path(path).is_absolute() else str(Path(PROJECT_ROOT) / path)
    
    # ======================== 資料載入 ========================
    
    def _load_arrow_data(self, dir_path: str) -> Optional[np.ndarray]:
        """
        從指定資料夾載入向量，支援 HF datasets 與 Feather 格式。
        """
        # * 嘗試以 HF datasets 格式載入
        hf_markers = ["dataset_info.json", "state.json"]
        if any(exists(join_path(dir_path, m)) for m in hf_markers):
            try:
                ds = load_dataset(dir_path)
                vec_col, _ = _infer_vector_dim(ds)
                if vec_col and vec_col in ds.column_names:
                    return np.array(ds[vec_col])
                return ds.to_pandas().values
            except Exception as e:
                print(f"    [Warning] HF 載入失敗 {dir_path}: {e}")
        
        # * 嘗試以 Feather 格式讀取 Arrow 檔案
        for fname in ["data.arrow", "data-00000-of-00001.arrow"]:
            fpath = join_path(dir_path, fname)
            if exists(fpath):
                try:
                    table = feather.read_table(fpath)
                    return self._table_to_numpy(table)
                except Exception:
                    pass
        
        return None

    def _table_to_numpy(self, table: pa.Table) -> np.ndarray:
        """將 Arrow Table 轉為 numpy，自動偵測向量欄位。"""
        for col in ["embedding", "vector", "log_vector", "concept_vector"]:
            if col in table.column_names:
                return np.array(table[col].to_pylist())
        if "template_embedding" in table.column_names:
            return np.array(table["template_embedding"].to_pylist())
        if "param_embedding" in table.column_names:
            return np.array(table["param_embedding"].to_pylist())
        return table.to_pandas().values
    
    def load_external_knowledge(
        self,
        external_knowledge_dir: str = EXTERNAL_KNOWLEDGE_DIR,
    ) -> Optional[np.ndarray]:
        """
        載入所有外部知識向量（MITRE、CAPEC 等）作為語義錨點。
        
        這些向量會與每個 Dataset 聯合訓練 NMF，用於：
        1. 提供跨 Technique 的語義基準
        2. 幫助 NMF 學習「Dataset 與已知攻擊模式的關聯」
        
        Returns:
            外部知識向量矩陣，若無則回傳 None
        """
        external_knowledge_dir = self._resolve_path(external_knowledge_dir)
        
        if not exists(external_knowledge_dir):
            print(f"    [Warning] 找不到外部知識目錄: {external_knowledge_dir}")
            return None
        
        all_vectors = []
        ref_dim: Optional[int] = None
        
        for subdir in get_dirs(external_knowledge_dir):
            ext_path = join_path(external_knowledge_dir, subdir)
            vectors = self._load_arrow_data(ext_path)
            
            if vectors is None:
                continue
                
            vectors = np.asarray(vectors)
            if vectors.ndim == 1:
                vectors = vectors.reshape(-1, 1)
            if vectors.ndim != 2:
                continue
            
            # 驗證維度一致性
            if ref_dim is None:
                ref_dim = vectors.shape[1]
            elif vectors.shape[1] != ref_dim:
                print(f"    [Warning] 維度不符 {subdir}: {vectors.shape[1]} != {ref_dim}")
                continue
            
            all_vectors.append(vectors)
            print(f"    載入外部知識: {subdir}, shape={vectors.shape}")
        
        if not all_vectors:
            return None
        
        self._external_vectors = np.vstack(all_vectors)

        # TF-IDF weighting disabled to preserve embedding geometry
        
        print(f"    外部知識總計: {self._external_vectors.shape}")
        return self._external_vectors

    def _apply_external_tfidf_weighting(self, vectors: np.ndarray, base_dir: str) -> np.ndarray:
        """Apply TF-IDF weighting to external knowledge embeddings."""
        try:
            import scipy.sparse
            # Try to find MITRE_TFIDF matrix
            candidates = [
                join_path(base_dir, "MITRE_TFIDF", "mitre_tfidf_matrix.pkl"),
                join_path(str(Path(base_dir).parent), "MITRE_TFIDF", "mitre_tfidf_matrix.pkl"),
                join_path(str(Path(base_dir).parent), "data", "ExternalKnowledge", "MITRE_TFIDF", "mitre_tfidf_matrix.pkl")
            ]
            
            tfidf_path = None
            for p in candidates:
                if exists(p):
                    tfidf_path = p
                    break
            
            if tfidf_path:
                with open(tfidf_path, "rb") as f:
                    tfidf_matrix = pickle.load(f)
                
                if tfidf_matrix.shape[0] != vectors.shape[0]:
                    print(f"    [Info] 外部 TF-IDF shape {tfidf_matrix.shape} != Embedding shape {vectors.shape} (可能包含非 MITRE 來源)，略過加權")
                    return vectors
                
                if scipy.sparse.issparse(tfidf_matrix):
                    norms = scipy.sparse.linalg.norm(tfidf_matrix, axis=1)
                else:
                    norms = np.linalg.norm(tfidf_matrix, axis=1)
                
                if norms.max() > 0:
                    norms = norms / norms.max()
                
                # Use same weighting factor as logs: 1.0 + 0.5 * norm
                scaling_factors = 1.0 + (0.5 * norms)
                weighted_vectors = vectors * scaling_factors[:, np.newaxis]
                
                print(f"    [Ext-TF-IDF] 已應用外部知識 TF-IDF 加權。Avg factor: {scaling_factors.mean():.4f}")
                return weighted_vectors
                
        except Exception as e:
            print(f"    [Warning] 外部 TF-IDF 加權失敗: {e}")
            
        return vectors
    
    def load_dataset_vectors(self, dataset_path: str) -> np.ndarray:
        """
        載入單一 Dataset 的 Log Vectors。
        
        Args:
            dataset_path: Dataset 資料夾路徑
            
        Returns:
            Log Vectors 矩陣 (n_samples, n_features)
        """
        dataset_path = self._resolve_path(dataset_path)
        vectors = self._load_arrow_data(dataset_path)
        
        if vectors is None:
            raise FileNotFoundError(f"找不到資料: {dataset_path}")
        
        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(-1, 1)
        
        return vectors
    
    # ======================== Per-Dataset 模型訓練 ========================
    
    def fit_local_model(
        self,
        dataset_vectors: np.ndarray,
        external_vectors: Optional[np.ndarray] = None,
        dataset_id: str = "unknown",
    ) -> "ConceptExtractor":
        """
        針對單一 Dataset 訓練局部 NMF 模型（Per-Dataset 策略）。
        
        策略：
        1. 將 Dataset + External Knowledge 聯合訓練 NMF
        2. External Knowledge 作為語義錨點，凸顯 Dataset 的特異性
        3. 訓練後僅對 Dataset 部分進行轉換與後續 HMM
        
        這樣萃取出的概念會反映：
        - 該 Dataset 與已知攻擊模式的相似性
        - 該 Dataset 的獨特行為模式
        
        Args:
            dataset_vectors: 單一 Dataset 的 Log Vectors (n_samples, n_features)
            external_vectors: 外部知識向量（若為 None 則使用快取的 _external_vectors）
            dataset_id: Dataset 識別碼（用於日誌輸出）
            
        Returns:
            self（已訓練完成的萃取器）
        """
        print(f"\n    [Per-Dataset NMF] 訓練 {dataset_id}...")
        
        # * 使用外部向量（優先傳入參數，其次使用快取）
        ext_vectors = external_vectors if external_vectors is not None else self._external_vectors
        
        # * 聯合訓練資料（Dataset + External）
        dataset_vectors = np.asarray(dataset_vectors, dtype=np.float64)
        self._dataset_sample_count = len(dataset_vectors)
        
        if ext_vectors is not None:
            # 驗證維度
            if dataset_vectors.shape[1] != ext_vectors.shape[1]:
                raise ValueError(
                    f"維度不符: Dataset={dataset_vectors.shape[1]}, "
                    f"External={ext_vectors.shape[1]}"
                )
            X_train = np.vstack([dataset_vectors, ext_vectors])
            print(f"    Dataset: {self._dataset_sample_count} 筆, External: {len(ext_vectors)} 筆")
        else:
            X_train = dataset_vectors
            print(f"    Dataset: {self._dataset_sample_count} 筆 (無外部知識)")
        
        # * 動態調整概念數量：避免概念數超過樣本數
        n_samples = len(X_train)
        effective_n_concepts = min(self.n_concepts, n_samples - 1, dataset_vectors.shape[1])
        if effective_n_concepts < self.n_concepts:
            print(f"    [Warning] 動態調整概念數: {self.n_concepts} → {effective_n_concepts}")
        self._effective_n_concepts = effective_n_concepts
        
        # * Min-Max 縮放確保非負性
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = np.clip(X_scaled, 0, None)
        
        # * 初始化並擬合模型
        if self.method == "nmf":
            use_gpu_nmf = self.use_gpu and _check_cuda_available()
            
            if use_gpu_nmf:
                self.model = NMFGpu(
                    n_components=effective_n_concepts,
                    l1_reg=self.l1_reg,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    epsilon=self.gpu_epsilon,
                    random_state=self.random_state,
                    batch_size=self.gpu_batch_size,
                    check_interval=self.gpu_check_interval,
                    verbose=self.gpu_verbose,
                )
                self._is_gpu_model = True
            else:
                if self.use_gpu:
                    print("    [Warning] GPU 不可用，改用 sklearn CPU NMF")
                self.model = NMF(
                    n_components=effective_n_concepts,
                    init=self.init,
                    solver="cd",
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )
                self._is_gpu_model = False
                
        elif self.method == "lda":
            self.model = LatentDirichletAllocation(
                n_components=effective_n_concepts,
                max_iter=self.max_iter,
                random_state=self.random_state,
                learning_method="batch",
            )
            self._is_gpu_model = False
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'nmf' or 'lda'.")
        
        self.model.fit(X_scaled)
        self._is_fitted = True
        
        device_info = "GPU" if self._is_gpu_model else "CPU"
        print(f"    [完成] {self.method.upper()} 訓練完成（{device_info}），概念數: {effective_n_concepts}")
        return self
    
    # ======================== 資料轉換 ========================
    
    def transform_local(self, X: np.ndarray) -> np.ndarray:
        """
        使用局部模型將向量投影至概念空間。
        
        Args:
            X: 輸入向量矩陣 (n_samples, n_features)
            
        Returns:
            概念向量矩陣 (n_samples, n_concepts)
        """
        if not self._is_fitted:
            raise RuntimeError("尚未訓練模型。請先呼叫 fit_local_model()。")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, 0, None)
        
        return self.model.transform(X_scaled)
    
    def transform_dataset_only(self, dataset_vectors: np.ndarray) -> np.ndarray:
        """
        僅轉換 Dataset 部分（排除 External Knowledge）。
        
        這是 Per-Dataset 策略的核心：
        - 訓練時使用 Dataset + External Knowledge
        - 轉換時僅對 Dataset 進行投影
        
        Args:
            dataset_vectors: 原始 Dataset 向量（與訓練時相同）
            
        Returns:
            Dataset 的概念向量矩陣
        """
        return self.transform_local(dataset_vectors)
    
    # ======================== 模型存取 ========================
    
    def save_local_model(
        self,
        output_dir: str,
        dataset_id: str,
    ) -> str:
        """
        儲存 Per-Dataset 模型至指定目錄。
        
        Args:
            output_dir: 輸出根目錄
            dataset_id: Dataset 識別碼
            
        Returns:
            模型檔案路徑
        """
        output_dir = self._resolve_path(output_dir)
        dataset_output_dir = join_path(output_dir, f"{dataset_id}_concepts")
        ensure_dir(dataset_output_dir)
        
        model_path = join_path(dataset_output_dir, "nmf_model.pkl")
        
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "n_concepts": self._effective_n_concepts,
                "method": self.method,
                "is_gpu_model": self._is_gpu_model,
                "dataset_sample_count": self._dataset_sample_count,
            }, f)
        
        print(f"    模型已儲存至 {model_path}")
        return model_path
    
    def load_local_model(
        self,
        model_path: str,
    ) -> "ConceptExtractor":
        """
        載入 Per-Dataset 模型。
        
        Args:
            model_path: 模型檔案路徑
            
        Returns:
            self
        """
        model_path = self._resolve_path(model_path)
        
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._effective_n_concepts = data["n_concepts"]
        self.method = data["method"]
        self._is_gpu_model = data.get("is_gpu_model", False)
        self._dataset_sample_count = data.get("dataset_sample_count", 0)
        self._is_fitted = True
        
        device_info = "GPU" if self._is_gpu_model else "CPU"
        print(f"    模型已從 {model_path} 載入（{device_info}）")
        return self
    
    # ======================== 單一 Dataset 完整流程 ========================
    
    def process_single_dataset(
        self,
        dataset_id: str,
        input_path: str,
        output_dir: str = CONCEPT_VECTORS_DIR,
        external_knowledge_dir: str = EXTERNAL_KNOWLEDGE_DIR,
        copy_metadata: bool = True,
        use_tfidf_weighting: bool = False,
    ) -> np.ndarray:
        """
        處理單一 Dataset 的完整流程：載入 → (選用 TF-IDF 加權) → 訓練 → 轉換 → 存檔。
        
        這是 Per-Dataset 策略的主入口點。
        
        Args:
            dataset_id: Dataset 識別碼
            input_path: 輸入 LogVectors 資料夾路徑
            output_dir: 輸出 ConceptVectors 根目錄
            external_knowledge_dir: 外部知識目錄
            copy_metadata: 是否複製 metadata 檔案
            use_tfidf_weighting: 是否使用 TF-IDF 調整嵌入權重
            
        Returns:
            Dataset 的概念向量矩陣
        """
        input_path = self._resolve_path(input_path)
        output_dir = self._resolve_path(output_dir)
        
        print(f"\n[Processing] {dataset_id}")
        
        # * Step 1: 載入外部知識（若尚未載入）
        if self._external_vectors is None:
            self.load_external_knowledge(external_knowledge_dir)
        
        # * Step 2: 載入 Dataset 向量
        dataset_vectors = self.load_dataset_vectors(input_path)
        print(f"    載入 Dataset: {dataset_vectors.shape}")
        
        # * [TF-IDF Integration] 權重調整
        if use_tfidf_weighting:
            dataset_vectors = self._apply_tfidf_weighting(dataset_id, dataset_vectors, input_path)

        # * Step 3: Per-Dataset NMF 訓練
        self.fit_local_model(
            dataset_vectors=dataset_vectors,
            dataset_id=dataset_id,
        )
        
        # * Step 4: 轉換至概念空間（僅 Dataset 部分）
        concept_vectors = self.transform_dataset_only(dataset_vectors)
        
        # * Step 5: 儲存結果
        output_name = dataset_id.replace("_logvectors", "").replace("_embeddings", "")
        dataset_output_dir = join_path(output_dir, f"{output_name}_concepts")
        ensure_dir(dataset_output_dir)
        
        # 儲存概念向量
        output_arrow = join_path(dataset_output_dir, "data-00000-of-00001.arrow")
        table = pa.table({"concept_vector": concept_vectors.tolist()})
        feather.write_feather(table, output_arrow)
        
        # 儲存模型
        self.save_local_model(output_dir, output_name)
        
        # 複製 metadata
        if copy_metadata:
            for meta_file in ["state.json", "dataset_info.json"]:
                src = join_path(input_path, meta_file)
                dst = join_path(dataset_output_dir, meta_file)
                if exists(src):
                    shutil.copy2(src, dst)
        
        print(f"    [完成] 概念向量已存至 {dataset_output_dir}, shape={concept_vectors.shape}")
        
        return concept_vectors
    
    def _apply_tfidf_weighting(self, dataset_id: str, vectors: np.ndarray, input_path: str) -> np.ndarray:
        """
        利用 TF-IDF 調整嵌入向量權重。
        
        策略:
        1. 載入該 Dataset 的預計算 TF-IDF 向量 (sparse matrix)
        2. 計算每筆資料的 TF-IDF 強度 (L2 norm 或 Max weight)
        3. 用強度縮放原始嵌入向量
        """
        try:
            import scipy.sparse
            # 尋找 tfidf.npz
            tfidf_path = join_path(input_path, "tfidf.npz")
            
            # Debug: Print checking paths
            print(f"    [Debug] Checking main path: {tfidf_path}")
            
            if not exists(tfidf_path):
                # 嘗試去除/增加後綴
                parent = str(Path(input_path).parent)
                candidates = [
                    join_path(input_path, "tfidf.npz"),
                    join_path(parent, f"{dataset_id}_embeddings", "tfidf.npz"),
                    join_path(parent, f"{dataset_id}_raw_events_embeddings", "tfidf.npz")
                ]
                print(f"    [Debug] Checking candidates: {candidates}")
                for c in candidates:
                    if exists(c):
                        tfidf_path = c
                        break
            
            if exists(tfidf_path):
                tfidf_matrix = scipy.sparse.load_npz(tfidf_path)
                if tfidf_matrix.shape[0] != vectors.shape[0]:
                    print(f"    [Warning] TF-IDF shape {tfidf_matrix.shape} != Embedding shape {vectors.shape}，略過加權")
                    return vectors
                    
                # 計算權重因子 (Normalized Sum of Top Keywords)
                # 假設: 有顯著關鍵字的 log 應該更重要
                # axis=1 sum gives rough importance
                # 這裡使用 1 + sigmoid(norm) 稍微放大重要 log 的向量
                
                # Simple approach: L2 Norm of TFIDF vector
                if scipy.sparse.issparse(tfidf_matrix):
                    norms = scipy.sparse.linalg.norm(tfidf_matrix, axis=1)
                else:
                    norms = np.linalg.norm(tfidf_matrix, axis=1)
                
                # Normalize norms to 0-1 range for safe scaling
                if norms.max() > 0:
                    norms = norms / norms.max()
                
                # Scaling factor: Base 1.0 + (0.0 ~ 0.5 boost based on TFIDF strength)
                # 這會讓包含強特徵關鍵字的 log 在 NMF 中影響力變大
                scaling_factors = 1.0 + (0.5 * norms)
                
                # Apply scaling row-wise
                weighted_vectors = vectors * scaling_factors[:, np.newaxis]
                
                print(f"    [TF-IDF] 已應用 TF-IDF 加權。Avg factor: {scaling_factors.mean():.4f}")
                return weighted_vectors
            else:
                 print(f"    [TF-IDF] 找不到 tfidf.npz，略過加權")
        except Exception as e:
            print(f"    [Warning] TF-IDF 加權失敗: {e}")
            
        return vectors

    # ======================== 分析工具 ========================
    
    def get_concept_basis(self) -> np.ndarray:
        """
        回傳已學得的概念基矩陣 W。
        
        Returns:
            W：NMF 的基矩陣 (n_concepts, n_features)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.components_
    
    def extract_representative_samples(
        self,
        concept_vectors: np.ndarray,
        top_n: int = 10,
    ) -> Dict[int, List[int]]:
        """
        提取每個概念的代表性樣本索引（Top-N 權重）。
        
        用於人工標註概念語義或自動生成概念標籤。
        
        Args:
            concept_vectors: 概念向量矩陣 (n_samples, n_concepts)
            top_n: 每個概念提取的代表性樣本數量
            
        Returns:
            {concept_idx: [sample_indices]}
        """
        representative_indices = {}
        n_concepts = concept_vectors.shape[1]
        
        for concept_idx in range(n_concepts):
            concept_weights = concept_vectors[:, concept_idx]
            top_indices = np.argsort(concept_weights)[::-1][:top_n]
            representative_indices[concept_idx] = top_indices.tolist()
        
        return representative_indices


# ======================== 便捷函式 ========================

def process_all_datasets(
    log_vectors_dir: str = LOG_VECTORS_DIR,
    concept_vectors_dir: str = CONCEPT_VECTORS_DIR,
    external_knowledge_dir: str = EXTERNAL_KNOWLEDGE_DIR,
    n_concepts: int = NMF_COMPONENTS,
    dataset_ids: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    批次處理所有 Dataset（Per-Dataset NMF 策略）。
    
    每個 Dataset 獨立訓練專屬 NMF 模型。
    
    Args:
        log_vectors_dir: LogVectors 根目錄
        concept_vectors_dir: ConceptVectors 輸出根目錄
        external_knowledge_dir: 外部知識目錄
        n_concepts: 概念數量
        dataset_ids: 指定處理的 Dataset ID（None 則處理全部）
        
    Returns:
        {dataset_id: concept_vectors}
    """
    log_vectors_dir = str(Path(log_vectors_dir)) if Path(log_vectors_dir).is_absolute() else str(Path(PROJECT_ROOT) / log_vectors_dir)
    
    if not exists(log_vectors_dir):
        raise FileNotFoundError(f"找不到 LogVectors 目錄: {log_vectors_dir}")
    
    # 取得所有 Dataset 目錄
    all_dirs = list(get_dirs(log_vectors_dir))
    if dataset_ids is not None:
        all_dirs = [d for d in all_dirs if any(did in d for did in dataset_ids)]
    
    total = len(all_dirs)
    print(f"\n{'=' * 60}")
    print(f"Per-Dataset 概念提取 - 共 {total} 個資料集")
    print(f"{'=' * 60}")
    
    results = {}
    
    # 建立共用的 extractor（外部知識只載入一次）
    extractor = ConceptExtractor(n_concepts=n_concepts)
    extractor.load_external_knowledge(external_knowledge_dir)
    
    for idx, log_id_dir in enumerate(all_dirs, 1):
        print(f"\n=== [{idx}/{total}] ===")
        
        dataset_id = log_id_dir.replace("_logvectors", "").replace("_embeddings", "")
        input_path = join_path(log_vectors_dir, log_id_dir)
        
        try:
            # 每個 Dataset 重新初始化模型（但共用外部知識快取）
            extractor.model = None
            extractor._is_fitted = False
            
            concept_vectors = extractor.process_single_dataset(
                dataset_id=dataset_id,
                input_path=input_path,
                output_dir=concept_vectors_dir,
                external_knowledge_dir=external_knowledge_dir,
            )
            results[dataset_id] = concept_vectors
            
        except Exception as e:
            print(f"    [Error] 處理 {dataset_id} 失敗: {e}")
            continue
    
    # 統計摘要
    print(f"\n{'=' * 60}")
    print(f"處理摘要")
    print(f"{'=' * 60}")
    print(f"成功處理: {len(results)}/{total} 個資料集")
    
    return results


if __name__ == "__main__":
    # 範例用法：Per-Dataset 概念提取
    results = process_all_datasets()
    
    if results:
        avg_concepts = np.mean([v.shape[1] for v in results.values()])
        print(f"\n平均概念數: {avg_concepts:.2f}")
    
    print("\n[完成] Per-Dataset 概念提取已完成。")
