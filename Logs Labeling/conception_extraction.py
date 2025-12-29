"""
ConceptExtractor：使用（NMF）或（LDA）將高維度的稠密向量映射至潛在的概念空間。

支援 GPU 加速（透過 PyTorch）與 CPU 回退：
    1. 優先使用 GPU 加速的 NMF（乘法更新規則）
    2. 自動處理 GPU 記憶體不足（Mini-batch 訓練）
    3. 若 GPU 不可用則自動回退至 sklearn CPU 實作
    4. 使用 joblib 對批次資料集轉換進行並行處理
"""

import sys
import shutil
import pickle
from pathlib import Path
from typing import Optional, List, Literal

import numpy as np
from joblib import Parallel, delayed
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
    NMF_MAX_ITER,
    NMF_TOL,
    NMF_INIT,
    CONCEPT_SAMPLE_RATIO,
    NMF_MODEL_PATH,
    CONCEPT_VECTORS_DIR,
    EXTERNAL_KNOWLEDGE_DIR,
    SEED,
    CONCEPT_BATCH_N_JOBS,
    NMF_USE_GPU,
    NMF_GPU_BATCH_SIZE,
    NMF_GPU_EPSILON,
    NMF_GPU_CHECK_INTERVAL,
    NMF_GPU_VERBOSE,
)

# 匯入 GPU NMF 模組
sys.path.insert(0, str(Path(CURRENT_DIR) / "models"))
from models.NMF_gpu import NMFGpu, _check_cuda_available


class ConceptExtractor:
    """
    使用 NMF 或 LDA 從日誌向量中萃取潛在概念。
    
    流程：
        1. prepare_training_data()：聚合並抽樣向量以建立訓練資料
        2. fit_global_model()：訓練 NMF/LDA 以學習概念基矩陣 W
        3. transform_dataset()：將單一資料集投影至概念空間
    
    GPU 加速：
        - NMF 訓練：優先使用 PyTorch GPU 加速（乘法更新規則）
        - 自動處理 OOM：動態調整 batch size 以避免記憶體不足
        - CPU 回退：若 GPU 不可用則自動使用 sklearn 實作
        - 批次轉換：透過 n_jobs 參數設定多資料集並行處理
    """
    
    def __init__(
        self,
        n_concepts: int = NMF_COMPONENTS,
        method: Literal["nmf", "lda"] = "nmf",
        max_iter: int = NMF_MAX_ITER,
        tol: float = NMF_TOL,
        init: str = NMF_INIT,
        random_state: int = SEED,
        model_path: str = NMF_MODEL_PATH,
        n_jobs: int = CONCEPT_BATCH_N_JOBS,
        # GPU 相關參數
        use_gpu: bool = NMF_USE_GPU,
        gpu_batch_size: Optional[int] = NMF_GPU_BATCH_SIZE,
        gpu_epsilon: float = NMF_GPU_EPSILON,
        gpu_check_interval: int = NMF_GPU_CHECK_INTERVAL,
        gpu_verbose: bool = NMF_GPU_VERBOSE,
    ):
        self.n_concepts = n_concepts
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.model_path = model_path
        self.n_jobs = n_jobs
        
        # GPU 設定
        self.use_gpu = use_gpu
        self.gpu_batch_size = gpu_batch_size
        self.gpu_epsilon = gpu_epsilon
        self.gpu_check_interval = gpu_check_interval
        self.gpu_verbose = gpu_verbose
        
        self.model = None
        self.scaler = MinMaxScaler()
        self._is_fitted = False
        self._is_gpu_model = False  # 標記是否使用 GPU 模型

    def _resolve_path(self, path: str) -> str:
        """將相對路徑轉為專案根目錄下的絕對路徑。"""
        return str(Path(path)) if Path(path).is_absolute() else str(Path(PROJECT_ROOT) / path)
    
    # ======================== 資料準備（Data Preparation） ========================
    
    def _validate_and_append_vectors(
        self,
        vectors: np.ndarray,
        ref_dim: Optional[int],
        source_name: str,
        all_vectors: List[np.ndarray],
    ) -> Optional[int]:
        """
        驗證向量維度並加入集合（內部輔助方法）。
        
        回傳：
            更新後的參考維度，若跳過則回傳原本的 ref_dim
        """
        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(-1, 1)
        if vectors.ndim != 2:
            print(f"跳過 {source_name}: 非預期維度 {vectors.ndim}")
            return ref_dim
        
        if ref_dim is None:
            ref_dim = vectors.shape[1]
        elif vectors.shape[1] != ref_dim:
            print(f"跳過 {source_name}: 維度 {vectors.shape[1]} != {ref_dim}")
            return ref_dim
        
        all_vectors.append(vectors)
        return ref_dim
    
    def prepare_training_data(
        self,
        log_vectors_dir: str = LOG_VECTORS_DIR,
        external_knowledge_dir: Optional[str] = EXTERNAL_KNOWLEDGE_DIR,
        sample_ratio: float = CONCEPT_SAMPLE_RATIO,
    ) -> np.ndarray:
        """
        從多個來源聚合並抽樣向量，建立全域訓練資料。
        
        參數：
            log_vectors_dir：包含各 LogID 子資料夾的根目錄
            external_knowledge_dir：外部知識向量的目錄（如 MITRE）
            sample_ratio：每個資料集抽樣比例
            
        回傳：
            X_train：聚合後的訓練矩陣（n_samples, n_features）
        """
        log_vectors_dir = self._resolve_path(log_vectors_dir)
        external_knowledge_dir = (
            self._resolve_path(external_knowledge_dir)
            if external_knowledge_dir else None
        )
        all_vectors = []
        np.random.seed(self.random_state)
        ref_dim: Optional[int] = None
        
        # * 載入外部知識向量（若存在）
        if external_knowledge_dir and exists(external_knowledge_dir):
            for subdir in get_dirs(external_knowledge_dir):
                ext_path = join_path(external_knowledge_dir, subdir)
                vectors = self._load_arrow_data(ext_path)
                if vectors is not None:
                    ref_dim = self._validate_and_append_vectors(
                        vectors, ref_dim, f"外部知識 {subdir}", all_vectors
                    )
                    if all_vectors and all_vectors[-1] is vectors:
                        print(f"載入外部知識: {subdir}, shape={vectors.shape}")
        
        # * 從每個 LogVectors 資料集進行抽樣
        if exists(log_vectors_dir):
            for log_id_dir in get_dirs(log_vectors_dir):
                dataset_path = join_path(log_vectors_dir, log_id_dir)
                vectors = self._load_arrow_data(dataset_path)
                if vectors is None:
                    continue

                old_len = len(all_vectors)
                ref_dim = self._validate_and_append_vectors(
                    vectors, ref_dim, log_id_dir, all_vectors
                )
                
                # 若成功加入，進行抽樣（取代原本完整加入的向量）
                if len(all_vectors) > old_len:
                    full_vectors = all_vectors.pop()
                    n_samples = max(1, int(len(full_vectors) * sample_ratio))
                    indices = np.random.choice(len(full_vectors), n_samples, replace=False)
                    all_vectors.append(full_vectors[indices])
                    print(f"抽樣 {n_samples}/{len(full_vectors)} 自 {log_id_dir}")
        
        if not all_vectors:
            raise ValueError("找不到訓練資料，請檢查輸入目錄。")
        
        # * 將所有向量垂直堆疊
        X_train = np.vstack(all_vectors)
        print(f"訓練資料準備完成: {X_train.shape}")
        return X_train
    
    def _load_arrow_data(self, dir_path: str) -> Optional[np.ndarray]:
        """
        從指定資料夾載入向量，支援 HF datasets 與 Feather 格式。
        """
        # 嘗試以 HF datasets 格式載入
        hf_markers = ["dataset_info.json", "state.json"]
        if any(exists(join_path(dir_path, m)) for m in hf_markers):
            try:
                ds = load_dataset(dir_path)
                vec_col, _ = _infer_vector_dim(ds)
                if vec_col and vec_col in ds.column_names:
                    return np.array(ds[vec_col])
                return ds.to_pandas().values
            except Exception as e:
                print(f"HF 載入失敗 {dir_path}: {e}")
        
        # 嘗試以 Feather 格式讀取 Arrow 檔案
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
        return table.to_pandas().values
    
    # ======================== 模型訓練（Model Training） ========================
    
    def fit_global_model(self, X_train: np.ndarray) -> "ConceptExtractor":
        """
        在聚合後的資料上訓練全域概念模型（NMF 或 LDA）。
        
        對於 NMF：
            - 優先使用 GPU 加速（若 use_gpu=True 且 CUDA 可用）
            - 自動處理 OOM 問題（動態調整 batch size）
            - GPU 不可用時自動回退至 sklearn CPU 實作
        
        對於 LDA：
            - 使用 sklearn 的 LatentDirichletAllocation
        
        參數：
            X_train：訓練矩陣（n_samples, n_features）
            
        回傳：
            self（已訓練完成的萃取器）
        """
        print(f"開始訓練 {self.method.upper()} 模型...")
        
        # * 以 Min-Max 縮放確保非負性
        print("正在進行 Min-Max 縮放...")
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = np.clip(X_scaled, 0, None)  # Extra safety for numerical stability
        print(f"縮放完成，開始擬合模型（n_concepts={self.n_concepts}, max_iter={self.max_iter}）...")
        
        # * 初始化並擬合模型
        if self.method == "nmf":
            # 檢查是否應該使用 GPU
            use_gpu_nmf = self.use_gpu and _check_cuda_available()
            
            if use_gpu_nmf:
                # 使用 GPU 加速的 NMF
                self.model = NMFGpu(
                    n_components=self.n_concepts,
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
                # 使用 sklearn CPU NMF
                if self.use_gpu:
                    print("⚠️ GPU 不可用，改用 sklearn CPU NMF")
                self.model = NMF(
                    n_components=self.n_concepts,
                    init=self.init,
                    solver="cd",
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )
                self._is_gpu_model = False
                
        elif self.method == "lda":
            # LDA 需要擬計數（非負整數或浮點數）
            self.model = LatentDirichletAllocation(
                n_components=self.n_concepts,
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
        print(f"✅ 全域 {self.method.upper()} 模型訓練完成（{device_info}），概念數量: {self.n_concepts}")
        return self
    
    def save_model(self, path: Optional[str] = None) -> None:
        """將訓練好的模型持久化至磁碟。"""
        save_path = self._resolve_path(path or self.model_path)
        ensure_dir(get_parent_dir(save_path))
        
        with open(save_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "n_concepts": self.n_concepts,
                "method": self.method,
                "n_jobs": self.n_jobs,
                "is_gpu_model": self._is_gpu_model,
                "use_gpu": self.use_gpu,
                "gpu_batch_size": self.gpu_batch_size,
            }, f)
        print(f"模型已儲存至 {save_path}")
    
    def load_model(self, path: Optional[str] = None) -> "ConceptExtractor":
        """從磁碟載入已訓練模型。"""
        load_path = self._resolve_path(path or self.model_path)
        
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.n_concepts = data["n_concepts"]
        self.method = data["method"]
        self.n_jobs = data.get("n_jobs", -1)  # 向後相容舊版模型
        self._is_gpu_model = data.get("is_gpu_model", False)
        self.use_gpu = data.get("use_gpu", self.use_gpu)
        self.gpu_batch_size = data.get("gpu_batch_size", self.gpu_batch_size)
        self._is_fitted = True
        
        device_info = "GPU" if self._is_gpu_model else "CPU"
        print(f"模型已從 {load_path} 載入（{device_info} 模型）")
        return self
    
    # ======================== 資料轉換（Transformation） ========================
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用凍結的基矩陣 W 將向量投影至概念空間。
        
        參數：
            X：輸入矩陣（n_samples, n_features）
            
        回傳：
            H：概念權重矩陣（n_samples, n_concepts）
        """
        if not self._is_fitted:
            raise RuntimeError("尚未訓練模型。請先呼叫 fit_global_model()。")
        
        # * 套用與訓練相同的縮放方式
        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, 0, None)
        
        return self.model.transform(X_scaled)
    
    def transform_dataset(
        self,
        input_path: str,
        output_path: str,
        copy_metadata: bool = True,
    ) -> None:
        """
        轉換單一資料集並儲存到結構化的輸出目錄。
        
        參數：
            input_path：輸入 LogVectors 資料夾路徑
            output_path：輸出 ConceptVectors 資料夾路徑
            copy_metadata：是否複製 state.json 與 dataset_info.json
        """
        if not self._is_fitted:
            raise RuntimeError("尚未訓練模型。請先載入或訓練模型。")
        
        input_path = self._resolve_path(input_path)
        output_path = self._resolve_path(output_path)

        # * 載入輸入向量
        X = self._load_arrow_data(input_path)
        if X is None:
            print(f"在 {input_path} 找不到資料，已跳過。")
            return
        
        # * 轉換至概念空間
        H = self.transform(X)
        
        # * 儲存轉換後資料
        ensure_dir(output_path)
        output_arrow = join_path(output_path, "data-00000-of-00001.arrow")
        
        table = pa.table({"concept_vector": H.tolist()})
        feather.write_feather(table, output_arrow)
        print(f"轉換完成 {input_path} -> {output_path}, shape={H.shape}")
        
        # * 複製中繼資料（metadata）檔案
        if copy_metadata:
            for meta_file in ["state.json", "dataset_info.json"]:
                src, dst = join_path(input_path, meta_file), join_path(output_path, meta_file)
                if exists(src):
                    shutil.copy2(src, dst)
    
    def _transform_single_dataset(
        self,
        log_id_dir: str,
        log_vectors_dir: str,
        concept_vectors_dir: str,
    ) -> Optional[str]:
        """
        轉換單一資料集（內部方法，供並行處理使用）。
        
        回傳：
            成功時回傳輸出路徑，失敗時回傳 None
        """
        input_path = join_path(log_vectors_dir, log_id_dir)
        output_name = log_id_dir.replace("_logvectors", "_concepts")
        output_path = join_path(concept_vectors_dir, output_name)
        
        try:
            self.transform_dataset(input_path, output_path)
            return output_path
        except Exception as e:
            print(f"轉換 {log_id_dir} 時發生錯誤: {e}")
            return None
    
    def batch_transform(
        self,
        log_vectors_dir: str = LOG_VECTORS_DIR,
        concept_vectors_dir: str = CONCEPT_VECTORS_DIR,
        n_jobs: Optional[int] = None,
    ) -> None:
        """
        將 LogVectors 目錄中的所有資料集轉換為 ConceptVectors。
        
        支援多 CPU 並行處理以加速批次轉換。
        
        參數：
            log_vectors_dir：輸入 LogVectors 的根目錄
            concept_vectors_dir：輸出 ConceptVectors 的根目錄
            n_jobs：並行工作數（None 時使用實例預設值，-1 使用所有 CPU）
        """
        log_vectors_dir = self._resolve_path(log_vectors_dir)
        concept_vectors_dir = self._resolve_path(concept_vectors_dir)
        n_jobs = n_jobs if n_jobs is not None else self.n_jobs

        if not exists(log_vectors_dir):
            raise FileNotFoundError(f"Input directory not found: {log_vectors_dir}")
        
        ensure_dir(concept_vectors_dir)
        
        log_id_dirs = list(get_dirs(log_vectors_dir))
        total = len(log_id_dirs)
        print(f"開始並行批次轉換：共 {total} 個資料集，使用 n_jobs={n_jobs}")
        
        # * 使用 joblib 進行並行處理
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self._transform_single_dataset)(
                log_id_dir, log_vectors_dir, concept_vectors_dir
            )
            for log_id_dir in log_id_dirs
        )
        
        # * 統計成功與失敗數量
        success_count = sum(1 for r in results if r is not None)
        print(f"批次轉換完成：成功 {success_count}/{total}，輸出目錄：{concept_vectors_dir}")
    
    # ======================== 分析工具（Analysis Utilities） ========================
    
    def get_concept_basis(self) -> np.ndarray:
        """
        回傳已學得的概念基矩陣 W。
        
        回傳：
            W：NMF 的基矩陣（n_concepts, n_features），
               或 LDA 的主題-詞彙分佈
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.components_
    
    def get_reconstruction_error(self, X: np.ndarray) -> float:
        """計算給定資料的重建誤差。"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, 0, None)
        H = self.model.transform(X_scaled)
        W = self.model.components_
        X_reconstructed = H @ W
        
        return np.mean((X_scaled - X_reconstructed) ** 2)


# ======================== 便捷函式（Convenience Functions） ========================

def train_concept_extractor(
    log_vectors_dir: str = LOG_VECTORS_DIR,
    external_knowledge_dir: Optional[str] = EXTERNAL_KNOWLEDGE_DIR,
    n_concepts: int = NMF_COMPONENTS,
    sample_ratio: float = CONCEPT_SAMPLE_RATIO,
    model_path: str = NMF_MODEL_PATH,
    method: Literal["nmf", "lda"] = "nmf",
    n_jobs: int = -1,
) -> ConceptExtractor:
    """
    端到端訓練流程：準備資料、訓練模型並儲存。
    
    參數：
        n_jobs：批次轉換時的並行工作數（-1 表示使用所有 CPU）
    """
    extractor = ConceptExtractor(
        n_concepts=n_concepts, 
        method=method, 
        model_path=model_path,
        n_jobs=n_jobs,
    )
    
    X_train = extractor.prepare_training_data(
        log_vectors_dir=log_vectors_dir,
        external_knowledge_dir=external_knowledge_dir,
        sample_ratio=sample_ratio,
    )
    
    extractor.fit_global_model(X_train)
    extractor.save_model()
    
    return extractor


def transform_all_datasets(
    model_path: str = NMF_MODEL_PATH,
    log_vectors_dir: str = LOG_VECTORS_DIR,
    concept_vectors_dir: str = CONCEPT_VECTORS_DIR,
    n_jobs: int = -1,
) -> None:
    """
    載入已訓練模型，並將所有 LogVectors 轉換為 ConceptVectors。
    
    參數：
        n_jobs：並行工作數（-1 表示使用所有 CPU）
    """
    extractor = ConceptExtractor(n_jobs=n_jobs)
    extractor.load_model(model_path)
    extractor.batch_transform(log_vectors_dir, concept_vectors_dir, n_jobs=n_jobs)


if __name__ == "__main__":
    # 範例用法
    extractor = train_concept_extractor()
    transform_all_datasets()
