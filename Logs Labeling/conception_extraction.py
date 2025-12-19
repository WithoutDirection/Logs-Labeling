"""
ConceptExtractor：使用非負矩陣分解（NMF）或隱含狄利克雷分佈（LDA）
將高維度的稠密向量映射至潛在的概念空間。
"""

import sys
import shutil
import pickle
from pathlib import Path
from typing import Optional, List, Literal, Union

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
    join_path, get_parent_dir, ensure_dir, exists, is_dir, get_dirs
)
from utils.dataset import load_dataset
from config import (
    LOG_VECTORS_DIR,
    DATA_DIR,
    NMF_COMPONENTS,
    NMF_MAX_ITER,
    NMF_TOL,
    NMF_INIT,
    CONCEPT_SAMPLE_RATIO,
    NMF_MODEL_PATH,
    CONCEPT_VECTORS_DIR,
    EXTERNAL_KNOWLEDGE_DIR,
    SEED,
)


class ConceptExtractor:
    """
    使用 NMF 或 LDA 從日誌向量中萃取潛在概念。
    
    流程：
        1. prepare_training_data()：聚合並抽樣向量以建立訓練資料
        2. fit_global_model()：訓練 NMF/LDA 以學習概念基矩陣 W
        3. transform_dataset()：將單一資料集投影至概念空間
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
    ):
        self.n_concepts = n_concepts
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.model_path = model_path
        
        self.model = None
        self.scaler = MinMaxScaler()
        self._is_fitted = False

    def _resolve_path(self, path: str) -> str:
        """將相對路徑轉為專案根目錄下的絕對路徑。"""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(Path(PROJECT_ROOT) / p)
    
    # ======================== 資料準備（Data Preparation） ========================
    
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
        
        # * 載入外部知識向量（若存在）
        if external_knowledge_dir and exists(external_knowledge_dir):
            for subdir in get_dirs(external_knowledge_dir):
                ext_path = join_path(external_knowledge_dir, subdir)
                vectors = self._load_arrow_data(ext_path)
                if vectors is not None:
                    all_vectors.append(vectors)
                    print(f"Loaded external knowledge: {subdir}, shape={vectors.shape}")
        
        # * 從每個 LogVectors 資料集進行抽樣
        if exists(log_vectors_dir):
            for log_id_dir in get_dirs(log_vectors_dir):
                dataset_path = join_path(log_vectors_dir, log_id_dir)
                    
                vectors = self._load_arrow_data(dataset_path)
                if vectors is None:
                    continue
                
                n_samples = max(1, int(len(vectors) * sample_ratio))
                indices = np.random.choice(len(vectors), n_samples, replace=False)
                sampled = vectors[indices]
                all_vectors.append(sampled)
                print(f"Sampled {n_samples}/{len(vectors)} from {log_id_dir}")
        
        if not all_vectors:
            raise ValueError("No training data found. Check input directories.")
        
        # * 將所有向量垂直堆疊
        X_train = np.vstack(all_vectors)
        print(f"Training data prepared: {X_train.shape}")
        return X_train
    
    def _load_arrow_data(self, dir_path: str) -> Optional[np.ndarray]:
        """
        從指定資料夾載入向量，支援 HF datasets、Feather、IPC file/stream。
        
        優先檢查是否為 HF datasets 格式（含 dataset_info.json 或 state.json），
        避免對 HF 分片檔案進行無意義的 Feather/IPC 嘗試。
        """
        # * 優先判斷是否為 HF datasets 格式
        hf_markers = ["dataset_info.json", "state.json"]
        is_hf_dataset = any(
            exists(join_path(dir_path, marker)) for marker in hf_markers
        )
        
        if is_hf_dataset:
            try:
                ds = load_dataset(dir_path)
                return self._dataset_to_numpy(ds)
            except Exception as e:
                print(f"HF load_from_disk failed for {dir_path}: {e}")
                # 若 HF 載入失敗，仍可嘗試其他方式
        
        # * 嘗試以 Feather/IPC 格式讀取單一 Arrow 檔案
        possible_names = ["data.arrow", "data-00000-of-00001.arrow"]
        
        for fname in possible_names:
            fpath = join_path(dir_path, fname)
            if not exists(fpath):
                continue

            try:
                table = feather.read_table(fpath)
                return self._table_to_numpy(table)
            except Exception:
                pass  # 靜默失敗，嘗試下一種方式

            try:
                with pa.memory_map(fpath, "r") as source:
                    reader = pa.ipc.open_file(source)
                    table = reader.read_all()
                return self._table_to_numpy(table)
            except Exception:
                pass

            try:
                with pa.memory_map(fpath, "r") as source:
                    reader = pa.ipc.open_stream(source)
                    table = reader.read_all()
                return self._table_to_numpy(table)
            except Exception:
                pass

        # * 最後嘗試 HF datasets（若前面未嘗試過）
        if not is_hf_dataset:
            try:
                ds = load_dataset(dir_path)
                return self._dataset_to_numpy(ds)
            except Exception as e:
                print(f"Failed to load dataset at {dir_path}: {e}")
        
        return None

    def _table_to_numpy(self, table: pa.Table) -> np.ndarray:
        """將 Arrow 表格轉換為 numpy 陣列，支援常見欄位名稱。"""
        if "embedding" in table.column_names:
            data = table["embedding"].to_pylist()
        elif "vector" in table.column_names:
            data = table["vector"].to_pylist()
        elif "log_vector" in table.column_names:
            data = table["log_vector"].to_pylist()
        else:
            return table.to_pandas().values
        return np.array(data)

    def _dataset_to_numpy(self, ds) -> np.ndarray:
        """將 HF Dataset 轉為 numpy，優先回傳向量欄位。"""
        for key in ("embedding", "vector", "log_vector"):
            if key in ds.column_names:
                return np.array(ds[key])
        return ds.to_pandas().values
    
    # ======================== 模型訓練（Model Training） ========================
    
    def fit_global_model(self, X_train: np.ndarray) -> "ConceptExtractor":
        """
        在聚合後的資料上訓練全域概念模型（NMF 或 LDA）。
        
        參數：
            X_train：訓練矩陣（n_samples, n_features）
            
        回傳：
            self（已訓練完成的萃取器）
        """
        # * 以 Min-Max 縮放確保非負性
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = np.clip(X_scaled, 0, None)  # Extra safety for numerical stability
        
        # * 初始化並擬合模型
        if self.method == "nmf":
            self.model = NMF(
                n_components=self.n_concepts,
                init=self.init,
                solver="cd",  # 座標下降法（Coordinate Descent）
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
        elif self.method == "lda":
            # LDA 需要擬計數（非負整數或浮點數）
            self.model = LatentDirichletAllocation(
                n_components=self.n_concepts,
                max_iter=self.max_iter,
                random_state=self.random_state,
                learning_method="batch",
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'nmf' or 'lda'.")
        
        self.model.fit(X_scaled)
        self._is_fitted = True
        print(f"Global {self.method.upper()} model fitted with {self.n_concepts} concepts")
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
            }, f)
        print(f"Model saved to {save_path}")
    
    def load_model(self, path: Optional[str] = None) -> "ConceptExtractor":
        """從磁碟載入已訓練模型。"""
        load_path = self._resolve_path(path or self.model_path)
        
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.n_concepts = data["n_concepts"]
        self.method = data["method"]
        self._is_fitted = True
        print(f"Model loaded from {load_path}")
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
        print(f"Transformed {input_path} -> {output_path}, shape={H.shape}")
        
        # * 複製中繼資料（metadata）檔案
        if copy_metadata:
            for meta_file in ["state.json", "dataset_info.json"]:
                src = join_path(input_path, meta_file)
                dst = join_path(output_path, meta_file)
                if exists(src):
                    shutil.copy2(src, dst)
    
    def batch_transform(
        self,
        log_vectors_dir: str = LOG_VECTORS_DIR,
        concept_vectors_dir: str = CONCEPT_VECTORS_DIR,
    ) -> None:
        """
        將 LogVectors 目錄中的所有資料集轉換為 ConceptVectors。
        
        參數：
            log_vectors_dir：輸入 LogVectors 的根目錄
            concept_vectors_dir：輸出 ConceptVectors 的根目錄
        """
        log_vectors_dir = self._resolve_path(log_vectors_dir)
        concept_vectors_dir = self._resolve_path(concept_vectors_dir)

        if not exists(log_vectors_dir):
            raise FileNotFoundError(f"Input directory not found: {log_vectors_dir}")
        
        ensure_dir(concept_vectors_dir)
        
        for log_id_dir in get_dirs(log_vectors_dir):
            input_path = join_path(log_vectors_dir, log_id_dir)
            
            # 維持目錄結構：{LogID}_logvectors -> {LogID}_concepts
            output_name = log_id_dir.replace("_logvectors", "_concepts")
            output_path = join_path(concept_vectors_dir, output_name)
            
            self.transform_dataset(input_path, output_path)
        
        print(f"Batch transformation complete: {concept_vectors_dir}")
    
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
) -> ConceptExtractor:
    """
    端到端訓練流程：準備資料、訓練模型並儲存。
    """
    extractor = ConceptExtractor(n_concepts=n_concepts, method=method, model_path=model_path)
    
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
) -> None:
    """
    載入已訓練模型，並將所有 LogVectors 轉換為 ConceptVectors。
    """
    extractor = ConceptExtractor()
    extractor.load_model(model_path)
    extractor.batch_transform(log_vectors_dir, concept_vectors_dir)


if __name__ == "__main__":
    # 範例用法
    extractor = train_concept_extractor()
    transform_all_datasets()
