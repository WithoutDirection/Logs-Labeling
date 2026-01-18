"""
GPU 加速 NMF 實作：PyTorch 乘法更新規則 + L1 稀疏性約束

# * 完全矩陣運算，適合 GPU 平行加速
# * 乘法更新規則自動保持非負性
# * L1 正則化強制概念稀疏性，提升解釋性
# * Mini-batch 處理避免 OOM
# * 自動回退至 CPU
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    import torch


def _check_torch_available() -> bool:
    # * 檢查 PyTorch 可用性
    try:
        import torch
        return True
    except ImportError:
        return False


def _check_cuda_available() -> bool:
    # * 檢查 CUDA 可用性
    if not _check_torch_available():
        return False
    import torch
    return torch.cuda.is_available()


class NMFGpu:
    """
    GPU 加速 NMF：乘法更新規則 + L1 稀疏性正則化
    
    # * 演算法：V ≈ W @ H
    # * V：輸入矩陣 (n_samples, n_features)
    # * W：權重矩陣 (n_samples, n_components)
    # * H：基矩陣 (n_components, n_features)
    
    # * 更新規則（含 L1 正則化）：
    # * H = H * (W^T @ V) / (W^T @ W @ H + l1_reg + eps)
    # * W = W * (V @ H^T) / (W @ H @ H^T + eps)
    
    參數：
        n_components：概念數量
        l1_reg：L1 正則化強度（控制稀疏度）
        max_iter：最大迭代次數
        tol：收斂容許誤差
        epsilon：數值穩定性常數
        random_state：隨機種子
        batch_size：Mini-batch 大小（None=自動）
        check_interval：收斂檢查間隔
        verbose：顯示進度
        device：運算裝置（'cuda'/'cpu'/None=自動）
    """
    
    def __init__(
        self,
        n_components: int = 50,
        l1_reg: float = 0.01,
        max_iter: int = 300,
        tol: float = 1e-3,
        epsilon: float = 1e-8,
        random_state: int = 42,
        batch_size: Optional[int] = None,
        check_interval: int = 10,
        verbose: bool = True,
        device: Optional[str] = None,
    ):
        self.n_components = n_components
        self.l1_reg = l1_reg  # * L1 正則化強度
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
        self.random_state = random_state
        self.batch_size = batch_size
        self.check_interval = check_interval
        self.verbose = verbose
        self._device_str = device
        
        # * 訓練後儲存的組件
        self.components_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.reconstruction_err_: Optional[float] = None
        self.n_iter_: int = 0
        
        self._torch = None
        self._device = None
    
    def __getstate__(self):
        # * 自訂 pickle 序列化，排除不可序列化的 torch 模組
        state = self.__dict__.copy()
        state['_torch'] = None
        state['_device'] = None
        return state
    
    def __setstate__(self, state):
        # * 自訂 pickle 反序列化，torch 將延遲初始化
        self.__dict__.update(state)
        
    def _init_torch(self) -> bool:
        # * 延遲初始化 PyTorch，回傳是否使用 GPU
        if self._torch is not None:
            return self._device.type == 'cuda'
        
        if not _check_torch_available():
            raise ImportError("需要安裝 PyTorch 才能使用 GPU NMF。請執行: pip install torch")
        
        import torch
        self._torch = torch
        
        # * 設定裝置
        use_cuda = False
        if self._device_str == 'cuda' or (self._device_str is None and torch.cuda.is_available()):
            if self._check_cuda_compatibility():
                use_cuda = True
            else:
                if self.verbose:
                    print(" GPU 不相容，改用 CPU")
        
        if self._device_str == 'cpu':
            self._device = torch.device('cpu')
        elif use_cuda:
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
            if self.verbose and self._device_str is None and not torch.cuda.is_available():
                print(" CUDA 不可用，改用 CPU")
        
        # * 設定隨機種子
        torch.manual_seed(self.random_state)
        if self._device.type == 'cuda':
            torch.cuda.manual_seed(self.random_state)
        
        return self._device.type == 'cuda'
    
    def _check_cuda_compatibility(self) -> bool:
        # * 檢查 GPU compute capability 相容性
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            
            # * 測試 CUDA kernel 是否可用
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_tensor = torch.zeros(2, 2, device='cuda')
                _ = test_tensor @ test_tensor
                del test_tensor
                torch.cuda.empty_cache()
            return True
        except RuntimeError as e:
            if "no kernel image" in str(e) or "CUDA" in str(e):
                return False
            raise
    
    def _to_tensor(self, X: np.ndarray, dtype=None) -> "torch.Tensor":
        # * 轉 numpy 為 Tensor 並移至裝置
        if dtype is None:
            dtype = self._torch.float32
        return self._torch.from_numpy(X.astype(np.float32)).to(dtype).to(self._device)
    
    def _to_numpy(self, tensor: "torch.Tensor") -> np.ndarray:
        # * 轉 Tensor 回 numpy
        return tensor.detach().cpu().numpy()
    
    def _compute_reconstruction_error_batched(
        self, X: np.ndarray, H: np.ndarray, batch_size: int = 10000
    ) -> float:
        # * 批次計算重建誤差（避免 OOM）
        n_samples = X.shape[0]
        total_error = 0.0
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X[start:end]
            
            V_batch = self._to_tensor(X_batch)
            H_tensor = self._to_tensor(H)
            W_batch = self._transform_batch(V_batch, H_tensor, n_iter=20)
            
            with self._torch.no_grad():
                reconstruction = W_batch @ H_tensor
                batch_error = self._torch.sum((V_batch - reconstruction) ** 2).item()
                total_error += batch_error
            
            del V_batch, H_tensor, W_batch, reconstruction
            if self._device.type == 'cuda':
                self._torch.cuda.empty_cache()
        
        return np.sqrt(total_error / n_samples)

    def _compute_reconstruction_error(
        self, V: "torch.Tensor", W: "torch.Tensor", H: "torch.Tensor"
    ) -> float:
        # * 計算單次重建誤差，用於收斂判斷
        torch = self._torch
        with torch.no_grad():
            reconstruction = W @ H
            error = torch.norm(V - reconstruction, p="fro").item()
        return error
    
    def _fit_full_batch(
        self, V: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        # * 全批次訓練（適合較小資料集）
        torch = self._torch
        n_samples, n_features = V.shape
        
        # * 隨機初始化 W 和 H
        W = torch.rand(n_samples, self.n_components, device=self._device, dtype=V.dtype)
        H = torch.rand(self.n_components, n_features, device=self._device, dtype=V.dtype)
        W = W + self.epsilon
        H = H + self.epsilon
        
        prev_error = float('inf')
        
        for iteration in range(self.max_iter):
            # * 更新 H：H = H * (W^T @ V) / (W^T @ W @ H + l1_reg + eps)
            numerator_H = W.T @ V
            denominator_H = W.T @ W @ H + self.l1_reg + self.epsilon
            H = H * (numerator_H / denominator_H)
            
            # * 更新 W：W = W * (V @ H^T) / (W @ H @ H^T + eps)
            numerator_W = V @ H.T
            denominator_W = W @ H @ H.T + self.epsilon
            W = W * (numerator_W / denominator_W)
            
            # * 定期檢查收斂
            if (iteration + 1) % self.check_interval == 0:
                error = self._compute_reconstruction_error(V, W, H)
                relative_change = abs(prev_error - error) / (prev_error + self.epsilon)
                
                if self.verbose:
                    print(f"  迭代 {iteration + 1}/{self.max_iter}, 誤差: {error:.6f}, 變化: {relative_change:.6f}")
                
                if relative_change < self.tol:
                    if self.verbose:
                        print(f"  在第 {iteration + 1} 次迭代收斂")
                    self.n_iter_ = iteration + 1
                    break
                
                prev_error = error
        else:
            self.n_iter_ = self.max_iter
        
        return W, H
    
    def _fit_mini_batch(
        self, V: "torch.Tensor", batch_size: int
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        # * Mini-batch 訓練（適合大資料集，避免 OOM）
        torch = self._torch
        n_samples, n_features = V.shape
        
        # * 初始化共享的 H
        H = torch.rand(self.n_components, n_features, device=self._device, dtype=V.dtype)
        H = H + self.epsilon
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        prev_total_error = float('inf')
        
        for iteration in range(self.max_iter):
            total_error = 0.0
            indices = torch.randperm(n_samples, device=self._device)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                V_batch = V[batch_indices]
                batch_n = V_batch.shape[0]
                
                # * 初始化 W_batch
                W_batch = torch.rand(batch_n, self.n_components, device=self._device, dtype=V.dtype)
                W_batch = W_batch + self.epsilon
                
                # * 內部迭代
                for _ in range(5):
                    numerator_W = V_batch @ H.T
                    denominator_W = W_batch @ H @ H.T + self.epsilon
                    W_batch = W_batch * (numerator_W / denominator_W)
                
                # * 更新共享 H（含 L1 正則化）
                numerator_H = W_batch.T @ V_batch
                denominator_H = W_batch.T @ W_batch @ H + self.l1_reg + self.epsilon
                H_update = H * (numerator_H / denominator_H)
                H = 0.9 * H + 0.1 * H_update  # * 平滑更新
                
                # * 計算 batch 誤差
                with torch.no_grad():
                    batch_error = torch.norm(V_batch - W_batch @ H, p='fro').item()
                    total_error += batch_error ** 2
            
            total_error = np.sqrt(total_error)
            
            # * 檢查收斂
            if (iteration + 1) % self.check_interval == 0:
                relative_change = abs(prev_total_error - total_error) / (prev_total_error + self.epsilon)
                
                if self.verbose:
                    print(f"  迭代 {iteration + 1}/{self.max_iter}, 誤差: {total_error:.6f}, 變化: {relative_change:.6f}")
                
                if relative_change < self.tol:
                    if self.verbose:
                        print(f"  在第 {iteration + 1} 次迭代收斂")
                    self.n_iter_ = iteration + 1
                    break
                
                prev_total_error = total_error
        else:
            self.n_iter_ = self.max_iter
        
        # * 最終計算 W（分批處理）
        W_list = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            V_batch = V[start_idx:end_idx]
            W_batch = self._transform_batch(V_batch, H)
            W_list.append(W_batch)
        
        W = torch.cat(W_list, dim=0)
        return W, H
    
    def _transform_batch(
        self, V_batch: "torch.Tensor", H: "torch.Tensor", n_iter: int = 50
    ) -> "torch.Tensor":
        # * 單批次 transform（固定 H，只更新 W）
        torch = self._torch
        batch_n = V_batch.shape[0]
        
        W_batch = torch.rand(batch_n, self.n_components, device=self._device, dtype=V_batch.dtype)
        W_batch = W_batch + self.epsilon
        H_HT = H @ H.T
        
        for _ in range(n_iter):
            numerator_W = V_batch @ H.T
            denominator_W = W_batch @ H_HT + self.epsilon
            W_batch = W_batch * (numerator_W / denominator_W)
        
        return W_batch
    
    def _estimate_memory_usage(self, n_samples: int, n_features: int) -> int:
        # * 估計 GPU 記憶體需求 (bytes)
        bytes_per_float = 4
        v_size = n_samples * n_features * bytes_per_float
        w_size = n_samples * self.n_components * bytes_per_float
        h_size = self.n_components * n_features * bytes_per_float
        intermediate = (v_size + w_size) * 2  # * 中間計算
        return v_size + w_size + h_size + intermediate
    
    def _auto_batch_size(self, n_samples: int, n_features: int) -> Optional[int]:
        # * 根據 GPU 記憶體自動決定 batch size
        if self._device.type != 'cuda':
            return None
        
        torch = self._torch
        total_memory = torch.cuda.get_device_properties(self._device).total_memory
        allocated_memory = torch.cuda.memory_allocated(self._device)
        free_memory = total_memory - allocated_memory
        usable_memory = int(free_memory * 0.8)  # * 保留 20% 緩衝
        
        estimated_usage = self._estimate_memory_usage(n_samples, n_features)
        
        if estimated_usage <= usable_memory:
            if self.verbose:
                print(f"  GPU 記憶體充足，使用全批次訓練")
            return None
        
        # * 計算安全 batch size
        per_sample_memory = self._estimate_memory_usage(1, n_features)
        h_size = self.n_components * n_features * 4
        usable_for_samples = usable_memory - h_size * 3
        
        if usable_for_samples <= 0:
            safe_batch_size = 64
        else:
            safe_batch_size = max(64, int(usable_for_samples / per_sample_memory))
        
        safe_batch_size = min(safe_batch_size, n_samples // 2)
        
        if self.verbose:
            print(f"  GPU 記憶體限制，使用 Mini-batch (batch_size={safe_batch_size})")
            print(f"  可用: {usable_memory / 1e9:.2f} GB, 估計: {estimated_usage / 1e9:.2f} GB")
        
        return safe_batch_size
    
    def fit(self, X: np.ndarray) -> "NMFGpu":
        # * 訓練 NMF 模型
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X 必須是 2D 矩陣，收到 {X.ndim}D")
        if np.any(X < 0):
            raise ValueError("NMF 要求輸入矩陣必須為非負")
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        is_gpu = self._init_torch()
        
        if self.verbose:
            device_name = "GPU" if is_gpu else "CPU"
            if is_gpu:
                gpu_name = self._torch.cuda.get_device_name(self._device)
                print(f"使用 {device_name} ({gpu_name}) 執行 NMF")
            else:
                print(f"使用 {device_name} 執行 NMF")
            print(f"  輸入: ({n_samples}, {n_features}), 概念數: {self.n_components}, L1={self.l1_reg}")
        
        # * 決定 batch size
        batch_size = self.batch_size
        if batch_size is None and is_gpu:
            batch_size = self._auto_batch_size(n_samples, n_features)
        
        V = self._to_tensor(X)
        
        # * 訓練
        try:
            if batch_size is None:
                W, H = self._fit_full_batch(V)
            else:
                W, H = self._fit_mini_batch(V, batch_size)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if self.verbose:
                    print(" GPU 記憶體不足，降低 batch size 重試...")
                self._torch.cuda.empty_cache()
                fallback_batch = max(32, (batch_size or n_samples) // 4)
                V = self._to_tensor(X)
                W, H = self._fit_mini_batch(V, fallback_batch)
            else:
                raise
        
        # * 儲存結果
        self.components_ = self._to_numpy(H)
        
        if is_gpu:
            del V, W, H
            self._torch.cuda.empty_cache()
        
        # * 計算重建誤差
        if self.verbose:
            print("計算重建誤差中...")
        self.reconstruction_err_ = self._compute_reconstruction_error_batched(X, self.components_)
        
        if self.verbose:
            print(f"NMF 訓練完成，迭代: {self.n_iter_}, 誤差: {self.reconstruction_err_:.6f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用已訓練的基矩陣 H 將新資料投影至概念空間。
        
        參數：
            X：輸入矩陣（n_samples, n_features）
            
        回傳：
            W：概念權重矩陣（n_samples, n_components）
        """
        if self.components_ is None:
            raise RuntimeError("模型尚未訓練。請先呼叫 fit()。")
        
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X 必須是 2D 矩陣，收到 {X.ndim}D")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"特徵維度不匹配: 預期 {self.n_features_in_}, 收到 {X.shape[1]}")
        
        n_samples = X.shape[0]
        
        # 初始化 PyTorch（若尚未初始化）
        is_gpu = self._init_torch()
        
        # 決定 batch size
        batch_size = self.batch_size
        if batch_size is None and is_gpu:
            batch_size = self._auto_batch_size(n_samples, self.n_features_in_)
        
        # 轉換為 Tensor
        V = self._to_tensor(X)
        H = self._to_tensor(self.components_)
        
        try:
            if batch_size is None:
                # 全批次 transform
                W = self._transform_batch(V, H)
                result = self._to_numpy(W)
            else:
                # Mini-batch transform
                n_batches = (n_samples + batch_size - 1) // batch_size
                W_list = []
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, n_samples)
                    V_batch = V[start_idx:end_idx]
                    
                    W_batch = self._transform_batch(V_batch, H)
                    W_list.append(self._to_numpy(W_batch))
                
                result = np.vstack(W_list)
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if self.verbose:
                    print(" Transform 時 GPU 記憶體不足，改用更小的 batch...")
                self._torch.cuda.empty_cache()
                
                # 使用更小的 batch
                fallback_batch = max(32, (batch_size or n_samples) // 4)
                V = self._to_tensor(X)
                H = self._to_tensor(self.components_)
                
                n_batches = (n_samples + fallback_batch - 1) // fallback_batch
                W_list = []
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * fallback_batch
                    end_idx = min((batch_idx + 1) * fallback_batch, n_samples)
                    V_batch = V[start_idx:end_idx]
                    
                    W_batch = self._transform_batch(V_batch, H)
                    W_list.append(self._to_numpy(W_batch))
                
                result = np.vstack(W_list)
            else:
                raise
        
        # 清理 GPU 記憶體
        if is_gpu:
            del V, H
            self._torch.cuda.empty_cache()
        
        return result
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        訓練模型並回傳訓練資料的概念權重。
        
        參數：
            X：輸入矩陣（n_samples, n_features）
            
        回傳：
            W：概念權重矩陣（n_samples, n_components）
        """
        self.fit(X)
        return self.transform(X)
    
    def get_params(self) -> Dict[str, Any]:
        # * 回傳模型參數
        return {
            "n_components": self.n_components,
            "l1_reg": self.l1_reg,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "epsilon": self.epsilon,
            "random_state": self.random_state,
            "batch_size": self.batch_size,
            "check_interval": self.check_interval,
            "verbose": self.verbose,
        }
    
    def set_params(self, **params) -> "NMFGpu":
        # * 設定模型參數
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


def create_nmf_model(
    n_components: int = 50,
    l1_reg: float = 0.01,
    max_iter: int = 300,
    tol: float = 1e-3,
    use_gpu: bool = True,
    batch_size: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
):
    # * 工廠函式：根據環境建立 NMF 模型
    if use_gpu and _check_cuda_available():
        return NMFGpu(
            n_components=n_components,
            l1_reg=l1_reg,
            max_iter=max_iter,
            tol=tol,
            batch_size=batch_size,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        # * 回退至 sklearn NMF
        if verbose:
            if use_gpu:
                print("CUDA 不可用，使用 sklearn CPU NMF")
            else:
                print("使用 sklearn CPU NMF")
        
        from sklearn.decomposition import NMF
        return NMF(
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            init="nndsvd",
            solver="cd",
        )
