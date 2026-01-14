"""
LogChunker：BiLSTM 區塊化模組

此模組負責：
1. 將嵌入向量切割成滑動視窗
2. 使用 BiLSTM + Attention 計算 Log Vector
3. 儲存為 HuggingFace Dataset 格式
"""
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional

from ._config import (
    DATA_DIR, LOG_VECTORS_DIR,
    SEQUENCE_WINDOW_SIZE, SEQUENCE_STRIDE,
    BILSTM_HIDDEN_SIZE, BILSTM_NUM_LAYERS, BILSTM_DROPOUT,
    FUSION_ENABLE, FUSION_OUTPUT_DIM
)
from utils.path import join_path, ensure_dir, get_filtered_dirs
from utils.dataset import save_dataset, load_embeddings
from models.BiLSTMAttention import BiLSTMAttention


class LogChunker:
    """
    透過 BiLSTM + Attention 將日誌嵌入序列轉換為 Log Vector。
    
    使用滑動視窗將連續的日誌嵌入分組，
    再透過 BiLSTM 捕捉序列資訊，最後使用 Attention 聚合。
    
    Example:
        >>> chunker = LogChunker(window_size=50, stride=25)
        >>> chunker.chunk_logs(num=10)
    """
    
    def __init__(
        self,
        embeddings_dir: str = None,
        output_dir: str = None,
        window_size: int = None,
        stride: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        dropout: float = None,
        fusion_enable: bool = None,
        fusion_dim: int = None
    ):
        """
        初始化 LogChunker。
        
        Args:
            embeddings_dir: 嵌入向量目錄
            output_dir: Log Vector 輸出目錄
            window_size: 滑動視窗大小
            stride: 滑動步長
            hidden_size: BiLSTM 隱藏層維度
            num_layers: BiLSTM 層數
            dropout: Dropout 比率
            fusion_enable: 是否啟用融合層
            fusion_dim: 融合輸出維度
        """
        self.embeddings_dir = embeddings_dir or join_path(DATA_DIR, "Embeddings")
        self.output_dir = output_dir or LOG_VECTORS_DIR
        self.window_size = window_size or SEQUENCE_WINDOW_SIZE
        self.stride = stride or SEQUENCE_STRIDE
        self.hidden_size = hidden_size or BILSTM_HIDDEN_SIZE
        self.num_layers = num_layers or BILSTM_NUM_LAYERS
        self.dropout = dropout or BILSTM_DROPOUT
        self.fusion_enable = fusion_enable if fusion_enable is not None else FUSION_ENABLE
        self.fusion_dim = fusion_dim or FUSION_OUTPUT_DIM
        
        self.model = None
        self.device = None
    
    def _init_model(self, embedding_dim: int, has_parsing: bool):
        """初始化 BiLSTM + Attention 模型。"""
        import torch
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = BiLSTMAttention(
            embedding_dim=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            has_parsing=has_parsing,
            fusion_enable=self.fusion_enable,
            fusion_dim=self.fusion_dim
        ).to(self.device)
        self.model.eval()
    
    def _load_embeddings(self, embedding_path: str) -> Tuple[List[str], np.ndarray, Optional[np.ndarray], bool]:
        """載入嵌入向量資料。"""
        return load_embeddings(embedding_path)
    
    def _create_windows(
        self, 
        template_embeddings: np.ndarray, 
        param_embeddings: Optional[np.ndarray],
        log_ids: List[str]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[dict]]:
        """
        將嵌入向量切割成重疊視窗。
        
        Returns:
            (template_windows, param_windows, window_info)
        """
        template_windows = []
        param_windows = [] if param_embeddings is not None else None
        window_info = []
        
        num_logs = len(template_embeddings)
        
        # 滑動視窗切割
        for start_idx in range(0, num_logs - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            template_windows.append(template_embeddings[start_idx:end_idx])
            if param_embeddings is not None:
                param_windows.append(param_embeddings[start_idx:end_idx])
            window_info.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_log_id': log_ids[start_idx],
                'end_log_id': log_ids[end_idx - 1]
            })
        
        # 處理尾部
        if num_logs > self.window_size and (num_logs - self.window_size) % self.stride != 0:
            start_idx = num_logs - self.window_size
            template_windows.append(template_embeddings[start_idx:])
            if param_embeddings is not None:
                param_windows.append(param_embeddings[start_idx:])
            window_info.append({
                'start_idx': start_idx,
                'end_idx': num_logs,
                'start_log_id': log_ids[start_idx],
                'end_log_id': log_ids[-1]
            })
        
        template_arr = np.array(template_windows) if template_windows else np.array([])
        param_arr = np.array(param_windows) if param_windows else None
        return template_arr, param_arr, window_info
    
    def _compute_log_vectors(
        self, 
        template_windows: np.ndarray, 
        param_windows: Optional[np.ndarray]
    ) -> np.ndarray:
        """使用 BiLSTM + Attention 計算 Log Vector。"""
        import torch
        
        if len(template_windows) == 0:
            return np.array([])
        
        with torch.no_grad():
            template_tensor = torch.FloatTensor(template_windows).to(self.device)
            
            if param_windows is not None:
                param_tensor = torch.FloatTensor(param_windows).to(self.device)
                log_vectors = self.model(template_tensor, param_tensor)
            else:
                log_vectors = self.model(template_tensor, None)
            
            return log_vectors.cpu().numpy()
    
    def _save_log_vectors(self, log_vectors: np.ndarray, window_info: List[dict], output_path: str):
        """儲存 Log Vector。"""
        data_dict = {
            'window_idx': list(range(len(log_vectors))),
            'start_idx': [info['start_idx'] for info in window_info],
            'end_idx': [info['end_idx'] for info in window_info],
            'start_log_id': [info['start_log_id'] for info in window_info],
            'end_log_id': [info['end_log_id'] for info in window_info],
            'log_vector': log_vectors.tolist()
        }
        save_dataset(data_dict, output_path)
    
    def chunk_logs(self, **kwargs):
        """
        批次處理嵌入向量並生成 Log Vector。
        
        Args:
            num: 要處理的檔案數量
            ratio: 要處理的檔案比例 (0-1)
        """
        import torch
        
        num = kwargs.get("num")
        ratio = kwargs.get("ratio")
        dirs = get_filtered_dirs(self.embeddings_dir, '_embeddings', num=num, ratio=ratio)
        
        # 顯示資訊
        if num:
            print(f"處理 {num} 個嵌入檔案...")
        elif ratio:
            print(f"處理 {ratio*100:.1f}% 的嵌入檔案...")
        else:
            print("處理所有嵌入檔案...")
        
        ensure_dir(self.output_dir)
        
        processed_count = 0
        total_windows = 0
        
        with tqdm(dirs, desc="生成 Log Vector", unit="檔案", dynamic_ncols=True) as pbar:
            for dir_name in pbar:
                embedding_path = join_path(self.embeddings_dir, dir_name)
                short_name = dir_name if len(dir_name) <= 35 else dir_name[:32] + "..."
                pbar.set_postfix({"檔案": short_name}, refresh=False)
                
                try:
                    log_ids, template_emb, param_emb, has_parsing = self._load_embeddings(embedding_path)
                    
                    if len(template_emb) < self.window_size:
                        continue
                    
                    # 初始化模型（首次）
                    if self.model is None:
                        embedding_dim = template_emb.shape[1]
                        self._init_model(embedding_dim, has_parsing)
                    
                    # 計算 Log Vector
                    template_windows, param_windows, window_info = self._create_windows(
                        template_emb, param_emb, log_ids
                    )
                    
                    if len(template_windows) == 0:
                        continue
                    
                    log_vectors = self._compute_log_vectors(template_windows, param_windows)
                    
                    # 儲存結果
                    output_name = dir_name.replace('_embeddings', '_logvectors')
                    output_path = join_path(self.output_dir, output_name)
                    self._save_log_vectors(log_vectors, window_info, output_path)
                    
                    processed_count += 1
                    total_windows += len(template_windows)
                    
                except Exception as e:
                    print(f"\n[Error] 處理 {dir_name} 失敗: {e}")
        
        # 顯示統計
        print(f"\nLog Vector 生成完成！")
        print(f"  已處理: {processed_count}/{len(dirs)} 個檔案")
        print(f"  總視窗數: {total_windows}")
        print(f"  視窗大小: {self.window_size}")
        print(f"  滑動步長: {self.stride}")
        print(f"  輸出位置: {self.output_dir}")
        if self.model:
            print(f"  Log Vector 維度: {self.model.output_dim}")
