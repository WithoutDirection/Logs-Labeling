
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# * 調整匯入路徑，確保能載入同專案上層的 config.py
CURRENT_DIR = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(CURRENT_DIR).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.path import (
    join_path, get_stem, split_extension, ensure_dir, 
    get_filtered_files, get_filtered_dirs
)
from utils.dataset import save_dataset, load_embeddings
from models.BiLSTMAttention import BiLSTMAttention

import config
import importlib

# * 配置參數
LOG_INPUT_PATH = config.INPUT_LOGS_DIR
LOG_INTERMEDIATE_PATH = config.INTERMEDIATE_DATA_DIR
LOG_OUTPUT_PATH = config.PROCESSED_LOGS_DIR
ENABLE_PARSER = config.ENABLE_PARSER  # 是否解析原始事件
LOG_PARSER = config.DEFAULT_PARSER  # 可選: 'drain', 'spell', 'lenma' (小寫模組名稱)


class LogLoader:
    """使用插件式解析器模組載入並解析日誌檔案。"""
    
    def __init__(
        self, 
        input_dir: str = LOG_INPUT_PATH, 
        output_dir: str = LOG_OUTPUT_PATH,
        enable_parser: bool = ENABLE_PARSER,
        parser_name: str = LOG_PARSER,
        use_registry_parser: bool = True,
        parser_config: dict = None
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.enable_parser = enable_parser
        self.parser_name = parser_name.lower() if parser_name else 'drain'
        self.use_registry_parser = use_registry_parser
        
        # * 步驟 1: 僅在啟用解析時初始化解析器
        if self.enable_parser:
            # * 步驟 1.1: 動態導入解析器模組
            try:
                # 優先從 preprocess 子套件導入
                parser_module = importlib.import_module(f'preprocess.{self.parser_name}')
            except (ImportError, ModuleNotFoundError):
                try:
                    # 備選: 直接導入 (從 preprocess 目錄執行時)
                    parser_module = importlib.import_module(self.parser_name)
                except (ImportError, ModuleNotFoundError):
                    print(f"警告: 無法導入解析器 '{parser_name}'，改用 'drain'。")
                    self.parser_name = 'drain'
                    try:
                        parser_module = importlib.import_module(f'preprocess.{self.parser_name}')
                    except (ImportError, ModuleNotFoundError):
                        parser_module = importlib.import_module(self.parser_name)
            
            # * 步驟 1.2: 根據模組中可用的類別初始化解析器
            parser_config = parser_config or {}
            standard_config = parser_config.get('standard', {'depth': 4, 'st': 0.5})
            registry_config = parser_config.get('registry', {'depth': 6, 'st': 0.5, 'registry_mode': True})
            
            # * 步驟 1.3: 建立標準解析器 (通常是 DrainParser 或類似的)
            if hasattr(parser_module, 'DrainParser'):
                self.standard_parser = parser_module.DrainParser(**standard_config)
            elif hasattr(parser_module, 'Parser'):
                self.standard_parser = parser_module.Parser(**standard_config)
            else:
                raise AttributeError(f"解析器模組 '{parser_name}' 必須包含 'DrainParser' 或 'Parser' 類別")
            
            # * 步驟 1.4: 若需要則建立註冊表解析器
            if use_registry_parser:
                if hasattr(parser_module, 'RegistryDrainParser'):
                    self.registry_parser = parser_module.RegistryDrainParser(**registry_config)
                elif hasattr(parser_module, 'RegistryParser'):
                    self.registry_parser = parser_module.RegistryParser(**registry_config)
                else:
                    # 若無註冊表專用解析器則回退為標準解析器
                    self.registry_parser = None
                    print(f"注意: 解析器 '{parser_name}' 無註冊表專用解析器，僅使用標準解析器")
            else:
                self.registry_parser = None
        else:
            self.standard_parser = None
            self.registry_parser = None
    
    def parse_log_row(self, row: pd.Series, columns: List[str]) -> Tuple[str, List[str], str]:
        """解析單行日誌或返回原始日誌。
        
        若解析器停用，返回空模板/參數與原始日誌。
        否則委託給解析器的 parse_log_row 方法。
        """
        # * 步驟 2.1: 若解析器停用，建立日誌訊息但不解析
        if not self.enable_parser:
            log_parts = []
            for col in columns:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val) and str(val).strip() and str(val).lower() != 'nan':
                        log_parts.append(str(val).strip())
            log_message = " ".join(log_parts)
            return "", [], log_message
        
        # * 步驟 2.2: 判斷使用哪個解析器
        operation = row.get('Operation', '')
        
        if (self.use_registry_parser and 
            self.registry_parser and 
            self.registry_parser.is_registry_operation(operation)):
            # 使用註冊表解析器
            return self.registry_parser.parse_log_row(row, columns)
        else:
            # 使用標準解析器
            return self.standard_parser.parse_log_row(row, columns)
    
    def parse_file(self, file_path: str, columns: List[str]) -> pd.DataFrame:
        """解析單個 CSV 檔案並提取模板和參數。"""
        # * 步驟 3.1: 讀取 CSV 檔案
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            print(f"讀取檔案錯誤 {file_path}: {e}")
            return None
        
        file_name = get_stem(file_path)
        
        # * 步驟 3.2: 若解析器停用，直接合併選定欄位
        if not self.enable_parser:
            log_ids = []
            concatenated_logs = []
            
            for idx, row in df.iterrows():
                log_ids.append(f"{file_name}_{idx}")
                
                # 合併選定的欄位
                log_parts = []
                for col in columns:
                    if col in row.index:
                        val = row[col]
                        if pd.notna(val) and str(val).strip() and str(val).lower() != 'nan':
                            log_parts.append(str(val).strip())
                concatenated_logs.append(" ".join(log_parts))
            
            # 建立簡化的 DataFrame（無模板和參數）
            result_df = pd.DataFrame({
                'LogID': log_ids,
                'ConcatenatedLog': concatenated_logs
            })
            return result_df
        
        # * 步驟 3.3: 若解析器啟用，解析每一行日誌
        log_ids = []
        templates = []
        parameters = []
        original_logs = []
        
        for idx, row in df.iterrows():
            template, params, log_msg = self.parse_log_row(row, columns)
            log_ids.append(f"{file_name}_{idx}")
            templates.append(template)
            parameters.append('|'.join(params) if params else "")
            original_logs.append(log_msg)
        
        # * 步驟 3.4: 建立包含模板和參數的 DataFrame
        result_df = pd.DataFrame({
            'LogID': log_ids,
            'Template': templates,
            'Parameters': parameters,
            'OriginalLog': original_logs
        })
        
        return result_df
    
    def load_logs(
        self, 
        columns: List[str] = None,
        **kwargs
    ):
        """
        載入並解析日誌檔案。
        
        Args:
            columns: 要解析的欄位名稱列表 (預設: Operation, Path, Result, Command Line)
            num: 要處理的檔案數量
            ratio: 要處理的檔案比例 (0-1)
        """
        # * 步驟 4.1: 設定要解析的欄位
        if columns is None:
            columns = ["Operation", "Path", "Result", "Command Line"]
        
        # * 步驟 4.2: 獲取篩選後的檔案列表
        num = kwargs.get("num")
        ratio = kwargs.get("ratio")
        files = get_filtered_files(self.input_dir, ".csv", num=num, ratio=ratio)
        
        if num:
            print(f"載入 {num} 個日誌檔案...")
        elif ratio:
            print(f"載入 {ratio*100:.1f}% 的日誌檔案...")
        else:
            print("載入所有日誌檔案...")
        
        # * 步驟 4.4: 建立輸出目錄
        ensure_dir(LOG_INTERMEDIATE_PATH)
        
        # * 步驟 4.5: 處理檔案
        parsed_dfs = []
        
        with tqdm(files, desc="解析日誌中", unit="檔案", dynamic_ncols=True) as pbar:
            for file in pbar:
                file_path = join_path(self.input_dir, file)
                short_name = file if len(file) <= 40 else file[:37] + "..."
                pbar.set_postfix({"解析器": self.parser_name, "檔案": short_name}, refresh=False)
                
                # * 步驟 4.6: 解析單個檔案
                parsed_df = self.parse_file(file_path, columns)
                
                if parsed_df is not None:
                    # * 步驟 4.7: 儲存解析結果
                    output_path = join_path(LOG_INTERMEDIATE_PATH, file)
                    parsed_df.to_csv(output_path, index=False, encoding='utf-8')
                    parsed_dfs.append(parsed_df)
        
        # * 步驟 4.8: 顯示處理統計資訊
        print(f"\n解析完成！")
        print(f"  已處理: {len(parsed_dfs)}/{len(files)} 個檔案")
        print(f"  輸出位置: {LOG_INTERMEDIATE_PATH}")
        
        if self.enable_parser:
            print(f"  解析器: {self.parser_name}")
            print(f"  標準模板數量: {len(self.standard_parser.get_clusters())}")
            if self.use_registry_parser and self.registry_parser:
                print(f"  註冊表模板數量: {len(self.registry_parser.get_clusters())}")
        else:
            print(f"  解析器: 已停用 (保留原始日誌)")
        
        return parsed_dfs

class LogEmbedder:
    """根據 LogLoader 的輸出計算嵌入向量並儲存。"""
    
    def __init__(
        self,
        intermediate_dir: str = LOG_INTERMEDIATE_PATH,
        output_dir: str = None,
        model_name: str = config.BERT_MODEL_NAME,
        cache_dir: str = config.BERT_CACHE_DIR,
        batch_size: int = 32,
        normalize: bool = True
    ):
        self.intermediate_dir = intermediate_dir
        self.output_dir = output_dir or join_path(config.DATA_DIR, "Embeddings")
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.bert_model = None
    
    def _load_model(self):
        """載入 BERT 模型。"""
        # * 步驟 1: 動態載入 BERT 模型
        from models.bert import get_bert_model
        print(f"正在載入 BERT 模型: {self.model_name}...")
        self.bert_model = get_bert_model(
            self.model_name, 
            cache_dir=self.cache_dir, 
            auto_load=True
        )
    
    def _prepare_text(self, df: pd.DataFrame) -> Tuple[List[str], Optional[List[str]], bool]:
        """根據 DataFrame 欄位決定嵌入內容。
        
        Returns:
            (template_texts, param_texts, has_parsing)
            - has_parsing=True: 返回 (template_texts, param_texts, True)
            - has_parsing=False: 返回 (concatenated_texts, None, False)
        """
        # * 步驟 2: 判斷使用 Template+Parameters 或 ConcatenatedLog
        if 'Template' in df.columns and 'Parameters' in df.columns:
            # 有 Parsing: 分別提取 Template 與 Parameters
            template_texts = []
            param_texts = []
            for _, row in df.iterrows():
                template = str(row['Template']) if pd.notna(row['Template']) else ""
                params = str(row['Parameters']) if pd.notna(row['Parameters']) else ""
                template_texts.append(template.strip())
                param_texts.append(params.strip())
            return template_texts, param_texts, True
        elif 'ConcatenatedLog' in df.columns:
            # 無 Parsing: 直接使用 ConcatenatedLog
            texts = df['ConcatenatedLog'].fillna("").astype(str).tolist()
            return texts, None, False
        else:
            raise ValueError("DataFrame 必須包含 'Template'+'Parameters' 或 'ConcatenatedLog' 欄位")
    
    def embed_file(self, file_path: str) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray], bool]:
        """對單一檔案進行嵌入計算。
        
        Returns:
            (df, template_embeddings, param_embeddings, has_parsing)
            - has_parsing=True: 返回 Template 與 Parameters 的嵌入
            - has_parsing=False: 返回 ConcatenatedLog 的嵌入 (param_embeddings=None)
        """
        df = pd.read_csv(file_path, encoding='utf-8')
        template_texts, param_texts, has_parsing = self._prepare_text(df)
        
        # * 步驟 3: 使用 BERT 模型計算嵌入向量
        template_embeddings = self.bert_model.embed(
            template_texts, 
            batch_size=self.batch_size, 
            normalize=self.normalize
        )
        
        if has_parsing and param_texts:
            param_embeddings = self.bert_model.embed(
                param_texts,
                batch_size=self.batch_size,
                normalize=self.normalize
            )
        else:
            param_embeddings = None
        
        return df, template_embeddings, param_embeddings, has_parsing
    
    def _save_embeddings(
        self, 
        df: pd.DataFrame, 
        template_embeddings: np.ndarray, 
        param_embeddings: Optional[np.ndarray],
        has_parsing: bool,
        output_path: str
    ):
        """將嵌入向量儲存為 Hugging Face Dataset 格式。"""
        if has_parsing and param_embeddings is not None:
            data_dict = {
                'LogID': df['LogID'].tolist(),
                'template_embedding': template_embeddings.tolist(),
                'param_embedding': param_embeddings.tolist()
            }
        else:
            data_dict = {
                'LogID': df['LogID'].tolist(),
                'embedding': template_embeddings.tolist()
            }
        save_dataset(data_dict, output_path)
    
    def embed_logs(self, **kwargs):
        """
        批次處理所有中間資料並計算嵌入向量。
        
        Args:
            num: 要處理的檔案數量
            ratio: 要處理的檔案比例 (0-1)
        """
        # * 步驟 1: 載入 BERT 模型
        if self.bert_model is None:
            self._load_model()
        
        # * 步驟 2: 獲取篩選後的檔案列表
        num = kwargs.get("num")
        ratio = kwargs.get("ratio")
        files = get_filtered_files(self.intermediate_dir, ".csv", num=num, ratio=ratio)
        
        if num:
            print(f"嵌入 {num} 個日誌檔案...")
        elif ratio:
            print(f"嵌入 {ratio*100:.1f}% 的日誌檔案...")
        else:
            print("嵌入所有日誌檔案...")
        
        # * 步驟 3: 建立輸出目錄
        ensure_dir(self.output_dir)
        
        # * 步驟 4: 批次處理檔案
        processed_count = 0
        
        with tqdm(files, desc="計算嵌入中", unit="檔案", dynamic_ncols=True) as pbar:
            for file in pbar:
                file_path = join_path(self.intermediate_dir, file)
                short_name = file if len(file) <= 40 else file[:37] + "..."
                pbar.set_postfix({"模型": self.model_name, "檔案": short_name}, refresh=False)
                
                try:
                    # * 步驟 5: 計算嵌入並儲存
                    df, template_emb, param_emb, has_parsing = self.embed_file(file_path)
                    
                    # 輸出檔名: 原檔名加上 _embeddings 後綴
                    output_name = split_extension(file)[0] + "_embeddings"
                    output_path = join_path(self.output_dir, output_name)
                    self._save_embeddings(df, template_emb, param_emb, has_parsing, output_path)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"\n處理 {file} 時發生錯誤: {e}")
        
        # * 步驟 6: 顯示處理統計
        print(f"\n嵌入完成！")
        print(f"  已處理: {processed_count}/{len(files)} 個檔案")
        print(f"  輸出位置: {self.output_dir}")
        print(f"  模型: {self.model_name}")
        print(f"  嵌入維度: {self.bert_model.get_embedding_dim()}")


class LogChunker:
    """透過 BiLSTM + Attention 將日誌嵌入序列轉換為 Log Vector。"""
    
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
        self.embeddings_dir = embeddings_dir or join_path(config.DATA_DIR, "Embeddings")
        self.output_dir = output_dir or config.LOG_VECTORS_DIR
        self.window_size = window_size or config.SEQUENCE_WINDOW_SIZE
        self.stride = stride or config.SEQUENCE_STRIDE
        self.hidden_size = hidden_size or config.BILSTM_HIDDEN_SIZE
        self.num_layers = num_layers or config.BILSTM_NUM_LAYERS
        self.dropout = dropout or config.BILSTM_DROPOUT
        self.fusion_enable = fusion_enable if fusion_enable is not None else config.FUSION_ENABLE
        self.fusion_dim = fusion_dim or config.FUSION_OUTPUT_DIM
        
        self.model = None
        self.device = None
    
    def _init_model(self, embedding_dim: int, has_parsing: bool):
        """初始化 BiLSTM + Attention 模型。"""
        import torch
        import torch.nn as nn
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # * 步驟 2.1: 建立 BiLSTM + Attention 模型
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
        """將嵌入向量切割成重疊視窗。
        
        Returns:
            (template_windows, param_windows, window_info)
        """
        # * 步驟 1.2: 滑動視窗切割
        template_windows = []
        param_windows = [] if param_embeddings is not None else None
        window_info = []
        
        num_logs = len(template_embeddings)
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
        
        # 處理尾部不足一個視窗的情況
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
        
        # * 步驟 2.2: 批次處理並計算 Log Vector
        with torch.no_grad():
            template_tensor = torch.FloatTensor(template_windows).to(self.device)
            
            if param_windows is not None:
                param_tensor = torch.FloatTensor(param_windows).to(self.device)
                log_vectors = self.model(template_tensor, param_tensor)
            else:
                log_vectors = self.model(template_tensor, None)
            
            return log_vectors.cpu().numpy()
    
    def _save_log_vectors(self, log_vectors: np.ndarray, window_info: List[dict], output_path: str):
        """儲存 Log Vector 至 Hugging Face Dataset 格式。"""
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
                    # 載入嵌入向量
                    log_ids, template_emb, param_emb, has_parsing = self._load_embeddings(embedding_path)
                    
                    if len(template_emb) < self.window_size:
                        continue
                    
                    # 初始化模型 (首次或維度變化時)
                    if self.model is None:
                        embedding_dim = template_emb.shape[1]
                        self._init_model(embedding_dim, has_parsing)
                    
                    # 建立視窗並計算 Log Vector
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
                    print(f"\n處理 {dir_name} 時發生錯誤: {e}")
        
        # 顯示統計資訊
        print(f"\nLog Vector 生成完成！")
        print(f"  已處理: {processed_count}/{len(dirs)} 個檔案")
        print(f"  總視窗數: {total_windows}")
        print(f"  視窗大小: {self.window_size}")
        print(f"  滑動步長: {self.stride}")
        print(f"  輸出位置: {self.output_dir}")
        if self.model:
            print(f"  Log Vector 維度: {self.model.output_dim}")


def main():
    # 1: 啟用解析 (預設)
    N = 100  # 處理所有檔案
    # loader = LogLoader(enable_parser=False)
    # loader.load_logs(num=N)
    
    # 2: 不解析 (保留原始日誌)
    loader = LogLoader(enable_parser=False)
    loader.load_logs(num=N)
    
    # 3: 計算嵌入向量
    embedder = LogEmbedder(normalize=False)
    embedder.embed_logs(num=N)
    
    # 4: 生成 Log Vector
    chunker = LogChunker()
    chunker.chunk_logs(num=N)

if __name__ == "__main__":
    main()