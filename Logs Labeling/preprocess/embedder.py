"""
LogEmbedder：BERT 嵌入計算模組

此模組負責：
1. 載入 BERT 模型
2. 將日誌文本轉換為嵌入向量
3. 儲存為 HuggingFace Dataset 格式
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional

from ._config import (
    LOG_INTERMEDIATE_PATH, DATA_DIR,
    BERT_MODEL_NAME, BERT_CACHE_DIR
)
from utils.path import join_path, split_extension, ensure_dir, get_filtered_files
from utils.dataset import save_dataset


class LogEmbedder:
    """
    根據 LogLoader 的輸出計算嵌入向量並儲存。
    
    支援兩種模式：
    - 有解析：分別計算 Template 與 Parameters 的嵌入
    - 無解析：計算 ConcatenatedLog 的嵌入
    
    Example:
        >>> embedder = LogEmbedder(model_name="sentence-bert")
        >>> embedder.embed_logs(num=10)
    """
    
    def __init__(
        self,
        intermediate_dir: str = LOG_INTERMEDIATE_PATH,
        output_dir: str = None,
        model_name: str = BERT_MODEL_NAME,
        cache_dir: str = BERT_CACHE_DIR,
        batch_size: int = 32,
        normalize: bool = True
    ):
        """
        初始化 LogEmbedder。
        
        Args:
            intermediate_dir: 中間資料目錄（LogLoader 輸出）
            output_dir: 嵌入輸出目錄
            model_name: BERT 模型名稱
            cache_dir: 模型快取目錄
            batch_size: 批次大小
            normalize: 是否正規化嵌入向量
        """
        self.intermediate_dir = intermediate_dir
        self.output_dir = output_dir or join_path(DATA_DIR, "Embeddings")
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.bert_model = None
    
    def _load_model(self):
        """載入 BERT 模型。"""
        from models.bert import get_bert_model
        print(f"正在載入 BERT 模型: {self.model_name}...")
        self.bert_model = get_bert_model(
            self.model_name, 
            cache_dir=self.cache_dir, 
            auto_load=True
        )
    
    def _prepare_text(self, df: pd.DataFrame) -> Tuple[List[str], Optional[List[str]], bool]:
        """
        根據 DataFrame 欄位決定嵌入內容。
        
        Returns:
            (template_texts, param_texts, has_parsing)
        """
        if 'Template' in df.columns and 'Parameters' in df.columns:
            # 有解析：分別提取
            template_texts = []
            param_texts = []
            for _, row in df.iterrows():
                template = str(row['Template']) if pd.notna(row['Template']) else ""
                params = str(row['Parameters']) if pd.notna(row['Parameters']) else ""
                template_texts.append(template.strip())
                param_texts.append(params.strip())
            return template_texts, param_texts, True
        elif 'ConcatenatedLog' in df.columns:
            # 無解析：直接使用
            texts = df['ConcatenatedLog'].fillna("").astype(str).tolist()
            return texts, None, False
        else:
            raise ValueError("DataFrame 必須包含 'Template'+'Parameters' 或 'ConcatenatedLog' 欄位")
    
    def embed_file(self, file_path: str) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray], bool]:
        """
        對單一檔案進行嵌入計算。
        
        Returns:
            (df, template_embeddings, param_embeddings, has_parsing)
        """
        df = pd.read_csv(file_path, encoding='utf-8')
        template_texts, param_texts, has_parsing = self._prepare_text(df)
        dataset_name = split_extension(file_path.split("/")[-1])[0]
        
        template_embeddings = self.bert_model.embed(
            template_texts, 
            batch_size=self.batch_size, 
            normalize=self.normalize,
            dataset_name=dataset_name,
        )
        
        param_embeddings = None
        if has_parsing and param_texts:
            param_embeddings = self.bert_model.embed(
                param_texts,
                batch_size=self.batch_size,
                normalize=self.normalize,
                dataset_name=dataset_name,
            )
        
        return df, template_embeddings, param_embeddings, has_parsing
    
    def _save_embeddings(
        self, 
        df: pd.DataFrame, 
        template_embeddings: np.ndarray, 
        param_embeddings: Optional[np.ndarray],
        has_parsing: bool,
        output_path: str
    ):
        """將嵌入向量儲存為 HuggingFace Dataset 格式。"""
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
        if self.bert_model is None:
            self._load_model()
        
        num = kwargs.get("num")
        ratio = kwargs.get("ratio")
        files = get_filtered_files(self.intermediate_dir, ".csv", num=num, ratio=ratio)
        
        # 顯示資訊
        if num:
            print(f"嵌入 {num} 個日誌檔案...")
        elif ratio:
            print(f"嵌入 {ratio*100:.1f}% 的日誌檔案...")
        else:
            print("嵌入所有日誌檔案...")
        
        ensure_dir(self.output_dir)
        processed_count = 0
        
        with tqdm(files, desc="計算嵌入中", unit="檔案", dynamic_ncols=True) as pbar:
            for file in pbar:
                file_path = join_path(self.intermediate_dir, file)
                short_name = file if len(file) <= 40 else file[:37] + "..."
                pbar.set_postfix({"模型": self.model_name, "檔案": short_name}, refresh=False)
                
                try:
                    df, template_emb, param_emb, has_parsing = self.embed_file(file_path)
                    
                    output_name = split_extension(file)[0] + "_embeddings"
                    output_path = join_path(self.output_dir, output_name)
                    self._save_embeddings(df, template_emb, param_emb, has_parsing, output_path)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"\n[Error] 處理 {file} 失敗: {e}")
        
        # 顯示統計
        print(f"\n嵌入完成！")
        print(f"  已處理: {processed_count}/{len(files)} 個檔案")
        print(f"  輸出位置: {self.output_dir}")
        print(f"  模型: {self.model_name}")
        print(f"  嵌入維度: {self.bert_model.get_embedding_dim()}")
