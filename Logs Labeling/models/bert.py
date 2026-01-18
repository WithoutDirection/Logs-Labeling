"""
BERT 嵌入 API (BERT Embedding API)

LogsLabeling 專案中用於載入和使用不同 BERT 模型進行文本嵌入的統一介面。

支援多種 BERT 變體：
- SentenceBERT (sentence-transformers)
- BERT Base (transformers)
- 安全領域專用 BERT：SecBERT, CTI-BERT, CYBERT, ATTACK-BERT, CySecBERT

使用方式：
    from models.bert import get_bert_model
    from config import BERT_MODEL_NAME
    
    bert = get_bert_model(BERT_MODEL_NAME)
    embeddings = bert.embed(["text1", "text2"])
    print(bert.get_info())
"""

import os
import numpy as np
from typing import List, Union, Optional, Dict, Any
from abc import ABC, abstractmethod
import warnings

# 嘗試匯入所需的函式庫
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    warnings.warn("未安裝 sentence-transformers。部分模型將無法使用。")

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("未安裝 transformers。部分模型將無法使用。")


class BaseBERTModel(ABC):
    """
    BERT 模型的抽象基底類別。
    
    所有 BERT 模型類別都應繼承此類別並實作：
    - load(): 載入模型
    - embed(): 從文本生成嵌入向量
    - get_info(): 返回模型資訊
    """
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        """
        初始化 BERT 模型。
        
        參數:
            model_name: BERT 模型的名稱或路徑
            cache_dir: 下載模型的快取目錄
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load(self) -> 'BaseBERTModel':
        """載入 BERT 模型。返回 self 以支援鏈式呼叫。"""
        pass
    
    @abstractmethod
    def embed(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        從文本生成嵌入向量。
        
        參數:
            texts: 單個文本或要嵌入的文本列表
            batch_size: 處理的批次大小
            show_progress: 是否顯示進度條
            normalize: 是否將嵌入向量正規化為單位長度
            
        返回:
            嵌入向量的 Numpy 陣列 (n_texts, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        獲取模型資訊。
        
        返回:
            包含模型元數據的字典
        """
        pass
    
    def get_embedding_dim(self) -> int:
        """獲取嵌入向量維度。"""
        if not self.is_loaded:
            raise RuntimeError("模型未載入。請先呼叫 load()。")
        return self._get_embedding_dim_impl()
    
    @abstractmethod
    def _get_embedding_dim_impl(self) -> int:
        """獲取嵌入向量維度的實作。"""
        pass
    
    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.__class__.__name__}(model='{self.model_name}', status={status})"


class SentenceBERTModel(BaseBERTModel):
    """
    使用 sentence-transformers 函式庫的 SentenceBERT 模型。
    
    最適合：語義相似度、句子嵌入
    模型：all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2 等
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[str] = None):
        """
        初始化 SentenceBERT 模型。
        
        參數:
            model_name: SentenceTransformer 模型名稱
            cache_dir: 模型快取目錄
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "SentenceBERTModel 需要 sentence-transformers。 "
                "請安裝：pip install sentence-transformers"
            )
        super().__init__(model_name, cache_dir)
    
    def load(self) -> 'SentenceBERTModel':
        """載入 SentenceBERT 模型。"""
        print(f"正在載入 SentenceBERT 模型：{self.model_name}...")
        try:
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            self.is_loaded = True
            print(f"  模型載入成功")
            print(f"  嵌入維度：{self.get_embedding_dim()}")
        except Exception as e:
            print(f"  模型載入失敗：{e}")
            raise
        return self
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """使用 SentenceBERT 生成嵌入向量。"""
        if not self.is_loaded:
            raise RuntimeError("模型未載入。請先呼叫 load()。")
        
        # 將單個文本轉換為列表
        if isinstance(texts, str):
            texts = [texts]
        
        # 生成嵌入向量
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def _get_embedding_dim_impl(self) -> int:
        """獲取嵌入向量維度。"""
        return self.model.get_sentence_embedding_dimension()
    
    def get_info(self) -> Dict[str, Any]:
        """獲取模型資訊。"""
        info = {
            'model_type': 'SentenceBERT',
            'model_name': self.model_name,
            'is_loaded': self.is_loaded,
            'cache_dir': self.cache_dir,
        }
        
        if self.is_loaded:
            info.update({
                'embedding_dim': self.get_embedding_dim(),
                'max_seq_length': self.model.max_seq_length,
            })
            
            # 嘗試獲取池化模式（並非所有模型都有此屬性）
            try:
                info['pooling_mode'] = str(self.model._first_module().pooling_mode)
            except (AttributeError, IndexError):
                info['pooling_mode'] = 'unknown'
        
        return info
    
    def show_info(self):
        """以可讀格式列印模型資訊。"""
        info = self.get_info()
        print("=" * 60)
        print("SentenceBERT 模型資訊")
        print("=" * 60)
        for key, value in info.items():
            print(f"  {key:20s}: {value}")
        print("=" * 60)


class TransformerBERTModel(BaseBERTModel):
    """
    使用 transformers 函式庫的 BERT 模型（平均池化）。
    
    最適合：自訂 BERT 變體、微調模型
    模型：bert-base-uncased, distilbert-base-uncased, roberta-base 等
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        cache_dir: Optional[str] = None,
        pooling: str = 'mean'
    ):
        """
        初始化 Transformer BERT 模型。
        
        參數:
            model_name: HuggingFace 模型名稱
            cache_dir: 快取目錄
            pooling: 池化策略 ('mean', 'cls', 'max')
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "TransformerBERTModel 需要 transformers。 "
                "請安裝：pip install transformers torch"
            )
        super().__init__(model_name, cache_dir)
        self.pooling = pooling
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load(self) -> 'TransformerBERTModel':
        """載入 transformer 模型和分詞器。"""
        print(f"正在載入 Transformer BERT 模型：{self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            # 優先使用 safetensors 格式以避免 PyTorch 安全警告
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    use_safetensors=True
                )
            except Exception:
                # 如果 safetensors 不可用，回退到標準載入
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    use_safetensors=False
                )
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"  模型在 {self.device} 上載入成功")
            print(f"  嵌入維度：{self.get_embedding_dim()}")
        except Exception as e:
            print(f"  模型載入失敗：{e}")
            raise
        return self
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """使用 transformer 模型生成嵌入向量。"""
        if not self.is_loaded:
            raise RuntimeError("模型未載入。請先呼叫 load()。")
        
        # 將單個文本轉換為列表
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # 分批處理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # 分詞
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 移動到裝置
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # 生成嵌入向量
            with torch.no_grad():
                outputs = self.model(**encoded)
                
                # 應用池化
                if self.pooling == 'mean':
                    # 平均池化與注意力遮罩
                    attention_mask = encoded['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                elif self.pooling == 'cls':
                    # CLS token 嵌入
                    embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.pooling == 'max':
                    # 最大池化
                    embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
                else:
                    raise ValueError(f"未知的池化方法：{self.pooling}")
                
                # 如果需要，進行正規化
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        # 連接所有批次
        return np.vstack(all_embeddings)
    
    def _get_embedding_dim_impl(self) -> int:
        """獲取嵌入向量維度。"""
        return self.model.config.hidden_size
    
    def get_info(self) -> Dict[str, Any]:
        """獲取模型資訊。"""
        info = {
            'model_type': 'TransformerBERT',
            'model_name': self.model_name,
            'is_loaded': self.is_loaded,
            'cache_dir': self.cache_dir,
            'pooling': self.pooling,
            'device': self.device,
        }
        
        if self.is_loaded:
            info.update({
                'embedding_dim': self.get_embedding_dim(),
                'num_layers': self.model.config.num_hidden_layers,
                'num_attention_heads': self.model.config.num_attention_heads,
                'max_position_embeddings': self.model.config.max_position_embeddings,
            })
        
        return info
    
    def show_info(self):
        """以可讀格式列印模型資訊。"""
        info = self.get_info()
        print("=" * 60)
        print("Transformer BERT 模型資訊")
        print("=" * 60)
        for key, value in info.items():
            print(f"  {key:20s}: {value}")
        print("=" * 60)


# ==================== 模型註冊表 (Model Registry) ====================

# 預定義的模型配置
MODEL_REGISTRY = {
    # SentenceBERT 模型
    'sentence-bert': {
        'class': SentenceBERTModel,
        'model_name': 'all-MiniLM-L6-v2',
        'description': '快速高效的句子嵌入 (384 維)'
    },
    'sentence-bert-multilingual': {
        'class': SentenceBERTModel,
        'model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
        'description': '多語言句子嵌入 (384 維)'
    },
    'sentence-bert-large': {
        'class': SentenceBERTModel,
        'model_name': 'all-mpnet-base-v2',
        'description': '高品質句子嵌入 (768 維)'
    },
    'bert-base-nli': {
        'class': SentenceBERTModel,
        'model_name': 'bert-base-nli-mean-tokens',
        'description': '在 NLI 上訓練的 BERT (768 維)'
    },
    
    # Transformer 模型
    'bert-base': {
        'class': TransformerBERTModel,
        'model_name': 'bert-base-uncased',
        'description': '原始 BERT 基礎模型 (768 維)'
    },
    
    'secbert': {
        'class': TransformerBERTModel,
        'model_name': 'jackaduma/SecBERT',
        'description': 'SecBERT - 在安全文本上訓練 (768 維)'
    },
    'securebert': {
        'class': TransformerBERTModel,
        'model_name': 'EhsanAghaei/SecureBERT',
        'description': 'SecureBERT - 針對 CTI 優化的 RoBERTa 模型 (768 維)'
    },

    'codebert': {
        'class': TransformerBERTModel,
        'model_name': 'microsoft/codebert-base',
        'description': 'CodeBERT - 理解程式碼與指令列 (768 維)'
    },

    
    'cysecbert': {
        'class': TransformerBERTModel,
        'model_name': 'Mikey/CySecBERT',
        'description': 'CySecBERT - 基於大量資安論文訓練 (768 維)'
    },
    'cybert': {
        'class': SentenceBERTModel,
        'model_name': 'markusbink/CyBERT',
        'description': 'CYBERT - 網路安全領域 BERT (768 維)'
    },
    'cybersec-bert': {
        'class': TransformerBERTModel,
        'model_name': 'GangJustice/cybersecurity-bert-base-uncased',
        'description': 'Cybersecurity BERT 基礎模型 (768 維)'
    },
    'vulbert': {
        'class': TransformerBERTModel,
        'model_name': 'snunlp/KR-BERT-char16424',
        'description': 'VulBERT - 漏洞檢測 BERT (768 維)'
    },
}


def get_bert_model(
    model_key: str,
    cache_dir: Optional[str] = None,
    auto_load: bool = True,
    **kwargs
) -> BaseBERTModel:
    """
    透過鍵值或自訂配置獲取 BERT 模型。
    
    參數:
        model_key: MODEL_REGISTRY 中的模型鍵值或自訂模型名稱
        cache_dir: 模型快取目錄
        auto_load: 是否自動載入模型
        **kwargs: 模型初始化的額外參數
        
    返回:
        BaseBERTModel 實例
        
    範例:
        # 使用預定義模型
        bert = get_bert_model('sentence-bert')
        
        # 使用自訂模型
        bert = get_bert_model('custom-model-name', model_class='SentenceBERT')
        
        # 手動載入
        bert = get_bert_model('sentence-bert', auto_load=False)
        bert.load()
    """
    # 檢查是否為註冊模型
    if model_key in MODEL_REGISTRY:
        config = MODEL_REGISTRY[model_key]
        model_class = config['class']
        model_name = config['model_name']
        
        # 建立實例
        model = model_class(model_name=model_name, cache_dir=cache_dir, **kwargs)
        
    else:
        # 嘗試推斷模型類型或使用自訂配置
        model_class_name = kwargs.pop('model_class', 'SentenceBERT')
        
        if model_class_name == 'SentenceBERT':
            model = SentenceBERTModel(model_name=model_key, cache_dir=cache_dir, **kwargs)
        elif model_class_name == 'TransformerBERT':
            model = TransformerBERTModel(model_name=model_key, cache_dir=cache_dir, **kwargs)
        else:
            raise ValueError(f"未知的模型類別：{model_class_name}")
    
    # 如果請求則自動載入
    if auto_load:
        model.load()
    
    return model


def list_available_models() -> Dict[str, str]:
    """
    列出所有可用的預定義模型。
    
    返回:
        將模型鍵值映射到描述的字典
    """
    return {key: config['description'] for key, config in MODEL_REGISTRY.items()}


def show_available_models():
    """以可讀格式列印所有可用模型。"""
    print("=" * 80)
    print("可用 BERT 模型")
    print("=" * 80)
    
    for key, config in MODEL_REGISTRY.items():
        print(f"\n[{key}]")
        print(f"  模型: {config['model_name']}")
        print(f"  類型: {config['class'].__name__}")
        print(f"  描述: {config['description']}")
    
    print("\n" + "=" * 80)
    print("用法: bert = get_bert_model('model_key')")
    print("=" * 80)


# ==================== 工具函數 (Utility Functions) ====================

def compare_models(
    texts: List[str],
    model_keys: List[str],
    cache_dir: Optional[str] = None
):
    """
    在相同文本上比較多個 BERT 模型。
    
    參數:
        texts: 要嵌入的文本列表
        model_keys: 要比較的模型鍵值列表
        cache_dir: 快取目錄
    """
    import time
    
    print("=" * 80)
    print(f"正在比較 {len(model_keys)} 個模型，共 {len(texts)} 條文本")
    print("=" * 80)
    
    results = {}
    
    for key in model_keys:
        print(f"\n[{key}]")
        try:
            # 載入模型
            start = time.time()
            model = get_bert_model(key, cache_dir=cache_dir, auto_load=True)
            load_time = time.time() - start
            
            # 生成嵌入向量
            start = time.time()
            embeddings = model.embed(texts)
            embed_time = time.time() - start
            
            results[key] = {
                'load_time': load_time,
                'embed_time': embed_time,
                'embedding_dim': embeddings.shape[1],
                'embeddings': embeddings
            }
            
            print(f"  載入時間:   {load_time:.2f}s")
            print(f"  嵌入時間:   {embed_time:.2f}s")
            print(f"  維度:       {embeddings.shape[1]}")
            
        except Exception as e:
            print(f"  錯誤: {e}")
    
    print("\n" + "=" * 80)
    return results


if __name__ == "__main__":
    # 示範用法
    print("BERT API 示範\n")
    
    # 顯示可用模型
    show_available_models()
    
    # 使用簡單模型進行測試
    print("\n\n使用 SentenceBERT 進行測試：")
    print("-" * 60)
    
    try:
        bert = get_bert_model('sentence-bert')
        bert.show_info()
        
        # 測試嵌入
        texts = [
            "這是一個測試句子。",
            "另一個用於嵌入的範例文本。",
            "BERT 模型對 NLP 任務很有用。"
        ]
        
        print(f"\n正在嵌入 {len(texts)} 條文本...")
        embeddings = bert.embed(texts)
        print(f"結果形狀: {embeddings.shape}")
        print(f"第一個嵌入向量 (截斷): {embeddings[0][:5]}...")
        
    except Exception as e:
        print(f"示範失敗: {e}")
        print("如果未安裝 sentence-transformers，這是預期的情況。")
