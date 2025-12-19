"""BiLSTM + Attention 模型，用於將日誌序列轉換為 Log Vector"""
from typing import Optional

import torch
import torch.nn as nn


class BiLSTMAttention(nn.Module):
    """雙向 LSTM + Attention 機制模型
    
    支援單流 (僅 Template) 或雙流 (Template + Parameters) 輸入模式，
    透過 Attention 機制將序列嵌入聚合為單一向量。
    
    Args:
        embedding_dim: 輸入嵌入維度
        hidden_size: LSTM 隱藏層大小
        num_layers: LSTM 層數
        dropout: Dropout 比例
        has_parsing: 是否啟用雙流模式 (Template + Parameters)
        fusion_enable: 是否啟用融合層
        fusion_dim: 融合層輸出維度
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        has_parsing: bool = False,
        fusion_enable: bool = True,
        fusion_dim: int = 256
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.has_parsing = has_parsing
        self.fusion_enable = fusion_enable
        
        # * Template BiLSTM + Attention
        self.template_bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.template_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        
        # * Parameters BiLSTM + Attention (僅在 has_parsing=True 時使用)
        if has_parsing:
            self.param_bilstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.param_attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False)
            )
            bilstm_output_dim = hidden_size * 4  # 串接 Template 與 Parameters
        else:
            self.param_bilstm = None
            self.param_attention = None
            bilstm_output_dim = hidden_size * 2
        
        # * 融合層 (可選)
        if fusion_enable:
            self.fusion = nn.Sequential(
                nn.Linear(bilstm_output_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.output_dim = fusion_dim
        else:
            self.fusion = None
            self.output_dim = bilstm_output_dim
    
    def _apply_bilstm_attention(
        self, 
        x: torch.Tensor, 
        bilstm: nn.LSTM, 
        attention: nn.Sequential
    ) -> torch.Tensor:
        """對輸入應用 BiLSTM + Attention"""
        lstm_out, _ = bilstm(x)  # (batch, seq_len, hidden*2)
        attn_weights = attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
        return context
    
    def forward(
        self, 
        template_x: torch.Tensor, 
        param_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向傳播
        
        Args:
            template_x: Template 嵌入序列 (batch, seq_len, embedding_dim)
            param_x: Parameters 嵌入序列 (batch, seq_len, embedding_dim)，可選
            
        Returns:
            Log Vector (batch, output_dim)
        """
        # * Template BiLSTM + Attention
        template_context = self._apply_bilstm_attention(
            template_x, self.template_bilstm, self.template_attention
        )
        
        # * Parameters BiLSTM + Attention (若有 parsing)
        if self.has_parsing and param_x is not None and self.param_bilstm is not None:
            param_context = self._apply_bilstm_attention(
                param_x, self.param_bilstm, self.param_attention
            )
            combined = torch.cat([template_context, param_context], dim=1)
        else:
            combined = template_context
        
        # * 融合層 (可選)
        if self.fusion:
            output = self.fusion(combined)
        else:
            output = combined
        
        return output
