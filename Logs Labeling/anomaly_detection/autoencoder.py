"""AutoEncoder 異常偵測模組

利用深度神經網路學習 Log Vector 的壓縮表示，
透過重構誤差識別異常模式。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class AutoEncoderConfig:
    """AutoEncoder 超參數設定"""
    latent_dim: int = 32
    hidden_dims: List[int] = None
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    device: str = "auto"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class AutoEncoderModel(nn.Module):
    """AutoEncoder 神經網路架構"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        
        # * 建立 Encoder：逐層壓縮至 Latent Space
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # * 建立 Decoder：從 Latent Space 還原
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AutoEncoderDetector:
    """AutoEncoder 異常偵測器
    
    學習正常資料的壓縮表示，對異常資料產生較大重構誤差。
    """
    
    def __init__(self, config: Optional[AutoEncoderConfig] = None):
        self.config = config or AutoEncoderConfig()
        self.model: Optional[AutoEncoderModel] = None
        self.device = torch.device(self.config.device)
        self._is_fitted = False
        self._input_dim: Optional[int] = None
    
    def fit(self, X: np.ndarray) -> "AutoEncoderDetector":
        """訓練 AutoEncoder 模型
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
        """
        self._input_dim = X.shape[1]
        
        # * 建立模型並移至指定裝置
        self.model = AutoEncoderModel(
            input_dim=self._input_dim,
            hidden_dims=self.config.hidden_dims,
            latent_dim=self.config.latent_dim
        ).to(self.device)
        
        # * 準備資料載入器
        tensor_X = torch.FloatTensor(X)
        dataset = TensorDataset(tensor_X)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # * 訓練迴圈：最小化重構誤差 (MSE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        
        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for batch in dataloader:
                batch_x = batch[0].to(self.device)
                if batch_x.shape[0] < 2:
                    continue
                
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        self._is_fitted = True
        return self
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """計算異常分數 (重構誤差)
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
            
        Returns:
            異常分數陣列（MSE），值越高代表越異常
        """
        if not self._is_fitted:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()")
        
        self.model.eval()
        tensor_X = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(tensor_X)
            # * 計算每筆樣本的 MSE 作為異常分數
            mse = torch.mean((tensor_X - reconstructed) ** 2, dim=1)
        
        return mse.cpu().numpy()
    
    def predict_labels(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """預測異常標籤
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
            threshold: 異常閾值，若為 None 則使用 mean + 2*std
            
        Returns:
            標籤陣列 (0: 正常, 1: 異常)
        """
        scores = self.predict_scores(X)
        
        if threshold is None:
            # * 使用統計方法自動決定閾值
            threshold = np.mean(scores) + 2 * np.std(scores)
        
        return (scores > threshold).astype(int)
    
    def fit_predict(self, X: np.ndarray, threshold: Optional[float] = None) -> Dict[str, np.ndarray]:
        """訓練並預測
        
        Args:
            X: 形狀為 (n_samples, n_features) 的 Log Vector 矩陣
            threshold: 異常閾值
            
        Returns:
            包含 'scores' 和 'labels' 的字典
        """
        self.fit(X)
        return {
            "scores": self.predict_scores(X),
            "labels": self.predict_labels(X, threshold)
        }
