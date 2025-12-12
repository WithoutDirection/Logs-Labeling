"""效能趨勢分析模組

繪製各模型效能指標隨資料規模變化的趨勢圖。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualization.aggregator import ResultAggregator

# 模型顏色配置
MODEL_COLORS = {
    "isolation_forest": "#2ecc71",
    "copod": "#3498db",
    "autoencoder": "#9b59b6",
    "pca_gmm": "#e74c3c",
    "ensemble": "#34495e"
}


def plot_trend_analysis(
    aggregator: "ResultAggregator",
    output_dir: str = "result/unsupervised_anomaly_dection",
    metrics: List[str] = None,
    figsize: tuple = (14, 5)
) -> str:
    """繪製效能趨勢折線圖
    
    Args:
        aggregator: 結果聚合器
        output_dir: 輸出目錄
        metrics: 要繪製的指標 ["score_gap", "anomaly_ratio", "score_std"]
        figsize: 圖表大小
        
    Returns:
        儲存路徑
    """
    metrics = metrics or ["score_gap", "anomaly_ratio", "score_std"]
    summary = aggregator.get_metrics_summary()
    model_names = aggregator.get_model_names()
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    metric_labels = {
        "score_gap": "Score Gap (Anomaly - Normal)",
        "anomaly_ratio": "Anomaly Ratio",
        "score_std": "Score Std Dev",
        "score_mean": "Score Mean"
    }
    
    for ax, metric in zip(axes, metrics):
        for model_name in model_names:
            data = summary.get(model_name, [])
            if not data:
                continue
            
            sizes = [d["dataset_size"] for d in data]
            values = [d[metric] for d in data]
            color = MODEL_COLORS.get(model_name, "#7f8c8d")
            
            # * 繪製趨勢線與數據點
            ax.plot(sizes, values, "-o", label=model_name.replace("_", " ").title(),
                   color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel("Number of Datasets")
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(f"{metric_labels.get(metric, metric)} vs Data Scale")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "trend_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return output_path


def plot_anomaly_count_trend(
    aggregator: "ResultAggregator",
    output_dir: str = "result/unsupervised_anomaly_dection",
    figsize: tuple = (10, 6)
) -> str:
    """繪製異常數量趨勢圖
    
    Args:
        aggregator: 結果聚合器
        output_dir: 輸出目錄
        figsize: 圖表大小
        
    Returns:
        儲存路徑
    """
    summary = aggregator.get_metrics_summary()
    model_names = aggregator.get_model_names()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name in model_names:
        data = summary.get(model_name, [])
        if not data:
            continue
        
        sizes = [d["dataset_size"] for d in data]
        anomalies = [d["n_anomalies"] for d in data]
        color = MODEL_COLORS.get(model_name, "#7f8c8d")
        
        ax.plot(sizes, anomalies, "-o", label=model_name.replace("_", " ").title(),
               color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel("Number of Datasets")
    ax.set_ylabel("Number of Anomalies Detected")
    ax.set_title("Anomaly Detection Count vs Data Scale")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "anomaly_count_trend.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return output_path
