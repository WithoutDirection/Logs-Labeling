"""分佈演變圖模組

繪製正常與異常分數分佈的演變，展示模型區辨能力的變化。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualization.aggregator import ResultAggregator, AggregatedResult

# 顏色配置
NORMAL_COLOR = "#3498db"
ANOMALY_COLOR = "#e74c3c"
CMAP_NORMAL = plt.cm.Blues
CMAP_ANOMALY = plt.cm.Reds


def plot_score_histogram(
    results: Dict[str, any],
    output_dir: str = "result/unsupervised_anomaly_dection",
    title: str = "Anomaly Score Distribution"
) -> str:
    """繪製單次實驗的分數分佈直方圖
    
    Args:
        results: LogDetector.fit_predict() 的輸出
        output_dir: 輸出目錄
        title: 圖表標題
        
    Returns:
        儲存路徑
    """
    model_names = [k for k in results.keys() if k != "ensemble"]
    n_models = len(model_names) + 1
    
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        scores = results[model_name]["normalized_scores"]
        labels = results[model_name]["labels"]
        
        # * 繪製正常與異常分數直方圖
        ax.hist(scores[labels == 0], bins=50, alpha=0.7, label="Normal", color=NORMAL_COLOR)
        ax.hist(scores[labels == 1], bins=50, alpha=0.7, label="Anomaly", color=ANOMALY_COLOR)
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("Score")
        ax.legend()
    
    # Ensemble
    ax = axes[len(model_names)]
    scores = results["ensemble"]["scores"]
    labels = results["ensemble"]["labels"]
    ax.hist(scores[labels == 0], bins=50, alpha=0.7, label="Normal", color=NORMAL_COLOR)
    ax.hist(scores[labels == 1], bins=50, alpha=0.7, label="Anomaly", color=ANOMALY_COLOR)
    ax.set_title("Ensemble")
    ax.set_xlabel("Score")
    ax.legend()
    
    for j in range(len(model_names) + 1, len(axes)):
        axes[j].axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "score_histogram.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path


def plot_distribution_evolution(
    aggregator: "ResultAggregator",
    output_dir: str = "result/unsupervised_anomaly_dection",
    max_samples: int = 10000,
    figsize_per_model: tuple = (10, 6)
) -> List[str]:
    """繪製各模型分佈演變圖（Ridge Plot 風格）
    
    Args:
        aggregator: 結果聚合器
        output_dir: 輸出目錄
        max_samples: 每個規模的最大採樣數（加速 KDE）
        figsize_per_model: 每張圖的大小
        
    Returns:
        儲存路徑列表
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    
    model_names = aggregator.get_model_names()
    dataset_sizes = aggregator.get_dataset_sizes()
    n_sizes = len(dataset_sizes)
    
    for model_name in model_names:
        fig, ax = plt.subplots(figsize=figsize_per_model)
        
        model_results = aggregator.get_by_model(model_name)
        model_results = sorted(model_results, key=lambda x: x.dataset_size)
        
        for idx, result in enumerate(model_results):
            offset = idx * 0.8  # Y 軸偏移量
            
            # * 降採樣以加速 KDE
            scores = result.normalized_scores
            labels = result.labels
            if len(scores) > max_samples:
                indices = np.random.choice(len(scores), max_samples, replace=False)
                scores = scores[indices]
                labels = labels[indices]
            
            # * 計算 KDE
            x_range = np.linspace(0, 1, 200)
            
            normal_scores = scores[labels == 0]
            anomaly_scores = scores[labels == 1]
            
            if len(normal_scores) > 1:
                kde_normal = stats.gaussian_kde(normal_scores)
                density_normal = kde_normal(x_range)
                # * 繪製正常分佈（填充）
                ax.fill_between(x_range, offset, offset + density_normal * 0.3,
                              alpha=0.6, color=NORMAL_COLOR, label="Normal" if idx == 0 else "")
                ax.plot(x_range, offset + density_normal * 0.3, color=NORMAL_COLOR, linewidth=0.5)
            
            if len(anomaly_scores) > 1:
                kde_anomaly = stats.gaussian_kde(anomaly_scores)
                density_anomaly = kde_anomaly(x_range)
                # * 繪製異常分佈（填充）
                ax.fill_between(x_range, offset, offset + density_anomaly * 0.3,
                              alpha=0.6, color=ANOMALY_COLOR, label="Anomaly" if idx == 0 else "")
                ax.plot(x_range, offset + density_anomaly * 0.3, color=ANOMALY_COLOR, linewidth=0.5)
            
            # Y 軸標籤
            ax.text(-0.05, offset + 0.15, f"{result.dataset_size} sets",
                   ha="right", va="center", fontsize=9)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.2, n_sizes * 0.8 + 0.5)
        ax.set_xlabel("Normalized Score")
        ax.set_ylabel("Dataset Scale →")
        ax.set_title(f"{model_name.replace('_', ' ').title()} - Distribution Evolution")
        ax.set_yticks([])
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="x")
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"evolution_{model_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        output_paths.append(output_path)
    
    return output_paths


def plot_comparison_violin(
    aggregator: "ResultAggregator",
    output_dir: str = "result/unsupervised_anomaly_dection",
    figsize: tuple = (14, 8)
) -> str:
    """繪製各模型各規模的小提琴對比圖
    
    Args:
        aggregator: 結果聚合器
        output_dir: 輸出目錄
        figsize: 圖表大小
        
    Returns:
        儲存路徑
    """
    model_names = [m for m in aggregator.get_model_names() if m != "ensemble"]
    dataset_sizes = aggregator.get_dataset_sizes()
    
    fig, axes = plt.subplots(1, len(model_names), figsize=figsize, sharey=True)
    if len(model_names) == 1:
        axes = [axes]
    
    for ax, model_name in zip(axes, model_names):
        positions = []
        data_normal = []
        data_anomaly = []
        
        for i, size in enumerate(dataset_sizes):
            results = [r for r in aggregator.get_by_model(model_name) if r.dataset_size == size]
            if not results:
                continue
            
            r = results[0]
            positions.append(i)
            data_normal.append(r.normalized_scores[r.labels == 0])
            data_anomaly.append(r.normalized_scores[r.labels == 1])
        
        # * 繪製小提琴圖
        if data_normal:
            parts_n = ax.violinplot(data_normal, positions=positions, widths=0.4,
                                   showmeans=True, showextrema=False)
            for pc in parts_n["bodies"]:
                pc.set_facecolor(NORMAL_COLOR)
                pc.set_alpha(0.6)
        
        if data_anomaly:
            parts_a = ax.violinplot(data_anomaly, positions=[p + 0.4 for p in positions],
                                   widths=0.4, showmeans=True, showextrema=False)
            for pc in parts_a["bodies"]:
                pc.set_facecolor(ANOMALY_COLOR)
                pc.set_alpha(0.6)
        
        ax.set_xticks([p + 0.2 for p in positions])
        ax.set_xticklabels([str(s) for s in dataset_sizes])
        ax.set_xlabel("Number of Datasets")
        ax.set_title(model_name.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis="y")
    
    axes[0].set_ylabel("Normalized Score")
    
    # 圖例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=NORMAL_COLOR, alpha=0.6, label="Normal"),
        Patch(facecolor=ANOMALY_COLOR, alpha=0.6, label="Anomaly")
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 0.99))
    
    plt.suptitle("Score Distribution Comparison Across Scales", y=1.02)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comparison_violin.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return output_path
