"""異常偵測視覺化與分析整合模組

整合所有異常偵測相關的視覺化功能，包括：
- 相關性分析
- 分數分布圖
- 趨勢分析
- 結果聚合
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# * 調整匯入路徑
CURRENT_DIR = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(CURRENT_DIR).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.path import join_path, ensure_dir
import config

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

# ==================== 顏色配置 ====================
NORMAL_COLOR = "#3498db"
ANOMALY_COLOR = "#e74c3c"
MODEL_COLORS = {
    "isolation_forest": "#2ecc71",
    "copod": "#3498db",
    "autoencoder": "#9b59b6",
    "pca_gmm": "#e74c3c",
    "ensemble": "#34495e"
}


# ==================== 結果聚合器 ====================

@dataclass
class AggregatedResult:
    """聚合後的實驗結果"""
    model_name: str
    dataset_size: int
    normalized_scores: np.ndarray
    labels: np.ndarray
    raw_scores: Optional[np.ndarray] = None
    
    @property
    def n_samples(self) -> int:
        return len(self.labels)
    
    @property
    def n_anomalies(self) -> int:
        return int(self.labels.sum())
    
    @property
    def anomaly_ratio(self) -> float:
        return self.n_anomalies / self.n_samples if self.n_samples > 0 else 0.0
    
    @property
    def score_gap(self) -> float:
        """正常與異常分數間距"""
        if self.n_anomalies == 0 or self.n_anomalies == self.n_samples:
            return 0.0
        normal_mean = self.normalized_scores[self.labels == 0].mean()
        anomaly_mean = self.normalized_scores[self.labels == 1].mean()
        return anomaly_mean - normal_mean


class ResultAggregator:
    """多規模實驗結果聚合器"""
    
    def __init__(self, scaler_type: str = "minmax"):
        self.scaler_type = scaler_type
        self.results: List[AggregatedResult] = []
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """將分數正規化至 [0, 1]"""
        if self.scaler_type == "robust":
            scaler = RobustScaler()
            scaled = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
            return np.clip(scaled, 0, 1)
        else:
            min_val, max_val = scores.min(), scores.max()
            if max_val - min_val == 0:
                return np.zeros_like(scores)
            return (scores - min_val) / (max_val - min_val)
    
    def add_experiment(self, results: Dict[str, Any], dataset_size: int, normalize: bool = True) -> None:
        """加入一次實驗結果"""
        for model_name, model_results in results.items():
            if model_name == "ensemble":
                scores = model_results["scores"]
                raw_scores = None
            else:
                scores = model_results["normalized_scores"]
                raw_scores = model_results.get("raw_scores")
            
            normalized = self._normalize(scores) if normalize else scores
            
            self.results.append(AggregatedResult(
                model_name=model_name,
                dataset_size=dataset_size,
                normalized_scores=normalized,
                labels=model_results["labels"],
                raw_scores=raw_scores
            ))
    
    def get_by_model(self, model_name: str) -> List[AggregatedResult]:
        """取得特定模型的所有結果"""
        return [r for r in self.results if r.model_name == model_name]
    
    def get_by_size(self, dataset_size: int) -> List[AggregatedResult]:
        """取得特定規模的所有結果"""
        return [r for r in self.results if r.dataset_size == dataset_size]
    
    def get_model_names(self) -> List[str]:
        """取得所有模型名稱"""
        return list(set(r.model_name for r in self.results))
    
    def get_dataset_sizes(self) -> List[int]:
        """取得所有資料集規模（已排序）"""
        return sorted(set(r.dataset_size for r in self.results))
    
    def get_metrics_summary(self) -> Dict[str, List[Dict[str, Any]]]:
        """取得各模型各規模的指標摘要"""
        summary = {}
        for model_name in self.get_model_names():
            model_metrics = []
            for r in self.get_by_model(model_name):
                model_metrics.append({
                    "dataset_size": r.dataset_size,
                    "n_samples": r.n_samples,
                    "n_anomalies": r.n_anomalies,
                    "anomaly_ratio": r.anomaly_ratio,
                    "score_gap": r.score_gap,
                    "score_mean": r.normalized_scores.mean(),
                    "score_std": r.normalized_scores.std()
                })
            summary[model_name] = sorted(model_metrics, key=lambda x: x["dataset_size"])
        return summary


# ==================== 基礎視覺化函數 ====================


def plot_global_correlation_heatmap(corr_matrix: pd.DataFrame, output_dir: str, name: str = "all_datasets") -> None:
    """生成單一全域相關性熱圖"""
    if corr_matrix is None or corr_matrix.empty:
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1
    )

    plt.title(
        f'Model Score Correlation: {name}\n({config.CORRELATION_METHOD.capitalize()})',
        fontsize=14,
        pad=20
    )
    plt.tight_layout()

    save_path = join_path(output_dir, f"{name}_correlation.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"相關性熱圖已儲存: {save_path}")


def plot_anomaly_scores_distribution(
    scores: Dict[str, list],
    labels: list,
    dataset_name: str,
    output_dir: str
) -> None:
    """繪製異常分數分布圖
    
    顯示各模型的分數分布，並標示異常/正常樣本。
    
    Args:
        scores: 各模型分數字典 {"model_name": [scores]}
        labels: 異常標籤列表 (0: 正常, 1: 異常)
        dataset_name: 資料集名稱
        output_dir: 輸出目錄
    """
    n_models = len(scores)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, score_values) in zip(axes, scores.items()):
        # 分離正常和異常樣本
        normal_scores = [s for s, l in zip(score_values, labels) if l == 0]
        anomaly_scores = [s for s, l in zip(score_values, labels) if l == 1]
        
        # 繪製直方圖
        ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue')
        ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red')
        
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model_name} Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = join_path(output_dir, f"{dataset_name}_score_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"分數分布圖已儲存: {save_path}")


def plot_ensemble_comparison(
    results: Dict[str, pd.DataFrame],
    output_dir: str,
    metric: str = "anomaly_ratio"
) -> None:
    """比較不同資料集的 Ensemble 偵測結果
    
    Args:
        results: 資料集名稱到結果 DataFrame 的映射
        output_dir: 輸出目錄
        metric: 比較指標 ("anomaly_ratio" | "anomaly_count")
    """
    dataset_names = []
    metrics_data = []
    
    for name, df in results.items():
        dataset_names.append(name)
        if metric == "anomaly_ratio":
            ratio = df['ensemble_label'].sum() / len(df) * 100
            metrics_data.append(ratio)
        else:
            metrics_data.append(df['ensemble_label'].sum())
    
    plt.figure(figsize=(12, 6))
    plt.bar(dataset_names, metrics_data, color='steelblue', alpha=0.7)
    plt.xlabel('Dataset')
    plt.ylabel('Anomaly Ratio (%)' if metric == "anomaly_ratio" else 'Anomaly Count')
    plt.title(f'Ensemble Detection Results Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    save_path = join_path(output_dir, f"ensemble_comparison_{metric}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Ensemble 比較圖已儲存: {save_path}")


# ==================== 分數分布視覺化 ====================

def plot_score_histogram(
    results: Dict[str, Any],
    output_dir: str = "result/unsupervised_anomaly_dection",
    title: str = "Anomaly Score Distribution"
) -> str:
    """繪製單次實驗的分數分布直方圖"""
    model_names = [k for k in results.keys() if k != "ensemble"]
    n_models = len(model_names) + 1
    
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        scores = results[model_name]["normalized_scores"]
        labels = results[model_name]["labels"]
        
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
    aggregator: ResultAggregator,
    output_dir: str = "result/unsupervised_anomaly_dection",
    max_samples: int = 10000
) -> List[str]:
    """繪製各模型分布演變圖（Ridge Plot 風格）"""
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    
    model_names = aggregator.get_model_names()
    dataset_sizes = aggregator.get_dataset_sizes()
    n_sizes = len(dataset_sizes)
    
    for model_name in model_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        model_results = sorted(aggregator.get_by_model(model_name), key=lambda x: x.dataset_size)
        
        for idx, result in enumerate(model_results):
            offset = idx * 0.8
            
            # 降採樣加速 KDE
            scores, labels = result.normalized_scores, result.labels
            if len(scores) > max_samples:
                indices = np.random.choice(len(scores), max_samples, replace=False)
                scores, labels = scores[indices], labels[indices]
            
            x_range = np.linspace(0, 1, 200)
            normal_scores = scores[labels == 0]
            anomaly_scores = scores[labels == 1]
            
            # KDE 需要有變異的樣本；若分數全相同會導致奇異矩陣錯誤
            if len(normal_scores) > 1 and np.std(normal_scores) > 0:
                kde_normal = stats.gaussian_kde(normal_scores)
                density_normal = kde_normal(x_range)
                ax.fill_between(x_range, offset, offset + density_normal * 0.3,
                              alpha=0.6, color=NORMAL_COLOR, label="Normal" if idx == 0 else "")
                ax.plot(x_range, offset + density_normal * 0.3, color=NORMAL_COLOR, linewidth=0.5)
            
            if len(anomaly_scores) > 1 and np.std(anomaly_scores) > 0:
                kde_anomaly = stats.gaussian_kde(anomaly_scores)
                density_anomaly = kde_anomaly(x_range)
                ax.fill_between(x_range, offset, offset + density_anomaly * 0.3,
                              alpha=0.6, color=ANOMALY_COLOR, label="Anomaly" if idx == 0 else "")
                ax.plot(x_range, offset + density_anomaly * 0.3, color=ANOMALY_COLOR, linewidth=0.5)
            
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
    aggregator: ResultAggregator,
    output_dir: str = "result/unsupervised_anomaly_dection"
) -> str:
    """繪製各模型各規模的小提琴對比圖"""
    model_names = [m for m in aggregator.get_model_names() if m != "ensemble"]
    dataset_sizes = aggregator.get_dataset_sizes()
    
    fig, axes = plt.subplots(1, len(model_names), figsize=(14, 8), sharey=True)
    if len(model_names) == 1:
        axes = [axes]
    
    for ax, model_name in zip(axes, model_names):
        positions_normal, data_normal = [], []
        positions_anomaly, data_anomaly = [], []
        
        for i, size in enumerate(dataset_sizes):
            results = [r for r in aggregator.get_by_model(model_name) if r.dataset_size == size]
            if not results:
                continue
            
            r = results[0]
            normal_vals = r.normalized_scores[r.labels == 0]
            anomaly_vals = r.normalized_scores[r.labels == 1]
            
            positions_normal.append(i)
            data_normal.append(normal_vals)
            if len(anomaly_vals) > 0:
                positions_anomaly.append(i + 0.4)
                data_anomaly.append(anomaly_vals)
        
        if data_normal:
            parts_n = ax.violinplot(data_normal, positions=positions_normal, widths=0.4,
                                   showmeans=True, showextrema=False)
            for pc in parts_n["bodies"]:
                pc.set_facecolor(NORMAL_COLOR)
                pc.set_alpha(0.6)
        
        if data_anomaly:
            parts_a = ax.violinplot(data_anomaly, positions=positions_anomaly,
                                   widths=0.4, showmeans=True, showextrema=False)
            for pc in parts_a["bodies"]:
                pc.set_facecolor(ANOMALY_COLOR)
                pc.set_alpha(0.6)
        
        tick_positions = positions_normal
        ax.set_xticks([p + 0.2 for p in tick_positions])
        ax.set_xticklabels([str(dataset_sizes[p]) for p in tick_positions])
        ax.set_xlabel("Number of Datasets")
        ax.set_title(model_name.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis="y")
    
    axes[0].set_ylabel("Normalized Score")
    
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


# ==================== 趨勢分析視覺化 ====================

def plot_trend_analysis(
    aggregator: ResultAggregator,
    output_dir: str = "result/unsupervised_anomaly_dection",
    metrics: List[str] = None
) -> str:
    """繪製效能趨勢折線圖"""
    metrics = metrics or ["score_gap", "anomaly_ratio", "score_std"]
    summary = aggregator.get_metrics_summary()
    model_names = aggregator.get_model_names()
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
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
    aggregator: ResultAggregator,
    output_dir: str = "result/unsupervised_anomaly_dection"
) -> str:
    """繪製異常數量趨勢圖"""
    summary = aggregator.get_metrics_summary()
    model_names = aggregator.get_model_names()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
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


# ==================== 完整摘要生成 ====================

def generate_detection_summary(
    results: Dict[str, Any], 
    output_dir: str = None, 
    generate_visualizations: bool = True,
    enable_advanced_plots: bool = True
) -> None:
    """生成偵測結果摘要與視覺化
    
    整合多種視覺化方法，包括：
    - 相關性熱圖
    - 分數分布圖
    - Ensemble 比較
    - 多規模趨勢分析（需多個資料集）
    - 小提琴對比圖（需多個資料集）
    
    Args:
        results: run_detection_pipeline 的返回結果
        output_dir: 輸出目錄
        generate_visualizations: 是否生成基礎視覺化（相關性、分布圖）
        enable_advanced_plots: 是否生成進階視覺化（趨勢、演變圖）
    """
    output_dir = output_dir or "result/unsupervised_anomaly_dection"
    ensure_dir(output_dir)
    
    if not results:
        print("警告：沒有可用的偵測結果")
        return
    
    print("\n" + "=" * 80)
    print("生成偵測結果摘要與視覺化")
    print("=" * 80)
    
    summary_data = []
    all_correlations = {}
    combined_scores = {}
    all_scores_dict = {}
    all_results_dict = {}
    
    # ========== 階段 1: 收集資料 ==========
    print("\n[階段 1/4] 收集並統計資料...")
    
    for dataset_name, dataset in results.items():
        n_samples = len(dataset)
        row = {"dataset": dataset_name, "n_samples": n_samples}
        
        # 收集模型分數與標籤
        model_scores = {}
        model_labels = {}
        model_results = {}
        
        for col in dataset.column_names:
            # 跳過原始分數欄位（*_raw_score），僅使用正規化後的分數
            if col.endswith("_score") and not col.startswith("ensemble"):
                if col.endswith("_raw_score"):
                    continue
                model_name = col.replace("_score", "")
                model_scores[model_name] = dataset[col]
            
            if col.endswith("_label"):
                model_name = col.replace("_label", "")
                model_labels[model_name] = dataset[col]
                n_anomalies = sum(dataset[col])
                row[f"{model_name}_anomalies"] = n_anomalies
                row[f"{model_name}_ratio"] = n_anomalies / n_samples * 100 if n_samples > 0 else 0
        
        # 組裝結果字典供視覺化使用
        for model_name in model_scores.keys():
            if model_name not in model_results:
                model_results[model_name] = {
                    "normalized_scores": np.array(model_scores[model_name]),
                    "labels": np.array(model_labels.get(model_name, []))
                }
            combined_scores.setdefault(model_name, []).extend(model_scores[model_name])
        
        # 加入 ensemble 結果
        if "ensemble_score" in dataset.column_names:
            model_results["ensemble"] = {
                "scores": np.array(dataset["ensemble_score"]),
                "labels": np.array(dataset["ensemble_label"])
            }
        
        summary_data.append(row)
        all_scores_dict[dataset_name] = model_scores
        all_results_dict[dataset_name] = model_results
        
        # 計算相關性矩陣
        if len(model_scores) >= 2:
            scores_df = pd.DataFrame(model_scores)
            corr_method = getattr(config, 'CORRELATION_METHOD', 'pearson')
            corr_matrix = scores_df.corr(method=corr_method)
            all_correlations[dataset_name] = corr_matrix

    global_corr_matrix = None
    if len(combined_scores) >= 2:
        combined_df = pd.DataFrame(combined_scores)
        corr_method = getattr(config, 'CORRELATION_METHOD', 'pearson')
        global_corr_matrix = combined_df.corr(method=corr_method)
    
    print(f"✓ 已收集 {len(results)} 個資料集的結果")
    
    # ========== 階段 2: 基礎視覺化 ==========
    if generate_visualizations:
        print("\n[階段 2/4] 生成基礎視覺化...")
        
        # 1. 相關性熱圖
        if global_corr_matrix is not None:
            print("  ├─ 生成相關性熱圖 (全資料集)...")
            plot_global_correlation_heatmap(global_corr_matrix, output_dir, name="all_datasets")
        
        # 2. 各資料集的分數分布圖
        print("  ├─ 生成分數分布圖...")
        for dataset_name, model_results in all_results_dict.items():
            if model_results:
                scores_dict = {k: v["normalized_scores"] for k, v in model_results.items() if k != "ensemble"}
                if "ensemble" in model_results:
                    labels = model_results["ensemble"]["labels"]
                    plot_anomaly_scores_distribution(
                        scores_dict,
                        labels.tolist(),
                        dataset_name,
                        output_dir
                    )
        
        # 3. Ensemble 比較圖
        if len(results) > 1:
            print("  ├─ 生成 Ensemble 比較圖...")
            ensemble_dfs = {}
            for dataset_name, dataset in results.items():
                if "ensemble_label" in dataset.column_names:
                    df = pd.DataFrame({
                        "ensemble_label": dataset["ensemble_label"]
                    })
                    ensemble_dfs[dataset_name] = df
            
            if ensemble_dfs:
                plot_ensemble_comparison(ensemble_dfs, output_dir, metric="anomaly_ratio")
                plot_ensemble_comparison(ensemble_dfs, output_dir, metric="anomaly_count")
        
        print("  └─ 基礎視覺化完成")
    
    # ========== 階段 3: 進階視覺化（多規模分析）==========
    if enable_advanced_plots and len(results) >= 2:
        print("\n[階段 3/4] 生成進階視覺化（多規模分析）...")
        
        # 使用 ResultAggregator 聚合結果
        aggregator = ResultAggregator(scaler_type="minmax")
        
        dataset_sizes = sorted(enumerate(results.keys()), key=lambda x: len(results[x[1]]))
        for idx, dataset_name in dataset_sizes:
            dataset = results[dataset_name]
            dataset_size = idx + 1  # 使用序號作為規模標記
            
            model_results = all_results_dict.get(dataset_name, {})
            if model_results:
                aggregator.add_experiment(model_results, dataset_size=dataset_size, normalize=False)
        
        print(f"  ├─ 已聚合 {len(results)} 個規模的實驗結果")
        
        # 4. 單個資料集的分數直方圖
        first_dataset_name = list(all_results_dict.keys())[0]
        first_results = all_results_dict[first_dataset_name]
        if first_results:
            print("  ├─ 生成分數直方圖（示例資料集）...")
            plot_score_histogram(
                first_results,
                output_dir,
                title=f"Anomaly Score Distribution - {first_dataset_name}"
            )
        
        # 5. 分布演變圖（Ridge Plot）
        print("  ├─ 生成分布演變圖...")
        evolution_paths = plot_distribution_evolution(aggregator, output_dir)
        print(f"     └─ 已生成 {len(evolution_paths)} 個模型的演變圖")
        
        # 6. 小提琴對比圖
        print("  ├─ 生成小提琴對比圖...")
        plot_comparison_violin(aggregator, output_dir)
        
        # 7. 趨勢分析圖
        print("  ├─ 生成趨勢分析圖...")
        plot_trend_analysis(aggregator, output_dir, metrics=["score_gap", "anomaly_ratio", "score_std"])
        
        # 8. 異常數量趨勢圖
        print("  ├─ 生成異常數量趨勢圖...")
        plot_anomaly_count_trend(aggregator, output_dir)
        
        print("  └─ 進階視覺化完成")
    elif enable_advanced_plots:
        print("\n[階段 3/4] 跳過進階視覺化（需要至少 2 個資料集）")
    
    # ========== 階段 4: 文字摘要 ==========
    print("\n[階段 4/4] 生成文字摘要...")
    print("\n" + "=" * 80)
    print("偵測結果摘要")
    print("=" * 80)
    
    for row in summary_data:
        print(f"\n資料集: {row['dataset']}")
        print(f"  樣本數: {row['n_samples']}")
        
        # 按模型分組顯示
        model_stats = {}
        for key, value in row.items():
            if key.endswith("_ratio"):
                model = key.replace("_ratio", "")
                if model not in model_stats:
                    model_stats[model] = {}
                model_stats[model]["ratio"] = value
            elif key.endswith("_anomalies"):
                model = key.replace("_anomalies", "")
                if model not in model_stats:
                    model_stats[model] = {}
                model_stats[model]["count"] = value
        
        for model, stats in sorted(model_stats.items()):
            count = stats.get("count", 0)
            ratio = stats.get("ratio", 0)
            print(f"  {model:20s}: {count:5d} 異常 ({ratio:6.2f}%)")
    
    print("\n" + "=" * 80)
    print(f"所有視覺化結果已儲存至: {output_dir}")
    print("=" * 80)


    
    