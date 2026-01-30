"""
TF-IDF 詞彙覆蓋視覺化工具

功能說明：
    分析日誌詞彙與 MITRE ATT&CK 高權重詞彙的重疊情況，
    透過 Venn Diagram 與 Bar Chart 視覺化詞彙覆蓋率。

輸出：
    result/tfidf_coverage/
    ├── vocabulary_coverage_bar.png          # 詞彙覆蓋率長條圖
    ├── vocabulary_venn.png                  # 詞彙交集 Venn 圖
    ├── top_overlapping_terms.csv            # 高重疊詞彙清單
    ├── coverage_by_dataset.csv              # 各資料集覆蓋率統計
    └── coverage_summary.json                # 整體摘要

使用方式：
    # Pipeline 整合
    from visualization.tfidf_coverage import run_tfidf_coverage_analysis
    result = run_tfidf_coverage_analysis(top_n_mitre=500)
    
    # 命令列執行
    python tfidf_coverage.py --top-n 500
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 路徑設定
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(WORKSPACE_ROOT))

import config
from utils.path import ensure_dir

# 預設輸出目錄
OUTPUT_DIR = os.path.join("result", "tfidf_coverage")


# =============================================================================
# 核心分析函數
# =============================================================================

def _load_mitre_vectorizer() -> Tuple[Any, List[str]]:
    """載入 MITRE TF-IDF Vectorizer 並提取詞彙表"""
    tfidf_dir = getattr(config, 'MITRE_TFIDF_DIR', None)
    if not tfidf_dir:
        raise FileNotFoundError("MITRE_TFIDF_DIR 未設定")
    
    vec_path = os.path.join(tfidf_dir, "tfidf_vectorizer.pkl")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"找不到 Vectorizer: {vec_path}")
    
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    # 提取詞彙表
    vocabulary = vectorizer.get_feature_names_out()
    return vectorizer, list(vocabulary)


def _load_mitre_tfidf_matrix() -> Tuple[Any, np.ndarray]:
    """載入 MITRE TF-IDF 矩陣並計算每個詞彙的平均權重"""
    tfidf_dir = getattr(config, 'MITRE_TFIDF_DIR', None)
    mat_path = os.path.join(tfidf_dir, "mitre_tfidf_matrix.pkl")
    
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"找不到 TF-IDF 矩陣: {mat_path}")
    
    with open(mat_path, "rb") as f:
        tfidf_matrix = pickle.load(f)
    
    # 計算每個詞彙在所有文件中的平均權重
    mean_weights = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    return tfidf_matrix, mean_weights


def _get_mitre_top_terms(vocabulary: List[str], weights: np.ndarray, top_n: int = 500) -> set:
    """取得 MITRE 高權重詞彙"""
    indices = np.argsort(weights)[::-1][:top_n]
    return set(vocabulary[i] for i in indices)


def _extract_log_vocabulary(dataset_id: str) -> set:
    """從單一資料集提取日誌詞彙"""
    from precompute_log_tfidf import _find_source_csv, _extract_text
    
    csv_path = _find_source_csv(dataset_id)
    if not csv_path:
        return set()
    
    try:
        df = pd.read_csv(csv_path)
        texts = _extract_text(df)
        
        # 簡單分詞：空白分割 + 小寫
        all_words = set()
        for text in texts:
            words = text.lower().split()
            # 過濾過短的詞
            words = [w.strip('.,;:!?()[]{}') for w in words if len(w) > 2]
            all_words.update(words)
        
        return all_words
    except Exception:
        return set()


def _compute_coverage(log_vocab: set, mitre_terms: set) -> Dict[str, Any]:
    """計算詞彙覆蓋率"""
    overlap = log_vocab & mitre_terms
    
    return {
        "log_vocab_size": len(log_vocab),
        "mitre_terms_size": len(mitre_terms),
        "overlap_size": len(overlap),
        "log_coverage_ratio": len(overlap) / len(log_vocab) if log_vocab else 0,
        "mitre_coverage_ratio": len(overlap) / len(mitre_terms) if mitre_terms else 0,
        "overlap_terms": list(overlap)[:100],  # 只保留前 100 個
    }


# =============================================================================
# 視覺化函數
# =============================================================================

def plot_coverage_bar(
    dataset_coverages: Dict[str, Dict],
    output_path: str,
    figsize: Tuple[int, int] = (14, 8)
) -> str:
    """繪製各資料集詞彙覆蓋率長條圖"""
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    dataset_ids = list(dataset_coverages.keys())
    log_ratios = [d["log_coverage_ratio"] * 100 for d in dataset_coverages.values()]
    mitre_ratios = [d["mitre_coverage_ratio"] * 100 for d in dataset_coverages.values()]
    
    # 取前 20 個資料集顯示
    if len(dataset_ids) > 20:
        # 依 log_coverage_ratio 排序
        sorted_items = sorted(dataset_coverages.items(), 
                              key=lambda x: x[1]["log_coverage_ratio"], reverse=True)
        dataset_ids = [x[0][:15] for x in sorted_items[:20]]
        log_ratios = [x[1]["log_coverage_ratio"] * 100 for x in sorted_items[:20]]
        mitre_ratios = [x[1]["mitre_coverage_ratio"] * 100 for x in sorted_items[:20]]
    else:
        dataset_ids = [d[:15] for d in dataset_ids]
    
    x = np.arange(len(dataset_ids))
    
    # Upper: Log vocab coverage in MITRE
    bars1 = axes[0].bar(x, log_ratios, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Coverage (%)', fontsize=12)
    axes[0].set_title('Log Vocabulary Coverage in MITRE Top Terms', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dataset_ids, rotation=45, ha='right', fontsize=9)
    axes[0].axhline(y=np.mean(log_ratios), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(log_ratios):.1f}%')
    axes[0].legend()
    
    # 加入數值標籤
    for bar, val in zip(bars1, log_ratios):
        height = bar.get_height()
        axes[0].annotate(f'{val:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
    
    # Lower: MITRE vocab coverage by Logs
    bars2 = axes[1].bar(x, mitre_ratios, color='darkorange', alpha=0.8)
    axes[1].set_ylabel('Coverage (%)', fontsize=12)
    axes[1].set_xlabel('Dataset', fontsize=12)
    axes[1].set_title('MITRE Top Terms Coverage by Log Vocabulary', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dataset_ids, rotation=45, ha='right', fontsize=9)
    axes[1].axhline(y=np.mean(mitre_ratios), color='red', linestyle='--',
                    label=f'Mean: {np.mean(mitre_ratios):.1f}%')
    axes[1].legend()
    
    for bar, val in zip(bars2, mitre_ratios):
        height = bar.get_height()
        axes[1].annotate(f'{val:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_venn_diagram(
    log_vocab: set,
    mitre_terms: set,
    output_path: str,
    figsize: Tuple[int, int] = (10, 8)
) -> str:
    """繪製詞彙交集 Venn 圖（使用 matplotlib 手繪）"""
    try:
        from matplotlib_venn import venn2, venn2_circles
        has_venn = True
    except ImportError:
        has_venn = False
    
    fig, ax = plt.subplots(figsize=figsize)
    
    overlap = log_vocab & mitre_terms
    log_only = len(log_vocab - mitre_terms)
    mitre_only = len(mitre_terms - log_vocab)
    both = len(overlap)
    
    if has_venn:
        # 使用 matplotlib-venn
        v = venn2(
            subsets=(log_only, mitre_only, both),
            set_labels=('Log Vocabulary', 'MITRE Top Terms'),
            set_colors=('steelblue', 'darkorange'),
            alpha=0.6,
            ax=ax
        )
        venn2_circles(subsets=(log_only, mitre_only, both), linestyle='solid', linewidth=2, ax=ax)
        
        # 調整標籤
        for text in v.set_labels:
            if text:
                text.set_fontsize(14)
                text.set_fontweight('bold')
        for text in v.subset_labels:
            if text:
                text.set_fontsize(12)
    else:
        # 手繪圓形 Venn
        from matplotlib.patches import Circle
        
        circle1 = Circle((0.35, 0.5), 0.3, alpha=0.6, color='steelblue', label='Log 詞彙')
        circle2 = Circle((0.65, 0.5), 0.3, alpha=0.6, color='darkorange', label='MITRE 高權重詞彙')
        
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        ax.text(0.2, 0.5, f'{log_only:,}', fontsize=14, ha='center', va='center')
        ax.text(0.5, 0.5, f'{both:,}', fontsize=14, ha='center', va='center', fontweight='bold')
        ax.text(0.8, 0.5, f'{mitre_only:,}', fontsize=14, ha='center', va='center')
        
        ax.text(0.2, 0.85, 'Log Vocab', fontsize=12, ha='center', fontweight='bold')
        ax.text(0.8, 0.85, 'MITRE Terms', fontsize=12, ha='center', fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # 標題與統計
    total_log = len(log_vocab)
    total_mitre = len(mitre_terms)
    log_coverage = (both / total_log * 100) if total_log else 0
    mitre_coverage = (both / total_mitre * 100) if total_mitre else 0
    
    ax.set_title(
        f'Vocabulary Coverage Analysis (All Datasets)\n'
        f'Log->MITRE: {log_coverage:.1f}% | MITRE->Log: {mitre_coverage:.1f}%',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_top_overlapping_terms(
    overlap_terms: List[str],
    vocabulary: List[str],
    weights: np.ndarray,
    output_path: str,
    top_n: int = 30,
    figsize: Tuple[int, int] = (12, 10)
) -> str:
    """繪製高重疊詞彙的 TF-IDF 權重分佈"""
    # 建立詞彙到權重的映射
    vocab_to_weight = {v: w for v, w in zip(vocabulary, weights)}
    
    # 計算重疊詞彙的權重
    term_weights = [(term, vocab_to_weight.get(term, 0)) for term in overlap_terms]
    term_weights = sorted(term_weights, key=lambda x: x[1], reverse=True)[:top_n]
    
    if not term_weights:
        return None
    
    terms = [t[0] for t in term_weights]
    weights_vals = [t[1] for t in term_weights]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(terms))
    bars = ax.barh(y_pos, weights_vals, color='teal', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('MITRE TF-IDF Mean Weight', fontsize=12)
    ax.set_title(f'TF-IDF Weights of Overlapping Terms (Top {top_n})', fontsize=14, fontweight='bold')
    
    # 數值標籤
    for bar, val in zip(bars, weights_vals):
        width = bar.get_width()
        ax.annotate(f'{val:.4f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


# =============================================================================
# Pipeline API
# =============================================================================

def run_tfidf_coverage_analysis(
    top_n_mitre: int = 500,
    output_dir: str = OUTPUT_DIR,
    verbose: bool = True,
    max_datasets: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Pipeline API：執行 TF-IDF 詞彙覆蓋率分析
    
    分析日誌詞彙與 MITRE ATT&CK 高權重詞彙的重疊情況，
    生成 Venn Diagram 與 Bar Chart 視覺化。
    
    Args:
        top_n_mitre: 取 MITRE 前 N 個高權重詞彙進行比較
        output_dir: 輸出目錄
        verbose: 是否輸出詳細資訊
        max_datasets: 最大處理資料集數（None = 全部）
        
    Returns:
        dict: {
            "n_datasets": 處理的資料集數,
            "avg_log_coverage": 平均日誌覆蓋率,
            "avg_mitre_coverage": 平均 MITRE 覆蓋率,
            "total_overlap_terms": 總重疊詞彙數,
            "output_dir": 輸出目錄路徑,
        }
    """
    ensure_dir(output_dir)
    
    if verbose:
        print("\n" + "=" * 60)
        print("TF-IDF 詞彙覆蓋率分析")
        print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: 載入 MITRE TF-IDF 資料
    # -------------------------------------------------------------------------
    try:
        vectorizer, vocabulary = _load_mitre_vectorizer()
        _, mean_weights = _load_mitre_tfidf_matrix()
        mitre_top_terms = _get_mitre_top_terms(np.array(vocabulary), mean_weights, top_n_mitre)
        
        if verbose:
            print(f"  MITRE 詞彙表大小: {len(vocabulary)}")
            print(f"  選取 Top-{top_n_mitre} 高權重詞彙")
    except FileNotFoundError as e:
        if verbose:
            print(f"  [Error] {e}")
        return {"error": str(e), "enabled": False}
    
    # -------------------------------------------------------------------------
    # Step 2: 遍歷資料集提取詞彙
    # -------------------------------------------------------------------------
    embeddings_dir = config.LOG_VECTORS_DIR
    if not os.path.exists(embeddings_dir):
        return {"error": f"找不到目錄: {embeddings_dir}", "enabled": False}
    
    subdirs = [d for d in os.listdir(embeddings_dir) 
               if os.path.isdir(os.path.join(embeddings_dir, d))]
    
    if max_datasets:
        subdirs = subdirs[:max_datasets]
    
    if verbose:
        print(f"  處理資料集: {len(subdirs)} 個\n")
    
    dataset_coverages = {}
    all_log_vocab = set()
    
    from tqdm import tqdm
    iterator = tqdm(subdirs, desc="分析詞彙覆蓋", disable=not verbose)
    
    for subdir in iterator:
        dataset_id = subdir.replace("_embeddings", "").replace("_logvectors", "")
        log_vocab = _extract_log_vocabulary(dataset_id)
        
        if log_vocab:
            coverage = _compute_coverage(log_vocab, mitre_top_terms)
            dataset_coverages[dataset_id] = coverage
            all_log_vocab.update(log_vocab)
    
    if not dataset_coverages:
        return {"error": "無法提取任何資料集的詞彙", "enabled": False}
    
    # -------------------------------------------------------------------------
    # Step 3: 計算整體統計
    # -------------------------------------------------------------------------
    total_overlap = all_log_vocab & mitre_top_terms
    avg_log_coverage = np.mean([d["log_coverage_ratio"] for d in dataset_coverages.values()])
    avg_mitre_coverage = np.mean([d["mitre_coverage_ratio"] for d in dataset_coverages.values()])
    
    if verbose:
        print(f"\n  整體統計:")
        print(f"    總日誌詞彙: {len(all_log_vocab):,}")
        print(f"    MITRE 高權重詞彙: {len(mitre_top_terms):,}")
        print(f"    重疊詞彙: {len(total_overlap):,}")
        print(f"    平均 Log→MITRE 覆蓋率: {avg_log_coverage * 100:.1f}%")
        print(f"    平均 MITRE→Log 覆蓋率: {avg_mitre_coverage * 100:.1f}%")
    
    # -------------------------------------------------------------------------
    # Step 4: 生成視覺化
    # -------------------------------------------------------------------------
    if verbose:
        print("\n  生成視覺化...")
    
    # Bar Chart
    bar_path = os.path.join(output_dir, "vocabulary_coverage_bar.png")
    plot_coverage_bar(dataset_coverages, bar_path)
    
    # Venn Diagram
    venn_path = os.path.join(output_dir, "vocabulary_venn.png")
    plot_venn_diagram(all_log_vocab, mitre_top_terms, venn_path)
    
    # Top overlapping terms
    terms_path = os.path.join(output_dir, "top_overlapping_terms.png")
    plot_top_overlapping_terms(
        list(total_overlap), 
        vocabulary, 
        mean_weights, 
        terms_path,
        top_n=30
    )
    
    # -------------------------------------------------------------------------
    # Step 5: 儲存統計資料
    # -------------------------------------------------------------------------
    # CSV: 各資料集覆蓋率
    coverage_df = pd.DataFrame([
        {
            "dataset_id": k,
            "log_vocab_size": v["log_vocab_size"],
            "overlap_size": v["overlap_size"],
            "log_coverage_ratio": v["log_coverage_ratio"],
            "mitre_coverage_ratio": v["mitre_coverage_ratio"],
        }
        for k, v in dataset_coverages.items()
    ])
    coverage_df.to_csv(os.path.join(output_dir, "coverage_by_dataset.csv"), index=False)
    
    # CSV: 高重疊詞彙
    vocab_to_weight = {v: w for v, w in zip(vocabulary, mean_weights)}
    overlap_df = pd.DataFrame([
        {"term": term, "mitre_weight": vocab_to_weight.get(term, 0)}
        for term in total_overlap
    ]).sort_values("mitre_weight", ascending=False)
    overlap_df.to_csv(os.path.join(output_dir, "top_overlapping_terms.csv"), index=False)
    
    # JSON: 摘要
    summary = {
        "n_datasets": len(dataset_coverages),
        "total_log_vocab": len(all_log_vocab),
        "mitre_top_n": top_n_mitre,
        "total_overlap_terms": len(total_overlap),
        "avg_log_coverage": avg_log_coverage,
        "avg_mitre_coverage": avg_mitre_coverage,
        "output_files": {
            "bar_chart": bar_path,
            "venn_diagram": venn_path,
            "terms_chart": terms_path,
        }
    }
    with open(os.path.join(output_dir, "coverage_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"\n  輸出目錄: {output_dir}")
        print("  ✓ vocabulary_coverage_bar.png")
        print("  ✓ vocabulary_venn.png")
        print("  ✓ top_overlapping_terms.png")
        print("  ✓ coverage_by_dataset.csv")
        print("  ✓ top_overlapping_terms.csv")
        print("  ✓ coverage_summary.json")
    
    return {
        "n_datasets": len(dataset_coverages),
        "avg_log_coverage": avg_log_coverage,
        "avg_mitre_coverage": avg_mitre_coverage,
        "total_overlap_terms": len(total_overlap),
        "output_dir": output_dir,
        "enabled": True,
    }


# =============================================================================
# 命令列入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TF-IDF 詞彙覆蓋率分析")
    parser.add_argument("--top-n", type=int, default=500,
                        help="取 MITRE 前 N 個高權重詞彙 (預設: 500)")
    parser.add_argument("--max-datasets", type=int, default=None,
                        help="最大處理資料集數 (預設: 全部)")
    parser.add_argument("-o", "--output-dir", type=str, default=OUTPUT_DIR,
                        help="輸出目錄")
    
    args = parser.parse_args()
    
    result = run_tfidf_coverage_analysis(
        top_n_mitre=args.top_n,
        output_dir=args.output_dir,
        verbose=True,
        max_datasets=args.max_datasets,
    )
    
    if result.get("enabled", False):
        print(f"\n完成！平均覆蓋率: Log→MITRE={result['avg_log_coverage']*100:.1f}%, "
              f"MITRE→Log={result['avg_mitre_coverage']*100:.1f}%")
    else:
        print(f"\n失敗: {result.get('error', '未知錯誤')}")
