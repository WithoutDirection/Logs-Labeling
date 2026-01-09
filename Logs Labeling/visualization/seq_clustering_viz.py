"""
序列分群視覺化模組

生成四類視覺化輸出：
1. 時序甘特圖 (State Timeline) - 展示攻擊階段隨時間演變
2. 狀態語義熱力圖 (State-Concept Heatmap) - 解釋各狀態語義
3. 狀態轉移矩陣圖 (Transition Matrix) - 展示攻擊路徑邏輯
4. 綜合分析表格 (Summary Report) - CSV 格式分析報告
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from sequence_clustering import load_concept_vectors


# ======================== 輸出路徑 ========================
OUTPUT_DIR = os.path.join(config.RESULT_DIR, "sequence_clustering")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================== 資料載入 ========================

def load_cluster_results(dataset_id: str) -> dict:
    """載入單一資料集的分群結果（模型 + 標籤）"""
    base_dir = os.path.join(config.CLUSTER_RESULTS_DIR, dataset_id)
    
    labels_path = os.path.join(base_dir, "labels.npy")
    model_path = os.path.join(base_dir, "model.pkl")
    
    if not os.path.exists(labels_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到分群結果: {base_dir}")
    
    labels = np.load(labels_path)
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    return {
        "labels": labels,
        "model": model_data["model"],
        "best_k": model_data["best_k"],
        "best_score": model_data["best_score"],
    }


# ======================== 視覺化函式 ========================

def plot_timeline(labels: np.ndarray, dataset_id: str, save_path: str):
    """
    時序甘特圖：展示隱藏狀態隨時間的變化
    X 軸為時間索引，顏色區塊代表不同狀態
    """
    fig, ax = plt.subplots(figsize=(15, 2))
    cmap = plt.get_cmap("tab10")
    
    # 繪製連續色塊（效率優化：合併相同狀態區段）
    current_state = labels[0]
    start_idx = 0
    
    for i, state in enumerate(labels):
        if state != current_state or i == len(labels) - 1:
            end_idx = i if state != current_state else i + 1
            ax.axvspan(start_idx, end_idx, color=cmap(current_state % 10), alpha=0.8)
            current_state = state
            start_idx = i
    
    # 圖例
    unique_states = np.unique(labels)
    patches = [mpatches.Patch(color=cmap(s % 10), label=f"State {s}") for s in unique_states]
    ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1, 0.5))
    
    ax.set_xlabel("Time (Log Index)")
    ax.set_yticks([])
    ax.set_xlim(0, len(labels))
    ax.set_title(f"Attack Stage Timeline - {dataset_id}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_state_concept_heatmap(
    concept_matrix: np.ndarray,
    labels: np.ndarray,
    dataset_id: str,
    save_path: str,
):
    """
    狀態語義熱力圖：每個狀態的平均概念強度
    X 軸為 NMF 概念，Y 軸為 HMM 狀態
    """
    unique_states = np.unique(labels)
    n_concepts = concept_matrix.shape[1]
    
    # 計算每個狀態的平均概念向量
    state_profiles = np.zeros((len(unique_states), n_concepts))
    for i, state in enumerate(unique_states):
        mask = labels == state
        state_profiles[i] = np.mean(concept_matrix[mask], axis=0)
    
    # 限制顯示概念數（過多時取變異度最高的）
    max_concepts = min(20, n_concepts)
    if n_concepts > max_concepts:
        concept_var = np.var(state_profiles, axis=0)
        top_idx = np.argsort(concept_var)[-max_concepts:]
        state_profiles = state_profiles[:, top_idx]
        concept_labels = [f"C{i}" for i in top_idx]
    else:
        concept_labels = [f"C{i}" for i in range(n_concepts)]
    
    # 繪製熱力圖
    fig, ax = plt.subplots(figsize=(max(10, max_concepts * 0.5), max(4, len(unique_states) * 0.8)))
    sns.heatmap(
        state_profiles,
        annot=True if state_profiles.size <= 100 else False,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=concept_labels,
        yticklabels=[f"State {s}" for s in unique_states],
        ax=ax,
    )
    ax.set_xlabel("NMF Concept ID")
    ax.set_ylabel("HMM State ID")
    ax.set_title(f"State Semantic Profile - {dataset_id}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_transition_matrix(model, dataset_id: str, save_path: str):
    """
    狀態轉移矩陣：視覺化攻擊路徑邏輯
    對角線代表狀態持續性，非對角線代表轉移機率
    """
    transmat = model.transmat_
    n_states = transmat.shape[0]
    
    fig, ax = plt.subplots(figsize=(max(6, n_states * 0.8), max(5, n_states * 0.7)))
    sns.heatmap(
        transmat,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[f"To {i}" for i in range(n_states)],
        yticklabels=[f"From {i}" for i in range(n_states)],
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title(f"State Transition Matrix - {dataset_id}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_state_durations(labels: np.ndarray) -> dict:
    """計算每個狀態的平均持續時間（連續出現次數）"""
    durations = {s: [] for s in np.unique(labels)}
    
    current_state = labels[0]
    current_len = 1
    
    for i in range(1, len(labels)):
        if labels[i] == current_state:
            current_len += 1
        else:
            durations[current_state].append(current_len)
            current_state = labels[i]
            current_len = 1
    durations[current_state].append(current_len)  # 最後一段
    
    return {s: np.mean(d) if d else 0 for s, d in durations.items()}


def generate_summary_table(
    concept_matrix: np.ndarray,
    labels: np.ndarray,
    model,
    dataset_id: str,
) -> pd.DataFrame:
    """
    生成綜合分析表格，包含：
    - 狀態 ID、核心概念、平均持續時間、高機率轉移目標
    """
    unique_states = np.unique(labels)
    transmat = model.transmat_
    durations = compute_state_durations(labels)
    
    rows = []
    for state in unique_states:
        mask = labels == state
        mean_concept = np.mean(concept_matrix[mask], axis=0)
        
        # 取權重最大的前 3 個概念
        top_k = min(3, len(mean_concept))
        top_concepts_idx = np.argsort(mean_concept)[-top_k:][::-1]
        top_concepts_str = ", ".join(
            [f"C{idx}({mean_concept[idx]:.2f})" for idx in top_concepts_idx]
        )
        
        # 最可能的下一狀態
        next_probs = transmat[state]
        next_state = np.argmax(next_probs)
        next_prob = next_probs[next_state]
        next_state_str = f"State {next_state} ({next_prob:.0%})"
        
        rows.append({
            "Dataset": dataset_id,
            "State": state,
            "Top Concepts": top_concepts_str,
            "Avg Duration (logs)": f"{durations[state]:.1f}",
            "Next Likely State": next_state_str,
            "State Persistence": f"{transmat[state, state]:.0%}",
        })
    
    return pd.DataFrame(rows)


# ======================== 批次處理 ========================

def visualize_all(
    concept_vectors: dict,
    cluster_labels: dict,
    cluster_models: dict = None,
):
    """
    批次處理所有資料集並生成視覺化
    
    Args:
        concept_vectors: {dataset_id: concept_matrix} 概念向量字典
        cluster_labels: {dataset_id: labels} 分群標籤字典
        cluster_models: {dataset_id: model} HMM 模型字典（若為 None 則從檔案載入）
    """
    if not cluster_labels:
        print("[Error] 無分群結果可視覺化")
        return
    
    print(f"\n[視覺化] 處理 {len(cluster_labels)} 個資料集")
    all_summaries = []
    
    for i, (dataset_id, labels) in enumerate(cluster_labels.items(), 1):
        print(f"[{i}/{len(cluster_labels)}] {dataset_id}...")
        
        try:
            # 取得概念向量
            concept_key = dataset_id.replace("_embeddings", "")
            if concept_key not in concept_vectors:
                print(f"    [Skip] 找不到概念向量")
                continue
            concept_matrix = concept_vectors[concept_key]
            
            # 長度檢查
            if len(labels) != len(concept_matrix):
                print(f"    [Skip] 長度不匹配")
                continue
            
            # 取得模型
            if cluster_models and dataset_id in cluster_models:
                model = cluster_models[dataset_id]
            else:
                result = load_cluster_results(dataset_id)
                model = result["model"]
            
            # 建立輸出目錄
            dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_id)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # 1. 時序甘特圖
            plot_timeline(labels, dataset_id, os.path.join(dataset_output_dir, "timeline.png"))
            
            # 2. 狀態語義熱力圖
            plot_state_concept_heatmap(concept_matrix, labels, dataset_id,
                                       os.path.join(dataset_output_dir, "state_concept_heatmap.png"))
            
            # 3. 狀態轉移矩陣
            plot_transition_matrix(model, dataset_id,
                                   os.path.join(dataset_output_dir, "transition_matrix.png"))
            
            # 4. 綜合分析表格
            summary_df = generate_summary_table(concept_matrix, labels, model, dataset_id)
            summary_df.to_csv(os.path.join(dataset_output_dir, "summary.csv"), index=False)
            all_summaries.append(summary_df)
            
        except Exception as e:
            print(f"    [Error] {e}")
            continue
    
    # 合併總表
    if all_summaries:
        combined_df = pd.concat(all_summaries, ignore_index=True)
        combined_df.to_csv(os.path.join(OUTPUT_DIR, "all_datasets_summary.csv"), index=False)
    
    print(f"[完成] 視覺化輸出至 {OUTPUT_DIR}")


# ======================== 主程式 ========================

if __name__ == "__main__":
    print("=" * 60)
    print("序列分群視覺化")
    print("=" * 60)
    # 獨立執行時從檔案載入
    vectors = load_concept_vectors()
    from sequence_clustering import SequenceClustering
    clusterer = SequenceClustering()
    labels_dict = {}
    models_dict = {}
    for dataset_id in vectors.keys():
        try:
            result = load_cluster_results(dataset_id)
            labels_dict[dataset_id] = result["labels"]
            models_dict[dataset_id] = result["model"]
        except FileNotFoundError:
            continue
    visualize_all(vectors, labels_dict, models_dict)
