"""HMM → MITRE ATT&CK Cosine Similarity Visualization

用途
- 讀取 Stage III 的 HMM labels（每筆 log 的 state/cluster）
- 讀取對應 dataset 的 log embeddings（通常是 Stage I 的 BERT embeddings）
- 載入 MITRE technique embeddings
- 計算每個 cluster centroid 與每個 MITRE technique 的 cosine similarity
- 產出簡報用：heatmap、Top‑K 表格、(可選) 指定 cluster 的 Top‑K bar chart

不重跑 Pipeline：僅依賴既有 artifacts
- data/SequenceClusters/{dataset_id}/labels.npy
- data/Embeddings/{dataset_id}_embeddings/data-00000-of-00001.arrow (或相近命名)
- data/ExternalKnowledge/... 的 MITRE embeddings (npy 或 arrow)

API (for Pipeline integration)
    from visualization.hmm_mitre_similarity_viz import generate_hmm_mitre_similarity_assets
    generate_hmm_mitre_similarity_assets(dataset_id)

CLI (repo style: run via file path)
    python "Logs Labeling/visualization/hmm_mitre_similarity_viz.py" --list
    python "Logs Labeling/visualization/hmm_mitre_similarity_viz.py" --dataset-id <uuid> --topk 5

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


# ---- path bootstrap (match existing visualization modules style) ----
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _read_arrow_vectors(dir_path: Path, col_candidates: Tuple[str, ...]) -> Optional[np.ndarray]:
    """Read Arrow/Feather vectors from a dataset folder."""
    try:
        import pyarrow as pa
        import pyarrow.feather as feather
        import pyarrow.ipc as ipc
    except Exception:
        return None

    # Common filenames in this repo
    arrow_candidates = [
        dir_path / "data-00000-of-00001.arrow",
        dir_path / "data.arrow",
    ]
    arrow_path = next((p for p in arrow_candidates if p.exists()), None)
    if arrow_path is None:
        return None

    table = None
    # 1) Feather
    try:
        table = feather.read_table(str(arrow_path))
    except Exception:
        table = None

    # 2) Arrow IPC stream
    if table is None:
        try:
            with pa.memory_map(str(arrow_path), "r") as source:
                table = ipc.open_stream(source).read_all()
        except Exception:
            table = None

    # 3) Arrow IPC file
    if table is None:
        try:
            with pa.memory_map(str(arrow_path), "r") as source:
                table = ipc.open_file(source).read_all()
        except Exception:
            return None

    for col in col_candidates:
        if col in table.column_names:
            return np.asarray(table[col].to_pylist())

    # fallback
    return table.to_pandas().values


def list_available_datasets() -> List[str]:
    """List dataset IDs based on existing SequenceClusters outputs."""
    cluster_dir = Path(config.CLUSTER_RESULTS_DIR)
    if not cluster_dir.exists():
        return []
    return sorted([p.name for p in cluster_dir.iterdir() if p.is_dir() and (p / "labels.npy").exists()])


def load_hmm_labels(dataset_id: str) -> np.ndarray:
    labels_path = Path(config.CLUSTER_RESULTS_DIR) / dataset_id / "labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"找不到 labels.npy: {labels_path}")
    return np.load(labels_path)


def load_dataset_embeddings(dataset_id: str) -> np.ndarray:
    """Load log embeddings for a dataset.

    Tries common folder naming patterns under config.LOG_VECTORS_DIR.
    """
    base = Path(config.LOG_VECTORS_DIR)

    candidates = [
        base / f"{dataset_id}_embeddings",
        base / f"{dataset_id}_logvectors",
        base / dataset_id,
    ]

    for p in candidates:
        if p.exists() and p.is_dir():
            vec = _read_arrow_vectors(p, ("embedding", "embeddings", "vector", "log_vector"))
            if vec is not None:
                return np.asarray(vec)

            # npy fallback
            for fname in ["embeddings.npy", "log_vectors.npy", "vectors.npy"]:
                npy = p / fname
                if npy.exists():
                    return np.load(npy)

    raise FileNotFoundError(
        f"找不到 embeddings: tried {', '.join(str(c) for c in candidates)}"
    )


def _infer_mitre_embeddings_dir() -> Path:
    # Prefer config-defined path, with fallbacks.
    candidates = []
    if hasattr(config, "MITRE_EXTERNAL_KNOWLEDGE_DIR"):
        candidates.append(Path(config.MITRE_EXTERNAL_KNOWLEDGE_DIR))
    # historical/common names
    candidates.append(Path(config.EXTERNAL_KNOWLEDGE_DIR) / "MITRE_RAW_EMBEDDINGS")
    candidates.append(Path(config.EXTERNAL_KNOWLEDGE_DIR) / "MITRE_ATTACK")

    for p in candidates:
        if p.exists() and p.is_dir():
            return p

    raise FileNotFoundError(
        f"找不到 MITRE embeddings dir. Tried: {', '.join(str(c) for c in candidates)}"
    )


def load_mitre_embeddings(embeddings_dir: Optional[str] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load MITRE technique embeddings + ids/names."""
    emb_dir = Path(embeddings_dir) if embeddings_dir else _infer_mitre_embeddings_dir()

    # 1) npy preferred
    for fname in ["embeddings.npy", "mitre_embeddings.npy", "technique_embeddings.npy"]:
        p = emb_dir / fname
        if p.exists():
            emb = np.load(p)
            tids, tnames = _load_mitre_metadata(emb_dir, n=len(emb))
            return np.asarray(emb), tids, tnames

    # 2) arrow dataset
    vec = _read_arrow_vectors(emb_dir, ("embedding", "embeddings", "vector", "concept_vector"))
    if vec is not None:
        emb = np.asarray(vec)
        tids, tnames = _load_mitre_metadata(emb_dir, n=len(emb))
        return emb, tids, tnames

    # 3) huggingface dataset fallback
    state_json = emb_dir / "state.json"
    if state_json.exists():
        try:
            from datasets import load_from_disk

            ds = load_from_disk(str(emb_dir))
            if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
                ds = ds[next(iter(ds.keys()))]

            embed_col = next((c for c in ["embedding", "embeddings", "vector", "concept_vector"] if c in ds.column_names), None)
            if embed_col is None:
                raise ValueError(f"MITRE dataset missing embedding column. Found: {ds.column_names}")

            emb = np.asarray(ds[embed_col])
            tids = list(ds["technique_id"]) if "technique_id" in ds.column_names else [str(i) for i in range(len(ds))]
            tnames = list(ds["technique"]) if "technique" in ds.column_names else tids
            return emb, tids, tnames
        except Exception:
            pass

    raise FileNotFoundError(f"找不到 MITRE embeddings in {emb_dir}")


def _load_mitre_metadata(emb_dir: Path, n: int) -> Tuple[List[str], List[str]]:
    meta_csv = emb_dir / "metadata.csv"
    if meta_csv.exists():
        try:
            df = pd.read_csv(meta_csv)
            tids = (df["technique_id"] if "technique_id" in df.columns else df.get("id")).astype(str).tolist()
            if "technique" in df.columns:
                tnames = df["technique"].astype(str).tolist()
            elif "name" in df.columns:
                tnames = df["name"].astype(str).tolist()
            else:
                tnames = tids
            if len(tids) == n:
                return tids, tnames
        except Exception:
            pass

    tids = [f"T{i:04d}" for i in range(n)]
    return tids, tids


def compute_cluster_centroids(labels: np.ndarray, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (unique_clusters, centroid_matrix)."""
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings)

    if len(labels) != len(embeddings):
        m = min(len(labels), len(embeddings))
        print(f"[Warning] labels/embeddings 長度不一致：{len(labels)} vs {len(embeddings)}，將截斷為 {m}")
        labels = labels[:m]
        embeddings = embeddings[:m]

    unique = np.unique(labels)
    centroids = []
    for cid in unique:
        mask = labels == cid
        centroids.append(np.mean(embeddings[mask], axis=0))
    return unique, np.asarray(centroids)


def build_topk_table(
    similarity: np.ndarray,
    cluster_ids: np.ndarray,
    technique_ids: List[str],
    technique_names: List[str],
    top_k: int,
) -> pd.DataFrame:
    rows = []
    for r, cid in enumerate(cluster_ids):
        idxs = np.argsort(similarity[r])[-top_k:][::-1]
        for rank, j in enumerate(idxs, 1):
            rows.append(
                {
                    "cluster_id": int(cid),
                    "rank": rank,
                    "technique_id": technique_ids[j] if j < len(technique_ids) else str(j),
                    "technique_name": technique_names[j] if j < len(technique_names) else str(j),
                    "cosine_similarity": float(similarity[r, j]),
                }
            )
    return pd.DataFrame(rows)


def plot_similarity_heatmap(
    similarity: np.ndarray,
    cluster_ids: np.ndarray,
    technique_names: List[str],
    topk_df: pd.DataFrame,
    save_path: str,
    max_techniques: int = 25,
) -> None:
    """Heatmap: clusters × selected techniques.

    Selection: union of top‑K techniques across clusters, then keep most frequent (cap).
    """
    # count frequency in topk
    freq = topk_df[topk_df["rank"] == 1]["technique_name"].value_counts()
    if freq.empty:
        return

    selected = list(freq.index[:max_techniques])

    # map names -> indices (first match)
    name_to_idx = {n: i for i, n in enumerate(technique_names)}
    selected_idx = [name_to_idx[n] for n in selected if n in name_to_idx]

    if not selected_idx:
        return

    sub = similarity[:, selected_idx]

    plt.figure(figsize=(max(10, len(selected_idx) * 0.5), max(4, len(cluster_ids) * 0.6)))
    sns.heatmap(
        sub,
        cmap="viridis",
        xticklabels=[selected[i] for i in range(len(selected_idx))],
        yticklabels=[f"C{int(c)}" for c in cluster_ids],
        vmin=0,
        vmax=1,
    )
    plt.title("HMM Cluster → MITRE Cosine Similarity (Top Techniques)")
    plt.xlabel("MITRE Technique")
    plt.ylabel("HMM Cluster")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cluster_topk_bar(
    similarity_row: np.ndarray,
    technique_names: List[str],
    cluster_id: int,
    save_path: str,
    top_k: int = 8,
) -> None:
    idxs = np.argsort(similarity_row)[-top_k:][::-1]
    vals = [float(similarity_row[i]) for i in idxs]
    names = [technique_names[i] if i < len(technique_names) else str(i) for i in idxs]

    plt.figure(figsize=(12, max(3.5, 0.35 * len(names) + 1)))
    sns.barplot(x=vals, y=names, orient="h", color="#2d74b2")
    plt.xlim(0, 1)
    plt.title(f"Cluster {cluster_id}: Top-{top_k} MITRE Techniques (Cosine Similarity)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Technique")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_hmm_mitre_similarity_assets(
    dataset_id: str,
    output_dir: Optional[str] = None,
    top_k: int = 5,
    max_techniques_heatmap: int = 25,
    focus_cluster: Optional[int] = None,
    mitre_embeddings_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Generate slide-ready assets for HMM→MITRE cosine similarity."""

    out_dir = output_dir or os.path.join(config.RESULT_DIR, "hmm_mitre_similarity", dataset_id)
    _ensure_dir(out_dir)

    labels = load_hmm_labels(dataset_id)
    embeddings = load_dataset_embeddings(dataset_id)
    mitre_emb, mitre_ids, mitre_names = load_mitre_embeddings(mitre_embeddings_dir)

    # centroid per cluster
    cluster_ids, centroids = compute_cluster_centroids(labels, embeddings)

    # cosine similarity matrix [n_clusters, n_techniques]
    sim = cosine_similarity(centroids, mitre_emb)

    # top-k table
    topk_df = build_topk_table(sim, cluster_ids, mitre_ids, mitre_names, top_k=top_k)
    topk_csv = os.path.join(out_dir, "cluster_mitre_topk.csv")
    topk_df.to_csv(topk_csv, index=False)

    artifacts: Dict[str, str] = {"cluster_mitre_topk": topk_csv}

    # heatmap
    heat_path = os.path.join(out_dir, "cluster_mitre_similarity_heatmap.png")
    plot_similarity_heatmap(sim, cluster_ids, mitre_names, topk_df, heat_path, max_techniques=max_techniques_heatmap)
    artifacts["heatmap"] = heat_path

    # focus cluster bar
    if focus_cluster is not None:
        # find row
        try:
            row_idx = int(np.where(cluster_ids == focus_cluster)[0][0])
            bar_path = os.path.join(out_dir, f"cluster_{focus_cluster}_topk_bar.png")
            plot_cluster_topk_bar(sim[row_idx], mitre_names, focus_cluster, bar_path, top_k=max(3, min(12, top_k)))
            artifacts["focus_cluster_bar"] = bar_path
        except Exception:
            print(f"[Warning] 找不到 focus_cluster={focus_cluster}，跳過 bar plot")

    # metadata
    meta_path = os.path.join(out_dir, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"dataset_id={dataset_id}\n")
        f.write(f"n_logs={len(labels)}\n")
        f.write(f"n_clusters={len(cluster_ids)}\n")
        f.write(f"embedding_dim={embeddings.shape[1]}\n")
        f.write(f"n_mitre={len(mitre_emb)}\n")
        f.write(f"top_k={top_k}\n")
    artifacts["meta"] = meta_path

    print(f"[HMM→MITRE] 已產生輸出: {out_dir}")
    return artifacts


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Visualize HMM cluster → MITRE cosine similarity (no pipeline rerun)")
    parser.add_argument("--list", action="store_true", help="list available dataset IDs (from SequenceClusters)")
    parser.add_argument("--dataset-id", type=str, help="dataset uuid/id (folder name under SequenceClusters)")
    parser.add_argument("--outdir", type=str, default=None, help="output directory")
    parser.add_argument("--topk", type=int, default=5, help="top-k techniques per cluster")
    parser.add_argument("--max-tech", type=int, default=25, help="max techniques in heatmap")
    parser.add_argument("--focus-cluster", type=int, default=None, help="also generate a top-k bar chart for this cluster id")
    parser.add_argument("--mitre-dir", type=str, default=None, help="override MITRE embeddings dir")

    args = parser.parse_args()

    if args.list:
        items = list_available_datasets()
        print("Available datasets (from SequenceClusters):")
        for x in items:
            print(f"- {x}")
        return

    if not args.dataset_id:
        raise SystemExit("請提供 --dataset-id，或先用 --list 取得清單")

    generate_hmm_mitre_similarity_assets(
        dataset_id=args.dataset_id,
        output_dir=args.outdir,
        top_k=args.topk,
        max_techniques_heatmap=args.max_tech,
        focus_cluster=args.focus_cluster,
        mitre_embeddings_dir=args.mitre_dir,
    )


if __name__ == "__main__":
    main()
