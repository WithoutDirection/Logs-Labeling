#!/usr/bin/env python3
"""Concept-based Caldera experiment.

This script compares *post concept-extraction* log concept vectors
(`data/ConceptVectors/*_concepts`) against MITRE ATT&CK technique concepts.

Preferred mode (matches the main pipeline):
- Load log `concept_vector` from `data/ConceptVectors/<LogID>_concepts/`.
- Load the saved global concept model `models/nmf_concept_model.pkl`.
- Load MITRE vectors from `data/ExternalKnowledge/MITRE_ATTACK/` and transform
    them into concept vectors using the same saved scaler + NMF model.

Legacy inputs (`--embeddings-dataset`, `--processed-csv`) are still supported,
but they *re-fit* NMF and are not the post-extraction comparison.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


def _ensure_project_root_on_path():
    # Allows running as a script from anywhere.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    import sys

    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _auto_find_text_column(df: pd.DataFrame) -> str:
    # Prefer the same priority as the preprocessing pipeline.
    for col in ["Template", "ConcatenatedLog", "OriginalLog", "Content", "Event"]:
        if col in df.columns:
            return col
    raise ValueError(f"No suitable text column found. Available: {df.columns.tolist()}")


def _auto_find_mitre_description_column(df: pd.DataFrame) -> str:
    # Mirror external_sources.source_manager priority.
    for col in ["description_raw", "all_text", "Description", "description", "description_clean"]:
        if col in df.columns:
            sample = df[col].dropna().head(1)
            if len(sample) and len(str(sample.iloc[0])) > 10:
                return col
    raise ValueError(f"No suitable MITRE description column found. Available: {df.columns.tolist()}")


def _load_embeddings_dataset(dataset_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    from datasets import load_from_disk

    ds = load_from_disk(dataset_path)

    # Keep a small metadata DF for output.
    meta = {}
    if "LogID" in ds.column_names:
        meta["LogID"] = ds["LogID"]

    df_meta = pd.DataFrame(meta)

    if "embedding" in ds.column_names:
        X = np.array(ds["embedding"], dtype=float)
        return df_meta, X

    if "template_embedding" in ds.column_names and "param_embedding" in ds.column_names:
        template = np.array(ds["template_embedding"], dtype=float)
        param = np.array(ds["param_embedding"], dtype=float)
        X = (template + param) / 2.0
        return df_meta, X

    raise ValueError(
        "Embeddings dataset must contain either 'embedding' or both "
        "'template_embedding' and 'param_embedding'. "
        f"Found: {ds.column_names}"
    )


def _load_concept_dataset(dataset_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load ConceptVectors dataset folder and return (meta_df, concept_matrix)."""
    import pyarrow.feather as feather

    possible = ["data-00000-of-00001.arrow", "data.arrow"]
    arrow_path = None
    for name in possible:
        p = os.path.join(dataset_dir, name)
        if os.path.exists(p):
            arrow_path = p
            break
    if arrow_path is None:
        raise FileNotFoundError(f"No Arrow shard found in {dataset_dir}")

    table = feather.read_table(arrow_path)
    if "concept_vector" not in table.column_names:
        raise ValueError(
            f"Expected 'concept_vector' column in {arrow_path}. "
            f"Found: {table.column_names}"
        )

    H = np.array(table["concept_vector"].to_pylist(), dtype=float)
    meta = pd.DataFrame({"row_idx": list(range(H.shape[0]))})
    return meta, H


def _load_vectors_dataset(dataset_dir: str) -> np.ndarray:
    """Load a dataset folder containing vectors.

    Supports:
    - HuggingFace `datasets` folder (preferred when dataset_info.json/state.json exist)
    - Feather file fallback
    """

    # HuggingFace datasets
    if os.path.exists(os.path.join(dataset_dir, "dataset_info.json")) or os.path.exists(
        os.path.join(dataset_dir, "state.json")
    ):
        from datasets import load_from_disk

        ds = load_from_disk(dataset_dir)
        for col in ["log_vector", "embedding", "vector"]:
            if col in ds.column_names:
                return np.array(ds[col], dtype=float)
        
        if "template_embedding" in ds.column_names:
            template = np.array(ds["template_embedding"], dtype=float)
            if "param_embedding" in ds.column_names:
                param = np.array(ds["param_embedding"], dtype=float)
                return (template + param) / 2.0
            return template
            
        raise ValueError(f"No known vector column in dataset {dataset_dir}. Columns: {ds.column_names}")

    # Feather fallback
    import pyarrow.feather as feather

    possible = ["data-00000-of-00001.arrow", "data.arrow"]
    arrow_path = None
    for name in possible:
        p = os.path.join(dataset_dir, name)
        if os.path.exists(p):
            arrow_path = p
            break
    if arrow_path is None:
        raise FileNotFoundError(f"No Arrow shard found in {dataset_dir}")

    table = feather.read_table(arrow_path)
    for col in ["log_vector", "embedding", "vector"]:
        if col in table.column_names:
            return np.array(table[col].to_pylist(), dtype=float)
            
    if "template_embedding" in table.column_names:
        template = np.array(table["template_embedding"].to_pylist(), dtype=float)
        if "param_embedding" in table.column_names:
            param = np.array(table["param_embedding"].to_pylist(), dtype=float)
            return (template + param) / 2.0
        return template

    raise ValueError(f"No known vector column found in {arrow_path}. Columns: {table.column_names}")


def _compute_mitre_concepts_from_saved_model(
    mitre_vectors_dir: str,
    nmf_model_path: str,
) -> np.ndarray:
    _ensure_project_root_on_path()
    from models.conception_extraction import ConceptExtractor

    X_mitre = _load_vectors_dataset(mitre_vectors_dir)
    extractor = ConceptExtractor()
    extractor.load_model(nmf_model_path)
    return extractor.transform(X_mitre)


def _embed_texts(texts: List[str], bert_model: str, bert_cache_dir: Optional[str], batch_size: int) -> np.ndarray:
    _ensure_project_root_on_path()
    from models.bert import get_bert_model

    bert = get_bert_model(bert_model, cache_dir=bert_cache_dir, auto_load=True)
    return bert.embed(texts, batch_size=batch_size, show_progress=True, normalize=True)


def _fit_nmf_concepts(
    X_logs: np.ndarray,
    X_mitre: np.ndarray,
    n_components: int,
    max_iter: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Ensure non-negativity in a shared way.
    scaler = MinMaxScaler()
    X_all = np.vstack([X_logs, X_mitre])
    X_all_scaled = scaler.fit_transform(X_all)

    n_samples = X_all_scaled.shape[0]
    if n_samples <= 1:
        raise ValueError("Not enough samples to fit NMF.")

    k = max(2, min(n_components, n_samples - 1))

    nmf = NMF(
        n_components=k,
        init="nndsvd",
        solver="cd",
        max_iter=max_iter,
        random_state=random_state,
    )
    H_all = nmf.fit_transform(X_all_scaled)

    H_logs = H_all[: X_logs.shape[0]]
    H_mitre = H_all[X_logs.shape[0] :]
    return H_logs, H_mitre


def main() -> int:
    _ensure_project_root_on_path()

    import config

    parser = argparse.ArgumentParser(description="Concept-space Caldera vs MITRE experiment")
    group = parser.add_mutually_exclusive_group(required=True)

    # Preferred: post concept-extraction inputs
    group.add_argument(
        "--concept-dataset",
        type=str,
        help="Path to ConceptVectors dataset folder (data/ConceptVectors/*_concepts)",
    )

    # Legacy (not post-extraction)
    group.add_argument(
        "--embeddings-dataset",
        type=str,
        help="Path to HuggingFace embeddings dataset folder (data/Embeddings/*_embeddings)",
    )
    group.add_argument(
        "--processed-csv",
        type=str,
        help="Path to processed/intermediate CSV (Template/Parameters or ConcatenatedLog)",
    )

    parser.add_argument(
        "--mitre-path",
        type=str,
        default=getattr(config, "MITRE_TECHNIQUES_CSV", os.path.join(config.REFERENCE_RESOURCES_DIR, "MitreTechniquesTokens_WithDB_V1.csv")),
        help="Path to MITRE techniques CSV",
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default=getattr(config, "EXTERNAL_SOURCES_BERT_MODEL_NAME", getattr(config, "BERT_MODEL_NAME", "sentence-bert")),
        help="BERT model key (config.BERT_MODEL_NAME) or HuggingFace model id",
    )
    parser.add_argument(
        "--bert-cache-dir",
        type=str,
        default=getattr(config, "EXTERNAL_SOURCES_BERT_CACHE_DIR", getattr(config, "BERT_CACHE_DIR", None)),
        help="Model cache dir (defaults to config.BERT_CACHE_DIR)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(getattr(config, "EXTERNAL_SOURCES_EMBED_BATCH_SIZE", 32)),
    )

    # Post concept-extraction: use saved concept model + external knowledge vectors
    parser.add_argument(
        "--nmf-model-path",
        type=str,
        default=getattr(config, "NMF_MODEL_PATH", os.path.join("models", "nmf_concept_model.pkl")),
        help="Path to saved global concept model (models/nmf_concept_model.pkl)",
    )
    parser.add_argument(
        "--mitre-vectors-dir",
        type=str,
        default=getattr(
            config,
            "MITRE_EXTERNAL_KNOWLEDGE_DIR",
            os.path.join(getattr(config, "EXTERNAL_KNOWLEDGE_DIR", os.path.join("data", "ExternalKnowledge")), "MITRE_ATTACK"),
        ),
        help="Path to MITRE vectors dataset folder (data/ExternalKnowledge/MITRE_ATTACK)",
    )

    parser.add_argument("--nmf-components", type=int, default=getattr(config, "NMF_COMPONENTS", 10))
    parser.add_argument("--nmf-max-iter", type=int, default=int(getattr(config, "NMF_MAX_ITER", 500)))
    parser.add_argument("--random-state", type=int, default=getattr(config, "SEED", 42))

    parser.add_argument("--top-k", type=int, default=3, help="Store top-k MITRE matches per row")

    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Output CSV path (default: external_sources/data/test_out/<name>_concept_analysis.csv)",
    )

    # Backwards-compatible alias (older commands / notes)
    parser.add_argument(
        "--out-csv",
        dest="out_path",
        type=str,
        default=None,
        help="Alias for --out-path",
    )

    args = parser.parse_args()

    # -----------------------
    # Load log representations
    # -----------------------
    if args.concept_dataset:
        df_meta, H_logs = _load_concept_dataset(args.concept_dataset)
        input_name = os.path.basename(os.path.normpath(args.concept_dataset))

        # MITRE concepts from saved model + stored vectors (post extraction comparison)
        H_mitre = _compute_mitre_concepts_from_saved_model(
            mitre_vectors_dir=args.mitre_vectors_dir,
            nmf_model_path=args.nmf_model_path,
        )

        sim = cosine_similarity(H_logs, H_mitre)

    elif args.embeddings_dataset:
        df_meta, X_logs = _load_embeddings_dataset(args.embeddings_dataset)
        input_name = os.path.basename(os.path.normpath(args.embeddings_dataset))
        # Legacy path: re-fit shared NMF on (logs + MITRE embeddings)
        mitre_df = pd.read_csv(args.mitre_path)
        mitre_col = _auto_find_mitre_description_column(mitre_df)
        mitre_texts = mitre_df[mitre_col].fillna("").astype(str).tolist()
        X_mitre = _embed_texts(mitre_texts, args.bert_model, args.bert_cache_dir, args.batch_size)

        H_logs, H_mitre = _fit_nmf_concepts(
            X_logs=X_logs,
            X_mitre=X_mitre,
            n_components=args.nmf_components,
            max_iter=args.nmf_max_iter,
            random_state=args.random_state,
        )
        sim = cosine_similarity(H_logs, H_mitre)

    else:
        df_logs = pd.read_csv(args.processed_csv)
        text_col = _auto_find_text_column(df_logs)
        texts = df_logs[text_col].fillna("").astype(str).tolist()
        X_logs = _embed_texts(texts, args.bert_model, args.bert_cache_dir, args.batch_size)

        df_meta = df_logs[[c for c in ["LogID", "Template", "Parameters", "ConcatenatedLog", "OriginalLog"] if c in df_logs.columns]].copy()
        input_name = os.path.splitext(os.path.basename(args.processed_csv))[0]

        if X_logs.ndim != 2 or X_logs.shape[0] == 0:
            raise ValueError(f"Invalid log embedding matrix shape: {X_logs.shape}")

        # Legacy path: re-fit shared NMF on (logs + MITRE embeddings)
        mitre_df = pd.read_csv(args.mitre_path)
        mitre_col = _auto_find_mitre_description_column(mitre_df)
        mitre_texts = mitre_df[mitre_col].fillna("").astype(str).tolist()
        X_mitre = _embed_texts(mitre_texts, args.bert_model, args.bert_cache_dir, args.batch_size)

        H_logs, H_mitre = _fit_nmf_concepts(
            X_logs=X_logs,
            X_mitre=X_mitre,
            n_components=args.nmf_components,
            max_iter=args.nmf_max_iter,
            random_state=args.random_state,
        )
        sim = cosine_similarity(H_logs, H_mitre)

    # Technique name/id columns for reporting
    mitre_df = None
    if args.mitre_path and os.path.exists(args.mitre_path):
        mitre_df = pd.read_csv(args.mitre_path)
    else:
        # If the MITRE CSV isn't available, try using metadata from the vectors dataset.
        try:
            from datasets import load_from_disk

            if args.mitre_vectors_dir and (
                os.path.exists(os.path.join(args.mitre_vectors_dir, "dataset_info.json"))
                or os.path.exists(os.path.join(args.mitre_vectors_dir, "state.json"))
            ):
                ds = load_from_disk(args.mitre_vectors_dir)
                cols = [c for c in ["technique", "technique_id"] if c in ds.column_names]
                if cols:
                    mitre_df = ds.select_columns(cols).to_pandas()
        except Exception:
            mitre_df = None

    tech_name_col = "technique" if mitre_df is not None and "technique" in mitre_df.columns else None
    tech_id_col = "technique_id" if mitre_df is not None and "technique_id" in mitre_df.columns else None

    best_idx = np.argmax(sim, axis=1)
    best_score = sim[np.arange(sim.shape[0]), best_idx]

    def _format_mitre_label(i: int) -> str:
        if mitre_df is None:
            return f"T{i}"

        name = str(mitre_df.iloc[i][tech_name_col]) if tech_name_col else f"T{i}"
        if tech_id_col:
            tid = str(mitre_df.iloc[i][tech_id_col])
            if tid and tid != "nan":
                return f"{tid} {name}".strip()
        return name

    best_label = [_format_mitre_label(int(i)) for i in best_idx]

    out_df = df_meta.copy()
    out_df["mitre_best_technique"] = best_label
    out_df["mitre_best_concept_similarity"] = best_score

    # Optional top-k list
    top_k = max(1, int(args.top_k))
    top_indices = np.argsort(sim, axis=1)[:, -top_k:][:, ::-1]
    top_scores = np.take_along_axis(sim, top_indices, axis=1)

    out_df["mitre_topk_techniques"] = [
        "; ".join(_format_mitre_label(int(j)) for j in row)
        for row in top_indices
    ]
    out_df["mitre_topk_scores"] = [
        "; ".join(f"{float(s):.4f}" for s in row)
        for row in top_scores
    ]

    # -----------------------
    # Write results
    # -----------------------
    if args.out_path is None:
        out_dir = os.path.join(os.path.dirname(__file__), "data", "test_out")
        os.makedirs(out_dir, exist_ok=True)
        args.out_path = os.path.join(out_dir, f"{input_name}_concept_analysis.csv")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    out_df.to_csv(args.out_path, index=False)

    print(f"Saved: {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
