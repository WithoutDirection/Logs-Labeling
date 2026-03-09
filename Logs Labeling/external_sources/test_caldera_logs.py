#!/usr/bin/env python3
"""
Evaluate raw log embeddings against MITRE ATT&CK descriptions (Caldera logs).

Uses the current pipeline's combined.csv reference (built by ReferenceBuilder)
and TextProcessor for embeddings. Produces:
  - BERT cosine similarity ranking (raw embedding space)
  - NMF concept-space similarity ranking
  - Per-row results CSV saved to --out-dir

Probe mode (--probe-rows):
  Combine a comma-separated list of row indices into a single text and print
  top-N MITRE matches to the terminal.  Useful for quick spot-checks.
  Example: --probe-rows 163,290,291
"""

import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from text_processor import TextProcessor
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_reference_csv(path: str, label: str) -> pd.DataFrame:
    """Load a reference CSV and pick a usable description column."""
    df = pd.read_csv(path)
    # Prefer rich combined text; fall back gracefully
    for col in ("description_raw", "description", "cleaned_tokens"):
        if col in df.columns:
            df["_embed_text"] = df[col].fillna("").astype(str)
            print(f"  [{label}] {len(df)} rows, using column '{col}'")
            return df
    raise ValueError(f"[{label}] No usable description column in {path}")


def _name_col(df: pd.DataFrame) -> str:
    """Return the technique/name column from a reference DataFrame."""
    for c in ("technique", "name", "technique_name", "title"):
        if c in df.columns:
            return c
    return df.columns[0]


def _top_n(sim_row: np.ndarray, df: pd.DataFrame, n: int = 5):
    """Return top-n (name, score) pairs for a similarity row."""
    col = _name_col(df)
    idxs = np.argsort(sim_row)[-n:][::-1]
    return [(str(df.iloc[i][col]), float(sim_row[i])) for i in idxs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Caldera logs vs MITRE ATT&CK")
    parser.add_argument(
        "--log-path",
        type=str,
        default=os.path.join(config.DATA_DIR, "Caldera_Ability_Statistics",
                             "4bfb5f265a5ce07af6bf10da113af7db_raw_events.csv"),
        help="Path to a Caldera _raw_events.csv log file",
    )
    parser.add_argument(
        "--mitre-csv",
        type=str,
        default=None,
        help="Override MITRE reference CSV (default: combined.csv → fallback to V5)",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="Content",
        help="Column in the log CSV that contains the event text (default: Content)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(config.DATA_DIR, "test_out"),
        help="Output directory for analysis CSV",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top MITRE matches to record per event (default: 5)",
    )
    parser.add_argument(
        "--nmf-components",
        type=int,
        default=20,
        help="Max NMF components for concept-space analysis (default: 20)",
    )
    parser.add_argument(
        "--probe-rows",
        type=str,
        default=None,
        help="Comma-separated row indices to combine into a single probe text and print top-N matches (e.g. 163,290,291)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ paths
    mitre_csv = args.mitre_csv
    if mitre_csv is None:
        combined = config.REFERENCE_COMBINED_CSV
        mitre_csv = combined if os.path.exists(combined) else config.MITRE_TECHNIQUES_CSV

    # ------------------------------------------------------------------ load logs
    print("=" * 80)
    print("Caldera Log Evaluation  —  MITRE ATT&CK Similarity")
    print("=" * 80)

    print(f"\n[1] Loading logs: {args.log_path}")
    df_logs = pd.read_csv(args.log_path)
    print(f"    {len(df_logs)} events | columns: {list(df_logs.columns)}")

    texts = df_logs[args.text_col].fillna("").astype(str).tolist()

    # ------------------------------------------------------------------ load MITRE
    print(f"\n[2] Loading MITRE reference: {os.path.basename(mitre_csv)}")
    df_mitre = _load_reference_csv(mitre_csv, "MITRE")
    mitre_texts = df_mitre["_embed_text"].tolist()

    # ------------------------------------------------------------------ embed
    bert_model = config.BERT_MODEL_NAME
    print(f"\n[4] Embedding with model: {bert_model}")
    tp = TextProcessor(bert_model_name=bert_model)

    print("    Embedding log events...")
    log_embs = tp.generate_embeddings(texts)
    print(f"    Log embeddings: {log_embs.shape}")

    print("    Embedding MITRE techniques...")
    mitre_embs = tp.generate_embeddings(mitre_texts)
    print(f"    MITRE embeddings: {mitre_embs.shape}")

    # ------------------------------------------------------------------ BERT similarity
    print("\n[5] Computing BERT cosine similarity (log → MITRE)...")
    bert_sim = cosine_similarity(log_embs, mitre_embs)  # (n_logs, n_mitre)

    # ------------------------------------------------------------------ NMF concept space
    print(f"\n[6] NMF concept-space similarity (log + MITRE, max {args.nmf_components} components)...")
    n_logs = log_embs.shape[0]
    stacked = np.vstack([log_embs, mitre_embs])
    stacked_nn = stacked - stacked.min()  # shift to non-negative
    n_comp = min(args.nmf_components, stacked_nn.shape[0] - 1, stacked_nn.shape[1])
    nmf = NMF(n_components=n_comp, random_state=42, max_iter=500)
    H = nmf.fit_transform(stacked_nn)
    log_concepts   = H[:n_logs]
    mitre_concepts = H[n_logs:]
    concept_sim = cosine_similarity(log_concepts, mitre_concepts)  # (n_logs, n_mitre)
    print(f"    NMF components used: {n_comp}")

    # ------------------------------------------------------------------ build results
    print("\n[8] Building results DataFrame...")
    results_df = df_logs.copy()
    name_col = _name_col(df_mitre)
    top_n = args.top_n

    bert_top_techs  = []
    bert_top_scores = []
    concept_top_techs  = []
    concept_top_scores = []

    for i in range(len(df_logs)):
        # BERT top-N
        bt = _top_n(bert_sim[i], df_mitre, top_n)
        bert_top_techs.append(" | ".join(t for t, _ in bt))
        bert_top_scores.append(" | ".join(f"{s:.4f}" for _, s in bt))

        # Concept top-N
        ct = _top_n(concept_sim[i], df_mitre, top_n)
        concept_top_techs.append(" | ".join(t for t, _ in ct))
        concept_top_scores.append(" | ".join(f"{s:.4f}" for _, s in ct))

    results_df[f"bert_top{top_n}_techniques"]  = bert_top_techs
    results_df[f"bert_top{top_n}_scores"]      = bert_top_scores
    results_df[f"concept_top{top_n}_techniques"] = concept_top_techs
    results_df[f"concept_top{top_n}_scores"]     = concept_top_scores
    results_df["bert_best_technique"]  = [t.split(" | ")[0] for t in bert_top_techs]
    results_df["bert_best_score"]      = [float(s.split(" | ")[0]) for s in bert_top_scores]
    results_df["concept_best_technique"] = [t.split(" | ")[0] for t in concept_top_techs]
    results_df["concept_best_score"]     = [float(s.split(" | ")[0]) for s in concept_top_scores]

    # ------------------------------------------------------------------ distribution summary
    print("\n" + "=" * 80)
    print("TOP-5 TECHNIQUE DISTRIBUTION")
    print("=" * 80)
    print(f"\n[BERT similarity] top-5 most assigned techniques:")
    for tech, cnt in results_df["bert_best_technique"].value_counts().head(5).items():
        print(f"  {cnt:4d}  {tech}")
    print(f"\n[Concept space]   top-5 most assigned techniques:")
    for tech, cnt in results_df["concept_best_technique"].value_counts().head(5).items():
        print(f"  {cnt:4d}  {tech}")

    # ------------------------------------------------------------------ probe mode
    if args.probe_rows:
        idxs = [int(x.strip()) for x in args.probe_rows.split(",")]
        print("\n" + "=" * 80)
        print(f"PROBE  —  rows {idxs}")
        print("=" * 80)

        subset = df_logs.iloc[idxs]
        for abs_i, row in zip(idxs, subset.itertuples()):
            cmd = str(getattr(row, "Command_Line", "") or "")[:100]
            proc = getattr(row, "Process_Name", abs_i)
            op   = getattr(row, "Operation", "")
            print(f"  row {abs_i}: {proc} / {op} / {cmd}")

        combined_text = " ".join(
            subset[args.text_col].fillna("").astype(str).tolist()
        )
        probe_emb = tp.generate_embeddings([combined_text])  # (1, d)

        # BERT
        probe_bert_sim = cosine_similarity(probe_emb, mitre_embs)[0]
        print(f"\nTop-{args.top_n} techniques (BERT similarity):")
        for rank, (tech, score) in enumerate(_top_n(probe_bert_sim, df_mitre, args.top_n), 1):
            print(f"  {rank}. {tech}  (sim={score:.4f})")

        # NMF concept space
        stacked_p = np.vstack([probe_emb, mitre_embs])
        stacked_p_nn = stacked_p - stacked_p.min()
        n_comp_p = min(args.nmf_components, stacked_p_nn.shape[0] - 1, stacked_p_nn.shape[1])
        H_p = NMF(n_components=n_comp_p, random_state=42, max_iter=500).fit_transform(stacked_p_nn)
        probe_concept_sim = cosine_similarity(H_p[:1], H_p[1:])[0]
        print(f"\nTop-{args.top_n} techniques (Concept space):")
        for rank, (tech, score) in enumerate(_top_n(probe_concept_sim, df_mitre, args.top_n), 1):
            print(f"  {rank}. {tech}  (sim={score:.4f})")

    # ------------------------------------------------------------------ save
    os.makedirs(args.out_dir, exist_ok=True)
    base_name   = os.path.splitext(os.path.basename(args.log_path))[0]
    output_path = os.path.join(args.out_dir, f"{base_name}_analysis.csv")
    results_df.to_csv(output_path, index=True)
    print(f"\n[9] Results saved → {output_path}")


if __name__ == "__main__":
    main()
