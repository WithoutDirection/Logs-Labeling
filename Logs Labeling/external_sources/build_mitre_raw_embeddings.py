#!/usr/bin/env python3
"""Build MITRE external knowledge vectors using raw BERT embeddings.

Goal:
- Create a vector dataset that contains raw BERT embeddings of MITRE techniques.

Output:
- HuggingFace Dataset folder: `data/ExternalKnowledge/MITRE_RAW_EMBEDDINGS`
- Columns: technique_id, technique, description, embedding
"""

import argparse
import os
import sys
import shutil
import numpy as np
import pandas as pd

def _ensure_project_root_on_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Also add the root of the workspace
    workspace_root = os.path.dirname(project_root)
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)


def build_mitre_raw_embeddings(
    mitre_csv: str | None = None,
    out_dir: str | None = None,
    bert_model: str | None = None,
    *,
    force_rebuild: bool = False,
) -> str:
    """Build (or reuse) MITRE raw embedding dataset.

    Returns the output directory containing the HuggingFace dataset.
    """
    _ensure_project_root_on_path()

    import config
    from datasets import Dataset
    from models.bert import get_bert_model

    mitre_csv = mitre_csv or (
        # Prefer the multi-source combined CSV if it exists, fall back to legacy single file
        os.path.join(config.REFERENCE_RESOURCES_DIR, "combined.csv")
        if os.path.exists(os.path.join(config.REFERENCE_RESOURCES_DIR, "combined.csv"))
        else os.path.join(config.REFERENCE_RESOURCES_DIR, "MitreTechniquesTokens_V5.csv")
    )
    out_dir = out_dir or os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_RAW_EMBEDDINGS")
    bert_model = bert_model or config.BERT_MODEL_NAME

    # If already built, skip unless force.
    if not force_rebuild and os.path.exists(out_dir) and os.path.exists(os.path.join(out_dir, "state.json")):
        print(f"[Info] MITRE raw embeddings already exist, skip build: {out_dir}")
        return out_dir

    if force_rebuild and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    print(f"Loading MITRE techniques from {mitre_csv}...")
    df = pd.read_csv(mitre_csv)

    desc_col = "cleaned_tokens"
    if desc_col not in df.columns:
        print(f"[Warning] Expected description column '{desc_col}' not found.")
        desc_col = "description"

    descriptions = df[desc_col].fillna("").astype(str).tolist()

    print(f"Embedding {len(descriptions)} techniques using {bert_model}...")
    bert = get_bert_model(bert_model, cache_dir=config.BERT_CACHE_DIR, auto_load=True)
    embeddings = bert.embed(descriptions, batch_size=64, show_progress=True, normalize=True)

    out_dict = {
        "technique_id": df["technique_id"].tolist() if "technique_id" in df.columns else [str(i) for i in range(len(df))],
        "technique": df["technique"].tolist() if "technique" in df.columns else ["" for _ in range(len(df))],
        "description": descriptions,
        "embedding": embeddings.tolist(),
    }

    ds = Dataset.from_dict(out_dict)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    ds.save_to_disk(out_dir)

    print(f"Saved MITRE raw embeddings to {out_dir}")
    print(f"  Rows: {len(ds)}")
    print(f"  Embedding dim: {len(ds['embedding'][0])}")
    return out_dir

def main():
    _ensure_project_root_on_path()
    
    import config

    parser = argparse.ArgumentParser(description="Build MITRE raw BERT embeddings")
    parser.add_argument(
        "--mitre-csv",
        type=str,
        default=os.path.join(config.REFERENCE_RESOURCES_DIR, "MitreTechniquesTokens_V5.csv"),
        help="MITRE techniques CSV path",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_RAW_EMBEDDINGS"),
        help="Output dataset folder",
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default=config.BERT_MODEL_NAME,
        help="BERT model name",
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if output already exists",
    )

    args = parser.parse_args()

    build_mitre_raw_embeddings(
        mitre_csv=args.mitre_csv,
        out_dir=args.out_dir,
        bert_model=args.bert_model,
        force_rebuild=args.force_rebuild,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
