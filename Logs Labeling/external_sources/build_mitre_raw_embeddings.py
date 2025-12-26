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

def main():
    _ensure_project_root_on_path()
    
    import config
    from datasets import Dataset
    from models.bert import get_bert_model

    parser = argparse.ArgumentParser(description="Build MITRE raw BERT embeddings")
    parser.add_argument(
        "--mitre-csv",
        type=str,
        default=os.path.join(config.REFERENCE_RESOURCES_DIR, "MitreTechniquesTokens_V6_Sanitized.csv"),
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

    args = parser.parse_args()

    if not os.path.exists(args.mitre_csv):
        print(f"Error: MITRE CSV not found at {args.mitre_csv}")
        # Fallback to V5 if V6 doesn't exist
        args.mitre_csv = os.path.join(config.REFERENCE_RESOURCES_DIR, "MitreTechniquesTokens_V5.csv")
        if not os.path.exists(args.mitre_csv):
            return 1

    print(f"Loading MITRE techniques from {args.mitre_csv}...")
    df = pd.read_csv(args.mitre_csv)
    
    # Determine which text to embed
    # If we have super_cleaned_tokens, we might want to join them back to text
    # or just use the raw description. Usually, for raw embeddings, we use the description.
    desc_col = "description_raw" if "description_raw" in df.columns else "description"
    if desc_col not in df.columns:
        desc_col = df.columns[0] # Fallback
    
    descriptions = df[desc_col].fillna("").astype(str).tolist()
    
    print(f"Embedding {len(descriptions)} techniques using {args.bert_model}...")
    bert = get_bert_model(args.bert_model, cache_dir=config.BERT_CACHE_DIR, auto_load=True)
    embeddings = bert.embed(descriptions, batch_size=64, show_progress=True, normalize=True)
    
    out_dict = {
        "technique_id": df["technique_id"].tolist() if "technique_id" in df.columns else [str(i) for i in range(len(df))],
        "technique": df["technique"].tolist() if "technique" in df.columns else ["" for _ in range(len(df))],
        "description": descriptions,
        "embedding": embeddings.tolist(),
    }
    
    ds = Dataset.from_dict(out_dict)
    os.makedirs(os.path.dirname(args.out_dir), exist_ok=True)
    ds.save_to_disk(args.out_dir)
    
    print(f"Saved MITRE raw embeddings to {args.out_dir}")
    print(f"  Rows: {len(ds)}")
    print(f"  Embedding dim: {len(ds['embedding'][0])}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
