"""
Build TF-IDF Vectorizer and Matrix for MITRE ATT&CK
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Adjust path 
CURRENT_DIR = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(CURRENT_DIR).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from utils.path import ensure_dir

def build_mitre_tfidf(
    out_dir: str,
    mitre_csv: str = config.MITRE_TECHNIQUES_CSV,
    max_features: int = 5000, # Increased from 1000 to capture more specific terms
    force_rebuild: bool = True # Default to True for first run
):
    ensure_dir(out_dir)
    
    vectorizer_path = os.path.join(out_dir, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(out_dir, "mitre_tfidf_matrix.pkl")
    metadata_path = os.path.join(out_dir, "metadata.csv")

    if not force_rebuild and os.path.exists(vectorizer_path) and os.path.exists(matrix_path):
        print(f"TF-IDF data already exists in {out_dir}")
        return

    print(f"Building TF-IDF from {mitre_csv}...")
    try:
        df = pd.read_csv(mitre_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Fill NaN
    if "description" not in df.columns:
        print("Error: 'description' column not found in CSV.")
        print(f"Columns: {df.columns}")
        return
        
    documents = df["description"].fillna("").astype(str).tolist()
    
    # Initialize and fit TF-IDF
    # Use English stop words
    print(f"Fitting TF-IDF vectorizer (max_features={max_features})...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print(f"Saving artifacts to {out_dir}...")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
        
    with open(matrix_path, "wb") as f:
        pickle.dump(tfidf_matrix, f)
        
    # Save metadata (technique IDs/names) to map back easily
    # Ensure ID and Name columns exist
    cols_to_save = []
    if "technique" in df.columns: cols_to_save.append("technique")
    if "technique_id" in df.columns: cols_to_save.append("technique_id")
    
    if cols_to_save:
        meta_df = df[cols_to_save].copy() 
        meta_df.to_csv(metadata_path, index=False)
    
    print("Done.")

if __name__ == "__main__":
    out_dir = getattr(config, 'MITRE_TFIDF_DIR', os.path.join(config.EXTERNAL_KNOWLEDGE_DIR, "MITRE_TFIDF"))
    build_mitre_tfidf(out_dir=out_dir)
