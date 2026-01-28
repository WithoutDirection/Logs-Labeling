import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm

# Add package root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def load_vectorizer():
    """Load the pre-trained MITRE TF-IDF vectorizer."""
    tfidf_path = os.path.join(config.MITRE_TFIDF_DIR, "tfidf_vectorizer.pkl")
    if not os.path.exists(tfidf_path):
        print(f"[Error] Vectorizer not found at {tfidf_path}")
        return None
    with open(tfidf_path, "rb") as f:
        return pickle.load(f)

def find_source_csv(dataset_id, config):
    """Locate the source CSV for a dataset ID."""
    cleaned_id = dataset_id
    for suffix in ["_embeddings", "_concepts", "_vectors"]:
        if cleaned_id.endswith(suffix):
            cleaned_id = cleaned_id[:-len(suffix)]
            
    candidates = [
        f"{dataset_id}.csv",
        f"{cleaned_id}.csv",
        f"{dataset_id}_raw_events.csv",
        f"{cleaned_id}_raw_events.csv",
        f"syslogs_{dataset_id}_audit_log.csv",
        f"syslogs_{cleaned_id}_audit_log.csv"
    ]
    
    # 1. Try Intermediate Data
    for cand in candidates:
        p = os.path.join(config.INTERMEDIATE_DATA_DIR, cand)
        if os.path.exists(p):
            return p
            
    # 2. Try Input Logs
    input_path = os.path.join(config.INPUT_LOGS_DIR, f"{dataset_id}.csv")
    if os.path.exists(input_path):
        return input_path
        
    return None

def extract_text(df):
    """Extract text content from dataframe using defined heuristics."""
    if "ConcatenatedLog" in df.columns:
        return df["ConcatenatedLog"].fillna("").astype(str).tolist()
    elif "Template" in df.columns and "Parameters" in df.columns:
        return (df["Template"].fillna("") + " " + df["Parameters"].fillna("")).astype(str).tolist()
    elif "Content" in df.columns:
        return df["Content"].fillna("").astype(str).tolist()
    elif "Event" in df.columns:
        return df["Event"].fillna("").astype(str).tolist()
    else:
        return df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()

def main():
    print("=== Pre-computing TF-IDF Vectors for Logs ===")
    
    vectorizer = load_vectorizer()
    if not vectorizer:
        return

    embeddings_dir = config.LOG_VECTORS_DIR
    print(f"Scanning Embeddings Directory: {embeddings_dir}")
    
    if not os.path.exists(embeddings_dir):
        print("Embeddings directory does not exist.")
        return

    subdirs = [d for d in os.listdir(embeddings_dir) if os.path.isdir(os.path.join(embeddings_dir, d))]
    
    for subdir in tqdm(subdirs, desc="Processing Datasets"):
        dataset_dir = os.path.join(embeddings_dir, subdir)
        dataset_id = subdir # subdirectory name is usually the dataset ID or close to it
        
        # Check if already done? (Optional: force overwrite since user explicitly requested this)
        # out_path = os.path.join(dataset_dir, "tfidf.npz")
        # if os.path.exists(out_path): continue
        
        csv_path = find_source_csv(dataset_id, config)
        if not csv_path:
            # Try stripping _embeddings if subdir has it
            if "_embeddings" in dataset_id:
                clean_id = dataset_id.replace("_embeddings", "")
                csv_path = find_source_csv(clean_id, config)
            
        if not csv_path:
            print(f"[Skip] Source CSV not found for {dataset_id}")
            continue
            
        try:
            df = pd.read_csv(csv_path)
            texts = extract_text(df)
            
            if not texts:
                print(f"[Skip] No text extracted for {dataset_id}")
                continue
                
            # Transform
            tfidf_matrix = vectorizer.transform(texts)
            
            # Save
            out_path = os.path.join(dataset_dir, "tfidf.npz")
            scipy.sparse.save_npz(out_path, tfidf_matrix)
            # print(f"[Saved] {out_path} shape={tfidf_matrix.shape}")
            
        except Exception as e:
            print(f"[Error] Failed processing {dataset_id}: {e}")

if __name__ == "__main__":
    main()
