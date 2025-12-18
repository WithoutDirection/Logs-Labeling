#!/usr/bin/env python3
"""
Test ConceptUML pipeline against Caldera attack logs.
Analyze per-row results with MITRE ATT&CK and CAPEC mappings.
"""

import sys
import os
import argparse

# Add parent directory for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import components directly
from text_processor import TextProcessor
from source_manager import ExternalSourceManager
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def main():
    # ============ CONFIGURATION ============
    # BERT model options (sentence-transformers compatible):
    #   - 'sentence-transformers/all-MiniLM-L6-v2' (fast, good quality) ✓ RECOMMENDED
    #   - 'sentence-transformers/all-mpnet-base-v2' (higher quality) ✓ RECOMMENDED  
    #   - 'bert-base-nli-mean-tokens' (default, general purpose)
    #
    # Security-focused (will use mean pooling automatically):
    #   - 'jackaduma/SecBERT' (security-focused, works with sentence-transformers)
    #
    # NOTE: 'ehsanaghaei/SecureBERT' is RoBERTa-based MLM model, not optimized for embeddings
    #       Use 'sentence-transformers/all-mpnet-base-v2' for better results
    BERT_MODEL = (
        getattr(config, "EXTERNAL_SOURCES_BERT_MODEL_NAME", None) if config else None
    ) or 'sentence-transformers/all-mpnet-base-v2'

    # CLI arguments: log path and output directory
    parser = argparse.ArgumentParser(description="ConceptUML Caldera log analysis")
    parser.add_argument(
        "--log-path",
        type=str,
        default=os.path.join(
            getattr(config, "DATA_DIR", "data"),
            "Caldera_Ability_Statistics",
            "4bfb5f265a5ce07af6bf10da113af7db_raw_events.csv",
        ),
        help="Path to a Caldera _raw_events.csv log file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(
            getattr(config, "DATA_DIR", "data"),
            "test_out",
        ),
        help="Directory to store per-log analysis CSVs",
    )
    args = parser.parse_args()

    # Path to Caldera logs
    log_path = args.log_path
    
    # External source paths
    # Three MITRE variants to compare, all under Logs Labeling/external_sources/data/reference_resources
    mitre_base_dir = (
        getattr(config, "REFERENCE_RESOURCES_DIR", None) if config else None
    ) or os.path.join(getattr(config, "DATA_DIR", "data"), "reference_resources")
    mitre_variants = {
        'orig': os.path.join(mitre_base_dir, 'MitreTechniquesTokens_V5.csv'),
        'with_code': os.path.join(mitre_base_dir, 'MitreTechniquesTokens_WithCode_V1.csv'),
        'with_db': os.path.join(mitre_base_dir, 'MitreTechniquesTokens_WithDB_V1.csv'),
    }

    # CAPEC path (shared across all experiments)
    capec_path = os.path.join(mitre_base_dir, 'CapecTokens_V5.csv')
    
    print("=" * 80)
    print("ConceptUML Pipeline - Caldera Log Analysis")
    print("=" * 80)
    
    # Load the logs
    print("\n[1] Loading Caldera logs...")
    df = pd.read_csv(log_path)
    print(f"    Loaded {len(df)} events")
    print(f"    Columns: {list(df.columns)}")
    
    # Show sample of key columns
    print("\n[2] Event Summary:")
    print(f"    Unique Processes: {df['Process Name'].unique().tolist()}")
    print(f"    Unique Operations: {df['Operation'].nunique()} types")
    print(f"    Event Classes: {df['Event Class'].unique().tolist()}")
    
    # The 'Content' column has the full event description
    text_column = 'Content'
    
    # Extract texts
    texts = df[text_column].fillna('').astype(str).tolist()
    
    # Initialize components
    print("\n[3] Initializing ConceptUML components...")
    print(f"    Using BERT model: {BERT_MODEL}")
    text_processor = TextProcessor(bert_model_name=BERT_MODEL)
    source_manager = ExternalSourceManager(bert_model=BERT_MODEL)
    
    # Load CAPEC once (MITRE techniques will be loaded per-variant later)
    print("\n[4] Loading CAPEC external source...")
    source_manager.load_source('capec', capec_path)
    print(f"    CAPEC patterns: {len(source_manager.sources['capec'])}")
    
    # Compute embeddings for CAPEC
    print("\n[4.5] Computing embeddings for CAPEC...")
    source_manager.compute_embeddings('capec')
    capec_df = source_manager.sources['capec']
    capec_embeddings = source_manager.embeddings['capec']
    
    # Step 1: Preprocess texts
    print("\n[5] Preprocessing texts...")
    processed_texts = [text_processor.clean_text(t) for t in texts]
    
    # Step 2: Get BERT embeddings
    print("\n[6] Computing BERT embeddings...")
    bert_embeddings = text_processor.generate_embeddings(texts)
    print(f"    BERT shape: {bert_embeddings.shape}")

    # Create base results DataFrame
    results_df = df.copy()

    # ------------------------------------------------------------------
    # CAPEC similarity (shared across all MITRE variants)
    # ------------------------------------------------------------------
    print("\n[6.5] Computing similarities to CAPEC (shared)...")
    per_row_capec_sim = cosine_similarity(bert_embeddings, capec_embeddings)

    capec_patterns = []
    capec_sims = []
    for i in range(len(df)):
        capec_best_idx = np.argmax(per_row_capec_sim[i])
        capec_best_sim = per_row_capec_sim[i][capec_best_idx]
        # Extract a short identifier from description or use index
        if 'Name' in capec_df.columns:
            capec_name = capec_df.iloc[capec_best_idx]['Name']
        else:
            desc = str(capec_df.iloc[capec_best_idx]['Description'])[:50]
            capec_name = f"CAPEC-{capec_best_idx}: {desc}..."

        capec_patterns.append(capec_name)
        capec_sims.append(capec_best_sim)

    results_df['capec_pattern'] = capec_patterns
    results_df['capec_similarity'] = capec_sims

    # ------------------------------------------------------------------
    # MITRE similarity / concepts for each variant
    # ------------------------------------------------------------------
    print("\n[7] Extracting concepts and similarities for each MITRE variant...")

    successful_variants = []

    for variant_key, mitre_path in mitre_variants.items():
        if not os.path.isfile(mitre_path):
            print(f"    [WARN] Skipping variant '{variant_key}': file not found: {mitre_path}")
            continue

        print(f"\n[7.{variant_key}] Loading MITRE variant '{variant_key}' from {mitre_path}...")
        source_manager.load_source('mitre', mitre_path)
        mitre_df = source_manager.sources['mitre']
        print(f"    MITRE techniques ({variant_key}): {len(mitre_df)}")

        # Compute embeddings for this MITRE variant
        source_manager.compute_embeddings('mitre')
        mitre_embeddings = source_manager.embeddings['mitre']

        # NMF concept extraction on stacked log + MITRE embeddings (Method 3)
        print("    Running NMF on log + MITRE embeddings...")
        n_logs = bert_embeddings.shape[0]
        all_embeddings = np.vstack([bert_embeddings, mitre_embeddings])

        all_min = all_embeddings.min()
        if all_min < 0:
            all_embeddings_nmf = all_embeddings - all_min
        else:
            all_embeddings_nmf = all_embeddings

        n_components = min(20, all_embeddings_nmf.shape[0] - 1) if all_embeddings_nmf.shape[0] > 1 else 1
        nmf_model = NMF(n_components=n_components, random_state=42, max_iter=500)
        all_H = nmf_model.fit_transform(all_embeddings_nmf)

        log_concepts = all_H[:n_logs]
        mitre_concepts = all_H[n_logs:]
        print(f"    Concepts (topics) from embeddings for {variant_key}: {n_components}")

        # Concept-space similarity between logs and MITRE techniques (Method 3)
        concept_mitre_sim = cosine_similarity(log_concepts, mitre_concepts)

        # Baseline BERT similarity (Method 1)
        per_row_mitre_sim = cosine_similarity(bert_embeddings, mitre_embeddings)

        # Helper to get top-N MITRE matches for this variant
        def get_top_mitre_matches(sim_row, n=3):
            top_indices = np.argsort(sim_row)[-n:][::-1]
            matches = []
            for idx in top_indices:
                tech_name = mitre_df.iloc[idx]['technique'] if 'technique' in mitre_df.columns else f"T{idx}"
                matches.append({'name': tech_name, 'sim': sim_row[idx], 'idx': idx})
            return matches

        mitre_techniques = []
        mitre_sims = []
        concept_techniques = []
        concept_scores = []

        for i in range(len(df)):
            # Top 3 MITRE matches (embedding space)
            top_mitre = get_top_mitre_matches(per_row_mitre_sim[i], n=3)
            mitre_best_idx = top_mitre[0]['idx']
            mitre_best_sim = top_mitre[0]['sim']
            mitre_name = top_mitre[0]['name']

            # Concept-based most likely MITRE technique (concept space)
            concept_row = concept_mitre_sim[i]
            best_concept_doc_idx = int(np.argmax(concept_row))
            best_concept_score = float(concept_row[best_concept_doc_idx])
            if 'technique' in mitre_df.columns:
                best_concept_technique = str(mitre_df.iloc[best_concept_doc_idx]['technique'])
            else:
                best_concept_technique = f"T{best_concept_doc_idx}"

            mitre_techniques.append(mitre_name)
            mitre_sims.append(mitre_best_sim)
            concept_techniques.append(best_concept_technique)
            concept_scores.append(best_concept_score)

        # Store variant-specific columns
        mitre_tech_col = f"mitre_technique_{variant_key}"
        mitre_sim_col = f"mitre_similarity_{variant_key}"
        concept_tech_col = f"concept_mitre_technique_{variant_key}"
        concept_sim_col = f"concept_mitre_score_{variant_key}"

        results_df[mitre_tech_col] = mitre_techniques
        results_df[mitre_sim_col] = mitre_sims
        results_df[concept_tech_col] = concept_techniques
        results_df[concept_sim_col] = concept_scores

        successful_variants.append(variant_key)

    # High-level technique distribution analysis per MITRE variant
    print("\n" + "=" * 80)
    print("TECHNIQUE DISTRIBUTION ANALYSIS (TOP-5 PER MITRE VARIANT)")
    print("=" * 80)

    if successful_variants:
        for key in successful_variants:
            base_col = f'mitre_technique_{key}'
            concept_col = f'concept_mitre_technique_{key}'

            # Skip if neither column exists for some reason
            if base_col not in results_df.columns and concept_col not in results_df.columns:
                continue

            print(f"\n[Variant: {key}] Top-5 techniques by assigned rows (BERT similarity):")
            print("-" * 80)
            if base_col in results_df.columns:
                base_counts = results_df[base_col].value_counts().head(5)
                for tech, cnt in base_counts.items():
                    print(f"  - {tech}: {cnt} rows")
            else:
                print("  (No BERT-based technique assignments available.)")

            print(f"\n[Variant: {key}] Top-5 techniques by assigned rows (Concept space):")
            print("-" * 80)
            if concept_col in results_df.columns:
                concept_counts = results_df[concept_col].value_counts().head(5)
                for tech, cnt in concept_counts.items():
                    print(f"  - {tech}: {cnt} rows")
            else:
                print("  (No concept-based technique assignments available.)")
    else:
        print("No MITRE variants available to analyze.")
    
    # Save detailed results
    # Derive output path from input log file name
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(log_path))[0]
    output_path = os.path.join(out_dir, f"{base_name}_analysis.csv")

    # Select columns to save
    save_cols = [
        'Process Name', 'PID', 'Operation', 'Path', 'Result', 
        'Event Class', 'Command Line', 
        'capec_pattern', 'capec_similarity'
    ]

    # Add per-variant MITRE columns (if present)
    for key in mitre_variants.keys():
        for col in [
            f'mitre_technique_{key}',
            f'mitre_similarity_{key}',
            f'concept_mitre_technique_{key}',
            f'concept_mitre_score_{key}',
        ]:
            if col in results_df.columns:
                save_cols.append(col)

    # Save with human-friendly column names
    # Build a rename map for variant columns to make the CSV easier to read
    variant_labels = {
        'orig': 'MITRE (orig)',
        'with_code': 'MITRE (+code)',
        'with_db': 'MITRE (+DB)'
    }

    rename_map = {
        'capec_pattern': 'CAPEC Pattern',
        'capec_similarity': 'CAPEC Similarity',
    }

    for key, label in variant_labels.items():
        mitre_tech_col = f'mitre_technique_{key}'
        mitre_sim_col = f'mitre_similarity_{key}'
        concept_tech_col = f'concept_mitre_technique_{key}'
        concept_sim_col = f'concept_mitre_score_{key}'

        if mitre_tech_col in results_df.columns:
            rename_map[mitre_tech_col] = f'{label} Technique'
        if mitre_sim_col in results_df.columns:
            rename_map[mitre_sim_col] = f'{label} Similarity'
        if concept_tech_col in results_df.columns:
            rename_map[concept_tech_col] = f'{label} (Concept) Technique'
        if concept_sim_col in results_df.columns:
            rename_map[concept_sim_col] = f'{label} (Concept) Score'

    save_df = results_df[save_cols].rename(columns=rename_map)
    save_df.to_csv(output_path, index=True)
    print(f"\n\n[18] Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
