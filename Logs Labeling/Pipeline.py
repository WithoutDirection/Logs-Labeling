"""
Logs Labeling Pipeline
=======================

完整的日誌自動標註流水線 (Refactored)

流水線階段:
    Stage I:   全量輸入處理 (Process All Inputs)
               - Log Datasets: Parse -> BERT Embedding -> Chunking -> Log Vectors
               - Reference Sources: BERT Embedding + TF-IDF Fingerprinting
               - Log TF-IDF Transformation
    Stage II:  異常檢測 (Anomaly Detection)
    Stage III: Per-Dataset 處理 (NMF -> HMM -> Auto Labeling) (原 Stage IV)

模組依賴:
    - preprocess/: 統一輸入處理模組 (含 Log 與 Reference)
    - anomaly_dection/: 異常檢測模組
    - conception_extraction: NMF 概念提取
    - sequence_clustering: HMM 序列分群
    - auto_labeling: 自動標註與混合評分

配置檔:
    config.py: 集中管理所有路徑與超參數

Usage:
    python Pipeline.py              # 預設處理所有設定的資料集
    python Pipeline.py --n 5        # 處理 5 個資料集
"""

import config
import os
import shutil
import numpy as np
from utils.path import *


# =============================================================================
# 初始化函數
# =============================================================================

def init():
    """初始化工作空間，清除舊的實驗數據"""
    if os.path.exists(config.DATA_DIR):
        PRESERVED_ITEMS = {
            os.path.basename(config.INPUT_LOGS_DIR),
            "reference_resources",
            "groundtruth"
        }
        
        for item in os.listdir(config.DATA_DIR):
            if item in PRESERVED_ITEMS:
                continue
            item_path = os.path.join(config.DATA_DIR, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
                print(f"已刪除: {item_path}")
            except Exception as e:
                print(f"刪除失敗 {item_path}: {e}")


# =============================================================================
# Stage I: 全量輸入處理 (Logs + References)
# =============================================================================

def STAGE_I(N: int, enable_tfidf: bool = True):
    """
    Stage I: 統一處理所有輸入源
    
    1. Log Datasets 處理 (Parse, Embedding, Chunking)
    2. Reference Sources 處理 (TF-IDF Fingerprint, Embedding)
    3. Log TF-IDF 轉換
    
    Args:
        N: 要處理的資料集數量
        enable_tfidf: 是否執行 TF-IDF 相關計算
    """
    from preprocess import process_all_inputs
    
    enable_parser = getattr(config, 'PREPROCESS_ENABLE_PARSER', False)
    model_name = getattr(config, 'BERT_MODEL_NAME', 'securebert2')
    
    print("=" * 60)
    print("STAGE I: 輸入資料處理 (Logs & References)")
    print("=" * 60)
    print(f"  資料集: {N if N else 'All'} | BERT: {model_name} | TF-IDF: {'On' if enable_tfidf else 'Off'}")
    
    results = process_all_inputs(
        n_datasets=N,
        enable_parser=enable_parser,
        model_name=model_name,
        enable_chunking=False, # 根據原 Pipeline 設為 False，視需求調整
        enable_tfidf=enable_tfidf,
        verbose=True
    )
    
    print(f"\n[Stage I 完成] 已處理 Log: {results.get('n_loaded', 0)} 個 | Ref Embedding: {'OK' if 'reference_embedding_path' in results else 'Skip'}")
    return results


# =============================================================================
# Stage II: 異常檢測
# =============================================================================

def STAGE_II():
    """
    Stage II: 異常檢測 (多模型整合)
    """
    from anomaly_dection import run_detection
    
    models = getattr(config, "DETECTION_MODELS", ["isolation_forest", "copod", "autoencoder", "pca_gmm"])
    
    print("\n" + "=" * 60)
    print("STAGE II: 異常檢測")
    print("=" * 60)
    
    result = run_detection(
        input_dir=config.LOG_VECTORS_DIR,
        output_dir=config.DETECTION_RESULTS_DIR,
        viz_dir=getattr(config, 'DETECTION_VIZ_DIR', None),
        models=models,
        generate_viz=True,
        verbose=True
    )
    
    print(f"\n[Stage II 完成] 已檢測: {result['n_datasets']} 個資料集")
    return result


# =============================================================================
# Stage III: Per-Dataset 處理
# =============================================================================

def STAGE_III():
    """
    Stage III: Per-Dataset 處理 (NMF -> HMM -> 自動標註)
    """
    from conception_extraction import ConceptExtractor
    from sequence_clustering import SequenceClustering
    from auto_labeling import AutoLabeler
    from utils.path import get_dirs, join_path, exists
    
    print("\n" + "=" * 60)
    print("STAGE III: Per-Dataset 處理 (NMF → HMM → Auto Labeling)")
    print("=" * 60)
    
    log_vectors_dir = config.LOG_VECTORS_DIR
    if not exists(log_vectors_dir):
        print(f"[Error] Log Vectors 目錄不存在: {log_vectors_dir}")
        return {}
    
    all_dirs = list(get_dirs(log_vectors_dir))
    total = len(all_dirs)
    print(f"  待處理資料集: {total} | NMF Concepts: {config.NMF_COMPONENTS}")
    
    # 初始化組件
    extractor = ConceptExtractor(n_concepts=config.NMF_COMPONENTS)
    extractor.load_external_knowledge(config.EXTERNAL_KNOWLEDGE_DIR)
    clusterer = SequenceClustering()
    labeler = AutoLabeler()
    
    # 載入 MITRE 嵌入與 TF-IDF
    try:
        labeler.load_mitre_embeddings()
        labeler.load_mitre_tfidf()
        print("  ✓ 已載入 MITRE 知識庫 (Embedding + TF-IDF)")
    except Exception as e:
        print(f"[Warning] MITRE 資料載入不完整: {e}")
    
    results = {}
    
    # 遍歷處理
    for idx, log_id_dir in enumerate(all_dirs, 1):
        dataset_id = log_id_dir.replace("_logvectors", "").replace("_embeddings", "")
        input_path = join_path(log_vectors_dir, log_id_dir)
        
        print(f"\n[{idx}/{total}] {dataset_id}")
        print("-" * 40)
        
        try:
            # 1. NMF 概念提取
            extractor.model = None; extractor._is_fitted = False
            concept_vectors = extractor.process_single_dataset(
                dataset_id=dataset_id,
                input_path=input_path,
                output_dir=config.CONCEPT_VECTORS_DIR,
                external_knowledge_dir=config.EXTERNAL_KNOWLEDGE_DIR,
            )
            
            # 2. HMM 序列分群
            cluster_labels = clusterer.process_single_dataset(
                dataset_id=dataset_id,
                concept_matrix=concept_vectors,
                output_dir=config.CLUSTER_RESULTS_DIR,
            )
            
            # 3. 自動標註
            labeling_result = labeler.process_single_dataset(
                dataset_id=dataset_id,
                concept_vectors=concept_vectors,
                cluster_labels=cluster_labels,
                output_dir=config.LABELING_RESULTS_DIR,
                nmf_extractor=extractor,
                log_vectors_path=input_path,
            )
            
            results[dataset_id] = True
            print(f"  ✓ 完成 ( clusters={len(np.unique(cluster_labels))} )")
            
        except Exception as e:
            print(f"  ✗ 失敗: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    print(f"\n[Stage III 完成] 成功處理: {len(results)}/{total}")
    return results


# =============================================================================
# 主程式
# =============================================================================

def main(n_datasets: int = 5, enable_tfidf: bool = True):
    print("\n╔" + "═" * 50 + "╗")
    print("║" + " Logs Labeling Pipeline (Refactored) ".center(50) + "║")
    print("╚" + "═" * 50 + "╝")
    
    STAGE_I(n_datasets, enable_tfidf=enable_tfidf)
    STAGE_II()
    STAGE_III() # 原 Stage IV
    
    print("\n>>> Pipeline 執行完畢 <<<")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-datasets", type=int, default=10, help="處理資料集數量")
    parser.add_argument("--skip-tfidf", action="store_true", help="跳過 TF-IDF 計算")

    
    args = parser.parse_args()
    
    
    init()
        
    main(n_datasets=args.n_datasets, enable_tfidf=not args.skip_tfidf)
