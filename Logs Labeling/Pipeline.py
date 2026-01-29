"""
Logs Labeling Pipeline
=======================

完整的日誌自動標註流水線，基於 Per-Dataset 策略處理。

設計理念:
    每個 Dataset 獨立進行 NMF → HMM → Auto Labeling，
    確保每個 Technique 的標註不會被其他 Dataset 混淆。

流水線階段:
    Stage I:   日誌預處理與嵌入（BERT + Per-Log TF-IDF）
    Stage II:  異常檢測（多模型整合）
    Stage III: 建立 MITRE 外部知識嵌入
    Stage IV:  Per-Dataset 處理 (NMF → HMM → 自動標註)

模組依賴:
    - preprocess/: 日誌預處理與 BERT 嵌入
    - precompute_log_tfidf.py: Per-Log TF-IDF 預計算
    - anomaly_dection/: 異常檢測模組
    - external_sources/: MITRE 外部知識建構
    - conception_extraction.py: NMF 概念提取
    - sequence_clustering.py: HMM 序列分群
    - auto_labeling.py: 自動標註與混合評分

配置檔:
    config.py: 集中管理所有路徑與超參數

Usage:
    python Pipeline.py              # 預設處理 10 個資料集
    python Pipeline.py --n 5        # 處理 5 個資料集
    python Pipeline.py --skip-tfidf # 跳過 TF-IDF 預計算
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
    """
    初始化工作空間
    
    清除 data 資料夾中的實驗結果，保留：
    - input_logs/: 原始日誌
    - ExternalKnowledge/: MITRE 外部知識
    - reference_resources/: 參考資源
    - groundtruth/: 標註資料
    """
    if os.path.exists(config.DATA_DIR):
        PRESERVED_ITEMS = {
            os.path.basename(config.INPUT_LOGS_DIR),
            "ExternalKnowledge",
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
# Stage I: 日誌預處理與嵌入（BERT + TF-IDF）
# =============================================================================

def STAGE_I(N: int, enable_tfidf: bool = True, enable_comparison: bool = False):
    """
    Stage I: 日誌預處理與嵌入
    
    將原始日誌轉換為向量表示：
    1. 載入日誌檔案（可選：解析模板）
    2. 計算 BERT 嵌入向量
    3. (可選) 計算 Per-Log TF-IDF 向量（用於 Stage IV 混合評分）
    4. (可選) 執行 BERT 模型比較分析
    
    Args:
        N: 要處理的資料集數量
        enable_tfidf: 是否執行 Per-Log TF-IDF 預計算（預設 True）
        enable_comparison: 是否執行 BERT 模型比較（預設 False，較耗時）
        
    Returns:
        dict: 包含 n_loaded, embedding_dim, tfidf_stats 等結果
    """
    from preprocess import run_preprocessing
    
    enable_parser = getattr(config, 'PREPROCESS_ENABLE_PARSER', False)
    model_name = getattr(config, 'BERT_MODEL_NAME', 'sentence-bert')
    
    print("=" * 60)
    print("STAGE I: 日誌預處理與嵌入")
    print("=" * 60)
    print(f"  資料集: {N} 個 | BERT: {model_name} | TF-IDF: {'啟用' if enable_tfidf else '停用'}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: BERT 嵌入
    # -------------------------------------------------------------------------
    results = run_preprocessing(
        n_datasets=N,
        enable_parser=enable_parser,
        model_name=model_name,
        normalize=False,
        enable_chunking=False,
    )
    
    print(f"  [BERT] 已處理: {results['n_loaded']} 個資料集, 維度: {results['embedding_dim']}")
    
    # -------------------------------------------------------------------------
    # Step 2: Per-Log TF-IDF 預計算（可選）
    # -------------------------------------------------------------------------
    tfidf_stats = None
    if enable_tfidf:
        from precompute_log_tfidf import run_log_tfidf_precompute
        tfidf_stats = run_log_tfidf_precompute(force_rebuild=False, verbose=False)
        
        if tfidf_stats.get("enabled", False):
            print(f"  [TF-IDF] 成功: {tfidf_stats['success']} | 快取: {tfidf_stats['skipped']} | 失敗: {tfidf_stats['failed']}")
        else:
            print(f"  [TF-IDF] 跳過（Vectorizer 未找到）")
    
    results['tfidf_stats'] = tfidf_stats
    
    # -------------------------------------------------------------------------
    # Step 3: BERT 模型比較（可選）
    # -------------------------------------------------------------------------
    if enable_comparison:
        print("\n--- 執行 BERT 模型比較分析 ---")
        from visualization.bert_comparison import BertEmbeddingComparator
        BertEmbeddingComparator(
            model_keys=['securebert', 'sentence-bert', 'bert-base-nli'],
            max_samples=1000,
        ).run(n=N)
    
    print(f"\n[Stage I 完成] {results['n_loaded']} 個資料集")
    return results


# =============================================================================
# Stage II: 異常檢測
# =============================================================================

def STAGE_II():
    """
    Stage II: 異常檢測
    
    使用多模型整合方法檢測日誌異常：
    1. 載入 Log Vector
    2. 執行多模型異常檢測（IsolationForest, COPOD, AutoEncoder, PCA+GMM）
    3. 整合分數並生成視覺化報告
    
    Returns:
        dict: 包含 n_datasets 等結果
    """
    from anomaly_dection import run_detection
    
    models = getattr(config, "DETECTION_MODELS", ["isolation_forest", "copod", "autoencoder", "pca_gmm"])
    
    print("=" * 60)
    print("STAGE II: 異常檢測")
    print("=" * 60)
    print(f"  模型: {', '.join(models)}")
    print()
    
    result = run_detection(
        input_dir=config.LOG_VECTORS_DIR,
        output_dir=config.DETECTION_RESULTS_DIR,
        viz_dir=getattr(config, 'DETECTION_VIZ_DIR', None),
        models=models,
        generate_viz=True,
        verbose=True
    )
    
    print(f"\n[Stage II 完成] 已處理: {result['n_datasets']} 個資料集")
    return result


# =============================================================================
# Stage III: 建立外部知識嵌入
# =============================================================================

def STAGE_III():
    """
    Stage III: 建立外部知識嵌入
    
    將 MITRE ATT&CK 技術描述轉換為嵌入向量，
    供後續概念提取與自動標註使用。
    
    Returns:
        dict: 包含 n_techniques, embedding_dim, cached 等結果
    """
    from external_sources import build_knowledge_base

    print("=" * 60)
    print("STAGE III: 建立外部知識嵌入")
    print("=" * 60)
    print(f"  知識來源: MITRE ATT&CK | BERT 模型: {getattr(config, 'BERT_MODEL_NAME', 'sentence-bert')}")
    print()
    
    result = build_knowledge_base(force_rebuild=False, verbose=True)
    
    status = "使用快取" if result.get("cached") else "新建完成"
    print(f"\n[Stage III 完成] {status} | 技術數量: {result['n_techniques']} | 嵌入維度: {result['embedding_dim']}")
    return result


# =============================================================================
# Stage IV: Per-Dataset 處理 (NMF → HMM → 自動標註)
# =============================================================================

def STAGE_IV():
    """
    Stage IV: Per-Dataset 處理 (NMF → HMM → 自動標註)
    
    對每個資料集獨立執行：
    1. NMF 概念提取：將嵌入向量降維至概念空間
    2. HMM 序列分群：使用 BIC 準則自動選擇最佳 K 值
    3. 自動標註：混合 Embedding + TF-IDF 評分匹配 MITRE 技術
    
    Returns:
        dict: {dataset_id: {concept_vectors, cluster_labels, labeling_result}}
    """
    from conception_extraction import ConceptExtractor
    from sequence_clustering import SequenceClustering
    from auto_labeling import AutoLabeler
    from utils.path import get_dirs, join_path, exists
    
    print("=" * 60)
    print("STAGE IV: Per-Dataset 處理 (NMF → HMM → 標註)")
    print("=" * 60)
    
    log_vectors_dir = config.LOG_VECTORS_DIR
    if not exists(log_vectors_dir):
        print(f"[Error] 找不到目錄: {log_vectors_dir}")
        return {}
    
    all_dirs = list(get_dirs(log_vectors_dir))
    total = len(all_dirs)
    print(f"  資料集數量: {total} | NMF 概念數: {config.NMF_COMPONENTS}")
    print()
    
    # -------------------------------------------------------------------------
    # 初始化共用組件
    # -------------------------------------------------------------------------
    extractor = ConceptExtractor(n_concepts=config.NMF_COMPONENTS)
    extractor.load_external_knowledge(config.EXTERNAL_KNOWLEDGE_DIR)
    clusterer = SequenceClustering()
    labeler = AutoLabeler()
    
    # 載入 MITRE 嵌入與 TF-IDF（用於混合評分）
    try:
        labeler.load_mitre_embeddings()
        labeler.load_mitre_tfidf()
    except Exception as e:
        print(f"[Warning] MITRE 資料載入失敗: {e}")
    
    results = {}
    
    # -------------------------------------------------------------------------
    # 遍歷每個資料集
    # -------------------------------------------------------------------------
    for idx, log_id_dir in enumerate(all_dirs, 1):
        dataset_id = log_id_dir.replace("_logvectors", "").replace("_embeddings", "")
        input_path = join_path(log_vectors_dir, log_id_dir)
        
        print(f"\n[{idx}/{total}] {dataset_id}")
        print("-" * 50)
        
        try:
            # -----------------------------------------------------------------
            # Step 4a: NMF 概念提取
            # -----------------------------------------------------------------
            extractor.model = None
            extractor._is_fitted = False
            concept_vectors = extractor.process_single_dataset(
                dataset_id=dataset_id,
                input_path=input_path,
                output_dir=config.CONCEPT_VECTORS_DIR,
                external_knowledge_dir=config.EXTERNAL_KNOWLEDGE_DIR,
            )
            
            # -----------------------------------------------------------------
            # Step 4b: HMM 序列分群
            # -----------------------------------------------------------------
            cluster_labels = clusterer.process_single_dataset(
                dataset_id=dataset_id,
                concept_matrix=concept_vectors,
                output_dir=config.CLUSTER_RESULTS_DIR,
            )
            
            # -----------------------------------------------------------------
            # Step 4c: 自動標註（Embedding + TF-IDF 混合評分）
            # -----------------------------------------------------------------
            labeling_result = labeler.process_single_dataset(
                dataset_id=dataset_id,
                concept_vectors=concept_vectors,
                cluster_labels=cluster_labels,
                output_dir=config.LABELING_RESULTS_DIR,
                nmf_extractor=extractor,
            )
            
            results[dataset_id] = {
                "concept_vectors": concept_vectors,
                "cluster_labels": cluster_labels,
                "labeling_result": labeling_result,
            }
            
            n_clusters = len(np.unique(cluster_labels))
            print(f"  ✓ {len(concept_vectors)} 筆日誌 → {n_clusters} 個群集")
            
        except Exception as e:
            print(f"  ✗ 失敗: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # -------------------------------------------------------------------------
    # 結果摘要
    # -------------------------------------------------------------------------
    print(f"\n[Stage IV 完成] 成功: {len(results)}/{total} 個資料集")
    if results:
        avg_clusters = np.mean([len(np.unique(r["cluster_labels"])) for r in results.values()])
        print(f"  平均群集數: {avg_clusters:.1f}")
    
    return results


# =============================================================================
# 主程式入口
# =============================================================================

def main(n_datasets: int = 5, enable_tfidf: bool = True):
    """
    主程式入口
    
    執行完整的日誌標註流程：
    - Stage I:   日誌預處理與嵌入（BERT + TF-IDF）
    - Stage II:  異常檢測（多模型整合）
    - Stage III: 建立 MITRE 外部知識嵌入
    - Stage IV:  Per-Dataset 處理 (NMF → HMM → 自動標註)
    
    Args:
        n_datasets: 要處理的資料集數量（預設 5）
        enable_tfidf: 是否執行 Per-Log TF-IDF 預計算（預設 True）
    """
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " Logs Labeling Pipeline ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # init()  # 取消註解以清除先前實驗結果
    
    # -------------------------------------------------------------------------
    # 執行各階段
    # -------------------------------------------------------------------------
    STAGE_I(n_datasets, enable_tfidf=enable_tfidf)
    STAGE_II()
    STAGE_III()
    STAGE_IV()
    
    # -------------------------------------------------------------------------
    # 完成提示
    # -------------------------------------------------------------------------
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " ✓ 全部流程完成 ".center(56) + "║")
    print("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Logs Labeling Pipeline")
    parser.add_argument("-n", "--n-datasets", type=int, default=10,
                        help="要處理的資料集數量（預設: 10）")
    parser.add_argument("--skip-tfidf", action="store_true",
                        help="跳過 Per-Log TF-IDF 預計算")
    parser.add_argument("--init", action="store_true",
                        help="清除先前實驗結果")
    
    args = parser.parse_args()
    
    if args.init:
        init()
    
    main(n_datasets=args.n_datasets, enable_tfidf=not args.skip_tfidf)
