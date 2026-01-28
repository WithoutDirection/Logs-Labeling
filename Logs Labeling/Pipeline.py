"""
Logs Labeling.Pipeline 的 Docstring
整合各個步驟的日誌標記流程

# * Per-Dataset 策略：每個 Dataset 獨立進行 NMF → HMM → Auto Labeling
# * 這確保每個 Technique 的標註不會被其他 Technique 混淆

Steps:
1. 預處理 (Preprocessing): Reffer to preprocess/preprocess.py
    配置參數: 
2. 異常檢測 (Anomaly Detection): Reffer to anomaly_detection/log_anomaly_detector.py
    配置參數:
3. 建立 MITRE Raw Embeddings: Reffer to external_sources/build_mitre_raw_embeddings.py
    配置參數: MITRE_TECHNIQUES_CSV, MITRE_EXTERNAL_KNOWLEDGE_DIR, BERT_MODEL_NAME
4. Per-Dataset 處理流程 (NMF → HMM → Auto Labeling):
    4a. 概念提取 (Concept Extraction): Reffer to conception_extraction.py
    4b. 序列分群 (Sequence Clustering): Reffer to sequence_clustering.py
    4c. 自動標註 (Auto Labeling): Reffer to auto_labeling.py
"""
import config
import os
import shutil
from typing import Optional, List, Dict
from utils.path import *
import numpy as np

def init():
    # * 0. 配置資料夾並清除先前實驗結果
    config.DATA_DIR = os.path.join("data")
    config.INPUT_LOGS_DIR = os.path.join(config.DATA_DIR, "input_logs")
    REFERENCE_RESOURCES_DIR = os.path.join(config.DATA_DIR, "reference_resources")
    # 清除 data 資料夾中除了 INPUT_LOGS_DIR 以外的所有檔案與資料夾
    if os.path.exists(config.DATA_DIR):
        input_logs_name = os.path.basename(config.INPUT_LOGS_DIR)
        ref_resources_name = os.path.basename(REFERENCE_RESOURCES_DIR)
        for item in os.listdir(config.DATA_DIR):
            item_path = os.path.join(config.DATA_DIR, item)
            if item != input_logs_name and item != ref_resources_name:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"已刪除資料夾: {item_path}")
                else:
                    os.remove(item_path)
                    print(f"已刪除檔案: {item_path}")

def STAGE_I(N: int, enable_comparison: bool = False):
    """
    Stage I: 日誌預處理與嵌入
    
    將原始日誌轉換為 BERT 嵌入向量：
    1. 載入日誌檔案（可選：解析模板）
    2. 計算 BERT 嵌入向量
    3. (可選) 執行模型比較分析
    
    Args:
        N: 要處理的資料集數量
        enable_comparison: 是否執行 BERT 模型比較（預設關閉，較耗時）
    """
    from preprocess import run_preprocessing
    
    # 設定預處理參數
    enable_parser = getattr(config, 'PREPROCESS_ENABLE_PARSER', False)
    model_name = getattr(config, 'BERT_MODEL_NAME', 'sentence-bert')
    
    print("=" * 60)
    print("STAGE I: 日誌預處理與嵌入")
    print("=" * 60)
    print(f"  資料集: {N} 個")
    print(f"  模型: {model_name}")
    print(f"  解析器: {'啟用' if enable_parser else '停用'}")
    print()
    
    results = run_preprocessing(
        n_datasets=N,
        enable_parser=enable_parser,
        model_name=model_name,
        normalize=False,
        enable_chunking=False,
    )
    
    print(f"\n[Stage I 完成]")
    print(f"  已處理: {results['n_loaded']} 個資料集")
    print(f"  嵌入維度: {results['embedding_dim']}")
    
    # 可選：BERT 模型比較分析
    if enable_comparison:
        print("\n--- 執行 BERT 模型比較分析 ---")
        from visualization.bert_comparison import BertEmbeddingComparator
        comparator = BertEmbeddingComparator(
            model_keys=['securebert', 'sentence-bert', 'bert-base-nli'],
            max_samples=1000,
        )
        comparator.run(n=N)
    
    return results
    
def STAGE_II():
    """
    Stage II: 異常檢測
    
    使用多模型整合方法檢測日誌異常：
    1. 載入 Log Vector
    2. 執行多模型異常檢測（IsolationForest, COPOD, AutoEncoder, PCA+GMM）
    3. 整合分數並生成視覺化報告
    """
    from anomaly_dection import run_detection
    
    # 取得配置
    models = getattr(config, "DETECTION_MODELS", ["isolation_forest", "copod", "autoencoder", "pca_gmm"])
    
    print("=" * 60)
    print("STAGE II: 異常檢測")
    print("=" * 60)
    print(f"  模型: {', '.join(models)}")
    print(f"  輸入: {config.LOG_VECTORS_DIR}")
    print(f"  輸出: {config.DETECTION_RESULTS_DIR}")
    print()
    
    result = run_detection(
        input_dir=config.LOG_VECTORS_DIR,
        output_dir=config.DETECTION_RESULTS_DIR,
        viz_dir=getattr(config, 'DETECTION_VIZ_DIR', None),
        models=models,
        generate_viz=True,
        verbose=True
    )
    
    print(f"\n[Stage II 完成]")
    print(f"  已處理: {result['n_datasets']} 個資料集")
    
    return result
    
def STAGE_III():
    """
    Stage III: 建立外部知識嵌入
    
    將 MITRE ATT&CK 技術描述轉換為嵌入向量，
    供後續概念提取與自動標註使用。
    """
    from external_sources import build_knowledge_base

    print("=" * 60)
    print("STAGE III: 建立外部知識嵌入")
    print("=" * 60)
    print(f"  知識來源: MITRE ATT&CK")
    print(f"  BERT 模型: {getattr(config, 'BERT_MODEL_NAME', 'sentence-bert')}")
    print()
    
    result = build_knowledge_base(force_rebuild=False, verbose=True)
    
    print(f"\n[Stage III 完成]")
    if result.get("cached"):
        print(f"  狀態: 使用快取")
    else:
        print(f"  狀態: 新建完成")
    print(f"  技術數量: {result['n_techniques']}")
    print(f"  嵌入維度: {result['embedding_dim']}")
    
    return result


def STAGE_IV():
    """
    Stage IV: Per-Dataset 處理流程
    
    對每個 Dataset 獨立執行 NMF → HMM → 自動標註：
    - 概念提取 (NMF)：與外部知識聯合訓練
    - 序列分群 (HMM)：基於概念向量進行時序分群
    - 自動標註：將分群結果與 MITRE 技術比對
    """
    from conception_extraction import ConceptExtractor
    from sequence_clustering import SequenceClustering
    from auto_labeling import AutoLabeler
    from utils.path import get_dirs, join_path, exists
    
    print("=" * 60)
    print("STAGE IV: Per-Dataset 處理 (NMF → HMM → 標註)")
    print("=" * 60)
    
    # 取得所有待處理的 Dataset
    log_vectors_dir = config.LOG_VECTORS_DIR
    if not exists(log_vectors_dir):
        print(f"[Error] 找不到目錄: {log_vectors_dir}")
        return {}
    
    all_dirs = list(get_dirs(log_vectors_dir))
    total = len(all_dirs)
    
    print(f"  資料集數量: {total}")
    print(f"  NMF 概念數: {config.NMF_COMPONENTS}")
    print()
    
    # 初始化共用組件
    extractor = ConceptExtractor(n_concepts=config.NMF_COMPONENTS)
    extractor.load_external_knowledge(config.EXTERNAL_KNOWLEDGE_DIR)
    
    clusterer = SequenceClustering()
    labeler = AutoLabeler()
    
    try:
        labeler.load_mitre_embeddings()
    except Exception as e:
        print(f"[Warning] MITRE 嵌入載入失敗: {e}")
    
    results = {}
    
    for idx, log_id_dir in enumerate(all_dirs, 1):
        dataset_id = log_id_dir.replace("_logvectors", "").replace("_embeddings", "")
        input_path = join_path(log_vectors_dir, log_id_dir)
        
        print(f"\n[{idx}/{total}] {dataset_id}")
        print("-" * 50)
        
        try:
            # Step 4a: NMF 概念提取
            extractor.model = None
            extractor._is_fitted = False
            
            concept_vectors = extractor.process_single_dataset(
                dataset_id=dataset_id,
                input_path=input_path,
                output_dir=config.CONCEPT_VECTORS_DIR,
                external_knowledge_dir=config.EXTERNAL_KNOWLEDGE_DIR,
            )
            
            # Step 4b: HMM 序列分群
            cluster_labels = clusterer.process_single_dataset(
                dataset_id=dataset_id,
                concept_matrix=concept_vectors,
                output_dir=config.CLUSTER_RESULTS_DIR,
            )
            
            # Step 4c: 自動標註
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
    
    # 處理摘要
    print(f"\n[Stage IV 完成]")
    print(f"  成功: {len(results)}/{total} 個資料集")
    
    if results:
        avg_clusters = np.mean([
            len(np.unique(r["cluster_labels"])) 
            for r in results.values()
        ])
        print(f"  平均群集數: {avg_clusters:.1f}")
    
    return results


def main(n_datasets: int = 5):
    """
    主程式入口
    
    執行完整的日誌標註流程：
    - Stage I:  日誌預處理與 BERT 嵌入
    - Stage II: 異常檢測（多模型整合）
    - Stage III: 建立 MITRE 外部知識嵌入
    - Stage IV: Per-Dataset 處理 (NMF → HMM → 自動標註)
    
    Args:
        n_datasets: 要處理的資料集數量（預設 5）
    """
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " Logs Labeling Pipeline ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    init()
    
    STAGE_I(n_datasets)
    STAGE_II()
    STAGE_III()
    STAGE_IV()
    
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " ✓ 全部流程完成 ".center(56) + "║")
    print("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main(167) 