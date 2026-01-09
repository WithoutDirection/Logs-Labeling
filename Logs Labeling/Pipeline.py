"""
Logs Labeling.Pipeline 的 Docstring
整合各個步驟的日誌標記流程
Steps:
1. 預處理 (Preprocessing): Reffer to preprocess/preprocess.py
    配置參數: 
2. 異常檢測 (Anomaly Detection): Reffer to anomaly_detection/log_anomaly_detector.py
    配置參數:
3. 概念提取 (Concept Extraction): Reffer to conception_extraction.py
    配置參數:
4. 序列區塊化 (Sequence Clustering): Reffer to sequence_clustering.py
    配置參數:
5. 建立 MITRE Raw Embeddings: Reffer to external_sources/build_mitre_raw_embeddings.py
    配置參數: MITRE_TECHNIQUES_CSV, MITRE_EXTERNAL_KNOWLEDGE_DIR, BERT_MODEL_NAME
6. 自動標註 (Auto Labeling): Reffer to auto_labeling.py
    配置參數: LABELING_SIMILARITY_THRESHOLD, LABELING_CONFIDENCE_THRESHOLD, 
              LABELING_ANOMALY_WEIGHT, LABELING_SIMILARITY_WEIGHT, LABELING_TOP_K
"""
import config
import os
import shutil
from utils.path import *
import numpy as np

def init():
    # * 0. 配置資料夾並清除先前實驗結果
    config.DATA_DIR = os.path.join("data")
    config.INPUT_LOGS_DIR = os.path.join(config.DATA_DIR, "input_logs")
    # 清除 data 資料夾中除了 INPUT_LOGS_DIR 以外的所有檔案與資料夾
    if os.path.exists(config.DATA_DIR):
        input_logs_name = os.path.basename(config.INPUT_LOGS_DIR)
        for item in os.listdir(config.DATA_DIR):
            item_path = os.path.join(config.DATA_DIR, item)
            if item != input_logs_name:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"已刪除資料夾: {item_path}")
                else:
                    os.remove(item_path)
                    print(f"已刪除檔案: {item_path}")

def STAGE_I(N:int ):
    
     # * 1. 配置預處理參數，並執行預處理
    from preprocess.preprocess import LogLoader, LogEmbedder, LogChunker
    # * 1.1 載入日誌
    config.PREPROCESS_ENABLE_PARSER = False
    loader = LogLoader(enable_parser=config.PREPROCESS_ENABLE_PARSER)
    loader.load_logs(num=N) # 只載入前15 Datasets
    
    # * 1.2 日誌嵌入
    config.BERT_MODEL_NAME = "sentence-bert"
    embedder = LogEmbedder(model_name=config.BERT_MODEL_NAME, normalize=False)
    embedder.embed_logs()
    
    # # * 1.3 日誌區塊化
    # chnuker = LogChunker(
    #     window_size=config.SEQUENCE_WINDOW_SIZE,
    #     stride=config.SEQUENCE_STRIDE,
    #     bilstm_hidden_size=config.BILSTM_HIDDEN_SIZE,
    #     bilstm_num_layers=config.BILSTM_NUM_LAYERS,
    #     bilstm_dropout=config.BILSTM_DROPOUT,
    #     fusion_enable=config.FUSION_ENABLE,
    #     fusion_output_dim=config.FUSION_OUTPUT_DIM,
    # )
    # chnuker.chunk_logs()

    # # * Result of Stage 1
    from visualization.bert_comparison import BertEmbeddingComparator
    comparator = BertEmbeddingComparator(
        model_keys=['securebert', 'sentence-bert', 'bert-base-nli'],
        max_samples=1000,
    )
    comparator.run(n=N)
    
def STAGE_II():
    # * 2. 執行異常檢測（批次模式：一次載入所有數據、合併訓練、整體視覺化）
    from anomaly_dection.log_detector import run_detection_pipeline, generate_detection_summary
    results = run_detection_pipeline(
        input_dir=config.LOG_VECTORS_DIR,
        output_dir=config.DETECTION_RESULTS_DIR,
        models=getattr(config, "DETECTION_MODELS", None),
        verbose=True
    )
    if results:
        # 生成整體視覺化與計算相關結果
        generate_detection_summary(
            results, 
            output_dir=config.DETECTION_VIZ_DIR,
            generate_visualizations=True,
            enable_advanced_plots=True
        )
    
    # * 2.1 載入異常分數作為後續標註階段的權重
    # 高異常分數的 log vectors 在後續標註階段會被賦予較高的權重
    # 這些分數會在 STAGE_VI 中被 AutoLabeler 載入並使用
    from auto_labeling import load_anomaly_weights
    anomaly_weights = load_anomaly_weights(config.DETECTION_RESULTS_DIR)
    print(f"[Info] 已準備 {len(anomaly_weights)} 個資料集的異常分數權重供後續標註使用")
    
def STAGE_III():
     # * 3. 執行概念提取
    from conception_extraction import ConceptExtractor, train_concept_extractor, transform_all_datasets
    extractor = ConceptExtractor()
    # prepare_training_data 目前不接受 num_datasets，直接使用預設來源與抽樣比例
    X_train = extractor.prepare_training_data()
    extractor.fit_global_model(X_train)
    extractor.save_model(config.NMF_MODEL_PATH)
    transform_all_datasets(model_path=config.NMF_MODEL_PATH)
    
    # * 3.1 概念提取成果
    from visualization.conception_extraction_viz import ConceptVisualization
    viz = ConceptVisualization(output_dir=os.path.join(config.RESULT_DIR, "conception_sequence_clustering"))
    viz.run_multi_dataset(n_datasets=5)
    
    

def STAGE_IV():
    # * 4. 執行序列區塊化
    from sequence_clustering import SequenceClustering, load_concept_vectors
    print("=" * 60)
    print("序列分群 - Per-Dataset HMM 策略")
    print("=" * 60)
    
    vectors = load_concept_vectors()
    
    
    
    # ===== 批次處理所有資料集 =====
    clusterer = SequenceClustering()
    results = clusterer.batch_process_all(vectors)
    
    if results:
        avg_clusters = np.mean([len(np.unique(labels)) for labels in results.values()])
        print(f"平均群集數: {avg_clusters:.2f}")
    
    print("\n[完成] 序列分群已完成。")


def STAGE_V():
    """Step 5: Build / ensure MITRE raw embeddings dataset exists."""
    from external_sources.build_mitre_raw_embeddings import build_mitre_raw_embeddings

    print("=" * 60)
    print("建立 MITRE Raw Embeddings")
    print("=" * 60)
    out_dir = build_mitre_raw_embeddings(
        mitre_csv=getattr(config, "MITRE_TECHNIQUES_CSV", None),
        out_dir=getattr(config, "MITRE_EXTERNAL_KNOWLEDGE_DIR", None),
        bert_model=getattr(config, "BERT_MODEL_NAME", None),
        force_rebuild=False,
    )
    print(f"\n[完成] MITRE raw embeddings 準備完成: {out_dir}")


def STAGE_VI():
    """
    Step 6: 自動標註階段
    
    將 HMM 分群結果與 MITRE ATT&CK 外部知識進行比對，
    自動標註每筆日誌對應的攻擊技術。
    
    流程:
    1. 載入 NMF 模型、概念向量、分群標籤、異常分數
    2. 載入 MITRE ATT&CK 嵌入並轉換至概念空間
    3. 計算各 Cluster 的 Centroid 與 MITRE 向量的相似度
    4. 根據相似度與異常分數產生最終標註
    5. 輸出標註結果至 CSV
    """
    from auto_labeling import AutoLabeler, run_auto_labeling
    
    print("=" * 60)
    print("自動標註 - MITRE ATT&CK 技術比對")
    print("=" * 60)
    
    # 執行自動標註流程
    results = run_auto_labeling(
        output_dir=config.LABELING_RESULTS_DIR,
    )
    
    if results:
        total_samples = sum(len(df) for df in results.values())
        print(f"\n[完成] 已標註 {len(results)} 個資料集，共 {total_samples} 筆日誌。")
    else:
        print("\n[Warning] 無標註結果產生，請檢查前置步驟是否完成。")
    
    print("\n[完成] 自動標註已完成。")


def main():
    
    #init()
    N = 50
    # STAGE_I(N)
    # STAGE_II()
    # STAGE_III()
    # STAGE_IV()
    STAGE_V()
    STAGE_VI()
    
   
    
    
    
    

if __name__ == "__main__":
    main()